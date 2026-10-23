"""
AI Studio data setup workflow.

Reads an upload, picks what a pipeline import brings over, and turns the
sidebar's choices (sample columns, conditions, exclusions, BQC) into the
renamed frame and the ExperimentConfig the data tabs work on. Pure logic —
no Streamlit dependencies.
"""
import io
from typing import Optional, Tuple

import pandas as pd

from ..models.experiment import ExperimentConfig
from ..services.format_detection import DataFormat, FormatDetectionService


class AIStudioWorkflow:
    """The AI Studio's data setup, as static methods."""

    # Standard formats that the main pipeline already handles end-to-end
    # (cleaning, zero filtering, normalization). When users upload these
    # directly to AI Studio, we redirect them rather than letting them try
    # to pick sample columns from raw, unfiltered data.
    BLOCKED_FORMATS: frozenset = frozenset({
        DataFormat.LIPIDSEARCH.value,
        DataFormat.MSDIAL.value,
        DataFormat.METABOLOMICS_WORKBENCH.value,
    })

    @staticmethod
    def is_blocked_standard_format(
        detected_format: Optional[str], pipeline_source: Optional[str],
    ) -> bool:
        """True when the uploaded file is a known standard format that should
        go through the main pipeline first. Pipeline imports are exempt: the
        user has already explicitly chosen to bring that data over."""
        if pipeline_source is not None:
            return False
        return detected_format in AIStudioWorkflow.BLOCKED_FORMATS

    @staticmethod
    def read_upload(data: bytes) -> Tuple[pd.DataFrame, Optional[str]]:
        """Read an uploaded CSV: (frame, detected format's value), the format
        None for an empty frame. Raises whatever pandas raises on a file it
        cannot read."""
        # Metabolomics Workbench files are multi-section text documents whose
        # real table only the main pipeline's MW parser can recover.
        # detect_format() recognizes them by scanning the raw text for the
        # MS_METABOLITE_DATA markers — it cannot see them from a DataFrame
        # (a plain read_csv yields a garbage "Unnamed" frame). Detect from the
        # text first so an MW upload is flagged as MW (and thus blocked) rather
        # than mis-read as a table the user is asked to hand-pick columns from.
        text = data.decode("utf-8", errors="replace")
        text_format = FormatDetectionService.detect_format(text)
        # LipidSearch 5.2 exports are tab-delimited with a .csv extension; sniff
        # the delimiter so they don't collapse into a single column.
        delimiter = FormatDetectionService.sniff_delimiter(text)
        df = pd.read_csv(io.BytesIO(data), sep=delimiter)
        if df.empty:
            return df, None
        detected = (
            text_format
            if text_format == DataFormat.METABOLOMICS_WORKBENCH
            else FormatDetectionService.detect_format(df)
        )
        return df, detected.value

    @staticmethod
    def raw_import_frame(
        processing_df: Optional[pd.DataFrame],
        standardized_df: Optional[pd.DataFrame],
        raw_df: Optional[pd.DataFrame],
    ) -> Optional[pd.DataFrame]:
        """The frame the "Raw" pipeline import brings over.

        `standardized_df` over `raw_df`: both contain the same unfiltered,
        un-normalized values, but the standardized version has the column
        rename applied (e.g. LipidSearch `MeanArea[s1]` → `intensity[s1]`),
        which lets AI Studio's auto-detection + experiment pre-populate run
        end-to-end. `raw_df` only if standardization hasn't happened yet.
        Once inputs are confirmed, the processing frame comes first: the
        pipeline's experiment describes it, not `standardized_df`, when
        samples were excluded in the sidebar.
        """
        return next(
            (df for df in (processing_df, standardized_df, raw_df)
             if df is not None),
            None,
        )

    @staticmethod
    def import_setup(
        source_key: str,
        experiment: Optional[ExperimentConfig],
        bqc_label: Optional[str],
        handoff_experiment: Optional[ExperimentConfig],
        handoff_bqc_label: Optional[str],
        qc_experiment: Optional[ExperimentConfig],
    ) -> Tuple[Optional[ExperimentConfig], Optional[str]]:
        """The experiment and BQC label a pipeline import brings over.

        Args:
            source_key: 'qc_filtered', 'normalized' or 'raw'.
            experiment, bqc_label: the main sidebar's current setup.
            handoff_experiment, handoff_bqc_label: the setup frozen when
                Quality Check opened (None before that).
            qc_experiment: Quality Check's experiment after PCA removal.
        """
        # Module 1's output belongs to the setup frozen when Quality Check
        # opened; the sidebar may have changed since (Quality Check only
        # warns), and the raw frame follows the sidebar.
        if source_key != 'raw' and handoff_experiment is not None:
            experiment = handoff_experiment
            bqc_label = handoff_bqc_label
        # QC output must pair with the QC-updated experiment: PCA sample
        # removal drops sample columns and renumbers the survivors, so the
        # pre-QC experiment would claim samples the data no longer has.
        if source_key == 'qc_filtered' and qc_experiment is not None:
            experiment = qc_experiment
        # PCA removal can drop the whole BQC condition. Left in place, the
        # Studio's BQC radio would fall back to its first option and treat
        # an ordinary condition as BQC.
        if experiment is not None and bqc_label not in experiment.conditions_list:
            bqc_label = None
        return experiment, bqc_label

    @staticmethod
    def derive_experiment(df: pd.DataFrame) -> ExperimentConfig:
        """Create a minimal single-condition ExperimentConfig from numeric columns.

        This allows the visualization chat to work without the user manually
        defining experimental conditions.
        """
        intensity_cols = [c for c in df.columns if c.startswith("intensity[")]
        if not intensity_cols:
            intensity_cols = df.select_dtypes(include="number").columns.tolist()

        n_samples = max(len(intensity_cols), 1)
        return ExperimentConfig(
            conditions_list=["All Samples"],
            number_of_samples_list=[n_samples],
        )

    @staticmethod
    def detect_sample_prefix(df: pd.DataFrame) -> Optional[str]:
        """Return 'intensity' or 'concentration' depending on which sample-column
        prefix the DataFrame uses. Module 1 emits one or the other depending on
        the normalization method (raw / total intensity → intensity[*]; internal
        standard, protein, or both → concentration[*]).

        Returns None if neither prefix is present.
        """
        if any(c.startswith('intensity[') and c.endswith(']') for c in df.columns):
            return 'intensity'
        if any(c.startswith('concentration[') and c.endswith(']') for c in df.columns):
            return 'concentration'
        return None

    @staticmethod
    def detected_sample_columns(df: pd.DataFrame) -> Tuple[Optional[str], list]:
        """(prefix, its '{prefix}[*]' columns), or (None, []) when the frame
        has neither intensity[*] nor concentration[*] columns."""
        detected_prefix = AIStudioWorkflow.detect_sample_prefix(df)
        if detected_prefix is None:
            return None, []
        head = f'{detected_prefix}['
        return detected_prefix, [
            c for c in df.columns
            if c.startswith(head) and c.endswith(']')
        ]

    @staticmethod
    def apply_user_experiment(
        df: pd.DataFrame,
        conditions_list: list,
        samples_per_condition: list,
        sample_to_col: dict,
        output_prefix: str,
    ) -> tuple:
        """Build an ExperimentConfig and a reordered DataFrame.

        The user picks samples per condition in arbitrary order (e.g. condition A
        gets s1+s3, condition B gets s2+s4). ExperimentConfig assumes conditions
        occupy contiguous sample blocks (s1..sN), so we rename the sample columns
        so condition A's samples become s1..sX and condition B's become
        s(X+1)..s(X+Y). The output uses {output_prefix}[sN] regardless of how the
        original columns were named, so override-picked raw columns are
        normalized to the same shape as auto-detected ones. Other
        {output_prefix}[...] columns, which the user did not pick as samples,
        are dropped: their names could repeat the new ones. Other non-sample
        columns are left untouched.

        Args:
            df: DataFrame containing the sample columns.
            conditions_list: condition names, in declared order.
            samples_per_condition: list of lists; samples_per_condition[i] is the
                list of sample names assigned to conditions_list[i].
            sample_to_col: maps each sample name to its original column in df.
            output_prefix: 'intensity' or 'concentration' — the prefix used for
                renamed sample columns.

        Returns:
            (reordered_df, experiment).
        """
        rename_map = {}
        sample_cols_in_order = []
        new_idx = 1
        for samples in samples_per_condition:
            for old_sample in samples:
                old_col = sample_to_col[old_sample]
                new_col = f'{output_prefix}[s{new_idx}]'
                rename_map[old_col] = new_col
                sample_cols_in_order.append(old_col)
                new_idx += 1

        sample_col_set = set(sample_cols_in_order)
        non_sample = [
            c for c in df.columns
            if c not in sample_col_set
            and not (c.startswith(f'{output_prefix}[') and c.endswith(']'))
        ]
        reordered = df[non_sample + sample_cols_in_order].rename(columns=rename_map)

        experiment = ExperimentConfig(
            conditions_list=conditions_list,
            number_of_samples_list=[len(s) for s in samples_per_condition],
        )
        return reordered, experiment

    @staticmethod
    def apply_snapshot(
        file_id: Optional[str],
        output_prefix: str,
        sample_columns: list,
        kept_conditions: list,
        kept_samples: list,
        bqc_label: Optional[str],
    ) -> tuple:
        """What the apply step applied. Streamlit reruns the script on every
        interaction and the Confirm box stays checked, so the sidebar
        re-applies (and restarts the data tabs' chats) only when this
        changes."""
        return (
            file_id,
            output_prefix,
            tuple(sample_columns),
            tuple(kept_conditions),
            tuple(tuple(s) for s in kept_samples),
            bqc_label,
        )

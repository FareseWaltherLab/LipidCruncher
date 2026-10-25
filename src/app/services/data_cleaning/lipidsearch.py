"""
LipidSearch format data cleaning.
Pure logic - no Streamlit dependencies.
"""
import pandas as pd
from typing import Dict, List, Optional, Tuple

from ...models.experiment import ExperimentConfig
from ...constants import (
    normalize_hydroxyl, sort_chains_lipid_maps,
    remove_phantom_chains, format_lipid_name,
    LYSO_CLASSES,
    LIPIDSEARCH_DEFAULT_GRADES, LIPIDSEARCH_RELAXED_GRADES,
    LIPIDSEARCH_RELAXED_GRADE_CLASSES,
)
from ..standards import StandardsService
from .base import BaseDataCleaner
from .configs import GradeFilterConfig
from .exceptions import EmptyDataError, ConfigurationError

# Starts the cleaning message that explains which adduct each class was
# quantified from, which the app shows on the page (the other cleaning
# messages are not).
ADDUCT_NOTE_PREFIX = "Adducts:"

# A class's adduct choice meaning: no single adduct, keep each species' best
# row whatever its adduct (as for a class without an internal standard).
ANY_ADDUCT = "Any adduct"


class LipidSearchCleaner(BaseDataCleaner):
    """
    Cleaner for LipidSearch format data.

    Handles:
    - Grade filtering (A/B/C/D)
    - FA key validation
    - Lipid name standardization
    - Best AUC selection per lipid
    """

    # Metadata columns to keep in cleaned output. LipidMolec is listed first
    # so the final projection preserves it as the leftmost column. Intensity
    # columns are appended after these by _final_cleanup.
    OUTPUT_COLUMNS: Tuple[str, ...] = (
        'LipidMolec', 'ClassKey', 'CalcMass', 'BaseRt', 'FAKey', 'TotalGrade',
        'MainIon',
    )

    @staticmethod
    def clean(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
        config: Optional[GradeFilterConfig] = None
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Clean LipidSearch format data.

        Args:
            df: Raw LipidSearch DataFrame.
            experiment: Experiment configuration.
            config: Optional grade filtering configuration.

        Returns:
            Tuple of (cleaned_df, messages_list).

        Raises:
            EmptyDataError: If dataset is empty or becomes empty after cleaning.
            ConfigurationError: If no species remain after grade filtering.
        """
        messages = []
        initial_count = len(df)

        # Validate input
        if LipidSearchCleaner.is_effectively_empty(df):
            raise EmptyDataError(
                "Dataset is empty. Please upload a valid LipidSearch data file."
            )

        # Step 1: Remove rows with missing FA keys
        df, msg = LipidSearchCleaner._step_remove_missing_fa_keys(df)
        if msg:
            messages.append(msg)

        # Step 2: Convert intensity columns to numeric
        df = LipidSearchCleaner.convert_columns_to_numeric(df, experiment.full_samples_list)

        # Step 3: Apply grade filtering
        grade_dict = config.grade_config if config else None
        df, msg = LipidSearchCleaner._step_apply_grade_filter(df, grade_dict)
        if msg:
            messages.append(msg)

        # Step 4: Standardize lipid names
        df = LipidSearchCleaner._standardize_lipid_names(df)

        # Step 5: Quantify classes with an internal standard in its adduct
        # (every row left is grade-eligible: step 3 applied the same mask)
        candidates = df
        keep_other = config.keep_other_adducts if config else False
        df, anchors, removed = LipidSearchCleaner._match_standard_adducts(
            candidates, keep_other, config.class_adducts if config else None,
        )

        # Step 6: Select best AUC for each lipid
        df, msg = LipidSearchCleaner._step_select_best_auc(df, experiment, grade_dict)
        if msg:
            messages.append(msg)

        # Step 7: Remove all-zero rows
        df, msg = LipidSearchCleaner._step_remove_zero_rows(df)
        if msg:
            messages.append(msg)

        if df.empty:
            raise EmptyDataError(
                "No lipid species with non-zero intensity values found."
            )

        # Say which adduct each class was quantified from, what was removed
        # for not matching its standard, and which classes still mix adducts.
        msg = LipidSearchCleaner._adduct_note(
            candidates, df, anchors, removed, keep_other,
            config.class_adducts if config else None,
        )
        if msg:
            messages.append(msg)

        # Step 8: Final cleanup
        df = LipidSearchCleaner._final_cleanup(df)

        messages.append(LipidSearchCleaner._make_summary_message(initial_count, len(df)))
        return df, messages

    # ==================== Adducts ====================

    @staticmethod
    def _best_first(df: pd.DataFrame) -> pd.DataFrame:
        """Rows ordered best grade first, then highest identification rate."""
        priority = df['TotalGrade'].map(LipidSearchCleaner._GRADE_PRIORITY).fillna(99)
        return df.assign(_priority=priority).sort_values(
            ['_priority', 'TotalSmpIDRate(%)'], ascending=[True, False],
        ).drop(columns=['_priority'])

    @staticmethod
    def _match_standard_adducts(
        df: pd.DataFrame,
        keep_other: bool,
        class_adducts: Optional[Dict[str, str]] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, List[str]], pd.DataFrame]:
        """Quantify each class that has its own internal standard in the
        standard's adduct (``MainIon``), or in the adduct the user chose.

        Species and standard must ionise alike for the standard to convert
        a species' signal to a concentration; a [M+HCOO]- species against a
        [M+H]+ standard compares different scales. For a species with a row
        in the standard's adduct, only those rows stay (the best of them is
        selected later). A species seen only in other adducts is removed,
        unless ``keep_other``. Classes without a standard of their own are
        unchanged. The standard's adduct is that of its best row.
        ``class_adducts`` overrides it per class, gives a class without a
        standard an adduct, or (``ANY_ADDUCT``) leaves a class unmatched.

        Returns:
            The rows to select from; each anchored class's adducts; and the
            best row of each species removed (or kept) for not matching.
        """
        if 'MainIon' not in df.columns or df.empty:
            return df, {}, df.iloc[0:0]
        df = df.assign(MainIon=df['MainIon'].astype(str).str.strip())
        is_standard = StandardsService.detect_standards(df)
        standards = LipidSearchCleaner._best_first(df[is_standard]).drop_duplicates(
            'LipidMolec',
        )
        anchors = {
            cls: sorted(set(group['MainIon']))
            for cls, group in standards.groupby('ClassKey')
        }
        for cls, adduct in (class_adducts or {}).items():
            if adduct == ANY_ADDUCT:
                anchors.pop(cls, None)
            else:
                anchors[cls] = [adduct]
        if not anchors:
            return df, {}, df.iloc[0:0]

        species = df[~is_standard]
        anchored = species['ClassKey'].isin(list(anchors))
        matches = pd.Series(
            [ion in anchors.get(cls, ()) for cls, ion in
             zip(species['ClassKey'], species['MainIon'])],
            index=species.index, dtype=bool,
        )
        has_match = matches.groupby(species['LipidMolec']).transform('any')
        other_only = anchored & ~has_match
        unmatched = LipidSearchCleaner._best_first(species[other_only]).drop_duplicates(
            'LipidMolec',
        )
        keep = ~anchored | matches | (other_only & keep_other)
        return pd.concat([df[is_standard], species[keep]]), anchors, unmatched

    @staticmethod
    def _adduct_note(
        candidates: pd.DataFrame,
        kept: pd.DataFrame,
        anchors: Dict[str, List[str]],
        unmatched: pd.DataFrame,
        keep_other: bool,
        class_adducts: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        """Explain which adduct each class was quantified from.

        ``candidates`` are the grade-eligible rows before one row per lipid
        was kept; ``kept`` the rows kept; ``anchors`` and ``unmatched`` come
        from ``_match_standard_adducts``. None when the export has no MainIon
        column or every class is measured in one adduct.
        """
        if 'MainIon' not in candidates.columns or 'MainIon' not in kept.columns:
            return None
        ions = kept.assign(MainIon=kept['MainIon'].astype(str).str.strip())
        species = ions[~StandardsService.detect_standards(ions)]

        def counts(frame) -> str:
            tally = frame['MainIon'].value_counts().sort_index().sort_values(
                ascending=False, kind='stable',
            )
            return ", ".join(f"{ion} {n}" for ion, n in tally.items())

        parts = []
        anchored = []
        for cls, adducts in sorted(anchors.items()):
            matched = species[(species['ClassKey'] == cls)
                              & species['MainIon'].isin(adducts)]
            other = unmatched[unmatched['ClassKey'] == cls]
            chosen = ", your choice" if cls in (class_adducts or {}) else ""
            text = f"{cls} ({' / '.join(adducts)}{chosen}): {len(matched)} species"
            if len(other):
                text += (
                    f"; {len(other)} seen only as other adducts "
                    f"{'kept' if keep_other else 'removed'} ({counts(other)})"
                )
            anchored.append(text)
        if anchored:
            parts.append(
                "Classes quantified in one adduct (their internal standard's, "
                "so species and standard respond alike, or the one you chose): "
                + "; ".join(anchored) + "."
            )
            if len(unmatched) and not keep_other:
                parts.append(
                    "To keep the removed species, turn on **Keep species "
                    "measured in other adducts** under Configure Adducts."
                )
            elif len(unmatched):
                parts.append(
                    "The species kept in other adducts are on a different scale "
                    "from their class's standard."
                )

        mixed = [
            f"{cls} ({counts(group)})"
            for cls, group in species.groupby('ClassKey')
            if group['MainIon'].nunique() > 1
        ]
        if mixed:
            parts.append(
                "Classes whose species come from different adducts: "
                + "; ".join(mixed)
                + ". Each species is quantified from one identification (its "
                "best grade), never a sum of adducts; compare these species "
                "across conditions rather than with each other."
            )
        if not parts:
            return None
        return f"{ADDUCT_NOTE_PREFIX} " + " ".join(parts)

    # ==================== Cleaning Steps ====================

    @staticmethod
    def _step_remove_missing_fa_keys(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
        """Remove rows with missing FA keys (except Ch class)."""
        initial_count = len(df)
        df = LipidSearchCleaner._remove_missing_fa_keys(df)

        if df.empty:
            raise EmptyDataError(
                "No valid lipid species found. "
                "Dataset contains no rows with valid fatty acid (FA) keys."
            )

        removed = initial_count - len(df)
        msg = f"Removed {removed} rows with missing FA keys" if removed > 0 else None
        return df, msg

    @staticmethod
    def _step_apply_grade_filter(
        df: pd.DataFrame,
        grade_config: Optional[Dict[str, List[str]]]
    ) -> Tuple[pd.DataFrame, Optional[str]]:
        """Apply grade filtering."""
        initial_count = len(df)
        df = LipidSearchCleaner._apply_grade_filter(df, grade_config)

        if df.empty:
            raise ConfigurationError(
                "No lipid species remain after grade filtering. "
                "Please adjust your grade filter settings."
            )

        removed = initial_count - len(df)
        msg = f"Grade filter: Removed {removed} entries" if removed > 0 else None
        return df, msg

    @staticmethod
    def _step_select_best_auc(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
        grade_config: Optional[Dict[str, List[str]]]
    ) -> Tuple[pd.DataFrame, Optional[str]]:
        """Select best AUC for each lipid."""
        df = LipidSearchCleaner._select_best_auc(df, experiment, grade_config)

        if df.empty:
            raise EmptyDataError(
                "No lipid species remain after quality filtering."
            )

        msg = f"AUC selection: Consolidated to {len(df)} unique lipids"
        return df, msg

    # ==================== FA Key Handling ====================

    @staticmethod
    def _remove_missing_fa_keys(df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows with missing FA keys, with exceptions for Ch class.

        Cholesterol molecules are kept even without FA keys.
        """
        if df.empty:
            return df

        df = df.copy()
        df['LipidMolec'] = df['LipidMolec'].fillna('').astype(str)

        keep_mask = (
            df['FAKey'].notna() |
            (df['ClassKey'] == 'Ch') |
            df['LipidMolec'].str.contains(r'^Ch-D', regex=True, na=False)
        )
        return df[keep_mask].copy()

    # ==================== Grade Filtering ====================

    @staticmethod
    def _default_grade_mask(df: pd.DataFrame) -> 'pd.Series':
        """Default grades: A/B for every class, plus C for LPC and SM."""
        return (
            df['TotalGrade'].isin(LIPIDSEARCH_DEFAULT_GRADES) |
            (df['TotalGrade'].isin(LIPIDSEARCH_RELAXED_GRADES) &
             df['ClassKey'].isin(LIPIDSEARCH_RELAXED_GRADE_CLASSES))
        )

    @staticmethod
    def _apply_grade_filter(
        df: pd.DataFrame,
        grade_config: Optional[Dict[str, List[str]]] = None
    ) -> pd.DataFrame:
        """Apply grade filtering to the dataframe."""
        if grade_config is None:
            return df[LipidSearchCleaner._default_grade_mask(df)].copy()

        # Custom filtering based on config
        filtered_dfs = []
        for class_key, grades in grade_config.items():
            if grades:
                class_df = df[
                    (df['ClassKey'] == class_key) &
                    (df['TotalGrade'].isin(grades))
                ]
                filtered_dfs.append(class_df)

        if filtered_dfs:
            return pd.concat(filtered_dfs, ignore_index=True)
        return df.iloc[0:0].copy()

    # ==================== Lipid Name Standardization ====================

    @staticmethod
    def _standardize_lipid_names(df: pd.DataFrame) -> pd.DataFrame:
        """Standardize lipid names in the LipidMolec column."""
        df = df.copy()
        df['LipidMolec_modified'] = df.apply(
            lambda row: LipidSearchCleaner._standardize_single_name(
                row['ClassKey'], row['FAKey']
            ),
            axis=1
        )
        df = df.drop(columns=['LipidMolec'])
        return df.rename(columns={'LipidMolec_modified': 'LipidMolec'})

    @staticmethod
    def _standardize_single_name(class_key: str, fa_key: object) -> str:
        """Standardize a single lipid name to LIPID MAPS shorthand notation.

        Args:
            class_key: Lipid class (e.g., 'PC', 'PE', 'Ch')
            fa_key: Fatty acid key (e.g., '18:0_20:4'). May be str, float (NaN), or None.

        Returns:
            Standardized lipid name (e.g., 'PC 18:0_20:4')
        """
        fa_key = "" if pd.isna(fa_key) else str(fa_key)

        # Special handling for Cholesterol
        if class_key == 'Ch':
            return LipidSearchCleaner._standardize_cholesterol_name(fa_key)

        # Handle internal standards (deuterated)
        fa_key, internal_standard = LipidSearchCleaner._extract_internal_standard_suffix(fa_key)

        if not fa_key or fa_key == '()':
            return f"{class_key}{internal_standard}" if internal_standard else class_key

        # Parse chains
        raw = fa_key.strip('()')
        chains = raw.split('_')

        # Remove phantom 0:0 chains for lyso-species
        if class_key in LYSO_CLASSES:
            chains = remove_phantom_chains(chains)

        # Normalize hydroxyl notation: ;2O → ;O2
        chains = [normalize_hydroxyl(c) for c in chains]

        # Sort per LIPID MAPS rules
        chains = sort_chains_lipid_maps(chains, class_key)

        return format_lipid_name(class_key, chains, internal_standard)

    @staticmethod
    def _standardize_cholesterol_name(fa_key: str) -> str:
        """Standardize cholesterol lipid name."""
        if not fa_key or fa_key.startswith('D'):
            if fa_key.startswith('D'):
                return f"Ch-{fa_key}"
            return "Ch"
        return f"Ch {fa_key}"

    @staticmethod
    def _extract_internal_standard_suffix(fa_key: str) -> Tuple[str, str]:
        """
        Extract internal standard suffix from FA key.

        Returns:
            Tuple of (fa_key_without_suffix, suffix)
        """
        if '+D' in fa_key:
            parts = fa_key.split('+', 1)
            return parts[0], '+' + parts[1]
        return fa_key, ''

    # ==================== AUC Selection ====================

    # Grade priority for sorting: lower number = better grade
    _GRADE_PRIORITY: Dict[str, int] = {'A': 1, 'B': 2, 'C': 3, 'D': 4}

    @staticmethod
    def _select_best_auc(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
        grade_config: Optional[Dict[str, List[str]]] = None
    ) -> pd.DataFrame:
        """
        Select best AUC for each lipid based on grade priority.

        When multiple entries exist for the same lipid, select based on:
        1. Best grade (A > B > C > D)
        2. Highest TotalSmpIDRate(%)

        Default mode: A/B for all classes, C only for LPC/SM.
        Custom mode: each class has its own allowed grades from config.
        """
        # Filter to eligible entries based on grade config
        eligible = LipidSearchCleaner._build_eligibility_mask(df, grade_config)
        df = df[eligible].copy()

        if df.empty:
            sample_cols = [f'intensity[{s}]' for s in experiment.full_samples_list]
            return pd.DataFrame(
                columns=['LipidMolec', 'ClassKey', 'CalcMass', 'BaseRt', 'TotalSmpIDRate(%)'] + sample_cols
            )

        # Keep the best entry (grade, then identification rate) per lipid
        df = LipidSearchCleaner._best_first(df)
        df = df.drop_duplicates(subset=['LipidMolec'], keep='first')
        return df.reset_index(drop=True)

    @staticmethod
    def _build_eligibility_mask(
        df: pd.DataFrame,
        grade_config: Optional[Dict[str, List[str]]]
    ) -> 'pd.Series':
        """Build boolean mask for entries eligible for AUC selection."""
        if grade_config is None:
            return LipidSearchCleaner._default_grade_mask(df)

        # Custom: each class has its own allowed grades
        eligible = pd.Series(False, index=df.index)
        for class_key, grades in grade_config.items():
            eligible = eligible | (
                df['ClassKey'].eq(class_key) & df['TotalGrade'].isin(grades)
            )
        return eligible

    # ==================== Final Cleanup ====================

    @staticmethod
    def _final_cleanup(df: pd.DataFrame) -> pd.DataFrame:
        """Project to canonical metadata + intensity columns, reset index.

        Drops any extra columns produced by extended LipidSearch exports
        (e.g. OrgMeanArea[*], MeanHeight[*], MeanConc[*], LipidMolecGroup, etc.)
        so the cleaned frame matches the schema expected by downstream steps.
        """
        metadata_cols = [c for c in LipidSearchCleaner.OUTPUT_COLUMNS if c in df.columns]
        intensity_cols = [c for c in df.columns if c.startswith('intensity[')]
        return df[metadata_cols + intensity_cols].reset_index(drop=True)

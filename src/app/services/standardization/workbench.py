"""
Metabolomics Workbench format: the MS_METABOLITE_DATA block of a text
export, with a sample row and a factors row above the data.
"""

import csv
from typing import Union

import pandas as pd

from app.services import lipid_names
from app.services.standardization.result import StandardizationResult


def process(
    text_data: Union[str, pd.DataFrame],
) -> StandardizationResult:
    """Validate and standardize Metabolomics Workbench text data."""
    if not isinstance(text_data, str):
        return StandardizationResult(
            False, "Invalid input type for Metabolomics Workbench format"
        )

    # Validate markers
    if 'MS_METABOLITE_DATA_START' not in text_data or 'MS_METABOLITE_DATA_END' not in text_data:
        return StandardizationResult(False, "Missing required data section markers")

    start_idx = text_data.find('MS_METABOLITE_DATA_START')
    end_idx = text_data.find('MS_METABOLITE_DATA_END')
    if start_idx >= end_idx:
        return StandardizationResult(False, "Invalid data section markers")

    try:
        data_section = text_data[start_idx:end_idx].strip().split('\n')
        data_section = [line.strip() for line in data_section if line.strip()]
        data_section = [
            line for line in data_section if 'MS_METABOLITE_DATA_START' not in line
        ]

        if len(data_section) < 3:
            return StandardizationResult(False, "Insufficient data rows")

        # csv.reader keeps a quoted name such as "1,2-DAG 34:1" as one
        # field; a plain split on ',' silently dropped those rows.
        parsed = list(csv.reader(data_section))
        samples_line = parsed[0]
        factors_line = parsed[1]

        sample_names = samples_line[1:]
        conditions = factors_line[1:]

        cleaned_conditions = []
        for cond in conditions:
            cond = cond.strip()
            if cond.startswith('Condition:'):
                cond = cond.replace('Condition:', '').strip()
            cleaned_conditions.append(cond)

        # A row with another field count cannot be lined up with the
        # samples; dropping it silently lost that lipid from the analysis.
        n_fields = len(sample_names) + 1
        misaligned = [values for values in parsed[2:] if len(values) != n_fields]
        if misaligned:
            listed = '; '.join(
                f"'{values[0]}' ({len(values)} fields)" for values in misaligned[:5]
            )
            more = f" and {len(misaligned) - 5} more" if len(misaligned) > 5 else ""
            return StandardizationResult(
                False,
                f"{len(misaligned)} data row(s) do not have {n_fields} fields "
                f"(the lipid name plus one value per sample): {listed}{more}. "
                "A lipid name that contains a comma must be quoted.",
            )
        data_rows = parsed[2:]

        if not data_rows:
            return StandardizationResult(False, "No valid data rows found")

        # Number samples condition by condition (stable within a condition):
        # the experiment assigns s1..sN to conditions contiguously, so an
        # interleaved factor order would otherwise put samples in the wrong
        # condition.
        first_seen = list(dict.fromkeys(cleaned_conditions))
        order = sorted(
            range(len(sample_names)),
            key=lambda i: first_seen.index(cleaned_conditions[i]),
        )
        sample_names = [sample_names[i] for i in order]
        cleaned_conditions = [cleaned_conditions[i] for i in order]
        data_rows = [[row[0]] + [row[i + 1] for i in order] for row in data_rows]

        columns = ['LipidMolec'] + [
            f'intensity[s{i + 1}]' for i in range(len(sample_names))
        ]
        df = pd.DataFrame(data_rows, columns=columns)

        # Standardize lipid names and infer ClassKey
        df['LipidMolec'] = df['LipidMolec'].apply(
            lipid_names.standardize_lipid_name
        )
        df['ClassKey'] = df['LipidMolec'].apply(
            lipid_names.infer_class_key
        )

        # Convert intensity columns to numeric
        intensity_cols = [c for c in df.columns if c.startswith('intensity[')]
        for col in intensity_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df[intensity_cols] = df[intensity_cols].fillna(0)

        wb_conditions = {
            f's{i + 1}': cond for i, cond in enumerate(cleaned_conditions)
        }
        wb_samples = {
            f's{i + 1}': name.strip() for i, name in enumerate(sample_names)
        }
        column_mapping = pd.DataFrame({
            'standardized_name': ['LipidMolec'] + intensity_cols,
            'original_name': [samples_line[0].strip()] + [name.strip() for name in sample_names],
        })

        return StandardizationResult(
            success=True,
            message="Data successfully standardized",
            standardized_df=df,
            column_mapping=column_mapping,
            n_intensity_cols=len(sample_names),
            workbench_conditions=wb_conditions,
            workbench_samples=wb_samples,
        )

    except (ValueError, KeyError, IndexError) as e:
        return StandardizationResult(
            False, f"Error standardizing Metabolomics Workbench data: {e}"
        )

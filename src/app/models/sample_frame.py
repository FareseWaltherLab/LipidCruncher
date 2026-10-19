"""
Sample frame model: data plus the identity of the samples it holds.
"""
import re
from typing import Dict, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, model_validator

from app.models.experiment import ExperimentConfig

# 'intensity[s3]' / 'concentration[s3]' -> 's3'; a bare 's3' is kept.
_LABEL_RE = re.compile(r'^(?:\w+\[)?(s\d+)\]?$')


def _label(key: str) -> str:
    match = _LABEL_RE.match(str(key))
    if not match:
        raise ValueError(f"Not a sample column or label: {key!r}")
    return match.group(1)


class SampleFrame(BaseModel):
    """A frame whose samples are labelled s1..sN, with what each label means.

    Regrouping and excluding samples renumber them (exclude s1 and the old s2
    becomes s1), so anything kept per sample must move with the renumbering.
    A SampleFrame keeps the data, its grouping, the display names and each
    sample's origin together, and ``renumbered`` moves them all at once.

    Attributes:
        df: Data with one ``prefix[sN]`` column per sample.
        experiment: Grouping of ``df``'s samples into conditions.
        names: ``{label -> display name}`` for the named samples.
        origin: ``{label -> label in the uploaded file}`` for every sample.
            The upload label never changes, so per-sample inputs (protein
            amounts, exclusions) are keyed by it.

    Do not compare frames with ``==``: ``df`` has no truth value.
    """
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    df: pd.DataFrame
    experiment: ExperimentConfig
    names: Dict[str, str] = {}
    origin: Dict[str, str]

    @model_validator(mode='after')
    def _origin_covers_every_sample(self) -> 'SampleFrame':
        if sorted(self.origin) != sorted(self.experiment.full_samples_list):
            raise ValueError(
                f"origin labels {sorted(self.origin)} do not match the "
                f"experiment's samples {self.experiment.full_samples_list}"
            )
        return self

    @classmethod
    def from_upload(
        cls,
        df: pd.DataFrame,
        experiment: ExperimentConfig,
        names: Optional[Dict[str, str]] = None,
    ) -> 'SampleFrame':
        """A frame still labelled as uploaded: every sample is its own origin."""
        return cls(
            df=df,
            experiment=experiment,
            names=names or {},
            origin={s: s for s in experiment.full_samples_list},
        )

    def renumbered(
        self,
        df: pd.DataFrame,
        experiment: ExperimentConfig,
        old_to_new: Dict[str, str],
    ) -> 'SampleFrame':
        """This frame's samples after an event that renumbered them.

        Args:
            df: The data after the event.
            experiment: Its grouping.
            old_to_new: Each surviving sample's column or label before ->
                after (``'intensity[s2]' -> 'intensity[s1]'`` or
                ``'s2' -> 's1'``); a sample missing from it was removed.
        """
        new = {_label(old): _label(after) for old, after in old_to_new.items()}
        return SampleFrame(
            df=df,
            experiment=experiment,
            names={new[s]: name for s, name in self.names.items() if s in new},
            origin={new[s]: o for s, o in self.origin.items() if s in new},
        )

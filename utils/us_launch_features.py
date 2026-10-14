"""Optional, launch-time feature experiments for the U.S. cohort.

Import this module only for training-only feature comparisons. The baseline
FeatureBuilder and global modeling notebooks remain unchanged.
"""

import numpy as np
import pandas as pd
from utils.preprocessing import FeatureBuilder


class USLaunchFeatureBuilder(FeatureBuilder):
    """Conservative extra features using only goal, duration, and state.

    `goal_vs_state` learns state-level medians only from the fold used by
    `fit`. No target labels or test observations are used for that statistic.
    Other features are deterministic from a proposed campaign's attributes.
    """

    def __init__(self, include_timing=False, include_flags=False,
                 exclude_renamed=True, include_reward=False,
                 include_extra_launch_features=True):
        super().__init__(include_timing=include_timing,
                         include_flags=include_flags,
                         exclude_renamed=exclude_renamed,
                         include_reward=include_reward)
        self.include_extra_launch_features = include_extra_launch_features

    @staticmethod
    def _state(X):
        return X["location"].astype("string").str.extract(
            r",\s*([A-Z]{2})\s*$", expand=False
        )

    def fit(self, X, y=None):
        super().fit(X, y)
        if self.include_extra_launch_features:
            goal = pd.to_numeric(X["goal"], errors="coerce")
            log_goal = np.log1p(goal.where(goal.ge(0)))
            by_state = pd.DataFrame({
                "state": self._state(X).to_numpy(),
                "log_goal": log_goal.to_numpy(),
            })
            self.state_median_log_goal_ = by_state.groupby(
                "state", dropna=True
            )["log_goal"].median()
            self.overall_median_log_goal_ = float(log_goal.median())
        return self

    def transform(self, X):
        result = super().transform(X)
        if not self.include_extra_launch_features:
            return result

        goal = pd.to_numeric(X["goal"], errors="coerce")
        duration = pd.to_numeric(X["duration"], errors="coerce")
        state = self._state(X)
        log_goal = np.log1p(goal.where(goal.ge(0)))
        # Training-fold reference only; new/missing states use training median.
        reference = state.map(self.state_median_log_goal_).fillna(
            self.overall_median_log_goal_
        )
        result["goal_vs_state"] = (log_goal - reference).to_numpy()
        result["goal_round_1000"] = goal.mod(1000).eq(0).where(
            goal.notna(), other=np.nan
        ).astype(float).to_numpy()
        result["duration_exact_30"] = duration.eq(30).where(
            duration.notna(), other=np.nan
        ).astype(float).to_numpy()
        result["duration_exact_60"] = duration.eq(60).where(
            duration.notna(), other=np.nan
        ).astype(float).to_numpy()
        return result

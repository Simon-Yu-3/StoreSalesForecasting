"""Guard the forecast boundary in the pooled-model feature pipeline."""

from dataclasses import replace

import numpy as np
import pandas as pd

from auxiliary.data import chronological_folds, make_fixture
from auxiliary.global_benchmark import origin_features, training_examples


def test_pooled_model_never_reads_sales_after_the_cutoff():
    panel = make_fixture()
    cutoff = panel.dates.get_loc(chronological_folds(panel)[0].train_end)
    sales = panel.sales.copy()
    sales[:, cutoff + 1 :] = 1_000_000
    changed = replace(panel, sales=sales)

    actual = origin_features(panel, cutoff, promotions=True)
    pd.testing.assert_frame_equal(
        actual, origin_features(changed, cutoff, promotions=True)
    )
    history_only = origin_features(panel, cutoff, promotions=False)
    assert set(actual.columns) - set(history_only.columns) == {
        "planned_promotion",
        "promotion_mean_28",
    }
    x, y = training_examples(panel, cutoff, promotions=True)
    changed_x, changed_y = training_examples(changed, cutoff, promotions=True)
    pd.testing.assert_frame_equal(x, changed_x)
    np.testing.assert_array_equal(y, changed_y)

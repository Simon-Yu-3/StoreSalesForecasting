"""Check competition-format inputs and the frozen log-scale blend."""

from dataclasses import replace
from hashlib import sha256

import numpy as np
import pandas as pd
import pytest

from auxiliary.competition_ensemble import _read_predictions, blend
from auxiliary.data import make_fixture


def test_blend_preserves_test_ids_and_records_input_hashes(tmp_path):
    panel = replace(
        make_fixture(), provenance={"files": {"test.csv": {"sha256": "fixture"}}}
    )
    ids = panel.test["id"]
    calendar = tmp_path / "calendar_hgb.csv"
    t0 = tmp_path / "t0_beta.csv"
    pd.DataFrame({"id": ids, "sales": 15}).to_csv(calendar, index=False)
    pd.DataFrame({"id": ids, "sales": 255}).to_csv(t0, index=False)
    manifest = blend(panel, calendar, t0, tmp_path)
    result = pd.read_csv(tmp_path / "submission.csv")

    np.testing.assert_array_equal(result["id"], ids)
    # 0.75 * log(16) + 0.25 * log(256) = log(32).
    np.testing.assert_allclose(result["sales"], 31)
    assert manifest["rows"] == len(ids)
    assert manifest["inputs_sha256"] == {
        "calendar_hgb": sha256(calendar.read_bytes()).hexdigest(),
        "t0_beta_promotion": sha256(t0.read_bytes()).hexdigest(),
    }
    assert (
        manifest["sha256"]
        == sha256((tmp_path / "submission.csv").read_bytes()).hexdigest()
    )


@pytest.mark.parametrize(
    ("contents", "error"),
    [
        ("id,sales\n2,10\n1,12\n", "match test ID order"),
        ("id,sales\n1,10\n1,12\n", "match test ID order"),
        ("id,sales\n0,1,10\n1,2,12\n", "match test ID order"),
        ("id,sales\n1,10\n2,-1\n", "finite nonnegative forecasts"),
        ("id,sales\n1,10\n2,inf\n", "finite nonnegative forecasts"),
        ("id,sales\n1,10\n2,\n", "finite nonnegative forecasts"),
    ],
)
def test_blend_input_validation(tmp_path, contents, error):
    path = tmp_path / "invalid.csv"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        _read_predictions(path, np.array([1, 2]))


def test_predictions_keep_round_trip_precision(tmp_path):
    path = tmp_path / "precise.csv"
    original = 2.1925341612053457
    path.write_text(f"id,sales\n1,{original}\n", encoding="utf-8")
    assert _read_predictions(path, np.array([1]))[0] == original

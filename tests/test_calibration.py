
import pytest

from src.evaluation.calibration import leave_one_out_calibration


def test_leave_one_out_calibration_excludes_current_question():
    scores = [10.0, 8.0, 2.0, 1.0]
    labels = [True, True, False, False]

    folds = leave_one_out_calibration(scores, labels)

    assert len(folds) == 4

    threshold, lo, hi = folds[0]

    # Current score is 10.0, so it must not define the calibration maximum.
    assert hi == 8.0


def test_leave_one_out_calibration_returns_threshold_for_each_question():
    scores = [10.0, 8.0, 6.0, 2.0, 1.0]
    labels = [True, True, True, False, False]

    folds = leave_one_out_calibration(scores, labels)

    assert len(folds) == len(scores)

    for threshold, lo, hi in folds:
        assert lo <= hi
        assert lo <= threshold <= hi


def test_leave_one_out_calibration_requires_both_classes_in_each_fold():
    scores = [5.0, 4.0, 1.0]
    labels = [True, True, False]

    with pytest.raises(ValueError):
        leave_one_out_calibration(scores, labels)

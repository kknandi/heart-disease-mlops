def test_deliberate_failure_for_ci_demo():
    assert 0.80 >= 0.85, "Deliberate failure: shows the pipeline stops on test errors"

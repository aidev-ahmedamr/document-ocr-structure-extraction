from src.extraction.compare import build_result, compare_results


def test_identical_fields_give_full_agreement():
    a = build_result("pipeline_a", "text", {"dates": ["2026-01-01"]}, 1.0)
    b = build_result("pipeline_b", "text", {"dates": ["2026-01-01"]}, 2.0)

    report = compare_results(a, b)
    assert report["field_agreement"]["dates"] == 1.0
    assert report["overall_agreement"] == 1.0


def test_disjoint_fields_give_zero_agreement():
    a = build_result("pipeline_a", "text", {"dates": ["2026-01-01"]}, 1.0)
    b = build_result("pipeline_b", "text", {"dates": ["2025-05-05"]}, 2.0)

    report = compare_results(a, b)
    assert report["field_agreement"]["dates"] == 0.0


def test_both_empty_counts_as_agreement():
    a = build_result("pipeline_a", "text", {"emails": []}, 1.0)
    b = build_result("pipeline_b", "text", {"emails": []}, 2.0)

    report = compare_results(a, b)
    assert report["field_agreement"]["emails"] == 1.0

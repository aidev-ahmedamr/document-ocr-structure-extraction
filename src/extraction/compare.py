def build_result(pipeline_name, raw_text, fields, processing_time_seconds, document_type="unknown"):
    return {
        "pipeline": pipeline_name,
        "document_type": document_type,
        "raw_text": raw_text,
        "fields": fields,
        "processing_time_seconds": round(processing_time_seconds, 3),
    }


def _overlap_ratio(list_a, list_b):
    if not list_a and not list_b:
        return 1.0
    set_a, set_b = set(list_a), set(list_b)
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def compare_results(result_a, result_b):
    """Compare two pipeline outputs for the same document, field by field."""

    fields_a = result_a["fields"]
    fields_b = result_b["fields"]

    all_keys = set(fields_a.keys()) | set(fields_b.keys())

    field_agreement = {
        key: round(_overlap_ratio(fields_a.get(key, []), fields_b.get(key, [])), 2)
        for key in all_keys
    }

    overall_agreement = (
        round(sum(field_agreement.values()) / len(field_agreement), 2)
        if field_agreement else 0.0
    )

    return {
        "pipeline_a": result_a["pipeline"],
        "pipeline_b": result_b["pipeline"],
        "field_agreement": field_agreement,
        "overall_agreement": overall_agreement,
        "speed": {
            result_a["pipeline"]: result_a["processing_time_seconds"],
            result_b["pipeline"]: result_b["processing_time_seconds"],
        },
    }

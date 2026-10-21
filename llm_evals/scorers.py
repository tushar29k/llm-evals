"""Deterministic scorers: exact / contains / regex / json_schema.

Every scorer returns (score, detail): score is 0.0 or 1.0 (no partial
credit at this layer — partial credit is where evals start lying),
detail is the human-readable reason, for the report table.
"""

import json
import re

# unknown scorer names fail the dataset load, not the run
KNOWN_SCORERS = ("exact", "contains", "regex", "json_schema", "llm_judge")


def _as_list(expected):
    # "list of acceptable strings" = alternatives, any one passes
    return expected if isinstance(expected, list) else [expected]


def _exact(case, output):
    for exp in _as_list(case["expected"]):
        if str(output).strip() == str(exp).strip():
            return 1.0, "exact match"
    return 0.0, f"got {str(output).strip()!r}, wanted one of {case['expected']!r}"


def _contains(case, output):
    params = case.get("scorer_params") or {}
    hay = str(output)
    if params.get("case_insensitive"):
        hay = hay.lower()
    for exp in _as_list(case["expected"]):
        needle = str(exp)
        if params.get("case_insensitive"):
            needle = needle.lower()
        if needle in hay:
            return 1.0, f"found {str(exp)!r}"
    return 0.0, f"none of {case['expected']!r} in output"


def _regex(case, output):
    params = case.get("scorer_params") or {}
    pattern = params.get("pattern")
    if not pattern:
        raise ValueError(f"case {case['id']}: regex scorer needs scorer_params.pattern")
    m = re.search(pattern, str(output))
    if m:
        return 1.0, f"matched {m.group(0)!r}"
    return 0.0, f"pattern {pattern!r} not found"


def _mini_validate(schema, value, path="$"):
    # tiny dependency-free validator: covers the shapes evals actually use.
    # if the real jsonschema lib is installed we defer to it instead.
    try:
        import jsonschema  # noqa

        jsonschema.validate(value, schema)
        return None
    except ImportError:
        pass
    except Exception as e:  # noqa — the real lib raises, we return strings
        return str(e).split("\n")[0]

    want = schema.get("type")
    if want == "object":
        if not isinstance(value, dict):
            return f"{path}: expected object, got {type(value).__name__}"
        for key in schema.get("required", []):
            if key not in value:
                return f"{path}: missing required key {key!r}"
        for key, subschema in schema.get("properties", {}).items():
            if key in value:
                err = _mini_validate(subschema, value[key], f"{path}.{key}")
                if err:
                    return err
    elif want == "array":
        if not isinstance(value, list):
            return f"{path}: expected array, got {type(value).__name__}"
        items = schema.get("items")
        if items:
            for i, v in enumerate(value):
                err = _mini_validate(items, v, f"{path}[{i}]")
                if err:
                    return err
    elif want == "string":
        if not isinstance(value, str):
            return f"{path}: expected string, got {type(value).__name__}"
        if "minLength" in schema and len(value) < schema["minLength"]:
            return f"{path}: shorter than minLength {schema['minLength']}"
    elif want == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            return f"{path}: expected integer, got {type(value).__name__}"
    elif want == "number":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return f"{path}: expected number, got {type(value).__name__}"
    elif want == "boolean":
        if not isinstance(value, bool):
            return f"{path}: expected boolean, got {type(value).__name__}"
    elif want == "null":
        if value is not None:
            return f"{path}: expected null"
    if "enum" in schema and value not in schema["enum"]:
        return f"{path}: {value!r} not in enum {schema['enum']}"
    if "minimum" in schema and isinstance(value, (int, float)) and value < schema["minimum"]:
        return f"{path}: below minimum {schema['minimum']}"
    return None


def _json_schema(case, output):
    params = case.get("scorer_params") or {}
    schema = params.get("schema")
    if not schema:
        raise ValueError(f"case {case['id']}: json_schema scorer needs scorer_params.schema")
    try:
        value = json.loads(str(output))
    except (json.JSONDecodeError, ValueError):
        return 0.0, "output is not valid JSON"
    err = _mini_validate(schema, value)
    if err:
        return 0.0, err
    return 1.0, "valid JSON, schema ok"


_DISPATCH = {
    "exact": _exact,
    "contains": _contains,
    "regex": _regex,
    "json_schema": _json_schema,
}


def score_case(case, output):
    """Score one model output against one case. Returns (score, detail)."""
    scorer = case["scorer"]
    if scorer == "llm_judge":
        raise ValueError(f"case {case['id']}: llm_judge ships on day 3")
    fn = _DISPATCH.get(scorer)
    if fn is None:
        raise ValueError(f"case {case['id']}: unknown scorer {scorer!r}")
    return fn(case, output)

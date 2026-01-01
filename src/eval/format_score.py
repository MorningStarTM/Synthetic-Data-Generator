import ast
import json
import re
from typing import Any, Dict, List, Tuple, Union

JSONLike = Union[Dict[str, Any], List[Any]]

def _strip_wrapper(text: str) -> str:
    """
    Extract the most likely JSON/Python literal payload from common wrappers:
      - qa_json="...".
      - qa_json='...'
      - qa_json = ...
      - backticks, stray parentheses, etc.
    """
    s = text.strip()

    # Try to extract qa_json="...".
    m = re.search(r'qa_json\s*=\s*([\'"])(.*?)\1\s*[\)\]]?\s*$', s, flags=re.DOTALL)
    if m:
        return m.group(2).strip()

    # Or qa_json=<literal> without quotes (rare)
    m2 = re.search(r'qa_json\s*=\s*(\{.*\}|\[.*\])\s*$', s, flags=re.DOTALL)
    if m2:
        return m2.group(1).strip()

    # If the whole thing contains a dict/list somewhere, take the largest bracketed block
    # (simple heuristic; good enough for LLM outputs)
    candidates = re.findall(r'(\{.*\}|\[.*\])', s, flags=re.DOTALL)
    if candidates:
        # pick the longest candidate
        return max(candidates, key=len).strip()

    return s


def _parse_jsonlike(text: str) -> Tuple[JSONLike, List[str]]:
    """
    Parse LLM output into dict/list.
    Tries JSON first, then Python literal eval.
    Returns (obj, parse_notes).
    Raises ValueError if cannot parse.
    """
    notes = []
    payload = _strip_wrapper(text)

    # Try strict JSON
    try:
        obj = json.loads(payload)
        notes.append("parsed_as_json")
        return obj, notes
    except Exception:
        notes.append("json_load_failed")

    # Try Python literal (handles single quotes, etc.)
    try:
        obj = ast.literal_eval(payload)
        notes.append("parsed_as_python_literal")
        return obj, notes
    except Exception as e:
        raise ValueError(f"Cannot parse output as JSON or Python literal. Error: {e}") from e


def _is_column_oriented(obj: Any, expected_keys: List[str]) -> bool:
    """
    Column-oriented means:
      - dict
      - keys include expected keys
      - values are lists (or list-like) of same length
    """
    if not isinstance(obj, dict):
        return False
    if not all(k in obj for k in expected_keys):
        # still could be column-oriented but missing keys; we treat as not column-oriented later via missing penalty
        pass
    # If most values are lists, likely column oriented
    listish = 0
    for v in obj.values():
        if isinstance(v, list):
            listish += 1
    return listish >= max(1, int(0.7 * max(1, len(obj))))


def _is_row_oriented(obj: Any, expected_keys: List[str]) -> bool:
    """
    Row-oriented means:
      - list of dicts
      - dict keys overlap with expected keys
    """
    if not isinstance(obj, list) or len(obj) == 0:
        return False
    if not all(isinstance(x, dict) for x in obj):
        return False
    # overlap check
    overlap = 0
    for k in expected_keys:
        if k in obj[0]:
            overlap += 1
    return overlap >= max(1, int(0.5 * len(expected_keys)))


def format_mismatch_score(
    llm_output: str,
    expected_schema: Dict[str, str],
    expected_num_samples: int,
    require_exact_keys: bool = True,
) -> Dict[str, Any]:
    """
    expected_schema: {column_name: "int"/"float"/"str"/"bool"/"category"/...}
      - used for basic type checks
    Returns a dict report:
      {
        "score": float,
        "issues": [...],
        "orientation": {"expected": "...", "got": "..."},
        "missing_keys": [...],
        "extra_keys": [...],
        "length_issues": {...},
        "type_issues": {...},
        "parse_notes": [...]
      }
    """
    expected_keys = list(expected_schema.keys())
    issues: List[str] = []
    score = 0.0

    try:
        obj, parse_notes = _parse_jsonlike(llm_output)
    except ValueError as e:
        return {
            "score": 100.0,
            "issues": [f"parse_error: {e}"],
            "orientation": {"expected": "column_oriented", "got": "unparseable"},
            "missing_keys": expected_keys,
            "extra_keys": [],
            "length_issues": {},
            "type_issues": {},
            "parse_notes": [],
        }

    # Expected orientation
    expected_orientation = "column_oriented"

    got_orientation = "unknown"
    if _is_column_oriented(obj, expected_keys):
        got_orientation = "column_oriented"
    elif _is_row_oriented(obj, expected_keys):
        got_orientation = "row_oriented"
    else:
        got_orientation = type(obj).__name__

    if got_orientation != expected_orientation:
        issues.append(f"orientation_mismatch: expected={expected_orientation}, got={got_orientation}")
        score += 30.0  # big penalty because your pipeline needs exact format

    # Key checks depend on orientation
    missing_keys: List[str] = []
    extra_keys: List[str] = []

    length_issues: Dict[str, Any] = {}
    type_issues: Dict[str, Any] = {}

    # --- If row-oriented, we can still compute missing/extra keys from row dicts
    if got_orientation == "row_oriented":
        rows: List[Dict[str, Any]] = obj  # type: ignore
        all_keys = set().union(*(r.keys() for r in rows))
        missing_keys = [k for k in expected_keys if k not in all_keys]
        extra_keys = [k for k in all_keys if k not in expected_keys]

        if len(rows) != expected_num_samples:
            issues.append(f"num_samples_mismatch: expected={expected_num_samples}, got={len(rows)}")
            score += 10.0 + abs(len(rows) - expected_num_samples) * 1.0

        # type checks on row orientation (spot-check each col)
        for col, t in expected_schema.items():
            if col not in all_keys:
                continue
            bad = 0
            for r in rows:
                if col not in r:
                    bad += 1
                    continue
                if not _value_matches_type(r[col], t):
                    bad += 1
            if bad > 0:
                type_issues[col] = {"bad_count": bad, "total": len(rows), "expected_type": t}
                score += min(10.0, bad * 0.5)

    # --- Column-oriented: strict checks
    elif isinstance(obj, dict):
        d: Dict[str, Any] = obj

        present_keys = set(d.keys())
        missing_keys = [k for k in expected_keys if k not in present_keys]
        extra_keys = [k for k in present_keys if k not in expected_keys]

        # key penalties
        if missing_keys:
            issues.append(f"missing_keys: {missing_keys}")
            score += 5.0 * len(missing_keys)
        if extra_keys:
            issues.append(f"extra_keys: {extra_keys}")
            score += 2.0 * len(extra_keys)

        # length checks
        col_lengths = {}
        for k in expected_keys:
            if k in d and isinstance(d[k], list):
                col_lengths[k] = len(d[k])

        if col_lengths:
            # all expected columns should match expected_num_samples
            for k, L in col_lengths.items():
                if L != expected_num_samples:
                    length_issues[k] = {"expected": expected_num_samples, "got": L}
                    score += 3.0 + abs(L - expected_num_samples) * 0.5

            # check consistency across columns
            lengths_set = set(col_lengths.values())
            if len(lengths_set) > 1:
                issues.append(f"inconsistent_column_lengths: {col_lengths}")
                score += 10.0
        else:
            issues.append("no_list_columns_found")
            score += 20.0

        # type checks
        for col, t in expected_schema.items():
            if col not in d or not isinstance(d[col], list):
                continue
            bad = 0
            for v in d[col]:
                if not _value_matches_type(v, t):
                    bad += 1
            if bad > 0:
                type_issues[col] = {"bad_count": bad, "total": len(d[col]), "expected_type": t}
                score += min(10.0, bad * 0.5)

        # strictness toggle
        if require_exact_keys and (missing_keys or extra_keys):
            issues.append("require_exact_keys_violation")
            score += 10.0

    else:
        issues.append(f"unexpected_top_level_type: {type(obj).__name__}")
        score += 40.0

    return {
        "score": round(score, 3),
        "issues": issues,
        "orientation": {"expected": expected_orientation, "got": got_orientation},
        "missing_keys": missing_keys,
        "extra_keys": extra_keys,
        "length_issues": length_issues,
        "type_issues": type_issues,
        "parse_notes": parse_notes,
    }


def _value_matches_type(v: Any, t: str) -> bool:
    t = t.lower().strip()
    if t in {"int", "integer"}:
        return isinstance(v, int) and not isinstance(v, bool)
    if t in {"float", "double", "number"}:
        return (isinstance(v, (int, float)) and not isinstance(v, bool))
    if t in {"str", "string", "category"}:
        return isinstance(v, str)
    if t in {"bool", "boolean"}:
        return isinstance(v, bool)
    # fallback: accept anything
    return True

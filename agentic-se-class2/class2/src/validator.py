"""Minimal JSON Schema (draft-07 subset) validator, stdlib only."""
import re


def _type_ok(v, t):
    if t == "null": return v is None
    if t == "boolean": return isinstance(v, bool)
    if t == "integer": return isinstance(v, int) and not isinstance(v, bool)
    if t == "number": return isinstance(v, (int, float)) and not isinstance(v, bool)
    if t == "string": return isinstance(v, str)
    if t == "object": return isinstance(v, dict)
    if t == "array": return isinstance(v, list)
    return False


def validate(inst, schema, path="$"):
    errs = []
    t = schema.get("type")
    if t is not None:
        ts = t if isinstance(t, list) else [t]
        if not any(_type_ok(inst, x) for x in ts):
            return [f"{path}: expected {ts}, got {type(inst).__name__}"]
    if "enum" in schema and inst not in schema["enum"]:
        errs.append(f"{path}: {inst!r} not in enum")
    if "const" in schema and inst != schema["const"]:
        errs.append(f"{path}: != const")
    if isinstance(inst, str):
        if "pattern" in schema and not re.search(schema["pattern"], inst):
            errs.append(f"{path}: pattern mismatch")
        if "minLength" in schema and len(inst) < schema["minLength"]:
            errs.append(f"{path}: too short")
    if isinstance(inst, dict):
        props = schema.get("properties", {})
        for r in schema.get("required", []):
            if r not in inst:
                errs.append(f"{path}: missing required {r!r}")
        for k, v in inst.items():
            if k in props:
                errs += validate(v, props[k], f"{path}.{k}")
            elif schema.get("additionalProperties", True) is False:
                errs.append(f"{path}: unexpected property {k!r}")
    return errs

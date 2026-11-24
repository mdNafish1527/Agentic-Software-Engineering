"""Contract test for the notice schema.
Accepts an additive change (new optional field); fails breaking changes
(new required field, type change, removed enum value, removed field).
Run: python -m unittest discover -s tests -v"""
import copy, json, pathlib, sys, unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from validator import validate

V1 = json.loads((ROOT / "schema/notice.schema.json").read_text())
KEYS = [json.loads(l)["fields"] for l in (ROOT / "corpus/answer_key.jsonl").read_text().splitlines() if l.strip()]


def golden():
    """One payload per notice type plus the full key as consumer fixtures."""
    return KEYS


def compatible(old_schema, new_schema):
    """Backward compatible = every payload valid under the old schema stays valid under the new one."""
    return all(not validate(p, new_schema) for p in golden() if not validate(p, old_schema))


def bump(s):
    s = copy.deepcopy(s)
    s["x-schema-version"] += 1
    return s


class Contract(unittest.TestCase):
    def test_v1_has_version_and_accepts_golden(self):
        self.assertIsInstance(V1["x-schema-version"], int)
        for p in golden():
            self.assertEqual(validate(p, V1), [])

    def test_additive_change_accepted(self):
        v2 = bump(V1)
        v2["properties"]["registrar_url"] = {"type": ["string", "null"]}   # new optional field
        self.assertTrue(compatible(V1, v2))

    def test_additive_enum_value_accepted(self):
        v2 = bump(V1)
        v2["properties"]["notice_type"]["enum"].append("redemption_notice")
        self.assertTrue(compatible(V1, v2))

    def test_breaking_new_required_field(self):
        v2 = bump(V1)
        v2["properties"]["notice_ref"] = {"type": "string"}
        v2["required"].append("notice_ref")
        self.assertFalse(compatible(V1, v2))

    def test_breaking_type_change(self):
        v2 = bump(V1)
        v2["properties"]["amount"] = {"type": ["string", "null"]}
        self.assertFalse(compatible(V1, v2))

    def test_breaking_enum_value_removed(self):
        v2 = bump(V1)
        v2["properties"]["notice_type"]["enum"].remove("dispute_notice")
        self.assertFalse(compatible(V1, v2))

    def test_breaking_field_removed(self):
        v2 = bump(V1)
        del v2["properties"]["currency"]          # additionalProperties:false -> old payloads rejected
        self.assertFalse(compatible(V1, v2))


if __name__ == "__main__":
    unittest.main()

import json
import sys

with open("decision/virtual_life/schema.json", encoding="utf-8") as fh:
    schema = json.load(fh)
for key, node in schema.items():
    print("##", key)
    for name, item in node.get("items", {}).items():
        if isinstance(item, dict) and item.get("type") == "object":
            print("   ", name, "{object}")
            for nested, meta in item.get("items", {}).items():
                print("      ", nested, "=", meta.get("default"), "|", (meta.get("description") or "")[:70])
        else:
            print("   ", name, "=", item.get("default"), "|", (item.get("description") or "")[:70])
sys.stdout.flush()

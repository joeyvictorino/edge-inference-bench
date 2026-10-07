#!/usr/bin/env python3
"""Minimal reader for models.yaml (no PyYAML dependency).

Supports exactly the shape used in this repository:

    models:
      - id: value
        key: value
        other: "quoted value"

Usage:
    modelsyaml.py ids                      -> one id per line
    modelsyaml.py get <id> <field>         -> field value (empty line if blank)
    modelsyaml.py json                     -> whole list as JSON
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PATH = os.path.join(HERE, "..", "..", "models.yaml")


def _unquote(v):
    v = v.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        return v[1:-1]
    return v


def load(path=DEFAULT_PATH):
    models = []
    current = None
    in_list = False
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.split("#", 1)[0].rstrip() if not raw.lstrip().startswith("#") else ""
            if not line.strip():
                continue
            if line.strip() == "models:":
                in_list = True
                continue
            if not in_list:
                continue
            stripped = line.strip()
            if stripped.startswith("- "):
                current = {}
                models.append(current)
                stripped = stripped[2:]
            if ":" not in stripped:
                raise ValueError("cannot parse line: %r" % raw)
            key, value = stripped.split(":", 1)
            if current is None:
                raise ValueError("key outside list item: %r" % raw)
            current[key.strip()] = _unquote(value)
    for m in models:
        for required in ("id", "repo", "file"):
            if not m.get(required):
                raise ValueError("model entry missing %r: %r" % (required, m))
        m.setdefault("sha256", "")
        m.setdefault("mlx_equivalent", "")
    return models


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    cmd = argv[1]
    models = load()
    if cmd == "ids":
        for m in models:
            print(m["id"])
        return 0
    if cmd == "json":
        print(json.dumps(models, indent=2, sort_keys=True))
        return 0
    if cmd == "get":
        if len(argv) != 4:
            print("usage: modelsyaml.py get <id> <field>", file=sys.stderr)
            return 2
        wanted, field = argv[2], argv[3]
        for m in models:
            if m["id"] == wanted:
                print(m.get(field, ""))
                return 0
        print("unknown model id: %s" % wanted, file=sys.stderr)
        return 1
    print("unknown command: %s" % cmd, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))

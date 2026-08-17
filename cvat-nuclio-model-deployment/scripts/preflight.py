#!/usr/bin/env python3
"""Read-only checks for a Docker-based CVAT Nuclio detector source tree."""

import argparse
import json
import shutil
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    required = ["function.yaml", "Dockerfile", "main.py"]
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise SystemExit(f"missing required files: {', '.join(missing)}")

    config = yaml.safe_load((source / "function.yaml").read_text())
    metadata = config.get("metadata", {})
    spec = config.get("spec", {})
    annotations = metadata.get("annotations", {})
    try:
        labels = json.loads(annotations.get("spec", "[]"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid annotations.spec JSON: {exc}") from exc

    print(f"source={source}")
    print(f"function={metadata.get('name')} project={metadata.get('namespace')}")
    print(f"runtime={spec.get('runtime')} handler={spec.get('handler')}")
    print(f"image={spec.get('build', {}).get('image')}")
    print(f"labels={json.dumps(labels, ensure_ascii=True)}")

    if not labels:
        raise SystemExit("annotations.spec has no labels")
    unsupported = [item for item in labels if item.get("type") not in {"rectangle", "polygon", "skeleton", "ellipse", "cuboid"}]
    if unsupported:
        raise SystemExit(f"unsupported CVAT shape types: {unsupported}")
    if not shutil.which("docker"):
        raise SystemExit("docker is not available")
    if not shutil.which("nuctl"):
        raise SystemExit("nuctl is not available")
    print("tools=docker,nuctl available")


if __name__ == "__main__":
    main()

"""MiCo environment installation and dependency validation."""

import argparse
import base64
import csv
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

NAME = "decord-0.6.0-py3-none-manylinux2010_x86_64.whl"
TAG = "decord-0.6.0.dist-info/WHEEL"
RECORD = "decord-0.6.0.dist-info/RECORD"
OLD = b"Tag: cp36-cp36m-manylinux2010_x86_64"
NEW = b"Tag: py3-none-manylinux2010_x86_64"


def configure_wheel(original, destination):
    with zipfile.ZipFile(original) as archive:
        contents = {n: archive.read(n) for n in archive.namelist()}
    if original.name != NAME or OLD not in contents[TAG]:
        raise ValueError("Unexpected wheel; refusing to change its compatibility metadata")
    contents[TAG] = contents[TAG].replace(OLD, NEW)
    rows = []
    for name, data in sorted(contents.items()):
        if name == RECORD:
            continue
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
        rows.append([name, "sha256=" + digest, str(len(data))])
    rows.append([RECORD, "", ""])
    stream = io.StringIO(newline="")
    csv.writer(stream, lineterminator="\n").writerows(rows)
    contents[RECORD] = stream.getvalue().encode()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in contents.items():
            archive.writestr(name, data)
    with zipfile.ZipFile(original) as a, zipfile.ZipFile(destination) as b:
        changed = [name for name in a.namelist() if a.read(name) != b.read(name)]
    assert set(changed) == {TAG, RECORD}, changed
    proof = {
        "original_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
        "configured_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "changed_members": changed,
        "runtime_files_unchanged": True,
    }
    print(json.dumps(proof, indent=2))
    return proof


COMPONENTS = {
    "LLaVA-MiCo": (3, 10),
    "LLaVA-Video-MiCo": (3, 10),
    "VLMEvalKit-MiCo": (3, 11),
    "VLMEvalKit-MiCo-Qwen3": (3, 12),
    "Qwen3.5-MiCo": (3, 12),
}


def configure_video_dependency():
    """Normalize the compatibility tag of the published decord 0.6.0 wheel."""
    try:
        distribution = importlib.metadata.distribution("decord")
    except importlib.metadata.PackageNotFoundError:
        return
    if distribution.version != "0.6.0" or OLD.decode() not in (
        distribution.read_text("WHEEL") or ""
    ):
        return
    with tempfile.TemporaryDirectory(prefix="mico-setup-") as directory:
        root = Path(directory)
        original = root / "download"
        configured = root / "configured" / NAME
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "download",
                "--no-deps",
                "decord==0.6.0",
                "--dest",
                str(original),
            ],
            check=True,
        )
        configure_wheel(original / NAME, configured)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--force-reinstall",
                "--no-deps",
                str(configured),
            ],
            check=True,
        )


def check_requirements(path):
    from packaging.requirements import Requirement

    problems = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "-")):
            continue
        requirement = Requirement(line)
        if requirement.marker and not requirement.marker.evaluate():
            continue
        try:
            version = importlib.metadata.version(requirement.name)
            if not requirement.specifier.contains(version):
                problems.append(
                    f"{requirement.name}: installed {version}, expected {requirement.specifier}"
                )
        except importlib.metadata.PackageNotFoundError:
            problems.append(f"{requirement.name}: not installed")
    if problems:
        raise SystemExit("\n".join(problems))


def main():
    parser = argparse.ArgumentParser(description="Install or verify a MiCo component environment.")
    parser.add_argument("--component", required=True, choices=COMPONENTS)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify installed dependencies without changing the environment",
    )
    args = parser.parse_args()
    expected = COMPONENTS[args.component]
    if sys.version_info[:2] != expected:
        parser.error(f"{args.component} requires Python {expected[0]}.{expected[1]}")
    component = Path(__file__).resolve().parent / args.component
    requirements = component / "requirements.txt"
    if not args.check:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements)], check=True
        )
        configure_video_dependency()
        flash = component / "requirements-flash-attn.txt"
        if flash.exists():
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--no-build-isolation", "-r", str(flash)],
                check=True,
            )
    check_requirements(requirements)
    flash = component / "requirements-flash-attn.txt"
    if flash.exists():
        check_requirements(flash)
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    print(f"{args.component}: environment ready")


if __name__ == "__main__":
    main()

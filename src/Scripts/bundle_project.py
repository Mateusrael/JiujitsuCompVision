"""Create a source-only ZIP for upload when a Git remote is unavailable."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import os
import zipfile


ROOT = Path(__file__).resolve().parents[2]
ROOT_FILES = {"README.md", ".gitignore", ".gitattributes", "bootstrap.sh"}
SOURCE_DIRS = {"install", "launchers", "src", "tests", "docs"}
SOURCE_SUFFIXES = {".py", ".sh", ".md", ".json", ".txt"}


def source_files(root):
    """Allow only source locations; never collect Data, venvs or run artifacts."""
    for name in sorted(ROOT_FILES):
        path = root / name
        if path.is_file() and not path.is_symlink():
            yield path
    for directory in sorted(SOURCE_DIRS):
        for path in sorted((root / directory).rglob("*")):
            relative = path.relative_to(root)
            if (path.is_file() and not path.is_symlink()
                    and not any(part.startswith(".") or part == "__pycache__"
                                for part in relative.parts)
                    and path.suffix in SOURCE_SUFFIXES
                    and path.name != "local_config.sh"
                    and not any(parent.is_symlink() for parent in path.parents
                                if parent != root and root in parent.parents)):
                yield path


def git_value(root, *args):
    try:
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                                text=True, timeout=10, check=True)
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def build_bundle(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    files = list(source_files(root))
    if output in files:
        raise ValueError("Bundle output cannot replace a source file")
    if output.exists():
        raise FileExistsError(f"Bundle already exists; choose another output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=output.parent, suffix=".zip.tmp")
    os.close(descriptor)
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            hashes = {}
            for path in files:
                relative = path.relative_to(root).as_posix()
                content = path.read_bytes()
                # Shell scripts must retain Linux newlines even from a Windows checkout.
                if path.suffix == ".sh":
                    content = content.replace(b"\r\n", b"\n")
                archive.writestr(relative, content)
                hashes[relative] = hashlib.sha256(content).hexdigest()
            provenance = {"git_commit": git_value(root, "rev-parse", "HEAD"),
                          "git_status": git_value(root, "status", "--porcelain"),
                          "sha256": hashes}
            archive.writestr("source_manifest.json", json.dumps(provenance, indent=2) + "\n")
        # Publish complete bytes atomically; never overwrite a concurrent bundle.
        os.link(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return len(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / "VisaoComp-source.zip")
    args = parser.parse_args()
    try:
        count = build_bundle(ROOT, args.output)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Bundle failed: {exc}\n")
    print(f"Bundled {count} source files: {args.output.resolve()}")


if __name__ == "__main__":
    main()

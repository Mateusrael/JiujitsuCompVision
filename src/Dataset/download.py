"""Resumable official ViCoS downloads and conservative ZIP extraction (stdlib)."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
import urllib.error
import urllib.request
import zipfile
import zlib


BASE_URL = "https://data.vicos.si/datasets/JuiJuitsu/"
SOURCES = {name: BASE_URL + name for name in ("annotations.json", "images.zip")}
CHUNK = 1024 * 1024


def atomic_json(path, value):
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(value, handle, indent=2)
            handle.write("\n")
        except BaseException:
            handle.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance(path, url, downloaded=False):
    sidecar = path.with_name(path.name + ".source.json")
    if sidecar.is_symlink():
        raise ValueError(f"Provenance must not be a symbolic link: {sidecar}")
    prior = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {}
    if not isinstance(prior, dict):
        raise ValueError(f"Invalid provenance record: {sidecar}")
    checksum = digest_file(path)
    if prior and (prior.get("source_url") != url or prior.get("sha256") != checksum):
        raise ValueError(f"Existing provenance conflicts with {path}; refusing to replace it.")
    now = datetime.now(timezone.utc).isoformat()
    report = {"source_url": url, "size_bytes": path.stat().st_size, "sha256": checksum,
              "checksum_scope": "Locally computed integrity hash; not an official publisher checksum.",
              "downloaded_at": prior.get("downloaded_at", now if downloaded else None), "verified_at": now}
    atomic_json(sidecar, report)
    return report


def ensure_space(directory, needed, action):
    free = shutil.disk_usage(directory).free
    margin = max(64 * CHUNK, needed // 20)
    print(f"{action}: filesystem free={free:,} bytes; payload needed={needed:,}; margin={margin:,}. Quota/persistence unknown.")
    if free < needed + margin:
        raise OSError(f"Insufficient filesystem space for {action}; quota may be more restrictive.")


def validate_download(path):
    if path.name.startswith("annotations.json"):
        with path.open(encoding="utf-8") as handle:
            records = json.load(handle)
        if not isinstance(records, list) or not records or not all(isinstance(row, dict) for row in records):
            raise ValueError("Annotations must be a nonempty JSON array of objects.")
    else:
        with zipfile.ZipFile(path) as archive:
            if not archive.infolist():
                raise ValueError("The images ZIP is empty.")


def publish_partial(partial, target, url, metadata):
    validate_download(partial)
    # Hard-link publication is atomic and fails if another process created target.
    os.link(partial, target)
    partial.unlink()
    metadata.unlink(missing_ok=True)
    return provenance(target, url, downloaded=True)


def download_file(name, data_dir, timeout=60):
    url = SOURCES[name]
    target = data_dir / name
    partial = target.with_name(name + ".part")
    metadata = target.with_name(name + ".part.json")
    if any(path.is_symlink() for path in (target, partial, metadata)):
        raise ValueError("Download paths must not be symbolic links.")
    if target.exists():
        validate_download(target)
        print(f"Preserving existing {target}.")
        return provenance(target, url)
    state = json.loads(metadata.read_text(encoding="utf-8")) if metadata.exists() else {}
    if not isinstance(state, dict):
        raise ValueError(f"Invalid partial metadata: {metadata}")
    if (state and state.get("source_url") != url) or (partial.exists() and partial.stat().st_size and not state):
        raise ValueError(f"Unrecognized partial download for {target}; move it aside before retrying.")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset and offset == state.get("total_bytes"):
        return publish_partial(partial, target, url, metadata)
    advertised = None
    try:
        request = urllib.request.Request(url, method="HEAD", headers={"Accept-Encoding": "identity"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            advertised = int(response.headers["Content-Length"]) if response.headers.get("Content-Length") else None
    except (OSError, urllib.error.URLError, ValueError) as exc:
        print(f"HEAD unavailable for {name}: {exc}; using GET headers.")
    print(f"{name}: HEAD advertised bytes={advertised if advertised is not None else 'unknown'}.")
    validator = state.get("etag") or state.get("last_modified")
    headers = {"Accept-Encoding": "identity"}
    resume = bool(offset and validator)
    if resume:
        headers.update({"Range": f"bytes={offset}-", "If-Range": validator})
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.headers.get("Content-Encoding", "identity") != "identity":
            raise ValueError("Encoded HTTP response is incompatible with byte-range integrity checks.")
        length = int(response.headers["Content-Length"]) if response.headers.get("Content-Length") else None
        etag = response.headers.get("ETag")
        etag = etag if etag and not etag.startswith("W/") else None
        modified = response.headers.get("Last-Modified")
        if response.status == 206:
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
            if not resume or not match:
                raise ValueError("Unexpected or invalid HTTP Content-Range.")
            start, end, total = map(int, match.groups())
            if start != offset or end < start or total <= end or (length is not None and length != end - start + 1):
                raise ValueError("HTTP Content-Range does not match the saved partial download.")
            if state.get("total_bytes") not in (None, total):
                raise ValueError("Remote size changed during a ranged download.")
            if (state.get("etag") and etag and state["etag"] != etag) or (
                    not state.get("etag") and modified and modified != validator):
                raise ValueError("Remote validator changed during a ranged download.")
            length = end - start + 1
        elif response.status == 200:
            offset = 0  # A server ignoring Range or changing If-Range restarts cleanly.
            total = length if length is not None else advertised
        else:
            raise ValueError(f"Unexpected HTTP status {response.status}.")
        if total is not None and total < offset:
            raise ValueError("Invalid advertised download size.")
        ensure_space(data_dir, max(0, (total or 0) - offset), f"Download {name}")
        received = 0
        with partial.open("ab" if offset else "wb") as handle:
            # Truncate stale bytes before saving a new validator for a 200 restart.
            atomic_json(metadata, {"source_url": url, "etag": etag or (state.get("etag") if offset else None),
                                  "last_modified": modified or (state.get("last_modified") if offset else None), "total_bytes": total})
            while chunk := response.read(CHUNK):
                if length is not None and received + len(chunk) > length:
                    raise ValueError("HTTP response exceeds its advertised length.")
                handle.write(chunk)
                received += len(chunk)
        if (length is not None and received != length) or (total is not None and offset + received != total):
            raise OSError(f"Incomplete download of {name}; partial bytes preserved for retry.")
    return publish_partial(partial, target, url, metadata)


def crc_file(path):
    crc = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            crc = zlib.crc32(chunk, crc)
    return crc & 0xFFFFFFFF


def safe_entries(archive, destination):
    entries, seen = [], {}
    for info in archive.infolist():
        name = info.filename.rstrip("/")
        parts = name.split("/")
        kind = stat.S_IFMT(info.external_attr >> 16)
        if (not name or info.filename != info.orig_filename or "\\" in name
                or PurePosixPath(name).is_absolute() or any(part in ("", ".", "..") or ":" in part for part in parts)
                or kind not in (0, stat.S_IFREG, stat.S_IFDIR)):
            raise ValueError(f"Unsafe ZIP entry: {info.filename!r}")
        key = name.casefold()
        if key in seen:
            raise ValueError(f"Duplicate ZIP target: {name}")
        seen[key] = info.is_dir()
        target = destination.joinpath(*parts)
        if any(path.is_symlink() for path in (target, *target.parents)):
            raise ValueError(f"ZIP target contains a symbolic link: {target}")
        entries.append((info, target))
    for key in seen:
        if any(seen.get(str(parent)) is False for parent in PurePosixPath(key).parents):
            raise ValueError(f"ZIP file/directory conflict: {key}")
    return entries


def extract_images(archive_path, destination):
    destination = Path(destination).absolute()
    if destination.is_symlink():
        raise ValueError("Images destination must not be a symbolic link.")
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        entries = safe_entries(archive, destination)
        needed = 0
        for info, target in entries:
            if target.exists():
                valid = target.is_dir() if info.is_dir() else (
                    target.is_file() and target.stat().st_size == info.file_size and crc_file(target) == info.CRC)
                if not valid:
                    raise ValueError(f"Existing extraction differs; refusing to overwrite {target}.")
            elif not info.is_dir():
                needed += info.file_size
        total = sum(info.file_size for info, _ in entries)
        print(f"ZIP uncompressed total={total:,} bytes; extracting under {destination}.")
        ensure_space(destination, needed, "Extract images")
        written = 0
        for info, target in entries:
            if target.exists():
                # Read ZIP data too, verifying its CRC even when output already exists.
                if not info.is_dir():
                    with archive.open(info) as source:
                        while source.read(CHUNK):
                            pass
                continue
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile("wb", dir=target.parent, prefix=".extract-", delete=False) as handle:
                    temporary = Path(handle.name)
                    with archive.open(info) as source:
                        shutil.copyfileobj(source, handle, CHUNK)  # ZipFile validates CRC on EOF.
                os.link(temporary, target)
                written += 1
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        roots = sorted({info.filename.split("/")[0] for info, _ in entries if "/" in info.filename.rstrip("/")})
        return {"images_dir": str(destination), "uncompressed_bytes": total,
                "files_written": written, "files_total": sum(not info.is_dir() for info, _ in entries),
                "archive_subdirectories": roots}


def prepare_data(data_dir, images=False, extract_only=False, timeout=60):
    data_dir = Path(data_dir).expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    report = {"data_dir": str(data_dir), "downloads": {}}
    if not extract_only:
        report["downloads"]["annotations"] = download_file("annotations.json", data_dir, timeout)
        if images:
            report["downloads"]["images"] = download_file("images.zip", data_dir, timeout)
    if images or extract_only:
        archive = data_dir / "images.zip"
        if not archive.is_file():
            raise ValueError(f"--extract-only requires an existing {archive}.")
        report["extraction"] = extract_images(archive, data_dir / "images")
        report["downloads"]["images"] = provenance(archive, SOURCES["images.zip"])
    return report

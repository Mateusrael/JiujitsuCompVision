import contextlib
import io
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
import urllib.error
import zipfile

from src.Dataset import download


class Response(io.BytesIO):
    def __init__(self, body=b"", status=200, **headers):
        super().__init__(body)
        self.status = status
        self.headers = headers


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.payload = b'[{"image":"0000001"}]'
        self.name = "annotations.json"
        self.url = download.SOURCES[self.name]
        self.quiet = contextlib.redirect_stdout(io.StringIO())
        self.quiet.__enter__()
        self.addCleanup(self.quiet.__exit__, None, None, None)

    def partial(self, size=5, url=None):
        (self.directory / (self.name + ".part")).write_bytes(self.payload[:size])
        (self.directory / (self.name + ".part.json")).write_text(json.dumps({
            "source_url": url or self.url, "etag": '"version1"', "total_bytes": len(self.payload)
        }), encoding="utf-8")

    def head(self):
        return Response(**{"Content-Length": str(len(self.payload))})

    def test_resume_validates_range_and_records_integrity(self):
        self.partial()
        response = Response(self.payload[5:], 206, **{
            "Content-Range": f"bytes 5-{len(self.payload)-1}/{len(self.payload)}",
            "Content-Length": str(len(self.payload)-5), "ETag": '"version1"'})
        with patch.object(download.urllib.request, "urlopen", side_effect=[self.head(), response]) as request:
            result = download.download_file(self.name, self.directory)
        headers = request.call_args_list[1].args[0].headers
        self.assertEqual(headers["Range"], "bytes=5-")
        self.assertEqual(headers["If-range"], '"version1"')
        self.assertEqual((self.directory / self.name).read_bytes(), self.payload)
        self.assertEqual(len(result["sha256"]), 64)
        self.assertFalse((self.directory / (self.name + ".part")).exists())

    def test_server_ignoring_range_restarts_without_duplicate_bytes(self):
        self.partial()
        with patch.object(download.urllib.request, "urlopen", side_effect=[self.head(), Response(self.payload, **{
                "Content-Length": str(len(self.payload)), "ETag": '"version2"'})]):
            download.download_file(self.name, self.directory)
        self.assertEqual((self.directory / self.name).read_bytes(), self.payload)

    def test_restart_metadata_failure_cannot_keep_stale_bytes_with_new_validator(self):
        self.partial()
        with patch.object(download.urllib.request, "urlopen", side_effect=[self.head(), Response(self.payload, **{
                "Content-Length": str(len(self.payload)), "ETag": '"version2"'})]), \
                patch.object(download, "atomic_json", side_effect=OSError("disk write failed")):
            with self.assertRaisesRegex(OSError, "disk write"):
                download.download_file(self.name, self.directory)
        self.assertEqual((self.directory / (self.name + ".part")).read_bytes(), b"")
        saved = json.loads((self.directory / (self.name + ".part.json")).read_text(encoding="utf-8"))
        self.assertEqual(saved["etag"], '"version1"')

    def test_incomplete_response_keeps_partial_and_no_final(self):
        with patch.object(download.urllib.request, "urlopen", side_effect=[self.head(), Response(self.payload[:5], **{
                "Content-Length": str(len(self.payload)), "ETag": '"version1"'})]):
            with self.assertRaisesRegex(OSError, "Incomplete"):
                download.download_file(self.name, self.directory)
        self.assertEqual((self.directory / (self.name + ".part")).read_bytes(), self.payload[:5])
        self.assertFalse((self.directory / self.name).exists())

    def test_invalid_range_preserves_partial(self):
        self.partial()
        with patch.object(download.urllib.request, "urlopen", side_effect=[self.head(), Response(self.payload, 206, **{
                "Content-Range": f"bytes 0-{len(self.payload)-1}/{len(self.payload)}"})]):
            with self.assertRaisesRegex(ValueError, "Content-Range"):
                download.download_file(self.name, self.directory)
        self.assertEqual((self.directory / (self.name + ".part")).read_bytes(), self.payload[:5])

    def test_unknown_source_partial_is_not_reused(self):
        self.partial(url="https://example.invalid/other")
        with patch.object(download.urllib.request, "urlopen") as request:
            with self.assertRaisesRegex(ValueError, "Unrecognized"):
                download.download_file(self.name, self.directory)
            request.assert_not_called()

    def test_head_failure_uses_get_headers(self):
        with patch.object(download.urllib.request, "urlopen", side_effect=[urllib.error.URLError("HEAD blocked"),
                Response(self.payload, **{"Content-Length": str(len(self.payload))})]):
            download.download_file(self.name, self.directory)
        self.assertEqual((self.directory / self.name).read_bytes(), self.payload)

    def test_invalid_existing_annotations_are_preserved(self):
        target = self.directory / self.name
        target.write_text("not json", encoding="utf-8")
        with patch.object(download.urllib.request, "urlopen") as request:
            with self.assertRaises(ValueError):
                download.download_file(self.name, self.directory)
            request.assert_not_called()
        self.assertEqual(target.read_text(encoding="utf-8"), "not json")

    def make_archive(self, names):
        path = self.directory / "images.zip"
        with zipfile.ZipFile(path, "w") as archive:
            for name in names:
                info = zipfile.ZipInfo(name) if isinstance(name, str) else name
                if isinstance(name, str):
                    info.filename = name  # Keep malicious backslashes verbatim on Windows.
                archive.writestr(info, b"image bytes")
        return path

    def test_extract_rejects_unsafe_paths_and_symlinks(self):
        link = zipfile.ZipInfo("link")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        for name in ("../escape.jpg", "/absolute.jpg", "C:/drive.jpg", "a\\b.jpg", link):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Unsafe"):
                download.extract_images(self.make_archive([name]), self.directory / "images")
        self.assertFalse((self.directory / "escape.jpg").exists())

    def test_extract_preserves_matches_and_refuses_modified_files(self):
        archive = self.make_archive(["0101166.jpg", "nested/another.jpg"])
        destination = self.directory / "images"
        self.assertEqual(download.extract_images(archive, destination)["files_written"], 2)
        self.assertEqual(download.extract_images(archive, destination)["files_written"], 0)
        (destination / "0101166.jpg").write_bytes(b"other bytes")
        with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
            download.extract_images(archive, destination)
        self.assertEqual((destination / "0101166.jpg").read_bytes(), b"other bytes")

    def test_case_duplicate_zip_targets_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            download.extract_images(self.make_archive(["image.jpg", "IMAGE.jpg"]), self.directory / "images")

    def test_corrupt_zip_crc_never_publishes_a_file(self):
        archive = self.make_archive(["image.jpg"])
        archive.write_bytes(archive.read_bytes().replace(b"image bytes", b"wrong bytes"))
        with self.assertRaises(zipfile.BadZipFile):
            download.extract_images(archive, self.directory / "images")
        self.assertEqual(list((self.directory / "images").iterdir()), [])

    def test_extract_only_never_uses_network(self):
        self.make_archive(["0101166.jpg"])
        with patch.object(download.urllib.request, "urlopen") as request:
            result = download.prepare_data(self.directory, extract_only=True)
            request.assert_not_called()
        self.assertEqual(result["extraction"]["files_written"], 1)


if __name__ == "__main__":
    unittest.main()

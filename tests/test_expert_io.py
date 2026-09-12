import io
import unittest
import zipfile
from unittest.mock import patch

from fly_fruit_fly.expert import _HTTPRangeReader


class Response(io.BytesIO):
    def __init__(self, data, status, content_range):
        super().__init__(data)
        self.status = status
        self.headers = {"Content-Range": content_range}


class RangeReaderTests(unittest.TestCase):
    def test_extracts_member_with_real_zip_parser(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("data/wing_pattern_fmech.npy", b"test pattern payload")
        data = buffer.getvalue()
        def respond(request, timeout):
            self.assertEqual(timeout, 60)
            start, end = map(int, request.headers["Range"].split("=")[1].split("-"))
            return Response(data[start:end+1], 206, f"bytes {start}-{end}/{len(data)}")
        with patch("urllib.request.urlopen", side_effect=respond):
            with zipfile.ZipFile(_HTTPRangeReader("https://example.test/zip", len(data))) as archive:
                self.assertEqual(archive.read("data/wing_pattern_fmech.npy"), b"test pattern payload")

    def test_refuses_full_download_response(self):
        with patch("urllib.request.urlopen", return_value=Response(b"1234", 200, "")):
            with self.assertRaises(RuntimeError):
                _HTTPRangeReader("https://example.test/zip", 4).read(4)

    def test_refuses_wrong_range_or_short_response(self):
        for data, header in [(b"1234", "bytes 1-4/4"), (b"1", "bytes 0-3/4")]:
            with patch("urllib.request.urlopen", return_value=Response(data, 206, header)):
                with self.assertRaises(RuntimeError):
                    _HTTPRangeReader("https://example.test/zip", 4).read(4)

    def test_bounds_and_eof(self):
        reader = _HTTPRangeReader("https://example.test/zip", 2_000_000)
        with self.assertRaises(RuntimeError):
            reader.read()
        with self.assertRaises(ValueError):
            reader.seek(-1)
        reader.seek(0, 2)
        self.assertEqual(reader.read(), b"")


if __name__ == "__main__":
    unittest.main()

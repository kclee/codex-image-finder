from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from image_finder.ui import copy_image_file_to_clipboard


class FakeClipboard:
    def __init__(self) -> None:
        self.image = None

    def setImage(self, image) -> None:  # type: ignore[no-untyped-def]
        self.image = image


class ClipboardTests(unittest.TestCase):
    def test_copy_image_places_decoded_pixels_without_modifying_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "synthetic.png"
            Image.new("RGB", (8, 6), "purple").save(source)
            original = source.read_bytes()
            clipboard = FakeClipboard()

            self.assertTrue(copy_image_file_to_clipboard(source, clipboard))
            self.assertIsNotNone(clipboard.image)
            self.assertEqual((clipboard.image.width(), clipboard.image.height()), (8, 6))
            self.assertEqual(source.read_bytes(), original)

    def test_copy_image_rejects_non_image_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "synthetic.txt"
            source.write_text("not an image", encoding="utf-8")
            original = source.read_bytes()
            clipboard = FakeClipboard()

            self.assertFalse(copy_image_file_to_clipboard(source, clipboard))
            self.assertIsNone(clipboard.image)
            self.assertEqual(source.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()

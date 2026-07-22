from __future__ import annotations

import unittest

import numpy as np

from episode_value_video.images import image_to_pil


class ImageConversionTests(unittest.TestCase):
    def test_hwc_uint8(self) -> None:
        array = np.zeros((12, 16, 3), dtype=np.uint8)
        array[..., 1] = 127
        image = image_to_pil(array)
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, (16, 12))
        self.assertEqual(image.getpixel((0, 0)), (0, 127, 0))

    def test_chw_float_zero_to_one(self) -> None:
        array = np.zeros((3, 8, 10), dtype=np.float32)
        array[0] = 1.0
        image = image_to_pil(array)
        self.assertEqual(image.size, (10, 8))
        self.assertEqual(image.getpixel((0, 0)), (255, 0, 0))

    def test_bad_range_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported float image range"):
            image_to_pil(np.full((8, 10, 3), -1.0, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()

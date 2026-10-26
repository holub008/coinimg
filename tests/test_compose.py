"""Regression coverage for contrast-independent coin location."""
import unittest

import numpy as np
from PIL import Image, ImageDraw

from src.compose import _detect_coin, concatenate


def scene(*, center=(180, 125), radii=(32, 32), color=(225, 185, 65), chipped=False):
    # The phone shadow is much darker and larger than the coin, and touches
    # the frame. The paper also has a gradual lighting change.
    paper = np.broadcast_to(np.linspace(170, 205, 300)[:, None, None], (300, 400, 3)).copy()
    paper[230:] -= 95
    image = Image.fromarray(paper.astype(np.uint8))
    draw = ImageDraw.Draw(image)
    cx, cy = center
    rx, ry = radii
    draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=color)
    # Surface markings should not be mistaken for separate small coins.
    draw.line((cx - rx // 2, cy, cx + rx // 2, cy), fill=(65, 60, 55), width=2)
    if chipped:
        draw.polygon([(cx + rx - 6, cy - 9), (cx + rx + 1, cy - 12),
                      (cx + rx + 1, cy + 6)], fill=(184, 184, 184))
    return image


class CoinDetectionTests(unittest.TestCase):
    def assert_location(self, image, center, radius, tolerance=4):
        cx, cy, detected = _detect_coin(image)
        self.assertAlmostEqual(cx, center[0], delta=tolerance)
        self.assertAlmostEqual(cy, center[1], delta=tolerance)
        self.assertGreaterEqual(detected, radius - 1)
        self.assertLessEqual(detected, radius + tolerance)

    def test_bright_dark_and_similar_luminance_coins(self):
        for color in ((240, 240, 240), (225, 185, 65), (90, 90, 90), (198, 198, 198)):
            with self.subTest(color=color):
                self.assert_location(scene(color=color), (180, 125), 32)

    def test_small_off_center_coin(self):
        self.assert_location(scene(center=(315, 65), radii=(12, 12)), (315, 65), 12)

    def test_chipped_oval_coin(self):
        self.assert_location(scene(radii=(40, 29), chipped=True), (180, 125), 40)

    def test_rotations_and_resolution(self):
        original = scene()
        for angle in (0, 90, 180, 270):
            for scale in (1, 3):
                with self.subTest(angle=angle, scale=scale):
                    image = original.resize((400 * scale, 300 * scale)).rotate(angle, expand=True)
                    center = {0: (180, 125), 90: (125, 219),
                              180: (219, 174), 270: (174, 180)}[angle]
                    # Pixel centers transform about the image's outer edges.
                    center = tuple((c + .5) * scale - .5 for c in center)
                    self.assert_location(image, center, 32 * scale, tolerance=4 * scale)

    def test_empty_shadow_and_non_round_objects_raise(self):
        blank = Image.new('RGB', (400, 300), 'white')
        shadow = blank.copy()
        draw = ImageDraw.Draw(shadow)
        draw.rectangle((0, 230, 399, 299), fill=(65, 65, 65))
        objects = shadow.copy()
        draw = ImageDraw.Draw(objects)
        draw.rectangle((45, 45, 105, 105), fill=(80, 80, 80))
        draw.rectangle((140, 80, 280, 85), fill=(30, 30, 30))
        for image in (blank, shadow, objects, Image.new('RGB', (1, 10))):
            with self.subTest(size=image.size):
                with self.assertRaisesRegex(ValueError, 'could not find a coin'):
                    _detect_coin(image)

    def test_concatenate_centers_and_matches_both_faces(self):
        pair = concatenate(scene(), scene(center=(310, 85), radii=(20, 20)), size=200)
        self.assertEqual(pair.size, (400, 200))
        for offset in (0, 200):
            cx, cy, radius = _detect_coin(pair.crop((offset, 0, offset + 200, 200)))
            self.assertAlmostEqual(cx, 100, delta=4)
            self.assertAlmostEqual(cy, 100, delta=4)
            self.assertAlmostEqual(radius, 200 / 2.32, delta=5)


if __name__ == '__main__':
    unittest.main()

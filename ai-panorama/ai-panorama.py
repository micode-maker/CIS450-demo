#!/usr/bin/env python3
"""
Create a panorama using OpenCV's Stitcher API.
Now optimized to stitch as many images as possible.

Usage:
  python ai-panorama/ai-panorama.py panorama ai-panorama/images/*.png
"""

import sys
import glob
import cv2 as cv
import os
import re


def numeric_sort_key(path):
    """
    Sort files like:
    1-640x480.png
    2-640x480.png
    ...
    10-640x480.png
    correctly by number.
    """
    filename = os.path.basename(path)
    match = re.match(r"(\d+)", filename)
    return int(match.group(1)) if match else filename


def expand_inputs(args):
    """Expand wildcard patterns into sorted file paths."""
    paths = []
    for a in args:
        matches = glob.glob(a)
        if matches:
            paths.extend(matches)
        else:
            paths.append(a)

    # Sort numerically (important!)
    return sorted(paths, key=numeric_sort_key)


def load_images(image_paths, max_width=1200):
    """
    Load and optionally resize images to improve stability.
    """
    imgs = []
    for p in image_paths:
        img = cv.imread(p)
        if img is None:
            print("Could not read:", p)
            continue

        # Resize large images to help stitch many files
        h, w = img.shape[:2]
        if w > max_width:
            scale = max_width / w
            img = cv.resize(img, None, fx=scale, fy=scale)
            print(f"Resized {p}")

        imgs.append(img)

    return imgs


def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python ai-panorama.py panorama <images...>")
        sys.exit(1)

    mode_arg = sys.argv[1].lower()
    if mode_arg not in ("panorama", "scans"):
        print("Mode must be 'panorama' or 'scans'")
        sys.exit(1)

    mode = cv.Stitcher_PANORAMA if mode_arg == "panorama" else cv.Stitcher_SCANS

    image_paths = expand_inputs(sys.argv[2:])

    if len(image_paths) < 2:
        print("Need at least 2 images.")
        sys.exit(1)

    print("Images to stitch:")
    for p in image_paths:
        print("  ", p)

    imgs = load_images(image_paths)

    if len(imgs) < 2:
        print("Not enough valid images to stitch.")
        sys.exit(1)

    # Create stitcher
    stitcher = cv.Stitcher_create(mode)

    # Improve robustness for many images
    stitcher.setPanoConfidenceThresh(0.6)

    print("Stitching", len(imgs), "images...")

    status, pano = stitcher.stitch(imgs)

    if status != cv.Stitcher_OK:
        print("❌ Stitching failed. OpenCV status code:", status)
        print("Common fixes:")
        print("- Ensure images overlap by 30-50%")
        print("- Rotate camera in place")
        print("- Avoid moving objects")
        print("- Keep exposure consistent")
        sys.exit(1)

    out_file = "ai-panorama.jpg"
    cv.imwrite(out_file, pano)

    print("✅ Panorama saved as:", out_file)


if __name__ == "__main__":
    main()

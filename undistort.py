#!.venv/bin/python

import argparse
from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
ALPHAS = (1, 0)


def process_images(image_dir, results_dir, camera_matrix, dist_coeffs, image_size):
    image_paths = sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".jpg"
    )
    if not image_paths:
        raise RuntimeError(f"No JPG images found in {image_dir}.")

    print(f"Processing images from {image_dir}", flush=True)
    undistorted_dir = results_dir / "undistorted"
    comparisons_dir = results_dir / "comparisons"
    comparisons_dir.mkdir(parents=True, exist_ok=True)
    new_camera_matrices = {}
    for alpha in ALPHAS:
        (undistorted_dir / f"alpha{alpha}").mkdir(parents=True, exist_ok=True)
        # Keep the original output size without cropping to the returned ROI.
        new_camera_matrices[alpha], _ = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, image_size, alpha=alpha, newImgSize=image_size
        )

    comparison_count = 0
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Could not read {image_path}.")
        height, width = image.shape[:2]
        if (width, height) != image_size:
            raise RuntimeError(
                f"{image_path.name}: image size {(width, height)} differs from "
                f"the calibration size {image_size}."
            )

        # Add labels above all three full-size images without covering pixels.
        header_height = 100
        comparison = np.zeros((height + header_height, 3 * width, 3), np.uint8)
        comparison[header_height:, :width] = image
        labels = ["Original"]
        for column, alpha in enumerate(ALPHAS, start=1):
            corrected = cv2.undistort(
                image, camera_matrix, dist_coeffs, None, new_camera_matrices[alpha]
            )
            output_path = undistorted_dir / f"alpha{alpha}" / image_path.name
            if not cv2.imwrite(str(output_path), corrected):
                raise RuntimeError(f"Could not save {output_path}.")
            comparison[header_height:, column * width:(column + 1) * width] = corrected
            labels.append(f"Undistorted (alpha={alpha})")

        for column, label in enumerate(labels):
            cv2.putText(
                comparison, label, (column * width + 40, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA
            )
        comparison_path = comparisons_dir / image_path.name
        if not cv2.imwrite(str(comparison_path), comparison):
            raise RuntimeError(f"Could not save {comparison_path}.")
        comparison_count += 1

        print(f"Undistorted: {image_path.name} ({width} x {height})", flush=True)

    print(f"Images processed: {len(image_paths)}")
    print(f"Comparisons saved: {comparison_count}")
    print(f"Results saved in {undistorted_dir} and {comparisons_dir}")


def undistort_camera(camera_id):
    image_dir = BASE_DIR / "images" / camera_id
    test_image_dir = BASE_DIR / "images_test" / camera_id
    results_dir = BASE_DIR / "results" / camera_id
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Camera image directory does not exist: {image_dir}")
    calibration_path = results_dir / "calibration.npz"
    if not calibration_path.is_file():
        raise FileNotFoundError(f"Calibration not found for {camera_id}: {calibration_path}")

    print(f"Undistorting camera: {camera_id}", flush=True)
    with np.load(calibration_path, allow_pickle=False) as parameters:
        camera_matrix = parameters["camera_matrix"]
        dist_coeffs = parameters["dist_coeffs"]
        image_size = tuple(int(value) for value in parameters["image_size"])

    if (
        camera_matrix.shape != (3, 3)
        or not np.isfinite(camera_matrix).all()
        or not np.isfinite(dist_coeffs).all()
    ):
        raise ValueError("The saved calibration parameters are invalid.")

    process_images(image_dir, results_dir, camera_matrix, dist_coeffs, image_size)
    if test_image_dir.is_dir():
        process_images(
            test_image_dir, results_dir / "tests", camera_matrix, dist_coeffs, image_size
        )


def main():
    parser = argparse.ArgumentParser(description="Undistort images using each camera's calibration.")
    parser.add_argument(
        "camera_id", nargs="?", help="Camera directory name (default: all cameras in images/)"
    )
    args = parser.parse_args()
    image_root = BASE_DIR / "images"
    if not image_root.is_dir():
        parser.error(f"Image directory does not exist: {image_root}")

    if args.camera_id is not None:
        camera_id = args.camera_id
        if not camera_id or camera_id in (".", "..") or Path(camera_id).name != camera_id:
            parser.error("camera_id must be a single directory name.")
        if not (image_root / camera_id).is_dir():
            parser.error(f"Camera image directory does not exist: {image_root / camera_id}")
        camera_ids = [camera_id]
    else:
        camera_ids = sorted(path.name for path in image_root.iterdir() if path.is_dir())
        if not camera_ids:
            parser.error(f"No camera directories found in {image_root}.")

    for camera_id in camera_ids:
        undistort_camera(camera_id)


if __name__ == "__main__":
    main()

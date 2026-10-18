#!.venv/bin/python

import argparse
from pathlib import Path

import cv2
import numpy as np


SQUARE_SIZE = 3.1 # cm
PATTERN_SIZE = (4, 4)  # Number of inner corners: columns, rows.
BASE_DIR = Path(__file__).resolve().parent


def calibrate_camera(camera_id):
    image_dir = BASE_DIR / "images" / camera_id
    results_dir = BASE_DIR / "results" / camera_id
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Camera image directory does not exist: {image_dir}")

    print(f"Calibrating camera: {camera_id}", flush=True)

    if not np.isfinite(SQUARE_SIZE) or SQUARE_SIZE <= 0:
        raise ValueError("SQUARE_SIZE must be finite and positive.")

    image_paths = sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".jpg"
    )
    if not image_paths:
        raise RuntimeError(f"No JPG images found in {image_dir}.")

    corners_dir = results_dir / "corners"
    corners_dir.mkdir(parents=True, exist_ok=True)

    object_grid = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
    object_grid[:, :2] = np.mgrid[
        0:PATTERN_SIZE[0], 0:PATTERN_SIZE[1]
    ].T.reshape(-1, 2)
    object_grid *= SQUARE_SIZE

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    object_points = []
    image_points = []
    image_names = []
    discarded = []
    image_size = None

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            reason = f"{image_path.name}: image could not be read"
            discarded.append(reason)
            print(f"Discarded {reason}", flush=True)
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        current_size = (gray.shape[1], gray.shape[0])
        if image_size is None:
            image_size = current_size
        elif current_size != image_size:
            raise RuntimeError(
                f"{image_path.name}: image size {current_size} differs from "
                f"{image_size}. All images must have the same original size."
            )

        found, corners = cv2.findChessboardCorners(gray, PATTERN_SIZE, flags)
        if not found:
            found, corners = cv2.findChessboardCornersSB(
                gray, PATTERN_SIZE, cv2.CALIB_CB_NORMALIZE_IMAGE
            )
        if not found:
            reason = f"{image_path.name}: chessboard not detected"
            discarded.append(reason)
            print(f"Discarded {reason}", flush=True)
            continue

        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        object_points.append(object_grid.copy())
        image_points.append(corners)
        image_names.append(image_path.name)

        cv2.drawChessboardCorners(image, PATTERN_SIZE, corners, True)
        output_path = corners_dir / image_path.name
        if not cv2.imwrite(str(output_path), image):
            raise RuntimeError(f"Could not save {output_path}.")
        print(f"Detected {len(corners)} corners: {image_path.name}", flush=True)

    if len(image_points) < 3:
        raise RuntimeError(
            f"Only {len(image_points)} of {len(image_paths)} images have a detected "
            "chessboard. At least three views are required by this script."
        )

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    if (
        not np.isfinite(rms)
        or not np.all(np.isfinite(camera_matrix))
        or not np.all(np.isfinite(dist_coeffs))
        or camera_matrix[0, 0] <= 0
        or camera_matrix[1, 1] <= 0
    ):
        raise RuntimeError("Calibration returned invalid parameters.")

    reprojection_errors = []
    for points_3d, detected, rvec, tvec in zip(
        object_points, image_points, rvecs, tvecs
    ):
        projected, _ = cv2.projectPoints(
            points_3d, rvec, tvec, camera_matrix, dist_coeffs
        )
        # Average the Euclidean distance of each corresponding pair in pixels.
        residuals = detected.reshape(-1, 2) - projected.reshape(-1, 2)
        error = np.linalg.norm(residuals, axis=1).mean()
        reprojection_errors.append(float(error))

    mean_error = float(np.mean(reprojection_errors))
    report_lines = [
        "Camera calibration",
        f"Camera ID: {camera_id}",
        f"Images processed: {len(image_points)}",
        f"Images discarded: {len(discarded)}",
        f"Resolution: {image_size}",
        f"Inner corners: {PATTERN_SIZE}",
        f"Square size: {SQUARE_SIZE}cm",
        "",
        "Camera intrinsic matrix:",
        np.array2string(camera_matrix, precision=10),
        "",
        "Distortion coefficients (k1, k2, p1, p2, k3):",
        np.array2string(dist_coeffs.ravel(), precision=10),
        "",
        f"RMS: {rms:.10f} pixels",
        f"Mean Euclidean reprojection error: {mean_error:.10f} pixels",
        "",
        "Mean Euclidean reprojection error per image (pixels):",
    ]
    report_lines.extend(
        f"{name}: {error:.10f}"
        for name, error in zip(image_names, reprojection_errors)
    )
    report_lines.extend(["", "Discarded images:"] + (discarded or ["None"]))
    report = "\n".join(report_lines) + "\n"

    np.savez(
        results_dir / "calibration.npz",
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_size=np.array(image_size),
        square_size=SQUARE_SIZE,
        pattern_size=np.array(PATTERN_SIZE),
        rms=rms,
        mean_reprojection_error=mean_error,
        reprojection_errors=np.array(reprojection_errors),
        image_names=np.array(image_names),
    )
    (results_dir / "log.txt").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"Results saved in {results_dir}")


def main():
    parser = argparse.ArgumentParser(description="Calibrate cameras from chessboard images.")
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
        calibrate_camera(camera_id)


if __name__ == "__main__":
    main()

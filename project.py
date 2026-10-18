#!.venv/bin/python

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
PATTERN_SIZE = (4, 4)
SQUARE_SIZE_CM = 3.1
VALIDATION_INDICES = np.array([0, 3, 12, 15])
POSE_INDICES = np.array([index for index in range(16) if index not in VALIDATION_INDICES])


def project_images(camera_id, image_dir, output_dir, camera_matrix, dist_coeffs, image_size):
    image_paths = sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".jpg"
    )
    if not image_paths:
        raise RuntimeError(f"No JPG images found in {image_dir}.")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nProjection dataset: {image_dir}", flush=True)

    # Match calibrate.py: X changes first, followed by Y, with all Z equal to zero.
    object_points = np.zeros((16, 3), dtype=np.float64)
    object_points[:, :2] = np.mgrid[0:4, 0:4].T.reshape(-1, 2)
    object_points *= SQUARE_SIZE_CM
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    all_errors = []
    image_errors = []
    csv_rows = []
    discarded = []
    detected_count = 0

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            reason = f"{image_path.name}: image could not be read"
            discarded.append(reason)
            print(f"Discarded {reason}", flush=True)
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if (gray.shape[1], gray.shape[0]) != image_size:
            raise RuntimeError(f"{image_path.name}: image size differs from {image_size}.")

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
        detected_count += 1
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        observed = corners.reshape(-1, 2).astype(np.float64)

        # Only the 12 correspondences can influence the estimated pose.
        solved, rvec, tvec = cv2.solvePnP(
            object_points[POSE_INDICES], observed[POSE_INDICES],
            camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not solved or not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
            reason = f"{image_path.name}: pose estimation failed"
            discarded.append(reason)
            print(f"Discarded {reason}", flush=True)
            continue

        projected, _ = cv2.projectPoints(
            object_points[VALIDATION_INDICES], rvec, tvec, camera_matrix, dist_coeffs
        )
        projected = projected.reshape(-1, 2)
        if not np.isfinite(projected).all():
            raise RuntimeError(f"{image_path.name}: non-finite projected coordinates.")
        errors = np.linalg.norm(observed[VALIDATION_INDICES] - projected, axis=1)
        all_errors.extend(errors.tolist())
        image_errors.append((image_path.name, float(errors.mean())))

        # Round only for drawing; calculations and CSV retain subpixel precision.
        for point in observed[POSE_INDICES]:
            pixel = tuple(int(value) for value in np.rint(point))
            cv2.circle(image, pixel, 10, (0, 255, 0), -1, cv2.LINE_AA)
        for point_index, prediction, error in zip(VALIDATION_INDICES, projected, errors):
            observation = observed[point_index]
            observed_pixel = tuple(int(value) for value in np.rint(observation))
            projected_pixel = tuple(int(value) for value in np.rint(prediction))
            cv2.line(image, observed_pixel, projected_pixel, (0, 255, 255), 3, cv2.LINE_AA)
            cv2.circle(image, observed_pixel, 18, (255, 0, 0), 3, cv2.LINE_AA)
            cv2.drawMarker(
                image, projected_pixel, (0, 0, 255), cv2.MARKER_CROSS, 30, 3, cv2.LINE_AA
            )
            cv2.putText(
                image, f"{point_index}: {error:.2f}px",
                (observed_pixel[0] + 24, observed_pixel[1] - 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA
            )
            csv_rows.append([
                image_path.name, int(point_index), *object_points[point_index],
                *observation, *prediction, float(error)
            ])
        output_path = output_dir / image_path.name
        if not cv2.imwrite(str(output_path), image):
            raise RuntimeError(f"Could not save {output_path}.")
        print(f"{image_path.name}: mean validation error = {errors.mean():.10f} pixels", flush=True)

    if not all_errors:
        raise RuntimeError("No images produced a valid pose and validation projections.")
    errors = np.array(all_errors, dtype=np.float64)
    report_lines = [
        "Board projection validation",
        f"Camera ID: {camera_id}",
        f"Image directory: {image_dir}",
        f"Images processed: {len(image_paths)}",
        f"Images with detected board: {detected_count}",
        f"Images used: {len(image_errors)}",
        f"Images discarded: {len(discarded)}",
        f"Image size: {image_size}",
        f"Square size: {SQUARE_SIZE_CM} cm",
        f"Pose points per image: {len(POSE_INDICES)}; indices: {POSE_INDICES.tolist()}",
        f"Validation points per image: {len(VALIDATION_INDICES)}; indices: {VALIDATION_INDICES.tolist()}",
        f"Total validation points: {len(errors)}",
        "Pose method: SOLVEPNP_ITERATIVE, using only the 12 pose points",
        "Coordinates are in the board frame; original distorted images are used",
        "",
        f"Mean projection error: {errors.mean():.10f} pixels",
        f"Projection RMSE: {np.sqrt(np.mean(errors ** 2)):.10f} pixels",
        f"Maximum projection error: {errors.max():.10f} pixels",
        "",
        "Mean validation error per image (pixels):",
    ]
    report_lines.extend(f"{name}: {error:.10f}" for name, error in image_errors)
    report_lines.extend(["", "Discarded images:"] + (discarded or ["None"]))
    report_lines.extend([
        "", "Markers: green filled circles = pose; blue rings = observed validation;",
        "red crosses = projected validation; yellow lines = observation to projection."
    ])
    report = "\n".join(report_lines) + "\n"
    with (output_dir / "points.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "image_name", "point_index", "x_cm", "y_cm", "z_cm",
            "observed_u_px", "observed_v_px", "projected_u_px", "projected_v_px", "error_px"
        ])
        writer.writerows(csv_rows)
    (output_dir / "summary.txt").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"Results saved in {output_dir}")
    return errors


def project_camera(camera_id):
    image_dir = BASE_DIR / "images" / camera_id
    results_dir = BASE_DIR / "results" / camera_id
    calibration_path = results_dir / "calibration.npz"
    if not image_dir.is_dir() or not calibration_path.is_file():
        raise FileNotFoundError(f"Images or saved calibration not found for {camera_id}.")
    with np.load(calibration_path, allow_pickle=False) as parameters:
        camera_matrix = parameters["camera_matrix"]
        dist_coeffs = parameters["dist_coeffs"]
        image_size = tuple(int(value) for value in parameters["image_size"])
    if (
        camera_matrix.shape != (3, 3)
        or not np.isfinite(camera_matrix).all()
        or not np.isfinite(dist_coeffs).all()
        or camera_matrix[0, 0] <= 0
        or camera_matrix[1, 1] <= 0
    ):
        raise ValueError("The saved calibration parameters are invalid.")

    output_dir = results_dir / "projections"
    calibration_errors = project_images(
        camera_id, image_dir, output_dir, camera_matrix, dist_coeffs, image_size
    )
    test_dir = BASE_DIR / "project_test" / camera_id
    if test_dir.is_dir():
        test_errors = project_images(
            camera_id, test_dir, output_dir / "tests", camera_matrix, dist_coeffs, image_size
        )
        comparison_lines = [
            "Projection comparison using the same saved calibration",
            f"Camera ID: {camera_id}",
            f"Calibration images used: {calibration_errors.size // len(VALIDATION_INDICES)}",
            f"Independent images used: {test_errors.size // len(VALIDATION_INDICES)}",
            "Each image uses 12 pose points and 4 withheld validation points.",
            "Independent images are never used to recalibrate the camera.",
            "",
        ]
        metrics = (
            ("Mean error", np.mean),
            ("RMSE", lambda errors: np.sqrt(np.mean(errors ** 2))),
            ("Maximum error", np.max),
        )
        for name, calculate in metrics:
            reference = float(calculate(calibration_errors))
            independent = float(calculate(test_errors))
            difference = independent - reference
            percentage = f"{100 * difference / reference:+.2f}%" if reference > 0 else "undefined"
            comparison_lines.append(
                f"{name}: calibration={reference:.10f} px; independent={independent:.10f} px; "
                f"difference={difference:+.10f} px ({percentage})"
            )
        comparison_lines.extend([
            "", "Metrics are computed separately from individual validation-point errors.",
            "Different board poses and a small test sample limit generalization conclusions."
        ])
        comparison = "\n".join(comparison_lines) + "\n"
        (output_dir / "comparison.txt").write_text(comparison, encoding="utf-8")
        print("\n" + comparison)


def main():
    parser = argparse.ArgumentParser(description="Validate board projections for cameras.")
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
        project_camera(camera_id)


if __name__ == "__main__":
    main()

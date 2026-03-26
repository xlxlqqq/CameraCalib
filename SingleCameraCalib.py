import cv2
import numpy as np
import glob
import json
from extractPoint import detect_and_draw_corners


def calibrate_single_camera(
    image_dir: str = "./left&right",
    pattern_size: tuple = (19, 15),
    square_size_mm: float = 5.0,
    save_path: str = "calib_result.json"
):
    """
    单目相机内参标定（Zhang标定法）。

    Parameters
    ----------
    image_dir     : 标定图像目录
    pattern_size  : 内角点数 (cols, rows)
    square_size_mm: 单格物理尺寸（mm）
    save_path     : 标定结果保存路径（json）
    """
    # ── 收集所有角点 ──────────────────────────
    image_paths = glob.glob(f"{image_dir}/*.bmp")
    if not image_paths:
        print(f"[ERROR] 未找到图像：{image_dir}/*.bmp")
        return

    print(f"[INFO] 共找到 {len(image_paths)} 张图像，开始提取角点...\n")

    all_image_points = []
    all_object_points = []
    image_shape = None

    for path in image_paths:
        img_pts, obj_pts = detect_and_draw_corners(
            image_path=path,
            pattern_size=pattern_size,
            square_size_mm=square_size_mm,
            save_path=path.replace(".bmp", "_corners.png")
        )
        if img_pts is not None:
            all_image_points.append(img_pts)
            all_object_points.append(obj_pts)
            if image_shape is None:
                img = cv2.imread(path)
                image_shape = img.shape[:2]  # (height, width)

    print(f"\n[INFO] 成功提取 {len(all_image_points)}/{len(image_paths)} 张图像的角点")

    if len(all_image_points) < 3:
        print("[ERROR] 有效图像不足3张，无法标定")
        return

    # ── Zhang标定 ────────────────────────────
    h, w = image_shape
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        all_object_points, all_image_points, (w, h), None, None
    )

    # ── 重投影误差评估 ─────────────────────────
    errors = []
    for obj_pts, img_pts, rvec, tvec in zip(all_object_points, all_image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
        err = cv2.norm(img_pts, projected, cv2.NORM_L2) / len(projected)
        errors.append(err)
    mean_error = np.mean(errors)

    # ── 终端输出 ──────────────────────────────
    print("\n" + "=" * 50)
    print("           标 定 结 果")
    print("=" * 50)
    print(f"有效图像数       : {len(all_image_points)}")
    print(f"\n内参矩阵 K:\n{K}")
    print(f"\n畸变系数 [k1, k2, p1, p2, k3]:\n{dist.ravel()}")
    print(f"\n平均重投影误差   : {mean_error:.4f} px")
    print(f"各图重投影误差   : {[round(e, 4) for e in errors]}")
    print("=" * 50)

    # ── 保存 JSON ─────────────────────────────
    result = {
        "valid_images": len(all_image_points),
        "image_size": {"width": w, "height": h},
        "camera_matrix": K.tolist(),
        "dist_coeffs": dist.ravel().tolist(),
        "reprojection_error": {
            "mean": round(float(mean_error), 6),
            "per_image": [round(float(e), 6) for e in errors]
        }
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    print(f"\n[INFO] 标定结果已保存至：{save_path}")

    return result


if __name__ == "__main__":
    calibrate_single_camera(
        image_dir="./left",
        pattern_size=(19, 15),
        square_size_mm=5.0,
        save_path="calib_result.json"
    )
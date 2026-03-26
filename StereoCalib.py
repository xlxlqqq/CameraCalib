import cv2
import numpy as np
import glob
import json
import os
from extractPoint import detect_and_draw_corners


def calibrate_stereo_camera(
    left_dir: str = "./left&right/left",
    right_dir: str = "./left&right/right",
    pattern_size: tuple = (19, 15),
    square_size_mm: float = 5.0,
    save_path: str = "stereo_calib_result.json"
):
    """
    双目相机标定（Zhang标定法）。

    Parameters
    ----------
    left_dir      : 左相机图像目录
    right_dir     : 右相机图像目录
    pattern_size  : 内角点数 (cols, rows)
    square_size_mm: 单格物理尺寸（mm）
    save_path     : 标定结果保存路径（json）
    """

    # ── 获取左右图像列表（按文件名排序，确保一一对应）──
    left_paths  = sorted(glob.glob(f"{left_dir}/*.bmp"))
    right_paths = sorted(glob.glob(f"{right_dir}/*.bmp"))

    if not left_paths or not right_paths:
        print("[ERROR] 未找到图像，请检查目录路径")
        return
    if len(left_paths) != len(right_paths):
        print(f"[ERROR] 左右图像数量不一致：左{len(left_paths)}张，右{len(right_paths)}张")
        return

    print(f"[INFO] 找到 {len(left_paths)} 对图像，开始提取角点...\n")

    # ── 提取左右角点 ──────────────────────────
    all_obj_pts   = []
    all_left_pts  = []
    all_right_pts = []
    image_shape   = None
    valid_count   = 0

    for i, (lp, rp) in enumerate(zip(left_paths, right_paths), 1):
        print(f"[{i:>3}/{len(left_paths)}] 处理左：{os.path.basename(lp)}  右：{os.path.basename(rp)}")

        l_pts, obj_pts = detect_and_draw_corners(
            image_path=lp,
            pattern_size=pattern_size,
            square_size_mm=square_size_mm,
            save_path=lp.replace(".bmp", "_corners.png")
        )
        r_pts, _ = detect_and_draw_corners(
            image_path=rp,
            pattern_size=pattern_size,
            square_size_mm=square_size_mm,
            save_path=rp.replace(".bmp", "_corners.png")
        )

        # 左右都成功才纳入标定
        if l_pts is not None and r_pts is not None:
            all_obj_pts.append(obj_pts)
            all_left_pts.append(l_pts)
            all_right_pts.append(r_pts)
            valid_count += 1
            if image_shape is None:
                img = cv2.imread(lp)
                image_shape = img.shape[:2]  # (height, width)
        else:
            print(f"  [SKIP] 该对图像角点提取失败，跳过")

    print(f"\n[INFO] 成功提取 {valid_count}/{len(left_paths)} 对图像的角点")

    if valid_count < 3:
        print("[ERROR] 有效图像对不足3对，无法标定")
        return

    h, w = image_shape

    # ── 单目预标定（为双目标定提供初始值）────────
    print("\n[INFO] 单目预标定中...")
    _, K1, dist1, _, _ = cv2.calibrateCamera(all_obj_pts, all_left_pts,  (w, h), None, None)
    _, K2, dist2, _, _ = cv2.calibrateCamera(all_obj_pts, all_right_pts, (w, h), None, None)

    # ── 双目标定 ──────────────────────────────
    print("[INFO] 双目标定中...")
    flags = (cv2.CALIB_FIX_INTRINSIC)   # 固定单目内参，只优化外参R/T
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

    ret, K1, dist1, K2, dist2, R, T, E, F = cv2.stereoCalibrate(
        all_obj_pts,
        all_left_pts,
        all_right_pts,
        K1, dist1,
        K2, dist2,
        (w, h),
        flags=flags,
        criteria=criteria
    )

    # ── 立体校正（用于评估极线对齐质量）─────────
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1, dist1, K2, dist2, (w, h), R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
    )

    # ── 重投影误差评估 ─────────────────────────
    # 左右相机各自的重投影误差
    _, _, _, rvecs_l, tvecs_l = cv2.calibrateCamera(
        all_obj_pts, all_left_pts, (w, h), K1, dist1,
        flags=cv2.CALIB_USE_INTRINSIC_GUESS
    )
    _, _, _, rvecs_r, tvecs_r = cv2.calibrateCamera(
        all_obj_pts, all_right_pts, (w, h), K2, dist2,
        flags=cv2.CALIB_USE_INTRINSIC_GUESS
    )

    left_errors, right_errors = [], []
    for obj_pts, l_pts, r_pts, rvl, tvl, rvr, tvr in zip(
        all_obj_pts, all_left_pts, all_right_pts, rvecs_l, tvecs_l, rvecs_r, tvecs_r
    ):
        proj_l, _ = cv2.projectPoints(obj_pts, rvl, tvl, K1, dist1)
        proj_r, _ = cv2.projectPoints(obj_pts, rvr, tvr, K2, dist2)
        left_errors.append(cv2.norm(l_pts, proj_l, cv2.NORM_L2) / len(proj_l))
        right_errors.append(cv2.norm(r_pts, proj_r, cv2.NORM_L2) / len(proj_r))

    mean_left  = float(np.mean(left_errors))
    mean_right = float(np.mean(right_errors))

    # 极线误差（评估双目对齐质量）
    epiline_errors = []
    for l_pts, r_pts in zip(all_left_pts, all_right_pts):
        pts_l = l_pts.reshape(-1, 2)
        pts_r = r_pts.reshape(-1, 2)
        pts_l_h = np.hstack([pts_l, np.ones((len(pts_l), 1))])
        pts_r_h = np.hstack([pts_r, np.ones((len(pts_r), 1))])
        lines_r = (F @ pts_l_h.T).T          # 左点对应的右图极线
        lines_l = (F.T @ pts_r_h.T).T        # 右点对应的左图极线
        # 点到极线距离
        def point_line_dist(pts, lines):
            d = np.abs(np.sum(pts * lines[:, :2], axis=1) + lines[:, 2])
            n = np.sqrt(lines[:, 0]**2 + lines[:, 1]**2)
            return np.mean(d / n)
        err = (point_line_dist(pts_r, lines_r) + point_line_dist(pts_l, lines_l)) / 2
        epiline_errors.append(float(err))
    mean_epiline = float(np.mean(epiline_errors))

    # 基线长度
    baseline_mm = float(np.linalg.norm(T))

    # ── 终端输出 ──────────────────────────────
    print("\n" + "=" * 55)
    print("              双 目 标 定 结 果")
    print("=" * 55)
    print(f"有效图像对数         : {valid_count}")
    print(f"图像尺寸             : {w} x {h}")
    print(f"\n左相机内参 K1:\n{K1}")
    print(f"左相机畸变系数        : {dist1.ravel()}")
    print(f"\n右相机内参 K2:\n{K2}")
    print(f"右相机畸变系数        : {dist2.ravel()}")
    print(f"\n旋转矩阵 R:\n{R}")
    print(f"\n平移向量 T (mm)      : {T.ravel()}")
    print(f"基线距离             : {baseline_mm:.3f} mm")
    print(f"\n【评估结果】")
    print(f"  左相机平均重投影误差  : {mean_left:.4f} px")
    print(f"  右相机平均重投影误差  : {mean_right:.4f} px")
    print(f"  双目平均极线误差      : {mean_epiline:.4f} px  (< 1.0 为良好)")
    print(f"  双目标定总误差 RMS    : {ret:.4f} px")
    print("=" * 55)

    # 质量评价
    if ret < 0.5 and mean_epiline < 0.5:
        grade = "优秀"
    elif ret < 1.0 and mean_epiline < 1.0:
        grade = "良好"
    else:
        grade = "较差，建议重新采集图像"
    print(f"  标定质量评级         : {grade}")
    print("=" * 55)

    # ── 保存 JSON ─────────────────────────────
    result = {
        "valid_pairs": valid_count,
        "image_size": {"width": w, "height": h},
        "left_camera": {
            "camera_matrix": K1.tolist(),
            "dist_coeffs": dist1.ravel().tolist()
        },
        "right_camera": {
            "camera_matrix": K2.tolist(),
            "dist_coeffs": dist2.ravel().tolist()
        },
        "stereo": {
            "R": R.tolist(),
            "T": T.ravel().tolist(),
            "E": E.tolist(),
            "F": F.tolist(),
            "baseline_mm": round(baseline_mm, 4)
        },
        "rectify": {
            "R1": R1.tolist(),
            "R2": R2.tolist(),
            "P1": P1.tolist(),
            "P2": P2.tolist(),
            "Q":  Q.tolist()
        },
        "evaluation": {
            "stereo_rms": round(ret, 6),
            "left_reproj_error_mean":  round(mean_left, 6),
            "right_reproj_error_mean": round(mean_right, 6),
            "epiline_error_mean": round(mean_epiline, 6),
            "grade": grade
        }
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    print(f"\n[INFO] 标定结果已保存至：{save_path}")

    return result


if __name__ == "__main__":
    calibrate_stereo_camera(
        left_dir="./left&right/left",
        right_dir="./left&right/right",
        pattern_size=(19, 15),
        square_size_mm=5.0,
        save_path="stereo_calib_result.json"
    )
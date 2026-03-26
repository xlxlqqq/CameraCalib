import cv2
import numpy as np


def detect_and_draw_corners(
    image_path: str,
    pattern_size: tuple = (19, 15),
    square_size_mm: float = 5.0,
    save_path: str = "result.png",
    scale: float = 0.5
):
    """
    检测棋盘格角点并绘制在原图上保存。

    Parameters
    ----------
    image_path    : 输入图像路径
    pattern_size  : 内角点数 (cols, rows)，20x16方格对应 (19, 15)
    square_size_mm: 单格物理尺寸（mm），用于构建物理坐标
    save_path     : 绘制结果的保存路径
    scale         : 粗检测缩放比例，默认0.5（缩小一半加速检测）

    Returns
    -------
    image_points  : 图像角点坐标，shape=(N,1,2) float32，失败返回 None
    object_points : 物理坐标（Z=0平面），shape=(N,1,3) float32，失败返回 None
    """
    # 读取图像
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 缩小图像用于粗检测
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # 在缩小图上检测角点
    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH
             | cv2.CALIB_CB_NORMALIZE_IMAGE
             | cv2.CALIB_CB_FAST_CHECK)
    found, corners = cv2.findChessboardCorners(small, pattern_size, flags)

    if not found:
        print(f"[WARN] 未检测到角点：{image_path}")
        return None, None

    # 将角点坐标映射回原图尺寸
    corners = corners / scale

    # 在原图上做亚像素精细化（保证精度）
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    # 构建物理坐标（Z=0）
    cols, rows = pattern_size
    obj_pts = np.zeros((cols * rows, 3), dtype=np.float32)
    obj_pts[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size_mm
    obj_pts = obj_pts.reshape(-1, 1, 3)

    # 绘制并保存
    cv2.drawChessboardCorners(image, pattern_size, corners, found)
    cv2.imwrite(save_path, image)
    print(f"[OK] 检测到 {cols * rows} 个角点，结果已保存至：{save_path}")

    return corners, obj_pts


if __name__ == "__main__":
    image_points, object_points = detect_and_draw_corners(
        image_path="Image_20260325200058001.bmp",
        pattern_size=(19, 15),
        square_size_mm=5.0,
        save_path="result.png",
        scale=0.5
    )
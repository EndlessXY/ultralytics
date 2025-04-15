from torchvision.datasets import Country211

from ultralytics import YOLO
from ultralytics import SAM
import cv2
import numpy as np


def load_models(det_model_path="./model/shl_best.pt", sam_model_path="./model/sam2.1_b.pt"):
    """
    加载YOLO检测模型和SAM模型

    参数:
      det_model_path: YOLO检测模型的路径
      sam_model_path: SAM模型的路径

    返回:
      det_model: YOLO检测模型对象
      sam_model: SAM模型对象
    """
    det_model = YOLO(det_model_path)
    sam_model = SAM(sam_model_path)
    return det_model, sam_model

def detect_bboxes(det_model, img):
    """
    使用YOLO模型对图像进行检测，返回检测到的边界框列表

    参数:
      det_model: YOLO检测模型对象
      img: 待检测的图像（numpy.ndarray）

    返回:
      bboxes_list: 边界框列表，每个边界框为 [x1, y1, x2, y2] 格式；若无检测则返回空列表
    """
    det_results = det_model(img, verbose=False)
    det_result = det_results[0]  # 获取第一个检测结果
    det_boxes = det_result.boxes.xyxy  # 边界框对象（Tensor）
    if det_boxes.numel() == 0:
        print("No boxes detected.")
        return []
    else:
        # 转换为列表格式
        return det_boxes.tolist()

def detect_mask(sam_model, img, bboxes_list):
    """
    使用SAM模型对给定图像和边界框进行mask检测

    参数:
      sam_model: SAM模型对象
      img: 待检测的图像（numpy.ndarray）
      bboxes_list: 边界框列表，格式为 [[x1, y1, x2, y2], ...]

    返回:
      mask: masks对象, ultralytics.engine.results.Masks
    """
    sam_results = sam_model(img, bboxes=bboxes_list, verbose=False)
    masks = sam_results[0].masks
    # 检查是否有mask产生
    if masks is None or masks.data.shape[0] == 0:
        print("No mask found.")
        return None
    return masks

def find_all_contours(mask):
    """
    从二值 mask 中查找所有轮廓，不做任何过滤操作

    参数:
      mask: 二值图像 (numpy.ndarray)，非零像素视为前景

    返回:
      contours: 轮廓列表
      hierarchy: 每个轮廓的层级信息
    """
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def get_largest_contour(mask):
    """
    从二值mask中找到最大的轮廓，仅返回面积最大的轮廓。

    参数:
      mask: 二值图像 (numpy.ndarray)，非零像素视为前景

    返回:
      largest_contour: 面积最大的轮廓；如果未找到轮廓，则返回 None
    """
    # 查找所有外部轮廓
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # 选择面积最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        return largest_contour
    return None

def filter_contours_by_area(mask, min_area=100):
    """
    从二值mask中查找所有轮廓，并根据设定面积阈值过滤掉较小的轮廓。

    参数:
      mask: 二值图像 (numpy.ndarray)，非零像素视为前景
      min_area: 最小轮廓面积阈值，只有面积大于该值的轮廓才会被返回

    返回:
      filtered_contours: 过滤后的轮廓列表
    """
    # 查找所有外部轮廓
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 过滤面积较小的轮廓
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]
    return filtered_contours

def get_contours_after_denoising(mask, kernel_size=(5, 5)):
    """
    对输入的二值mask进行形态学开运算去除噪声，再查找轮廓。

    参数:
      mask: 二值图像 (numpy.ndarray)，非零像素视为前景
      kernel_size: 形态学处理时使用的卷积核大小，默认为 (5, 5)

    返回:
      contours: 去噪后找到的轮廓列表
      denoised_mask: 经过形态学开运算处理后的二值mask
    """
    # 构造矩形卷积核
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    # 使用开运算（先腐蚀后膨胀）去除小区域噪声
    denoised_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # 查找轮廓
    contours, hierarchy = cv2.findContours(denoised_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, denoised_mask

def draw_contours_on_image(img, contours, output_path="./result/contour_result.jpg"):
    """
    在原图上绘制轮廓并保存

    参数:
      img: 图像Matrix（numpy.ndarray）
      contours: 轮廓列表，由 cv2.findContours 得到
      output_path: 绘制结果图像保存路径
    """
    # 在原图上绘制所有轮廓
    cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
    cv2.imwrite(output_path, img)
    print(f"轮廓图像已保存至 {output_path}")

def draw_mask(mask, output_path="./result/mask_result.jpg"):
    """
    保存 mask 图像

    参数:
      mask: 二值 mask 图像（uint8类型）
      output_path: mask 图像的保存路径
    """
    cv2.imwrite(output_path, mask)
    print(f"mask 图像已保存至 {output_path}")

def overlay_mask_on_image(image, mask, output_path="./result/overlay_result.jpg",
                          mask_color=(0, 255, 0), alpha=0.5):
    """
    将二值 mask 以指定颜色和透明度叠加在原图上，并保存结果。

    参数:
      image: 图像Matrix（numpy.ndarray）
      mask: 二值 mask 图像（uint8类型，非零像素代表 mask 区域）
      output_path: 结果图像保存路径
      mask_color: 覆盖 mask 的颜色，默认为绿色 (B, G, R)
      alpha: 叠加透明度（0.0 完全透明，1.0 完全不透明）

    返回:
      result: 叠加后的图像
    """
    # 确保 mask 与原图尺寸一致
    if (mask.shape[0] != image.shape[0]) or (mask.shape[1] != image.shape[1]):
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    # 创建一个与原图一样的彩色 mask 覆盖图，用指定颜色填充
    colored_mask = np.zeros_like(image, dtype=np.uint8)
    colored_mask[mask > 0] = mask_color

    # 叠加 mask 到原图上：alpha 表示 colored_mask 的权重
    result = cv2.addWeighted(colored_mask, alpha, image, 1 - alpha, 0)

    # 保存叠加结果
    cv2.imwrite(output_path, result)
    print(f"Overlay 图像已保存至 {output_path}")

    return result

def process_merged_masks(masks, open_kernel_size=5, close_kernel_size=5, min_contour_points=3, min_area=None):
    """
    对多个mask进行合并处理、形态学操作和轮廓提取过滤。

    参数:
        masks: mask数组，形状为 [n, h, w]
        open_kernel_size: 开运算操作的卷积核大小
        close_kernel_size: 闭运算操作的卷积核大小
        min_contour_points: 最小轮廓点数，小于此值的轮廓会被过滤
        min_area: 最小轮廓面积阈值，若为None则不进行面积过滤

    返回:
        final_mask: 经过处理后的最终mask
        filtered_contours: 过滤后的轮廓列表
    """
    # 1. 合并所有mask
    if masks.ndim == 3:
        merged_mask = np.zeros_like(masks[0], dtype=np.uint8)
        for mask in masks:
            merged_mask = np.logical_or(merged_mask, mask > 0).astype(np.uint8) * 255
    else:
        merged_mask = masks.copy()  # 如果只有一个mask，直接使用

    # 2. 开运算和闭运算操作
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel_size, open_kernel_size))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size))

    # 先开运算（去除小噪点）
    opened_mask = cv2.morphologyEx(merged_mask, cv2.MORPH_OPEN, open_kernel)
    # 再闭运算（填充小孔洞）
    processed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, close_kernel)

    # 3. 查找轮廓
    contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 4. 轮廓过滤
    filtered_contours = []
    for contour in contours:
        # 过滤掉点数少于min_contour_points的轮廓
        if len(contour) < min_contour_points:
            continue

        # 可选的面积过滤
        if min_area is not None and cv2.contourArea(contour) < min_area:
            continue

        filtered_contours.append(contour)

    # 5. 创建最终的mask
    final_mask = np.zeros_like(merged_mask, dtype=np.uint8)
    cv2.drawContours(final_mask, filtered_contours, -1, 255, -1)  # 填充轮廓内部

    return final_mask, filtered_contours

def draw_mask_and_contours(image, mask, contours, output_path="./result/mask_contours_result.jpg",
                           mask_color=(0, 255, 0), contour_color=(0, 0, 255),
                           mask_alpha=0.3, contour_thickness=2):
    """
    在图像上同时绘制半透明mask和轮廓线。

    参数:
        image: 原始图像 (numpy.ndarray)
        mask: 二值mask图像 (numpy.ndarray)
        contours: 轮廓列表
        output_path: 输出图像保存路径
        mask_color: mask覆盖颜色，BGR格式，默认为绿色(0, 255, 0)
        contour_color: 轮廓线颜色，BGR格式，默认为红色(0, 0, 255)
        mask_alpha: mask透明度，值范围[0, 1]，0为完全透明，1为完全不透明
        contour_thickness: 轮廓线宽度，默认为2

    返回:
        result: 绘制了mask和轮廓的结果图像
    """
    # 确保mask与原图尺寸一致
    if (mask.shape[0] != image.shape[0]) or (mask.shape[1] != image.shape[1]):
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    # 创建图像副本避免修改原图
    result = image.copy()

    # 只在掩码区域应用颜色混合
    mask_region = mask > 0
    if np.any(mask_region):
        # 在掩码区域进行混合
        result[mask_region] = cv2.addWeighted(
            np.full_like(result[mask_region], mask_color),
            mask_alpha,
            result[mask_region],
            1 - mask_alpha,
            0
        )

    # 在结果图像上绘制轮廓
    cv2.drawContours(result, contours, -1, contour_color, contour_thickness)

    # 保存结果图像
    cv2.imwrite(output_path, result)
    print(f"Mask和轮廓叠加图像已保存至 {output_path}")

    return result


if __name__ == "__main__":
    # 图像和模型路径配置
    img_path = "./img/shl6.jpg"
    det_model_path = "./model/shl_best.pt"
    sam_model_path = "./model/sam2.1_b.pt"
    img = cv2.imread(img_path)

    # 1. 加载模型
    det_model, sam_model = load_models(det_model_path, sam_model_path)

    # 2. bbox检测
    bboxes_list = detect_bboxes(det_model, img)
    if not bboxes_list:
        print("No bboxes detected.")
        exit(0)

    # 3. masks检测
    masks = detect_mask(sam_model, img, bboxes_list)
    if masks is None:
        exit(0)
        print("No mask detected.")

    # 4. 后处理：轮廓提取及可视化
    masks = masks.data.cpu().numpy()
    masks = (masks * 255).astype(np.uint8)

    final_mask, filtered_contours = process_merged_masks(masks)
    print(f"找到 {len(filtered_contours)} 个轮廓")
    overlay_mask_on_image(img, final_mask)
    draw_contours_on_image(img, filtered_contours)

    draw_mask_and_contours(img, final_mask, filtered_contours, output_path="./result/mask_contours_overlay.jpg")




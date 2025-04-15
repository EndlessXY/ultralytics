import os
import argparse
import cv2
from pathlib import Path

from ultralytics import YOLO
from ultralytics import FastSAM, SAM


def get_image_files(folder_path):
    """
    获取文件夹中所有图片文件

    参数:
        folder_path: 文件夹路径

    返回:
        图片文件路径列表
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    image_files = []

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path) and Path(file_path).suffix.lower() in image_extensions:
            image_files.append(file_path)

    return image_files

def load_models(model_path):
    """
    通过文件名自动选择加载YOLO、FastSAM、SAM模型

    参数:
      model_path: 模型文件路径

    返回:
        model: 加载的模型对象
    """
    if "yolo" in model_path:
        model = YOLO(model_path)
    elif "FastSAM" in model_path:
        model = FastSAM(model_path)
    elif "sam" in model_path:
        model = SAM(model_path)
    else:
        raise ValueError("Unsupported model type.")
    return model

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
        return None
    else:
        # 转换为列表格式
        return det_boxes.tolist(), det_result

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
        print("No masks detected.")
        return None
    else:
        return masks, sam_results[0]

def save_masks(sam_result, save_path):
    """
    保存mask到指定路径

    参数:
      masks: masks对象, ultralytics.engine.results.Masks
      img: 原始图像（numpy.ndarray）
      save_path: 保存路径
    """
    sam_result.save(save_path)
    print(f"Saved masks to {save_path}")

def save_bboxes(det_result, save_path):
    """
    保存检测到的边界框到指定路径

    参数:
      det_result: 检测结果对象
      save_path: 保存路径
    """
    det_result.save(save_path)
    print(f"Saved bounding boxes to {save_path}")

def save_img(img, save_path):
    """
    保存图像到指定路径

    参数:
      img: 原始图像（numpy.ndarray）
      save_path: 保存路径
    """
    cv2.imwrite(save_path, img)
    print(f"Saved image to {save_path}")

def get_filename_without_extension(file_path):
    # 创建Path对象并直接获取stem属性（文件名无后缀）
    return Path(file_path).stem

def infer(models, img_path, yolo_model):
    """
    对图像进行推断，检测边界框并进行mask分割

    参数:
      models: 模型字典
      img_path: 图像路径
    """
    img = cv2.imread(img_path)
    img_filename = Path(img_path).name

    for model_name, model_info in models.items():
        model = model_info["model"]
        output_dir = model_info["output"]

        # 1. 检测边界框
        yolo_results = detect_bboxes(yolo_model, img)
        if yolo_results is None:
            print(f"No bounding boxes detected for {img_filename} using {model_name}.")
            save_img(img, str(Path(output_dir) / Path(img_path).name))
            continue
        bboxes_list, det_result = yolo_results
        save_bboxes(det_result, str(Path(output_dir) / f"{Path(img_path).stem}_det{Path(img_path).suffix}"))

        # 2. 检测mask
        sam_results = detect_mask(model, img, bboxes_list)
        if sam_results is None:
            print(f"No masks detected for {img_filename} using {model_name}.")
            continue
        masks, sam_result = sam_results
        save_masks(sam_result, str(Path(output_dir) / f"{Path(img_path).stem}_mask{Path(img_path).suffix}"))


if __name__ == '__main__':
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='对图像进行检测和分割')
    parser.add_argument('--input_path', type=str, help='图像文件或包含图像的文件夹路径')
    args = parser.parse_args()

    img_path = "./img/shl6.jpg"
    det_model_path = "./model/shl_best.pt"
    fastsam_model_path = "./model/FastSAM-s.pt"
    sam_model_path = "./model/sam2_t.pt"
    mobile_sam_model_path = "./model/mobile_sam.pt"
    model_paths = [fastsam_model_path, sam_model_path, mobile_sam_model_path]
    models = {}
    result_folder = "./result"
    # 创建结果文件夹
    Path(result_folder).mkdir(parents=True, exist_ok=True)

    # 1. 加载模型
    for model_path in model_paths:
        model_name = get_filename_without_extension(model_path)
        model_info = {}
        model_predict_output = str(Path(result_folder) / model_name)
        Path(model_predict_output).mkdir(parents=True, exist_ok=True)
        model_info["output"] = model_predict_output
        model_info["model"] = load_models(model_path)
        models[model_name] = model_info

    # 2. 处理输入（文件或文件夹）
    input_path = args.input_path

    yolo_model = YOLO(det_model_path)
    if os.path.isfile(input_path):
        # 处理单个图像
        print(f"正在处理单个图像: {input_path}")
        infer(models, input_path, yolo_model)
    elif os.path.isdir(input_path):
        # 处理文件夹中的所有图像
        image_files = get_image_files(input_path)
        print(f"在文件夹 {input_path} 中找到 {len(image_files)} 个图像")

        for img_file in image_files:
            print(f"正在处理图像: {img_file}")
            infer(models, img_file, yolo_model)
    else:
        print(f"错误: 输入路径 {input_path} 不存在或无法访问")


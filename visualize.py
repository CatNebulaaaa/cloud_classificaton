import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import os
import cv2  
import numpy as np

def visualize_predictions(model, device, class_mapping, input_folder, output_folder):
    """
    对一个文件夹里的所有图片进行预测，并将结果可视化后保存。
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"创建输出文件夹: {output_folder}")

    # 定义图像预处理
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 设置模型为评估模式
    model.eval()

    # 遍历输入文件夹中的所有图片
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(supported_formats)]
    
    if not image_files:
        print(f"错误：在文件夹 '{input_folder}' 中没有找到支持的图片文件。")
        return

    print(f"找到 {len(image_files)} 张图片，开始进行预测和可视化...")

    for image_name in image_files:
        image_path = os.path.join(input_folder, image_name)
        
        # 使用PIL加载图片
        image = Image.open(image_path).convert("RGB")
        
        # 预处理图片以输入模型
        image_tensor = preprocess(image).unsqueeze(0).to(device)

        # 执行推理
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            top_prob, top_idx = torch.topk(probabilities, 1)
            
            pred_class_idx = top_idx.item()
            pred_class_name = class_mapping[str(pred_class_idx)]
            confidence = top_prob.item()

        # 使用OpenCV进行可视化
        # 将PIL Image转换为OpenCV格式 (RGB -> BGR)
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # 调整尺寸以匹配我们的标准显示（可选，但可以使标签大小更一致）
        h, w, _ = cv_image.shape
        scale_factor = 400 / h # 将高度缩放到400像素
        new_w, new_h = int(w * scale_factor), int(h * scale_factor)
        cv_image = cv2.resize(cv_image, (new_w, new_h))

        # 准备要绘制的文本
        label_text = f"{pred_class_name} ({confidence:.2%})"
        
        # 在图片顶部绘制一个黑色背景条
        cv2.rectangle(cv_image, (0, 0), (new_w, 30), (0, 0, 0), -1)
        # 将文本写在背景条上
        cv2.putText(
            cv_image, 
            label_text, 
            (10, 22), # 文本位置
            cv2.FONT_HERSHEY_SIMPLEX, # 字体
            0.7, # 字体大小
            (255, 255, 255), # 字体颜色 (白色)
            2 # 字体粗细
        )
        
        # 保存结果图片
        output_path = os.path.join(output_folder, image_name)
        cv2.imwrite(output_path, cv_image)

    print(f"\n处理完成！所有结果已保存到 '{output_folder}' 文件夹。")

if __name__ == '__main__':
    WEIGHTS_PATH = 'path/to/best_model.pth' # 你的权重文件路径
    CLASS_MAP_PATH = 'path/to/class_mapping.json' # 你的类别映射文件路径
    INPUT_FOLDER = 'input_images' # 存放待预测图片的文件夹
    OUTPUT_FOLDER = 'output_images' # 存放结果的文件夹
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载类别映射
    try:
        with open(CLASS_MAP_PATH, 'r') as f:
            class_mapping = json.load(f)
        num_classes = len(class_mapping)
    except FileNotFoundError:
        print(f"错误：类别映射文件 '{CLASS_MAP_PATH}' 未找到")
        exit()

    # 重新构建模型的分类头
    print("正在加载模型结构...")
    model = models.mobilenet_v2()
    num_features = model.classifier[-1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5), 
        nn.Linear(num_features, num_classes)
    )
    
    # 加载训练好的权重
    print(f"正在从 '{WEIGHTS_PATH}' 加载权重...")
    try:
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    except FileNotFoundError:
        print(f"错误：权重文件 '{WEIGHTS_PATH}' 未找到")
        exit()
    model.to(device)

    # 执行可视化预测
    visualize_predictions(model, device, class_mapping, INPUT_FOLDER, OUTPUT_FOLDER)
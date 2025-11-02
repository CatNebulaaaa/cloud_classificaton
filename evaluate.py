import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import json
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, test_loader, device, class_names):
    model.eval()
    total_loss = 0.0
    total_corrects = 0
    all_preds = []
    all_labels = []
    criterion = nn.CrossEntropyLoss()

    print("开始在测试集上进行评估...")
    start_time = time.time()
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            total_corrects += torch.sum(preds == labels.data)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    end_time = time.time()
    eval_time = end_time - start_time
    print(f"评估完成，耗时: {eval_time // 60:.0f}m {eval_time % 60:.0f}s")
    
    avg_loss = total_loss / len(test_loader.dataset)
    avg_acc = total_corrects.double() / len(test_loader.dataset)
    
    print(f"\n--- 总体性能 ---")
    print(f"测试集平均损失 (Avg Loss): {avg_loss:.4f}")
    print(f"测试集准确率 (Accuracy): {avg_acc:.2%}")
    print("--------------------")

    print("\n--- 分类报告 ---")
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print(report)
    print("--------------------")

    print("\n正在生成混淆矩阵...")
    cm = confusion_matrix(all_labels, all_preds)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Labels')
    plt.xlabel('Predicted Labels')
    plt.savefig("confusion_matrix.png")
    print("混淆矩阵已保存为 'confusion_matrix.png'")
    
    return avg_loss, avg_acc


if __name__ == '__main__':
    TEST_DIR = 'path/to/test'  # 测试集文件夹路径
    WEIGHTS_PATH = 'path/to/best_model.pth' # 权重文件路径
    CLASS_MAP_PATH = 'path/to/class_mapping.json' # 类别映射文件路径

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载类别映射
    try:
        with open(CLASS_MAP_PATH, 'r') as f:
            idx_to_class = json.load(f)
        num_classes = len(idx_to_class)
        class_names = [idx_to_class[str(i)] for i in range(num_classes)]
    except FileNotFoundError:
        print(f"错误：类别映射文件 '{CLASS_MAP_PATH}' 未找到")
        exit()

    # 重新构建模型结构
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

    # 定义测试集的数据变换和加载器
    test_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        test_dataset = datasets.ImageFolder(TEST_DIR, test_transforms)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
        print(f"从 '{TEST_DIR}' 加载了 {len(test_dataset)} 张测试图片。")
    except FileNotFoundError:
        print(f"错误：测试集文件夹 '{TEST_DIR}' 未找到")
        exit()
        
    # 执行评估
    evaluate_model(model, test_loader, device, class_names)



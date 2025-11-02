# 基于 MobileNetV2 的云朵分类项目

本项目提供了一个使用 PyTorch 和 MobileNetV2 架构进行云朵分类模型训练、评估和可视化的完整流程。

## 项目简介

项目包含三个核心脚本：

*   `train.py`: 在GCD数据集上训练 MobileNetV2 模型。脚本会自动划分训练集和验证集，应用数据增强，并根据验证集准确率保存最优模型。
*   `evaluate.py`: 在测试集上评估已训练好的模型。脚本会计算准确率、生成详细的分类报告和混淆矩阵图。
*   `visualize.py`: 对指定文件夹中的图片进行推理预测，并将带有预测类别和置信度的结果图保存下来。

## 环境与准备

### 先决条件

*   Python 3

### 环境安装

1.  建议创建一个虚拟环境。
2.  安装所有必需的库：
    ```bash
    pip install torch torchvision pandas seaborn matplotlib scikit-learn opencv-python numpy
    ```

### 目录结构

本项目使用[TJNU Ground‑based Cloud Dataset (GCD)](https://github.com/shuangliutjnu/TJNU-Ground-based-Cloud-Dataset)	数据集，包含中国多个省份地面拍摄的云朵图像。按 WMO 云属标准分 7 类，约有19,000张图像。
在项目根目录下，请解压GCD.zip获得train和test文件夹并手动创建input_images和output_images文件夹：

*   `train/`: 存放训练图片，每个子文件夹代表一个类别，文件夹名为类别名。
*   `test/`: 存放测试图片，目录结构与 `train` 文件夹相同。
*   `input_images/`: 存放需要进行可视化预测的单张图片。
*   `output_images/`: 可视化后的结果图片将保存在这里。
*   `train.py`：模型训练脚本
*   `visualize.py`：图片预测结果可视化
*   `evaluate.py`：模型评估脚本


## 如何运行

#### 1. 训练模型

*   配好上述环境，设置好路径
*   运行训练脚本：
    ```bash
    python train.py
    ```
*   训练完成后，会在根目录生成一个 `best_model.pth` 文件，其中包含了效果最好的模型权重。

#### 2. 评估模型

*   确保根目录下已有 `best_model.pth` 和 `class_mapping.json` 文件。也可以直接用我训练好的权重文件，不过他的正确率只有91.8%
*   运行评估脚本：
    ```bash
    python evaluate.py
    ```
*   评估结果（准确率、分类报告）将直接打印在控制台，同时会生成一张 `confusion_matrix.png` 混淆矩阵图。

#### 3. 可视化预测

*   将您想预测的图片放入 `input_images` 文件夹。
*   确保根目录下已有 `best_model.pth` 和 `class_mapping.json` 文件。
*   运行可视化脚本：
    ```bash
    python visualize.py
    ```
*   带有预测结果的图片将会被保存在 `output_images` 文件夹中。

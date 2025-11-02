import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import datasets, models, transforms 
import time
import copy
import os
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(224), # 直接Resize到224，因为MobileNetV2也用224
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

path_train = 'path/to/train'

# 因为只有训练集，所以要从训练集分割出验证集
train_dataset = datasets.ImageFolder(root=path_train, transform=data_transforms['train'])
val_dataset = datasets.ImageFolder(root=path_train, transform=data_transforms['val'])
val_split = 0.2
dataset_size = len(train_dataset)
indices = list(range(dataset_size))
split_point = int(np.floor(val_split * dataset_size))
np.random.seed(42)
np.random.shuffle(indices)
train_indices, val_indices = indices[split_point:], indices[:split_point]
train_subset = Subset(train_dataset, train_indices)
val_subset = Subset(val_dataset, val_indices)
image_datasets = {'train': train_subset, 'val': val_subset}
dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=32, shuffle=True, num_workers=4),
    'val': DataLoader(image_datasets['val'], batch_size=32, shuffle=False, num_workers=4),
}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = train_dataset.classes
num_classes = len(class_names)


print("Using MobileNetV2 as the base model.")
# 加载预训练的 MobileNetV2
weights = models.MobileNet_V2_Weights.IMAGENET1K_V2
model = models.mobilenet_v2(weights=weights)

# 解冻所有层进行全模型微调 (或者只解冻最后几层，可以先尝试全解冻)
for param in model.parameters():
    param.requires_grad = True

# 替换 MobileNetV2 的分类头
# MobileNetV2的分类器是一个单独的'classifier'模块
num_features = model.classifier[-1].in_features # 获取最后一个线性层的输入特征数
model.classifier = nn.Sequential(
    nn.Dropout(p=0.5), # 降低Dropout，因为模型本身没那么复杂了
    nn.Linear(num_features, num_classes)
)
model = model.to(device)


# 优化器和调度器
criterion = nn.CrossEntropyLoss()

# 初始学习率可以稍微调高一点，因为是新模型
optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)

# 智能调度器保持不变
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)

def train_model(model, criterion, optimizer, scheduler, num_epochs=100):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    patience = 15
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            current_lr = optimizer.param_groups[0]['lr']
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} LR: {current_lr:.1e}')

            if phase == 'val':
                scheduler.step(epoch_acc)
                if epoch_acc > best_acc:
                    print(f"Validation accuracy improved ({best_acc:.4f} --> {epoch_acc:.4f}). Saving model...")
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {patience} epochs with no improvement.")
            break
        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    save_path = 'best_model.pth'
    print(f'Saving best model weights to {save_path}')
    torch.save(best_model_wts, save_path)
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    model_trained = train_model(model, criterion, optimizer, scheduler, num_epochs=100)
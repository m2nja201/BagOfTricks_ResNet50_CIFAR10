# Additional Imports
import torch.nn.functional as F
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import argparse
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from tqdm.auto import tqdm
from torchvision import datasets, models
import copy

from model.resnet import ResNet50
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import _LRScheduler
import math
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class LinearWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, total_epochs, last_epoch=-1):
        self.total_epochs = total_epochs
        super(LinearWarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.last_epoch / self.total_epochs for base_lr in self.base_lrs]

# Custom CosineAnnealingWarmRestarts
class CustomCosineAnnealingWarmRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, gamma=1.0, last_epoch=-1):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.gamma = gamma
        self.T_i = T_0
        self.cycle = 0
        self.last_restart = 0
        super(CustomCosineAnnealingWarmRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == self.last_restart:
            self.T_i = self.T_i * self.T_mult
            self.last_restart = self.last_epoch

        elif self.last_epoch == self.last_restart + self.T_i:
            self.cycle += 1
            self.last_restart = self.last_epoch
            self.base_lrs = [base_lr * self.gamma for base_lr in self.base_lrs]

        return [base_lr * (1 + math.cos(math.pi * (self.last_epoch - self.last_restart) / self.T_i)) / 2
                for base_lr in self.base_lrs]
    

# Label Smoothing
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.05, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
    
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
    

# MixUp
def mixup_data(x, y, alpha=0.1, use_cuda=True):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# Validation Function
def validate(model, val_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    val_running_loss = 0.0
    val_correct_predictions = 0
    val_total_predictions = 0

    with torch.no_grad():  # No need to track gradients
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            val_total_predictions += labels.size(0)
            val_correct_predictions += (predicted == labels).sum().item()

    val_loss = val_running_loss / len(val_loader)
    val_accuracy = val_correct_predictions / val_total_predictions * 100
    return val_loss, val_accuracy


import torch.nn.functional as F

def distillation_loss(student_outputs, teacher_outputs, temperature):
    """Calculate the distillation loss."""
    return F.kl_div(F.log_softmax(student_outputs / temperature, dim=1),
                    F.softmax(teacher_outputs / temperature, dim=1), 
                    reduction='batchmean')


# Train Function
def train(model, teacher_model, train_loader, test_loader, criterion, optimizer, total_epochs, device, temperature=3.0, alpha=0.5):
    #warmup_scheduler = warmup_scheduler = LinearWarmupScheduler(optimizer, 5)
    #cosine_scheduler = CustomCosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2, gamma=0.5)
    scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=100, cycle_mult=1.0, max_lr=0.05, min_lr=0, warmup_steps=5, gamma=0.5)

    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    model.to(device)
    for epoch in range(total_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()

            # Forward pass through student
            student_outputs = model(inputs)

            # Forward pass through teacher
            with torch.no_grad():
                teacher_outputs = teacher_model(inputs)

            if np.random.rand() < 0.1:  # Apply mixup
                inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, alpha=0)
                student_outputs = model(inputs)
                loss = mixup_criterion(criterion, student_outputs, targets_a, targets_b, lam)
            else:
                # Standard loss
                loss = criterion(student_outputs, labels)

            # Distillation loss
            dist_loss = distillation_loss(student_outputs, teacher_outputs, temperature)
            loss += alpha * dist_loss

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accuracy calculation
            _, predicted = torch.max(student_outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct_predictions / total_predictions * 100

        val_loss, val_acc = validate(model, test_loader, criterion, device)
        last_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        # tensorBoard
        writer.add_scalar('lr', last_lr, epoch)
        writer.add_scalar('loss/train', epoch_loss, epoch)
        writer.add_scalar('accuracy/train', epoch_acc, epoch)
        writer.add_scalar('loss/val', val_loss, epoch)
        writer.add_scalar('accuracy/val', val_acc, epoch)

        # update
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), f"/root/workspace/minjae/BagOfTricks_restart/best/{args.name}.pt")
            print(f"{epoch} is the Best accuracy!")

        print(f'Epoch {epoch+1}/{total_epochs}, LR: {last_lr}\nLoss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
        print(f'Val_Loss: {val_loss:.2f}, Val_Acc: {val_acc:.2f}%\n')

    print(f'Best Validation Accuracy: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), f"/root/workspace/minjae/BagOfTricks_restart/best/{args.name}.pt")
    print("Traning Ends...")
    writer.close()
    return model


# Final performance
def final_acc(model):
    all_labels = []
    all_preds = []

    model.eval()
    for images, labels in tqdm(test_loader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        pred = torch.argmax(outputs, dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(pred.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    return all_labels, all_preds


if __name__ == '__main__':
    # 실행 title 설정
    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('--name', type=str, help='An input name')

    args = parser.parse_args()
    if args.name:
        print(f"I will try <<{args.name}>>! Training Starts..")
    else:
        args.name = 'none_name'


    # TensorBoard 정의
    layout = {
        "Baseline" : {
            "loss" : ["Multiline", ["loss/train", "loss/val"]],
            "accuracy" : ["Multiline", ["accuracy/train", "accuracy/val"]],
            "learning rate" : ["Multiline", ["lr"]]
        },
    }
    writer = SummaryWriter(f"./logs/{args.name}") 
    writer.add_custom_scalars(layout=layout)


    # CIFAR-10 mean, std 정의
    train_mean = [0.4913997645378113, 0.48215836706161499, 0.44653093814849854]
    train_std = [0.24703224098682404, 0.24348513793945312, 0.2615878584384918]

    test_mean = [0.4942141773700714, 0.4851310842037201, 0.45040971088409424]
    test_std = [0.2466523642539978, 0.24289205610752106, 0.2615927150249481]


    # Transforms 정의
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=(0.6, 1.4)),
        transforms.ToTensor(),
        transforms.Normalize(train_mean, train_std)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(test_mean, test_std)
    ])


    # Dataset 정의 (CIFAR-10)
    trainset = datasets.CIFAR10(root='../data', train=True, transform=train_transform, download=True)
    testset = datasets.CIFAR10(root='../data', train=False, transform=test_transform, download=True)


    # Device 연결
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)


    # Assuming you have a pre-trained teacher model
    teacher_model = models.resnet152(pretrained=True) # Load your pre-trained teacher model here
    teacher_model.fc = nn.Linear(teacher_model.fc.in_features, 10)
    teacher_model.to(device=device)
    teacher_model.eval()  # Teacher model should be in eval mode



    # 초기 모델
    model = ResNet50()
    model = model.to(device=device)


    # Parameter 설정
    batch_size = 128
    learning_rate = 0.05
    #criterion = nn.CrossEntropyLoss()   # Label Smoothing을 사용하지 않을 경우
    criterion = LabelSmoothingLoss(classes=10, smoothing=0.05)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)
    

    # Dataset Loader
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=16)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=16)


    ## Train
    model = train(model=model, teacher_model=teacher_model, train_loader=train_loader, test_loader=test_loader, criterion=criterion, optimizer=optimizer, total_epochs=100, device=device, temperature=2)
    torch.save(model.state_dict(), f"/root/workspace/minjae/BagOfTricks_restart/best/{args.name}.pt")


    ## Test
    test_labels, test_preds = final_acc(model)
    print(f"Accuracy : {(accuracy_score(test_labels, test_preds) * 100):.2f}%")
    print(f"Precision : {(precision_score(test_labels, test_preds, average='macro')):.4f}")
    print(f"Recall : {(recall_score(test_labels, test_preds, average='macro')):.4f}")
    print(f"F1 Score : {(f1_score(test_labels, test_preds, average='macro')):.4f}")




# Modify the Train Function
def train(model, teacher_model, train_loader, test_loader, criterion, optimizer, total_epochs, device):
    ...
    for epoch in range(total_epochs):
        model.train()
        ...
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass through the teacher model with no gradients
            with torch.no_grad():
                teacher_outputs = teacher_model(inputs)

            # Forward pass through the student model
            student_outputs = model(inputs)

            # Compute the distillation loss
            loss = distillation_loss(student_outputs, labels, teacher_outputs)

            # Rest of your training loop remains the same
            ...

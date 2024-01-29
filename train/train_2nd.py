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

from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingWarmRestarts
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Warm Up + CosineAnnealingWarmRestarts
class LinearWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, total_epochs, last_epoch=-1):
        self.total_epochs = total_epochs
        super(LinearWarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.last_epoch / self.total_epochs for base_lr in self.base_lrs]


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


# Train Function
def train(model, train_loader, test_loader, criterion, optimizer, total_epochs, device):
    #warmup_epochs = 5
    #warmup_scheduler = LinearWarmupScheduler(optimizer, warmup_epochs)
    #cosine_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

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
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Add accuracy calculation
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct_predictions / total_predictions * 100

        val_loss, val_acc = validate(model, test_loader, criterion, device)

        last_lr = optimizer.param_groups[0]['lr']
        #if epoch < warmup_epochs:
        #    warmup_scheduler.step()
        #else:
        #    cosine_scheduler.step()

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
    writer = SummaryWriter("./logs/{args.name}") 
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


    # 초기 모델
    model = ResNet50()
    model = model.to(device=device)


    # Parameter 설정
    batch_size = 128
    learning_rate = 0.05
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-5)
    

    # Dataset Loader
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=16)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=16)


    ## Train
    model = train(model=model, train_loader=train_loader, test_loader=test_loader, criterion=criterion, optimizer=optimizer, total_epochs=100, device=device)
    torch.save(model.state_dict(), "/root/workspace/minjae/BagOfTricks_restart/best/{args.name}")


    ## Test
    test_labels, test_preds = final_acc(model)
    print(f"Accuracy : {(accuracy_score(test_labels, test_preds) * 100):.2f}%")
    print(f"Precision : {(precision_score(test_labels, test_preds, average='macro')):.4f}")
    print(f"Recall : {(recall_score(test_labels, test_preds, average='macro')):.4f}")
    print(f"F1 Score : {(f1_score(test_labels, test_preds, average='macro')):.4f}")

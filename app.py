import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

from PIL import Image


class BaselineCNN(nn.Module):
    def __init__(self, n): 
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*16*16,128), nn.ReLU(),
            nn.Linear(128,n)
        )
    def forward(self,x): return self.classifier(self.features(x))

class BaselineBN(BaselineCNN):
    def __init__(self,n):
        super().__init__(n)
        self.features = nn.Sequential(
            nn.Conv2d(3,16,3,1,1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,1,1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,1,1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2)
        )


class LeakyCNN(BaselineCNN):
    def __init__(self,n):
        super().__init__(n)
        self.features = nn.Sequential(
            nn.Conv2d(3,16,3,1,1), nn.LeakyReLU(0.01), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,1,1), nn.LeakyReLU(0.01), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,1,1), nn.LeakyReLU(0.01), nn.MaxPool2d(2)
        )

class Dropout25CNN(BaselineCNN):
    def __init__(self,n):
        super().__init__(n)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*16*16,128), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(128,n)
        )

class DeeperCNN(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,16,3,1,1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,1,1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,1,1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,1,1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*8*8,256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256,n)
        )
    def forward(self,x): 
        return self.classifier(self.features(x))

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//reduction, channels, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.fc(x)

class SECNN(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1), nn.ReLU(), SEBlock(16), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(), SEBlock(32), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), SEBlock(64), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*16*16,128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128,n)
        )
    def forward(self, x):
        return self.classifier(self.conv(x))


def train_epoch(model, loader, opt, crit, device):
    model.train()
    loss_sum, preds_all, labs_all = 0, [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        out = model(xb)
        loss = crit(out, yb)
        loss.backward()
        opt.step()
        loss_sum += loss.item() * xb.size(0)
        preds_all += out.argmax(1).cpu().tolist()
        labs_all  += yb.cpu().tolist()
    return loss_sum / len(loader.dataset), accuracy_score(labs_all, preds_all)

@torch.no_grad()
def eval_with_metrics(model, loader, crit, device):
    model.eval()
    loss_sum, preds_all, labs_all = 0, [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        loss_sum += crit(out, yb).item() * xb.size(0)
        preds_all += out.argmax(1).cpu().tolist()
        labs_all  += yb.cpu().tolist()
    acc = accuracy_score(labs_all, preds_all)
    p, r, f1, _ = precision_recall_fscore_support(labs_all, preds_all, average=None, zero_division=0)
    return loss_sum / len(loader.dataset), acc, p, r, f1, labs_all, preds_all

def train_and_eval(model, name, trn_ld, val_ld, tst_ld, epochs, lr, device, classes):
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    print(f"\n-- {name} --")
    
    for e in range(1, epochs+1):
        tl, ta = train_epoch(model, trn_ld, opt, crit, device)
        vl, va, *_ = eval_with_metrics(model, val_ld, crit, device)
        
        train_losses.append(tl)
        train_accs.append(ta)
        val_losses.append(vl)
        val_accs.append(va)
        print(f"Ep{e}: training_loss={tl:.4f} training_accuracy={ta:.4f} | validation_loss={vl:.4f} validation_accuracy={va:.4f}")

    tl, ta, p, r, f1, ytrue, ypred = eval_with_metrics(model, tst_ld, crit, device)
    print(f"\n{name} Test Acc: {ta:.4f}\n" + str(classification_report(ytrue, ypred, target_names=classes)))
    
    return model, {
        'acc_overall': ta, 'p_per': p, 'r_per': r, 'f1_per': f1,
        'train_loss': train_losses, 'train_acc': train_accs,
        'val_loss': val_losses, 'val_acc': val_accs
    }


def predict_color(image_path, model, transform, device, classes, topk=1):
    model.eval()
    img = Image.open(image_path).convert('RGB')
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    idxs = np.argsort(probs)[::-1][:topk]
    return [(classes[i], float(probs[i])) for i in idxs]


def train_manual_models(manual_defs, epochs, lr, train_loader, val_loader, test_loader, device, classes):
    manual_results = {}
    for name, mdl in manual_defs.items():
        print(f"\n[MAIN] Manual model: {name}")
        _, metrics = train_and_eval(
            mdl, name,
            train_loader, val_loader, test_loader,
            epochs, lr,
            device=device, classes=classes
        )
        manual_results[name] = metrics
    best = max(manual_results, key=lambda k: manual_results[k]['acc_overall'])
    return manual_results, best


def train_transfer_models(tl_defs, epochs, lr, train_loader, val_loader, test_loader, device, classes):
    transfer_results = {}
    for name, mdl in tl_defs.items():
        print(f"\nTransfer model: {name}")
        if name is 'resnet34':
            mdl.fc = nn.Linear(mdl.fc.in_features, len(classes))
        else:
            mdl.classifier[1] = nn.Linear(mdl.classifier[1].in_features, len(classes))
        _, metrics = train_and_eval(
            mdl, name,
            train_loader, val_loader, test_loader,
            epochs, lr,
            device=device, classes=classes
        )
        transfer_results[name] = metrics
    return transfer_results

def hyperparameter_tuning(param_grid, manual_defs, best_manual, train_loader, val_loader, classes, device):
    grid_search_results = []
    best_val_acc = 0.0
    best_params = {}
    print("\nStarting hyperparameter grid search...")

    for lr, opt_name in param_grid:
        print(f"Testing lr={lr}, optimizer={opt_name}")
        model = manual_defs[best_manual].__class__(len(classes)).to(device)

        if opt_name == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=lr)
        elif opt_name == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=lr)
        else:
            optimizer = optim.AdamW(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for _ in range(5):
            train_epoch(model, train_loader, optimizer, criterion, device)

        _, val_acc, *_ = eval_with_metrics(model, val_loader, criterion, device)
        print(f"--> Val Acc: {val_acc:.4f}\n")

        grid_search_results.append({'lr': lr, 'optimizer': opt_name, 'val_acc': val_acc})
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = {'lr': lr, 'optimizer': opt_name}

    return grid_search_results, best_params, best_val_acc

def train_model_best_hyperparameters(model_best, best_params, epochs, criterion, train_loader, device):
    opt_name, lr = best_params['optimizer'], best_params['lr']

    optimizer = (
        optim.SGD(model_best.parameters(), lr=lr) if opt_name=='SGD' else
        optim.Adam(model_best.parameters(), lr=lr) if opt_name=='Adam'else 
        optim.AdamW(model_best.parameters(), lr=lr)
    )

    for epoch in range(1, epochs+1):
        train_epoch(model_best, train_loader, optimizer, criterion, device)

def plot_manual_results(manual_results):
    for name in manual_results:
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.plot(manual_results[name]['train_loss'], label='Train')
        plt.plot(manual_results[name]['val_loss'], label='Validation')
        plt.title(f'{name} - Loss')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.subplot(1,2,2)
        plt.plot(manual_results[name]['train_acc'], label='Train')
        plt.plot(manual_results[name]['val_acc'], label='Validation') 
        plt.title(f'{name} - Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        plt.tight_layout()
        plt.show()

def plot_accurcy_comparison(manual_results, transfer_results):
    all_models = list(manual_results.keys()) + list(transfer_results.keys())
    all_acc = [manual_results[m]['acc_overall'] for m in manual_results] + [transfer_results[m]['acc_overall'] for m in transfer_results]
    plt.figure(figsize=(10,5))
    plt.bar(all_models, all_acc)
    plt.title('Model Comparison: Test Accuracy')
    plt.xticks(rotation=45)
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.show()    

def plot_table_classification_report(ytrue, ypred, classes):
    report = classification_report(ytrue, ypred, target_names=classes, output_dict=True)
    df_report = pd.DataFrame(report).transpose().round(4)
    plt.figure(figsize=(8,3))
    plt.table(cellText=df_report.values, colLabels=df_report.columns, 
             rowLabels=df_report.index, cellLoc='center', loc='center')
    plt.axis('off')
    plt.title('Classification Report')
    plt.tight_layout()
    plt.show()

def plot_metrics_per_class(p, r, f1, best_manual, classes):
    plt.figure(figsize=(12,6))
    x = np.arange(len(classes))
    width = 0.25
    plt.bar(x-width, p, width, label='Precision')
    plt.bar(x, r, width, label='Recall')
    plt.bar(x+width, f1, width, label='F1-Score')
    plt.xticks(x, classes, rotation=45)
    plt.title(f'Per-Class Metrics ({best_manual} Tuned)')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(ytrue, ypred, best_manual, classes):
    cm = confusion_matrix(ytrue, ypred)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.title(f'Confusion Matrix ({best_manual} Tuned)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def plot_table_best_hyperparameters(best_params):
    df_params = pd.DataFrame(list(best_params.items()), columns=['Parameter', 'Value'])
    plt.figure(figsize=(6,2))
    plt.table(
        cellText=df_params.values,
        colLabels=df_params.columns,
        cellLoc='center',
        loc='center'
    )
    plt.axis('off')
    plt.title('Best Hyperparameters')
    plt.tight_layout()
    plt.show()

def plot_hyperparameter_results(results):
    df_grid = pd.DataFrame(results)
    pivot = df_grid.pivot(index='lr', columns='optimizer', values='val_acc')

    plt.figure(figsize=(8,6))
    sns.heatmap(pivot, annot=True, fmt='.4f', cmap='Blues')
    plt.title('Validation Accuracy for Grid Search Hyperparameters')
    plt.xlabel('Optimizer')
    plt.ylabel('Learning Rate')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[MAIN] Using device: {device}")

    DATA_DIR = '.'
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LR_BASE = 1e-3

    base_tf = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2,0.2),
        transforms.ToTensor()
    ])

    train_ds = datasets.ImageFolder(os.path.join(DATA_DIR,'train'), transform=base_tf)
    val_ds   = datasets.ImageFolder(os.path.join(DATA_DIR,'val'),   transform=base_tf)
    test_ds  = datasets.ImageFolder(os.path.join(DATA_DIR,'test'),  transform=base_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    classes = train_ds.classes
    num_classes = len(classes)
    print(f"[MAIN] Found {num_classes} classes: {classes}")

    manual_defs = {
        'baseline': BaselineCNN(num_classes),
        'batchnorm': BaselineBN(num_classes),
        'leakyrelu': LeakyCNN(num_classes),
        'dropout25': Dropout25CNN(num_classes),
        'deeper': DeeperCNN(num_classes),
        'secnn': SECNN(num_classes)
    }

    tl_defs = {
      'resnet34': models.resnet34(pretrained=True),
      'mobilenet_v2': models.mobilenet_v2(pretrained=True)
    }
        
    manual_results, best_manual = train_manual_models(
        manual_defs, NUM_EPOCHS, LR_BASE, train_loader, val_loader, test_loader, device, classes
    )
    print(f"[MAIN] Best manual model: {best_manual}")

    transfer_results = train_transfer_models(
        tl_defs, NUM_EPOCHS, LR_BASE, train_loader, val_loader, test_loader, device, classes
    )

    plot_manual_results(manual_results)
    plot_accurcy_comparison(manual_results, transfer_results)

    LR_GRID = [1e-4, 1e-3, 5e-3, 1e-2]
    OPT_GRID = ['SGD', 'Adam', 'AdamW']
    param_grid = list(itertools.product(LR_GRID, OPT_GRID))
    grid_search_results, best_params, best_val_acc = hyperparameter_tuning(
        param_grid, manual_defs, best_manual, train_loader, val_loader, classes, device
    )
    
    print(f"[GRID SEARCH] Best Params: {best_params}, Best Val Acc: {best_val_acc:.4f}")

    model_best = manual_defs[best_manual].__class__(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    train_model_best_hyperparameters(model_best, best_params, NUM_EPOCHS, criterion, train_loader, device)

    _, tacc, p, r, f1, ytrue, ypred = eval_with_metrics(model_best, test_loader, criterion, device)
    print(f"\nTuned {best_manual} Test Acc: {tacc:.4f}")
    print(classification_report(ytrue, ypred, target_names=classes))

    plot_confusion_matrix(ytrue, ypred, best_manual, classes)
    plot_metrics_per_class(p, r, f1, best_manual, classes)
    plot_hyperparameter_results(grid_search_results)
    plot_table_classification_report(ytrue, ypred, classes)
    plot_table_best_hyperparameters(best_params)

    sample = os.path.join(DATA_DIR,'test', classes[0], os.listdir(os.path.join(DATA_DIR,'test',classes[0]))[0])
    print("[MAIN] Sample prediction:", predict_color(sample, manual_defs[best_manual], transform=base_tf, device=device, classes=classes))

if __name__ == "__main__":
    main()

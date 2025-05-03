import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)
from PIL import Image
import optuna
from optuna.samplers import TPESampler
import seaborn as sns  # ADDED
from optuna.visualization.matplotlib import plot_param_importances  # ADDED


# === Utility Functions ===

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
        # ADDED: Store metrics
        train_losses.append(tl)
        train_accs.append(ta)
        val_losses.append(vl)
        val_accs.append(va)
        print(f"Ep{e}: trL={tl:.4f} trA={ta:.4f} | valL={vl:.4f} valA={va:.4f}")

    # Test evaluation remains the same
    tl, ta, p, r, f1, ytrue, ypred = eval_with_metrics(model, tst_ld, crit, device)
    print(f"\n{name} Test Acc: {ta:.4f}\n" + str(classification_report(ytrue, ypred, target_names=classes)))
    
    # ADDED: Return training history
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


# === Model Definitions ===

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


# === Main Entry Point ===

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

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    classes     = train_ds.classes
    num_classes = len(classes)
    print(f"[MAIN] Found {num_classes} classes: {classes}")

    # Manual model definitions
    manual_defs = {
        'baseline':  BaselineCNN(num_classes),
        'batchnorm': BaselineBN(num_classes),
        'leakyrelu': LeakyCNN(num_classes),
        'dropout25': Dropout25CNN(num_classes),
        'deeper':    DeeperCNN(num_classes),
        'secnn': SECNN(num_classes)
    }

    # Train & evaluate manual models
    manual_results, manual_hist = {}, {}
    for name, mdl in manual_defs.items():
        print(f"\n[MAIN] Manual model: {name}")
        _, metrics = train_and_eval(
            mdl, name,
            train_loader, val_loader, test_loader,
            NUM_EPOCHS, lr=LR_BASE,
            device=device, classes=classes
        )
        manual_results[name] = metrics

    # Identify best manual model
    best_manual = max(manual_results, key=lambda k: manual_results[k]['acc_overall'])
    print(f"[MAIN] Best manual model: {best_manual}")

    # Transfer learning definitions
    tl_defs = {
      'resnet18':     models.resnet18(pretrained=True),
      'resnet34':     models.resnet34(pretrained=True),
      'mobilenet_v2': models.mobilenet_v2(pretrained=True)
    }
    # Train & evaluate transfer models
    transfer_results, transfer_hist = {}, {}
    for name, mdl in tl_defs.items():
        print(f"\n[MAIN] Transfer model: {name}")
        if 'resnet' in name:
            mdl.fc = nn.Linear(mdl.fc.in_features, num_classes)
        else:
            mdl.classifier[1] = nn.Linear(mdl.classifier[1].in_features, num_classes)
        _, metrics = train_and_eval(
            mdl, name,
            train_loader, val_loader, test_loader,
            epochs=NUM_EPOCHS, lr=LR_BASE,
            device=device, classes=classes
        )
        transfer_results[name] = metrics

    # 1. Training Curves for Manual Models
    for name in manual_results:
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.plot(manual_results[name]['train_loss'], label='Train')
        plt.plot(manual_results[name]['val_loss'], label='Validation')
        plt.title(f'{name} - Loss'); plt.xlabel('Epoch'); plt.legend()
        
        plt.subplot(1,2,2)
        plt.plot(manual_results[name]['train_acc'], label='Train')
        plt.plot(manual_results[name]['val_acc'], label='Validation') 
        plt.title(f'{name} - Accuracy'); plt.xlabel('Epoch'); plt.legend()
        plt.tight_layout(); plt.show()

    # 2. Test Accuracy Comparison
    all_models = list(manual_results.keys()) + list(transfer_results.keys())
    all_acc = [manual_results[m]['acc_overall'] for m in manual_results] + \
              [transfer_results[m]['acc_overall'] for m in transfer_results]
    plt.figure(figsize=(10,5))
    plt.bar(all_models, all_acc)
    plt.title('Model Comparison: Test Accuracy'); plt.xticks(rotation=45)
    plt.ylabel('Accuracy'); plt.tight_layout(); plt.show()

    # Hyperparameter tuning with Optuna TPESampler
    def objective(trial):
        # Suggest hyperparams
        lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
        optimizer_name = trial.suggest_categorical('optimizer', ['SGD', 'Adam'])

        # Create fresh model
        model = manual_defs[best_manual].__class__(num_classes).to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr) if optimizer_name == 'SGD' else optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        # Warmup training
        for _ in range(5):
            train_epoch(model, train_loader, optimizer, criterion, device)

        # Validation accuracy
        _, val_acc, _, _, _, _, _ = eval_with_metrics(model, val_loader, criterion, device)
        return val_acc

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=20)

    print('\n[MAIN] Optuna best params: ', study.best_params)
    print('[MAIN] Best validation accuracy: ', study.best_value)

    # Evaluate best trial on test set
    best_trial = study.best_trial
    model_best = manual_defs[best_manual].__class__(num_classes).to(device)
    opt_name = best_trial.params['optimizer']
    lr_best = best_trial.params['lr']
    optimizer = optim.SGD(model_best.parameters(), lr=lr_best) if opt_name == 'SGD' else optim.Adam(model_best.parameters(), lr=lr_best)
    criterion = nn.CrossEntropyLoss()

    # Full training
    for epoch in range(1, NUM_EPOCHS+1):
        train_epoch(model_best, train_loader, optimizer, criterion, device)
    tl, ta, p, r, f1, ytrue, ypred = eval_with_metrics(model_best, test_loader, criterion, device)
    print(f"\nTuned {best_manual} Test Acc: {ta:.4f}")
    print(classification_report(ytrue, ypred, target_names=classes))

    # 3. Confusion Matrix for Tuned Model
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(ytrue, ypred)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.title(f'Confusion Matrix ({best_manual} Tuned)'); plt.xlabel('Predicted'); plt.ylabel('True')
    plt.show()

    # 4. Per-Class Metrics
    plt.figure(figsize=(12,6))
    x = np.arange(len(classes))
    width = 0.25
    plt.bar(x-width, p, width, label='Precision')
    plt.bar(x, r, width, label='Recall')
    plt.bar(x+width, f1, width, label='F1-Score')
    plt.xticks(x, classes, rotation=45)
    plt.title(f'Per-Class Metrics ({best_manual} Tuned)'); plt.legend()
    plt.tight_layout(); plt.show()

    # 5. Hyperparameter Importance
    plot_param_importances(study)

    # === TABLES ===
    # 1. Classification Report
    report = classification_report(ytrue, ypred, target_names=classes, output_dict=True)
    df_report = pd.DataFrame(report).transpose().round(4)
    plt.figure(figsize=(8,3))
    plt.table(cellText=df_report.values, colLabels=df_report.columns, 
             rowLabels=df_report.index, cellLoc='center', loc='center')
    plt.axis('off'); plt.title('Classification Report'); plt.tight_layout(); plt.show()

    # 2. Best Hyperparameters
    best_params = study.best_params
    df_params = pd.DataFrame(best_params.items(), columns=['Parameter','Value'])
    plt.figure(figsize=(6,2))
    plt.table(cellText=df_params.values, colLabels=df_params.columns, 
             cellLoc='center', loc='center')
    plt.axis('off'); plt.title('Best Hyperparameters'); plt.tight_layout(); plt.show()
     
    # Demo prediction
    sample = os.path.join(DATA_DIR,'test', classes[0], os.listdir(os.path.join(DATA_DIR,'test',classes[0]))[0])
    print("[MAIN] Sample prediction:", predict_color(sample, manual_defs[best_manual], transform=base_tf, device=device, classes=classes))

if __name__ == "__main__":
    main()

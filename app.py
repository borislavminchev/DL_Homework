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
    opt   = optim.Adam(model.parameters(), lr=lr)
    crit  = nn.CrossEntropyLoss()
    print(f"\n-- {name} --")
    for e in range(1, epochs+1):
        tl, ta = train_epoch(model, trn_ld, opt, crit, device)
        vl, va, *_ = eval_with_metrics(model, val_ld, crit, device)
        print(f"Ep{e}: trL={tl:.4f} trA={ta:.4f} | valL={vl:.4f} valA={va:.4f}")
    tl, ta, p, r, f1, ytrue, ypred = eval_with_metrics(model, tst_ld, crit, device)
    print(f"\n{name} Test Acc: {ta:.4f}\n" + str(classification_report(ytrue, ypred, target_names=classes)))
    return model, {'acc_overall': ta, 'p_per': p, 'r_per': r, 'f1_per': f1}

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
            nn.Linear(64*16*16,128), nn.ReLU(), nn.Dropout(0.5),
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
    # Step 0: Device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[MAIN] Using device: {device}")

    # Step 1: Data paths, transforms, loaders
    DATA_DIR   = '.'  # ← update this
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LR_BASE    = 1e-3

    # base_tf = transforms.Compose([
    #     transforms.Resize((128,128)),
    #     transforms.ToTensor()
    # ])
    base_tf = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2,0.2),
        transforms.ToTensor()
    ])

    train_ds     = datasets.ImageFolder(os.path.join(DATA_DIR,'train'), transform=base_tf)
    val_ds       = datasets.ImageFolder(os.path.join(DATA_DIR,'val'),   transform=base_tf)
    test_ds      = datasets.ImageFolder(os.path.join(DATA_DIR,'test'),  transform=base_tf)

    train_loader     = DataLoader(train_ds,     batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
    val_loader       = DataLoader(val_ds,       batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader      = DataLoader(test_ds,      batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    classes     = train_ds.classes
    num_classes = len(classes)
    print(f"[MAIN] Found {num_classes} classes: {classes}")

    # Manual model definitions
    manual_defs = {
        # 'baseline':  BaselineCNN(num_classes),
        # 'batchnorm': BaselineBN(num_classes),
        # 'leakyrelu': LeakyCNN(num_classes),
        # 'dropout25': Dropout25CNN(num_classes),
        # 'deeper':    DeeperCNN(num_classes),
        'secnn': SECNN(num_classes)
    }

    # Train & evaluate manual models
    manual_results = {}
    for name, mdl in manual_defs.items():
        print(f"\n[MAIN] Manual model: {name}")
        _, metrics = train_and_eval(
            mdl, name,
            train_loader, val_loader, test_loader,
            NUM_EPOCHS, LR_BASE,
            device, classes
        )
        manual_results[name] = metrics

    # Transfer learning definitions
    tl_defs = {
      'resnet18':     models.resnet18(pretrained=True),
      'resnet34':     models.resnet34(pretrained=True),
      'mobilenet_v2': models.mobilenet_v2(pretrained=True)
    }
    # Train & evaluate transfer models
    transfer_results = {}
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

    # Hyperparameter tuning example
    tune_results = []
    for lr in [1e-2,1e-3,1e-4]:
        for optn in ['SGD','Adam']:
            print(f"\n[MAIN] Tuning lr={lr}, opt={optn}")
            m = BaselineCNN(num_classes).to(device)
            opt = optim.SGD(m.parameters(),lr=lr) if optn=='SGD' else optim.Adam(m.parameters(),lr=lr)
            crit = nn.CrossEntropyLoss()
            for _ in range(3):
                train_epoch(m, train_loader, opt, crit, device)
            _, acc, p, r, f1, *_ = eval_with_metrics(m, val_loader, crit, device)
            tune_results.append({'lr':lr,'opt':optn,'val_acc':acc,'val_f1_mean':f1.mean()})
    print("\n[MAIN] Hyperparameter tuning results:")
    print(pd.DataFrame(tune_results))

    # Summary plot: manual vs transfer accuracy
    df_m = pd.DataFrame([{'model':k,'acc':v['acc_overall'],'type':'manual'} for k,v in manual_results.items()])
    df_t = pd.DataFrame([{'model':k,'acc':v['acc_overall'],'type':'transfer'} for k,v in transfer_results.items()])
    df_all = pd.concat([df_m,df_t],ignore_index=True)
    plt.figure()
    for t in df_all['type'].unique():
        sub = df_all[df_all['type']==t]
        plt.plot(sub['model'], sub['acc'], marker='o', label=t)
    plt.xticks(rotation=45)
    plt.ylabel('Test Accuracy')
    plt.title('Manual vs Transfer')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Demo prediction
    best = max(manual_results, key=lambda k: manual_results[k]['acc_overall'])
    print(f"\n[MAIN] Best manual model: {best}")
    sample = os.path.join(DATA_DIR,'test', classes[0], os.listdir(os.path.join(DATA_DIR,'test',classes[0]))[0])
    print("[MAIN] Sample prediction:", predict_color(sample, manual_defs[best], transform=base_tf, device=device, classes=classes))

if __name__ == "__main__":
    main()

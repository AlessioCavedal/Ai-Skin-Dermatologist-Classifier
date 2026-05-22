'''
train_skin_cancer_fast.py

Fast training version for testing - optimized for quick results
'''

import torch
from torch.utils.data import DataLoader
from torch import nn
import torchmetrics
from ai_derm_dataloader import SkinCancerDataloader
import random
import pandas as pd
import matplotlib.pyplot as plt
from ai_derm_model import OurCNN

if __name__ == "__main__":

    try:
        if torch.backends.mps.is_available():
            dev = "mps"
        elif torch.cuda.is_available():
            dev = "cuda"
        elif hasattr(torch.version, "hip") and torch.version.hip:
            dev = "hip"
        else:
            dev = "cpu"
    except AttributeError:
        dev = "cpu"

    print(f"Best device for your config is: {dev}")
    
    # FAST TRAINING CONFIGURATION
    bs = 4  # Very small batch size for speed
    lr = 0.002  # Higher learning rate for faster convergence
    max_epochs = 10  # Very few epochs for quick test
    
    seed = random.randint(0, 1000)
    train_ds = SkinCancerDataloader("../Skin cancer/train", transform=None, seed=seed)
    test_ds = SkinCancerDataloader("../Skin cancer/test", transform=None, seed=seed)
    
    print("="*50)
    print("🚀 FAST SKIN CANCER CLASSIFIER TRAINING")
    print("="*50)
    print(f"📊 Dataset: {len(train_ds)} train, {len(test_ds)} test images")
    print(f"🖥️  Device: {dev.upper()}")
    print(f"⚙️  Batch size: {bs}, Learning rate: {lr}")
    print(f"🔄 Max epochs: {max_epochs} (FAST MODE)")
    print("⏱️  Estimated time: 5-10 minutes")
    print("="*50)

    # Optimize DataLoader for fast CPU training
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0, pin_memory=False)
    test_dl = DataLoader(test_ds, batch_size=bs, num_workers=0, pin_memory=False)

    model = OurCNN().to(dev)
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    acc_train = torchmetrics.Accuracy(task='multiclass', num_classes=2).to(dev)
    acc_test = torchmetrics.Accuracy(task='multiclass', num_classes=2).to(dev)

    best_acc = 0.0
    wait = 0
    patience = 3  # Early stopping after 3 epochs

    losses = []
    accs_train = []
    accs_test = []

    print("\n🚀 Starting FAST training...")
    for ep in range(max_epochs):
        print(f"\nEPOCH: {ep+1}/{max_epochs}")

        model.train()
        total_loss = 0

        for b, (x, y) in enumerate(train_dl):
            x, y = x.to(dev), y.to(dev)
            optim.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optim.step()

            total_loss += loss.item()
            acc_train.update(out, y)

            if b % 50 == 0:  # Less frequent updates for speed
                print(f"  Batch: {b}, Loss: {loss.item():.4f}")

        ep_acc = acc_train.compute()
        acc_train.reset()
        avg_loss = total_loss / len(train_dl)
        print(f"📊 Train Loss: {avg_loss:.4f}, Train Accuracy: {ep_acc:.4f}")

        losses.append(avg_loss)
        accs_train.append(ep_acc.item())

        model.eval()
        with torch.no_grad():
            for x, y in test_dl:
                x, y = x.to(dev), y.to(dev)
                out = model(x)
                acc_test.update(out, y)

        val_acc = acc_test.compute()
        acc_test.reset()
        print(f"🎯 Test Accuracy: {val_acc:.4f}")
        accs_test.append(val_acc.item())

        if val_acc > best_acc:
            best_acc = val_acc
            wait = 0
            torch.save(model.state_dict(), '../Model/model_skin_cancer_fast.pth')
            print(f"💾 Model saved! Best accuracy: {best_acc:.4f}")
        else:
            wait += 1

        if wait >= patience:
            print(f"⏹️  Early stopping after {ep+1} epochs")
            break

    print(f"\n🎉 FAST TRAINING COMPLETED!")
    print(f"🏆 Best accuracy achieved: {best_acc:.4f}")
    print(f"💾 Model saved as: model_skin_cancer_fast.pth")

    # Plotting
    epochs = range(1, len(losses) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, label='Train Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Fast Training - Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accs_train, label='Train Acc', marker='o')
    plt.plot(epochs, accs_test, label='Test Acc', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Fast Training - Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('../Model/skin_cancer_fast_accuracy.png')
    plt.show()
    
    print("\n🚀 Ready to test! Run: python skin_cancer_gui_fast.py")

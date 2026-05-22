'''

train_test.py

This file contains the function for train and test the model.

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
    
    # Optimized for high accuracy (95%+) training
    bs = 16  # Larger batch size for better gradients
    lr = 0.0005  # Lower learning rate for stable training
    max_epochs = 20  # Reduced epochs as requested
    
    seed = random.randint(0, 1000)
    train_ds = SkinCancerDataloader("../Skin cancer/train", transform=None, seed=seed)
    test_ds = SkinCancerDataloader("../Skin cancer/test", transform=None, seed=seed)
    
    print("="*50)
    print("🔬 SKIN CANCER CLASSIFIER TRAINING")
    print("🎯 TARGET: 95%+ ACCURACY")
    print("="*50)
    print(f"📊 Dataset: {len(train_ds)} train, {len(test_ds)} test images")
    print(f"🖥️  Device: {dev.upper()}")
    print(f"⚙️  Batch size: {bs}, Learning rate: {lr}")
    print(f"🔄 Max epochs: {max_epochs}")
    print(f"🎯 Target accuracy: 95%+")
    print(f"⏳ Estimated time: 15-25 minutes")
    print("="*50)

    '''
    
    This training function includes early stopping: if the test 
    accuracy doesn't improve for a "patience" epochs, the
    training process stops automatically.
    (usually 9/10 Epochs are sufficent)

    '''

    # Optimize DataLoader for high accuracy training
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=2, pin_memory=False, drop_last=True)
    test_dl = DataLoader(test_ds, batch_size=bs, num_workers=2, pin_memory=False)

    model = OurCNN().to(dev)
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='max', factor=0.5, patience=3, verbose=True)

    acc_train = torchmetrics.Accuracy(task='multiclass', num_classes=2).to(dev)
    acc_test = torchmetrics.Accuracy(task='multiclass', num_classes=2).to(dev)

    best_acc = 0.0
    wait = 0
    patience = 8  # High patience for achieving 95% accuracy

    losses = []
    accs_train = []
    accs_test = []

    for ep in range(max_epochs):
        print(f"EPOCH: {ep+1}/{max_epochs}")

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

            if b % 20 == 0:  # More frequent updates for CPU training
                print(f"B: {b}, L: {loss.item():.4f}")

        ep_acc = acc_train.compute()
        acc_train.reset()
        avg_loss = total_loss / len(train_dl)
        print(f"L: {avg_loss:.4f}, TRAIN ACCURACY: {ep_acc:.4f}")

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
        print(f"TEST ACCURACY: {val_acc:.4f}")
        accs_test.append(val_acc.item())
        
        # Update learning rate scheduler
        scheduler.step(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            wait = 0
            torch.save(model.state_dict(), '../Model/model_skin_cancer.pth')
            print("Model correctly SAVED in ../Model/model_skin_cancer.pth")
            
            # Check if we reached 95% accuracy
            if val_acc >= 0.95:
                print("🎉🎉🎉 CONGRATULATIONS! 🎉🎉🎉")
                print(f"🏆 TARGET REACHED: {val_acc*100:.2f}% accuracy!")
                print("🎯 Model successfully achieved 95%+ accuracy!")
        else:
            wait += 1

        if wait >= patience:
            print("Stop")
            break

    print("="*50)
    print("🏁 TRAINING FINISHED!")
    print("="*50)
    print(f"🏆 FINAL BEST ACCURACY: {best_acc*100:.2f}%")
    
    if best_acc >= 0.95:
        print("🎉🎉🎉 SUCCESS! 🎉🎉🎉")
        print("✅ Target of 95%+ accuracy ACHIEVED!")
    else:
        print("⚠️  Target of 95% accuracy not reached.")
        print(f"📈 Best achieved: {best_acc*100:.2f}%")
        print("💡 Consider increasing epochs or adjusting hyperparameters")
    
    print("="*50)

    # Plotting
    epochs = range(1, len(losses) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Skin Cancer Classification - Training Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accs_train, label='Train Acc')
    plt.plot(epochs, accs_test, label='Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Skin Cancer Classification - Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('../Model/skin_cancer_accuracy.png')
    plt.show()

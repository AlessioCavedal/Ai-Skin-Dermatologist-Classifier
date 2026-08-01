"""
Addestramento.

Il test set non viene mai letto qui: early stopping, scheduler e taratura della
soglia usano un validation ritagliato dal train. Il test lo guardo una volta
sola, in evaluate.py.

Ogni architettura porta la propria ricetta in config.RECIPES, quindi
`python train.py --arch NOME` riproduce esattamente la run che ho tenuto.
"""

from __future__ import annotations

import argparse
import csv
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

import config
import dataset as ds
import engine
from model import ARCHITECTURES, build_model, count_parameters


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Addestramento del classificatore")
    p.add_argument("--arch", default=config.DEFAULT_ARCH, choices=ARCHITECTURES)
    p.add_argument("--epochs", type=int, default=None, help="sovrascrive la ricetta")
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--workers", type=int, default=config.NUM_WORKERS)
    p.add_argument("--tag", default=None, help="etichetta per i file di output")
    args = p.parse_args()

    epochs_override = args.epochs
    for key, value in config.RECIPES[args.arch].items():
        setattr(args, key, value)
    if epochs_override is not None:
        args.epochs = epochs_override
    return args


def build_loaders(args, splits):
    train_tf = (ds.train_transform(args.img_size, args.normalize) if args.augment
                else ds.eval_transform(args.img_size, args.normalize))
    eval_tf = ds.eval_transform(args.img_size, args.normalize)

    train_ds = ds.SkinLesionDataset(splits["train"], train_tf)
    val_ds = ds.SkinLesionDataset(splits["val"], eval_tf)

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    common = dict(num_workers=args.workers, pin_memory=False,
                  persistent_workers=args.workers > 0)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                          worker_init_fn=engine.seed_worker, generator=generator, **common)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **common)
    return train_ds, val_ds, train_dl, val_dl


def build_scheduler(args, optimizer, steps_per_epoch):
    """OneCycle avanza a ogni batch, Plateau a ogni epoca sulla metrica."""
    if args.scheduler == "onecycle":
        sched = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=steps_per_epoch,
            pct_start=min(0.3, config.WARMUP_EPOCHS / max(args.epochs, 1)))
        return sched, None
    return None, torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3)


def main() -> None:
    args = parse_args()
    run_name = args.tag or args.arch

    engine.set_seed(args.seed)
    dev = config.device()
    splits = ds.build_splits(config.VAL_FRACTION, args.seed)
    train_ds, val_ds, train_dl, val_dl = build_loaders(args, splits)
    model = build_model(args.arch).to(dev)

    print("=" * 70)
    print(f"ADDESTRAMENTO  |  {run_name}")
    print("=" * 70)
    print(f"  architettura : {args.arch} ({count_parameters(model):,} parametri)")
    print(f"  device       : {dev}")
    print(f"  risoluzione  : {args.img_size}px   batch {args.batch_size}   lr {args.lr}")
    print(f"  augmentation : {'attiva' if args.augment else 'disattivata'}"
          f"   normalizzazione: {'attiva' if args.normalize else 'disattivata'}")
    print(f"  scheduler    : {args.scheduler}")
    print(f"  seed         : {args.seed}")
    print(f"  train / val  : {len(train_ds)} / {len(val_ds)}   {train_ds.class_counts()}")
    print(f"  test         : {len(splits['test'])} immagini, non usate qui")
    print(f"  epoche max   : {args.epochs} (early stopping patience {args.patience})")
    print("=" * 70)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=config.WEIGHT_DECAY)
    batch_sched, epoch_sched = build_scheduler(args, optimizer, len(train_dl))

    ckpt_path = config.MODEL_DIR / f"{run_name}.pth"
    history: list[dict] = []
    best_score, best_epoch, wait = -1.0, -1, 0
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss = engine.train_one_epoch(model, train_dl, loss_fn, optimizer, dev, batch_sched)
        metrics = engine.evaluate(model, val_dl, dev)
        score = getattr(metrics, config.SELECTION_METRIC)
        if epoch_sched is not None:
            epoch_sched.step(score)

        history.append({"epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
                        "train_loss": train_loss,
                        **{f"val_{k}": v for k, v in metrics.as_dict().items()}})

        marker = ""
        if score > best_score:
            best_score, best_epoch, wait = score, epoch, 0
            torch.save({
                "state_dict": model.state_dict(),
                "arch": args.arch,
                "img_size": args.img_size,
                "normalize": args.normalize,
                "augmentation": args.augment,
                "scheduler": args.scheduler,
                "classes": config.CLASSES,
                "seed": args.seed,
                "epoch": epoch,
                "selection_metric": config.SELECTION_METRIC,
                "val_metrics": metrics.as_dict(),
                "threshold": 0.5,
            }, ckpt_path)
            marker = "  <-- best (salvato)"
        else:
            wait += 1

        print(f"[{epoch:3d}/{args.epochs}] train_loss {train_loss:.4f} | "
              f"val {metrics.summary()}{marker}")

        if wait >= args.patience:
            print(f"\nEarly stopping all'epoca {epoch}: nessun miglioramento da {wait} epoche.")
            break

    elapsed = time.time() - started
    log_path = config.LOG_DIR / f"{run_name}.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    val_probs, val_targets, _ = engine.predict_probs(model, val_dl, dev)
    threshold, thr_score = engine.tune_threshold(val_probs, val_targets)

    ckpt["threshold"] = threshold
    ckpt["train_seconds"] = elapsed
    torch.save(ckpt, ckpt_path)

    print("\n" + "=" * 70)
    print(f"FINE in {elapsed / 60:.1f} min")
    print(f"  epoca migliore        : {best_epoch}  (val/{config.SELECTION_METRIC} {best_score:.4f})")
    print(f"  soglia tarata su val  : {threshold:.3f}  (bal-acc {thr_score:.4f})")
    print(f"  checkpoint            : {ckpt_path.name}")
    print(f"  log                   : {log_path.name}")
    print(f"\nProssimo passo:  python evaluate.py --checkpoint '{ckpt_path.name}'")
    print("=" * 70)

    plot_history(history, run_name, args)


def plot_history(history: list[dict], run_name: str, args) -> None:
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].plot(epochs, [h["train_loss"] for h in history], label="train")
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="validation")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Loss")

    axes[1].plot(epochs, [h["val_accuracy"] for h in history], label="accuracy")
    axes[1].plot(epochs, [h["val_balanced_accuracy"] for h in history], label="balanced accuracy")
    axes[1].set_ylabel("valore")
    axes[1].set_title("Accuracy (validation)")

    axes[2].plot(epochs, [h["val_sensitivity"] for h in history], label="sensibilita")
    axes[2].plot(epochs, [h["val_specificity"] for h in history], label="specificita")
    axes[2].plot(epochs, [h["val_auroc"] for h in history], label="AUROC")
    axes[2].set_ylabel("valore")
    axes[2].set_title("Metriche cliniche (validation)")

    for ax in axes:
        ax.set_xlabel("epoca")
        ax.legend()
        ax.grid(alpha=.3)

    fig.suptitle(f"{args.arch}  |  seed {args.seed}", fontsize=10)
    fig.tight_layout()
    out = config.FIG_DIR / f"{run_name}_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  grafici               : {out.name}")


if __name__ == "__main__":
    main()

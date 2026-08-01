"""
Valutazione finale sul test set.

E' l'unico script che legge il test. Modello e soglia arrivano gia' decisi dal
training, quindi il numero prodotto qui e' una stima onesta.

Uso:
    python evaluate.py --checkpoint NOME.pth
    python evaluate.py --checkpoint NOME.pth --clean-test
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

import config
import dataset as ds
import engine
from model import build_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Valutazione sul test set")
    p.add_argument("--checkpoint", required=True, help="nome file dentro Model/")
    p.add_argument("--clean-test", action="store_true",
                   help="esclude le immagini di test near-duplicate del train")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=config.NUM_WORKERS)
    return p.parse_args()


def leaked_test_images() -> set[str]:
    path = config.DATASET_DIR / "test_leaked_images.txt"
    if not path.exists():
        raise FileNotFoundError(f"{path.name} non trovato. Esegui prima: python audit_dataset.py")
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def test_loader(transform, batch_size: int, workers: int, clean: bool) -> DataLoader:
    items = ds.scan_split(config.TEST_DIR)
    if clean:
        leaked = leaked_test_images()
        items = [it for it in items if f"{it[0].parent.name}/{it[0].name}" not in leaked]
    return DataLoader(ds.SkinLesionDataset(items, transform), batch_size=batch_size,
                      shuffle=False, num_workers=workers, persistent_workers=workers > 0)


def load_checkpoint(name: str, device: str):
    ckpt = torch.load(config.MODEL_DIR / name, map_location=device, weights_only=False)
    model = build_model(ckpt["arch"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt, ds.eval_transform(ckpt["img_size"], ckpt.get("normalize", True))


def roc_curve(probs: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_pos = max(int((targets == 1).sum()), 1)
    n_neg = max(int((targets == 0).sum()), 1)
    tpr, fpr = [], []
    for t in np.unique(np.concatenate([[0.0, 1.0], probs]))[::-1]:
        preds = probs >= t
        tpr.append((preds & (targets == 1)).sum() / n_pos)
        fpr.append((preds & (targets == 0)).sum() / n_neg)
    return np.array(fpr), np.array(tpr)


def plot_report(name: str, probs: np.ndarray, targets: np.ndarray, m, threshold: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    cm = np.array([[m.tn, m.fp], [m.fn, m.tp]])
    axes[0].imshow(cm, cmap="Blues")
    axes[0].set_xticks([0, 1], ["pred. benigno", "pred. maligno"])
    axes[0].set_yticks([0, 1], ["vero benigno", "vero maligno"])
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=14,
                         color="white" if cm[i, j] > cm.max() / 2 else "black")
    axes[0].set_title(f"Matrice di confusione (soglia {threshold:.2f})")

    fpr, tpr = roc_curve(probs, targets)
    axes[1].plot(fpr, tpr, label=f"AUROC = {m.auroc:.4f}")
    axes[1].plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    axes[1].set_xlabel("1 - specificita (falsi positivi)")
    axes[1].set_ylabel("sensibilita (veri positivi)")
    axes[1].set_title("Curva ROC")
    axes[1].legend()
    axes[1].grid(alpha=.3)

    fig.suptitle(f"Test set - {name}", fontsize=11)
    fig.tight_layout()
    out = config.FIG_DIR / f"test_{name}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"    figura: {out.relative_to(config.ROOT)}")


def main() -> None:
    args = parse_args()
    dev = config.device()
    model, ckpt, transform = load_checkpoint(args.checkpoint, dev)
    loader = test_loader(transform, args.batch_size, args.workers, args.clean_test)
    name = args.checkpoint.replace(".pth", "") + ("_clean" if args.clean_test else "")

    print("=" * 70)
    print("VALUTAZIONE SUL TEST SET")
    print(f"  device: {dev} | modello: {ckpt['arch']} | immagini: {len(loader.dataset)}"
          + ("  (depurato dai near-duplicate)" if args.clean_test else ""))
    print("=" * 70)

    threshold = float(ckpt.get("threshold", 0.5))
    results = {"n_test_images": len(loader.dataset), "clean_test": args.clean_test}

    probs, targets, loss = engine.predict_probs(model, loader, dev)
    for thr in dict.fromkeys([0.5, threshold]):
        m = engine.compute_metrics(probs, targets, thr, loss)
        print(f"  soglia {thr:.2f}: {m.summary()}")
        results[f"thr{thr:.3f}"] = m.as_dict()

    plot_report(name, probs, targets,
                engine.compute_metrics(probs, targets, threshold, loss), threshold)

    results["checkpoint_meta"] = {k: v for k, v in ckpt.items() if k != "state_dict"}
    report = {"timestamp": datetime.now().isoformat(timespec="seconds"), "runs": {name: results}}
    out = config.REPORT_DIR / f"test_metrics_{name}.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nReport salvato: {out.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()

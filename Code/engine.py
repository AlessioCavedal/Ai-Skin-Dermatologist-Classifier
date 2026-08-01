"""
Cicli di training e valutazione, metriche, taratura della soglia.

Per uno screening oncologico l'accuracy da sola non basta: un falso negativo
non e' paragonabile a un falso positivo, quindi calcolo anche sensibilita,
specificita e AUROC.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def check_targets(targets: np.ndarray, n_classes: int = 2) -> None:
    valid = np.isin(targets, np.arange(n_classes))
    if not valid.all():
        raise RuntimeError(
            f"Etichette corrotte: {(~valid).sum()} valori fuori da [0, {n_classes - 1}], "
            f"per esempio {targets[~valid][:5].tolist()}."
        )


@dataclass
class Metrics:
    loss: float = 0.0
    accuracy: float = 0.0
    balanced_accuracy: float = 0.0
    sensitivity: float = 0.0
    specificity: float = 0.0
    precision: float = 0.0
    f1: float = 0.0
    auroc: float = 0.0
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    def as_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        return (f"loss {self.loss:.4f} | acc {self.accuracy:.4f} | "
                f"bal-acc {self.balanced_accuracy:.4f} | sens {self.sensitivity:.4f} | "
                f"spec {self.specificity:.4f} | auroc {self.auroc:.4f}")


def _auroc(probs: np.ndarray, targets: np.ndarray) -> float:
    """Statistica U di Mann-Whitney, con media dei ranghi sui pari merito."""
    n_pos, n_neg = int((targets == 1).sum()), int((targets == 0).sum())
    if not n_pos or not n_neg:
        return 0.0

    order = np.argsort(probs)
    ranks = np.empty(len(probs), dtype=float)
    ranks[order] = np.arange(1, len(probs) + 1)

    sorted_probs = probs[order]
    i = 0
    while i < len(sorted_probs):
        j = i
        while j + 1 < len(sorted_probs) and sorted_probs[j + 1] == sorted_probs[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2
        i = j + 1

    return float((ranks[targets == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def compute_metrics(probs: np.ndarray, targets: np.ndarray,
                    threshold: float = 0.5, loss: float = 0.0) -> Metrics:
    """probs = probabilita della classe maligna."""
    preds = (probs >= threshold).astype(int)
    tp = int(((preds == 1) & (targets == 1)).sum())
    tn = int(((preds == 0) & (targets == 0)).sum())
    fp = int(((preds == 1) & (targets == 0)).sum())
    fn = int(((preds == 0) & (targets == 1)).sum())

    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    denom = precision + sensitivity

    return Metrics(
        loss=float(loss),
        accuracy=(tp + tn) / len(targets),
        balanced_accuracy=(sensitivity + specificity) / 2,
        sensitivity=sensitivity,
        specificity=specificity,
        precision=precision,
        f1=2 * precision * sensitivity / denom if denom else 0.0,
        auroc=_auroc(probs, targets),
        tp=tp, fp=fp, tn=tn, fn=fn,
    )


def train_one_epoch(model, loader, loss_fn, optimizer, device, scheduler=None) -> float:
    model.train()
    running, seen = 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        running += loss.item() * x.size(0)
        seen += x.size(0)
    return running / max(seen, 1)


@torch.no_grad()
def predict_probs(model, loader, device):
    """
    Restituisce (probabilita della classe maligna, etichette, loss media).

    Le etichette non passano mai per il device: MPS non supporta int64 e la
    copia asincrona puo' restituire memoria non inizializzata, in modo
    intermittente e senza sollevare errori.
    """
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    all_probs, all_targets = [], []
    running, seen = 0.0, 0

    for x, y_cpu in loader:
        all_targets.append(y_cpu.numpy().copy())
        x = x.to(device, non_blocking=True)
        y = y_cpu.to(device)

        logits = model(x)
        probs = torch.softmax(logits, dim=1)

        running += loss_fn(logits, y).item() * x.size(0)
        seen += x.size(0)
        all_probs.append(probs[:, 1].detach().cpu().numpy().copy())

    targets = np.concatenate(all_targets)
    check_targets(targets)
    return np.concatenate(all_probs), targets, running / max(seen, 1)


def evaluate(model, loader, device, threshold: float = 0.5) -> Metrics:
    probs, targets, loss = predict_probs(model, loader, device)
    return compute_metrics(probs, targets, threshold, loss)


def tune_threshold(probs: np.ndarray, targets: np.ndarray,
                   objective: str = "balanced_accuracy") -> tuple[float, float]:
    """Cerca la soglia migliore sul validation. Usare 0.5 sarebbe arbitrario."""
    candidates = np.unique(np.concatenate([probs, np.linspace(0.05, 0.95, 91)]))
    best_thr, best_score = 0.5, -1.0
    for thr in candidates:
        score = getattr(compute_metrics(probs, targets, float(thr)), objective)
        if score > best_score:
            best_score, best_thr = score, float(thr)
    return best_thr, best_score

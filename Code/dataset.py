"""Dataset, split a tre vie e trasformazioni."""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import config
from dedup import IMG_EXTENSIONS, load_groups

Item = tuple[Path, int]


def scan_split(split_dir: Path) -> list[Item]:
    """Elenca (percorso, etichetta) da una cartella con benign/ e malignant/."""
    items: list[Item] = []
    for label, class_name in enumerate(config.CLASSES):
        class_dir = split_dir / class_name
        if class_dir.is_dir():
            files = sorted((p for p in class_dir.iterdir()
                            if p.suffix.lower() in IMG_EXTENSIONS), key=lambda p: p.name)
            items += [(p, label) for p in files]
    if not items:
        raise FileNotFoundError(f"Nessuna immagine trovata in {split_dir}")
    return items


def stratified_split(items: list[Item], val_fraction: float, seed: int) -> tuple[list[Item], list[Item]]:
    """
    Divide train e validation mantenendo le proporzioni di classe.

    E' group-aware: le immagini quasi identiche formano un gruppo e finiscono
    tutte dallo stesso lato, altrimenti il validation conterrebbe gemelli di
    immagini viste in addestramento.
    """
    groups = load_groups()
    if groups is None:
        print("ATTENZIONE: gruppi di duplicati non trovati, split puramente casuale.\n"
              "            Esegui prima: python dedup.py")

    def bucket_key(path: Path) -> str:
        key = f"{path.parent.name}/{path.name}"
        gid = groups.get(key) if groups else None
        return f"g{gid}" if gid is not None else f"s{key}"

    buckets: dict[str, list[Item]] = {}
    for item in items:
        buckets.setdefault(bucket_key(item[0]), []).append(item)

    by_class: dict[int, list[list[Item]]] = {i: [] for i in range(len(config.CLASSES))}
    for members in buckets.values():
        majority = Counter(label for _, label in members).most_common(1)[0][0]
        by_class[majority].append(members)

    rng = random.Random(seed)
    train_items: list[Item] = []
    val_items: list[Item] = []
    for bucket_list in by_class.values():
        rng.shuffle(bucket_list)
        target = round(sum(len(b) for b in bucket_list) * val_fraction)
        taken = 0
        for bucket in bucket_list:
            if taken < target:
                val_items += bucket
                taken += len(bucket)
            else:
                train_items += bucket

    rng.shuffle(train_items)
    rng.shuffle(val_items)
    return train_items, val_items


def build_splits(val_fraction: float = config.VAL_FRACTION,
                 seed: int = config.SEED) -> dict[str, list[Item]]:
    train_items, val_items = stratified_split(scan_split(config.TRAIN_DIR), val_fraction, seed)
    return {"train": train_items, "val": val_items, "test": scan_split(config.TEST_DIR)}


def _normalize(enabled: bool) -> list:
    return [transforms.Normalize(config.NORM_MEAN, config.NORM_STD)] if enabled else []


def train_transform(img_size: int, normalize: bool = True) -> transforms.Compose:
    """
    Le lesioni non hanno un orientamento canonico, quindi flip e rotazioni sono
    trasformazioni neutre. Il colore invece e' diagnostico, percio' tengo hue
    molto contenuto.
    """
    return transforms.Compose([
        transforms.Resize((int(img_size * 1.15), int(img_size * 1.15))),
        transforms.RandomRotation(180),
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.9, 1.11)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.20, hue=0.02),
        transforms.ToTensor(),
        *_normalize(normalize),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.12)),
    ])


def eval_transform(img_size: int, normalize: bool = True) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        *_normalize(normalize),
    ])


class SkinLesionDataset(Dataset):
    def __init__(self, items: list[Item], transform):
        self.items = list(items)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        return self.transform(Image.open(path).convert("RGB")), label

    def class_counts(self) -> dict[str, int]:
        counts = Counter(label for _, label in self.items)
        return {name: counts[i] for i, name in enumerate(config.CLASSES)}

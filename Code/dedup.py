"""
Rilevamento delle immagini quasi identiche.

Nel dataset la stessa lesione compare piu' volte, fotografata in condizioni
quasi uguali. Se i gemelli finiscono su lati opposti dello split, il modello
riconosce in validation immagini gia' viste in addestramento e le metriche si
gonfiano.

Procedo in due stadi: il difference hash produce i candidati a basso costo, poi
confronto i pixel per confermarli. Il solo hash non basta, perche' queste
immagini si somigliano tutte e produce centinaia di falsi positivi.

Uso:
    python dedup.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

import config

HAMMING_CANDIDATE = 8
CORR_NEAR_DUPLICATE = 0.97      # coppie casuali arrivano al massimo a 0.90
CORR_ALMOST_IDENTICAL = 0.99
THUMB = 32

GROUPS_PATH = config.DATASET_DIR / "train_duplicate_groups.json"
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def dhash(image: Image.Image, size: int = 8) -> np.ndarray:
    img = image.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    px = np.asarray(img, dtype=np.int16)
    return np.packbits((px[:, 1:] > px[:, :-1]).flatten())


def thumbnail_vector(image: Image.Image, size: int = THUMB) -> np.ndarray:
    """Miniatura normalizzata: invariante a luminosita e contrasto."""
    g = np.asarray(image.convert("L").resize((size, size), Image.Resampling.LANCZOS),
                   dtype=np.float32)
    return ((g - g.mean()) / (g.std() + 1e-6)).ravel()


def hamming_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _POPCOUNT[np.bitwise_xor(a[:, None, :], b[None, :, :])].sum(axis=2)


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / a.size)


def fingerprint(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    dh = np.zeros((len(paths), 8), dtype=np.uint8)
    th = np.zeros((len(paths), THUMB * THUMB), dtype=np.float32)
    for i, path in enumerate(paths):
        with Image.open(path) as im:
            dh[i] = dhash(im)
            th[i] = thumbnail_vector(im)
    return dh, th


def random_pair_correlation(thumbs, idx_a, idx_b, n: int = 4000, seed: int = 0) -> dict:
    """Correlazione fra coppie casuali: mi serve per tarare la soglia."""
    rng = np.random.default_rng(seed)
    corr = (thumbs[rng.choice(idx_a, n)] * thumbs[rng.choice(idx_b, n)]).mean(axis=1)
    return {"mean": float(corr.mean()), "max": float(corr.max())}


class UnionFind:
    """Se A somiglia a B e B a C, li tengo tutti e tre nello stesso gruppo."""

    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def scan_train_pool() -> list[tuple[str, Path]]:
    out = []
    for class_name in config.CLASSES:
        d = config.TRAIN_DIR / class_name
        if d.is_dir():
            out += [(f"{class_name}/{p.name}", p) for p in sorted(d.iterdir())
                    if p.suffix.lower() in IMG_EXTENSIONS]
    return out


def compute_groups(verbose: bool = True) -> dict:
    files = scan_train_pool()
    n = len(files)
    if verbose:
        print(f"Analisi di {n} immagini del pool di training...")

    dh, th = fingerprint([p for _, p in files])
    dist = hamming_matrix(dh, dh)
    candidates = np.argwhere(np.triu(dist <= HAMMING_CANDIDATE, k=1))

    uf = UnionFind(n)
    confirmed = 0
    for i, j in candidates:
        if correlation(th[i], th[j]) >= CORR_NEAR_DUPLICATE:
            uf.union(int(i), int(j))
            confirmed += 1

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(uf.find(i), []).append(i)
    groups = sorted((sorted(v) for v in clusters.values() if len(v) > 1), key=len, reverse=True)
    grouped = sum(len(g) for g in groups)

    if verbose:
        sizes = Counter(len(g) for g in groups)
        print(f"  candidati dall'hash : {len(candidates)}")
        print(f"  coppie confermate   : {confirmed}")
        print(f"  gruppi di duplicati : {len(groups)}")
        print(f"  immagini coinvolte  : {grouped} su {n} ({100 * grouped / n:.1f}%)")
        print("  dimensioni gruppi   : " +
              ", ".join(f"{k} immagini x{v}" for k, v in sorted(sizes.items())))
        print("\n  I gruppi piu' grandi:")
        for g in groups[:5]:
            print(f"    {', '.join(files[i][0] for i in g)}")

    return {
        "params": {"hamming_candidate": HAMMING_CANDIDATE,
                   "corr_threshold": CORR_NEAR_DUPLICATE,
                   "thumb_size": THUMB},
        "n_images": n,
        "n_groups": len(groups),
        "n_images_grouped": grouped,
        "groups": [[files[i][0] for i in g] for g in groups],
    }


def load_groups() -> dict[str, int] | None:
    """Mappa 'classe/file' -> gruppo. None se non sono ancora stati calcolati."""
    if not GROUPS_PATH.exists():
        return None
    payload = json.loads(GROUPS_PATH.read_text())
    return {key: gid for gid, group in enumerate(payload["groups"]) for key in group}


if __name__ == "__main__":
    GROUPS_PATH.write_text(json.dumps(compute_groups(), indent=2))
    print(f"\nSalvato in: {GROUPS_PATH.relative_to(config.ROOT)}")

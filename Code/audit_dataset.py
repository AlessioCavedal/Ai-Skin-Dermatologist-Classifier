"""
Verifiche di integrita sul dataset, con attenzione al leakage fra train e test.

Controlla conteggi e bilanciamento, file illeggibili, dimensioni, duplicati
esatti e near-duplicate. Produce un report markdown in Results/dataset/ e
l'elenco delle immagini di test contaminate che evaluate.py --clean-test
rilegge.

Uso:
    python audit_dataset.py
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

import config
from dedup import (CORR_ALMOST_IDENTICAL, CORR_NEAR_DUPLICATE, HAMMING_CANDIDATE,
                   IMG_EXTENSIONS, THUMB, dhash, hamming_matrix, random_pair_correlation,
                   thumbnail_vector)


def collect() -> list[dict]:
    records = []
    for split, split_dir in (("train", config.TRAIN_DIR), ("test", config.TEST_DIR)):
        for class_name in config.CLASSES:
            class_dir = split_dir / class_name
            if not class_dir.is_dir():
                continue
            for path in sorted(class_dir.iterdir()):
                if path.suffix.lower() in IMG_EXTENSIONS:
                    records.append({"path": path, "split": split, "class": class_name})
    return records


def write_leak_list(paths: set[Path]) -> None:
    out = config.DATASET_DIR / "test_leaked_images.txt"
    out.write_text("\n".join(sorted(f"{p.parent.name}/{p.name}" for p in paths)))
    print(f"Elenco immagini di test contaminate: {out.relative_to(config.ROOT)}")


def main() -> None:
    print("Audit del dataset in corso...")
    records = collect()
    lines: list[str] = []
    problems: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    emit("# Audit del dataset")
    emit()
    emit(f"Generato il {datetime.now():%d/%m/%Y %H:%M}")
    emit()

    emit("## 1. Conteggi e bilanciamento")
    emit()
    emit("| Split | benign | malignant | totale | % maligni |")
    emit("|---|---:|---:|---:|---:|")
    for split in ("train", "test"):
        c = Counter(r["class"] for r in records if r["split"] == split)
        total = sum(c.values())
        emit(f"| {split} | {c['benign']} | {c['malignant']} | {total} | "
             f"{100 * c['malignant'] / total:.1f}% |")
    emit(f"| **totale** | | | **{len(records)}** | |")
    emit()
    emit("Lo sbilanciamento e' lieve (circa 55/45): non serve oversampling, ma "
         "giustifica l'uso della balanced accuracy accanto all'accuracy.")
    emit()

    emit("## 2. Integrita e proprieta delle immagini")
    emit()
    sizes: Counter = Counter()
    modes: Counter = Counter()
    corrupt: list[Path] = []
    hashes: dict[str, list[dict]] = defaultdict(list)
    dhashes = np.zeros((len(records), 8), dtype=np.uint8)
    thumbs = np.zeros((len(records), THUMB * THUMB), dtype=np.float32)

    for i, record in enumerate(records):
        try:
            with Image.open(record["path"]) as im:
                im.load()
                sizes[im.size] += 1
                modes[im.mode] += 1
                dhashes[i] = dhash(im)
                thumbs[i] = thumbnail_vector(im)
        except Exception as exc:                              # noqa: BLE001
            corrupt.append(record["path"])
            problems.append(f"immagine illeggibile: {record['path'].name} ({exc})")
            continue
        hashes[hashlib.md5(record["path"].read_bytes()).hexdigest()].append(record)

    emit(f"- Immagini illeggibili o corrotte: **{len(corrupt)}**")
    emit("- Dimensioni presenti: " +
         ", ".join(f"`{w}x{h}` ({n})" for (w, h), n in sizes.most_common()))
    emit("- Modalita colore: " + ", ".join(f"`{m}` ({n})" for m, n in modes.most_common()))
    emit()
    if len(sizes) == 1 and next(iter(sizes)) == (224, 224):
        emit("Tutte le immagini sono 224x224: ridimensionare a 128 butta via circa il "
             "67% dei pixel, ed e' il prezzo che paga cnn_v1.")
    emit()

    emit("## 3. Duplicati esatti (MD5)")
    emit()
    cross_exact = [v for v in hashes.values() if len({r["split"] for r in v}) > 1]
    within_exact = [v for v in hashes.values()
                    if len(v) > 1 and len({r["split"] for r in v}) == 1]
    label_conflict = [v for v in hashes.values() if len({r["class"] for r in v}) > 1]

    emit(f"- File unici: **{len(hashes)}** su {len(records) - len(corrupt)} analizzati")
    emit(f"- Duplicati esatti **train <-> test**: **{len(cross_exact)}**")
    emit(f"- Duplicati esatti interni a uno split: **{len(within_exact)}**")
    emit(f"- Stessa immagine con etichette diverse: **{len(label_conflict)}**")
    emit()
    for group in within_exact:
        emit(f"  - `{'`, `'.join(r['path'].name for r in group)}` "
             f"({group[0]['split']}/{group[0]['class']})")
    for group in cross_exact:
        problems.append("LEAK esatto: " +
                        ", ".join(f"{r['split']}/{r['class']}/{r['path'].name}" for r in group))
        emit(f"  - **LEAK**: `{'`, `'.join(r['split'] + '/' + r['path'].name for r in group)}`")
    emit()

    emit("## 4. Near-duplicate tra train e test")
    emit()
    corrupt_set = set(corrupt)
    idx_train = np.array([i for i, r in enumerate(records)
                          if r["split"] == "train" and r["path"] not in corrupt_set])
    idx_test = np.array([i for i, r in enumerate(records)
                         if r["split"] == "test" and r["path"] not in corrupt_set])

    dist = hamming_matrix(dhashes[idx_test], dhashes[idx_train])
    candidates = np.argwhere(dist <= HAMMING_CANDIDATE)
    random_corr = random_pair_correlation(thumbs, idx_test, idx_train)

    near_pairs = []
    for ti, tj in candidates:
        corr = float(thumbs[idx_test[ti]] @ thumbs[idx_train[tj]] / thumbs.shape[1])
        if corr >= CORR_NEAR_DUPLICATE:
            near_pairs.append((records[idx_test[ti]], records[idx_train[tj]], corr))
    near_pairs.sort(key=lambda x: -x[2])

    affected = {p[0]["path"] for p in near_pairs}
    almost_identical = [p for p in near_pairs if p[2] >= CORR_ALMOST_IDENTICAL]
    cross_class = [p for p in near_pairs if p[0]["class"] != p[1]["class"]]

    emit("Metodo a due stadi: il difference hash produce i candidati "
         f"(Hamming <= {HAMMING_CANDIDATE}), poi ogni candidato viene verificato sulla "
         "correlazione fra miniature normalizzate. Il solo hash non basta: queste "
         "immagini sono strutturalmente molto simili e genera centinaia di falsi positivi.")
    emit()
    emit(f"Riferimento empirico: coppie **casuali** raggiungono al massimo una "
         f"correlazione di {random_corr['max']:.3f} (media {random_corr['mean']:.3f}). "
         f"La soglia e' quindi fissata a {CORR_NEAR_DUPLICATE}.")
    emit()
    emit(f"- Candidati dal solo hash: {len(candidates)} (in gran parte collisioni)")
    emit(f"- Coppie confermate (corr >= {CORR_NEAR_DUPLICATE}): **{len(near_pairs)}**")
    emit(f"- Di cui praticamente identiche (corr >= {CORR_ALMOST_IDENTICAL}): "
         f"**{len(almost_identical)}**")
    emit(f"- **Immagini di test coinvolte: {len(affected)} su {len(idx_test)} "
         f"({100 * len(affected) / len(idx_test):.1f}% del test set)**")
    emit(f"- Coppie con classi discordanti: {len(cross_class)}")
    emit()

    if near_pairs:
        emit("Le prime coppie per correlazione:")
        emit()
        emit("| correlazione | immagine di test | immagine di train | stessa classe |")
        emit("|---:|---|---|---|")
        for test_rec, train_rec, corr in near_pairs[:30]:
            same = "si" if test_rec["class"] == train_rec["class"] else "**NO**"
            emit(f"| {corr:.4f} | {test_rec['class']}/{test_rec['path'].name} | "
                 f"{train_rec['class']}/{train_rec['path'].name} | {same} |")
        emit()
        emit("Sono la stessa lesione fotografata due volte, finita per un lato nel train "
             "e per l'altro nel test. **Il problema e' presente all'origine nel dataset**, "
             "non e' stato introdotto qui.")
        emit()
        emit(f"Su queste {len(affected)} immagini il modello e' avvantaggiato. "
             "`evaluate.py --clean-test` ricalcola le metriche escludendole.")
        problems.append(f"{len(near_pairs)} coppie near-duplicate train/test che coinvolgono "
                        f"{len(affected)} immagini di test "
                        f"({100 * len(affected) / len(idx_test):.1f}%)")
        write_leak_list(affected)
    else:
        emit("Nessuna sovrapposizione rilevata: il test set e' indipendente dal train.")
    emit()

    emit("## Esito")
    emit()
    if problems:
        emit("Problemi rilevati:")
        emit()
        for problem in problems:
            emit(f"- {problem}")
    else:
        emit("Nessun problema bloccante rilevato.")
    emit()

    out = config.DATASET_DIR / "dataset_audit.md"
    out.write_text("\n".join(lines))
    print(f"\nReport salvato in: {out.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()

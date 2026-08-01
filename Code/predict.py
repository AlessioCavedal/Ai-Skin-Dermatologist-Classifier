"""
Classificazione di singole immagini.

La soglia e' quella tarata sul validation, non un 0.5 arbitrario. Con due
classi e softmax la probabilita dell'argmax e' sempre >= 0.5, quindi una
"soglia al 50%" non scarterebbe mai nulla: sotto ABSTENTION_THRESHOLD il
predictor si astiene invece di forzare una risposta.

Uso:
    python predict.py immagine.jpg [altre...] [--checkpoint NOME.pth]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

import config
import dataset as ds
from model import build_model


@dataclass
class Prediction:
    label: str
    prob_malignant: float
    confidence: float
    abstained: bool
    threshold: float

    def describe(self) -> str:
        if self.abstained:
            return (f"INCERTO (confidenza {self.confidence * 100:.1f}% sotto la soglia di "
                    f"astensione {config.ABSTENTION_THRESHOLD * 100:.0f}%) - "
                    f"p(maligno) = {self.prob_malignant * 100:.1f}%")
        return (f"{self.label.upper()} - p(maligno) = {self.prob_malignant * 100:.1f}% "
                f"(confidenza {self.confidence * 100:.1f}%, soglia {self.threshold:.2f})")


class SkinLesionPredictor:
    def __init__(self, checkpoint: str | Path | None = None, device: str | None = None):
        self.device = device or config.device()

        path = self._resolve_checkpoint(checkpoint)
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        self.model = build_model(ckpt["arch"]).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

        self.checkpoint_path = path
        self.arch = ckpt["arch"]
        self.img_size = ckpt["img_size"]
        self.threshold = float(ckpt.get("threshold", 0.5))
        self.classes = ckpt.get("classes", config.CLASSES)
        self.val_metrics = ckpt.get("val_metrics", {})
        self.transform = ds.eval_transform(self.img_size, ckpt.get("normalize", True))

    @staticmethod
    def _resolve_checkpoint(checkpoint) -> Path:
        if checkpoint is not None:
            path = Path(checkpoint)
            if not path.is_absolute():
                path = config.MODEL_DIR / path
            if not path.exists():
                raise FileNotFoundError(f"Checkpoint non trovato: {path}")
            return path
        candidates = sorted(config.MODEL_DIR.glob("*.pth"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise FileNotFoundError(
                f"Nessun checkpoint in {config.MODEL_DIR}. Esegui prima: python train.py")
        return candidates[-1]

    @torch.no_grad()
    def predict_image(self, image: Image.Image) -> Prediction:
        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)

        probs = torch.softmax(self.model(x), dim=1)[0]

        p_malignant = float(probs[1])
        label = self.classes[1] if p_malignant >= self.threshold else self.classes[0]
        confidence = p_malignant if label == self.classes[1] else 1.0 - p_malignant

        return Prediction(
            label=label,
            prob_malignant=p_malignant,
            confidence=confidence,
            abstained=confidence < config.ABSTENTION_THRESHOLD,
            threshold=self.threshold,
        )

    def predict_path(self, path: str | Path) -> Prediction:
        return self.predict_image(Image.open(path))


def main() -> None:
    p = argparse.ArgumentParser(description="Classifica una o piu' immagini")
    p.add_argument("images", nargs="+")
    p.add_argument("--checkpoint", default=None)
    args = p.parse_args()

    predictor = SkinLesionPredictor(args.checkpoint)
    print(f"Modello: {predictor.checkpoint_path.name} ({predictor.arch}, "
          f"{predictor.img_size}px, soglia {predictor.threshold:.2f})\n")

    for image_path in args.images:
        print(f"{Path(image_path).name:40s} {predictor.predict_path(image_path).describe()}")

    print("\nStrumento di supporto: non sostituisce una valutazione dermatologica.")


if __name__ == "__main__":
    main()

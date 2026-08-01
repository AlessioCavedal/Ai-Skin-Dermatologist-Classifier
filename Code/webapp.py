"""
Interfaccia web locale.

Le immagini caricate non vengono salvate su disco: restano in memoria per il
tempo della predizione, e l'anteprima la genera il browser dal file locale.

Uso:
    python webapp.py                 # http://127.0.0.1:7860
    python webapp.py --port 8080
"""

from __future__ import annotations

import argparse
import io
import traceback

import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError

import config
from predict import SkinLesionPredictor

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

_predictors: dict[str, SkinLesionPredictor] = {}


def available_models() -> list[dict]:
    models = []
    for path in sorted(config.MODEL_DIR.glob("*.pth")):
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:                                     # noqa: BLE001
            continue
        val = ckpt.get("val_metrics", {})
        models.append({
            "file": path.name,
            "arch": ckpt.get("arch", "?"),
            "img_size": ckpt.get("img_size"),
            "threshold": round(float(ckpt.get("threshold", 0.5)), 3),
            "epoch": ckpt.get("epoch"),
            "val_balanced_accuracy": round(val.get("balanced_accuracy", 0), 4) or None,
        })
    return models


def get_predictor(name: str) -> SkinLesionPredictor:
    if name not in _predictors:
        _predictors[name] = SkinLesionPredictor(name)
    return _predictors[name]


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/models")
def api_models():
    models = available_models()
    if not models:
        return jsonify({"models": [],
                        "error": "Nessun checkpoint in Model/. Esegui prima: python train.py"})
    return jsonify({"models": models})


@app.post("/api/predict")
def api_predict():
    if "image" not in request.files or not request.files["image"].filename:
        return jsonify({"error": "Nessuna immagine ricevuta."}), 400
    model_name = request.form.get("model") or ""
    if not model_name:
        return jsonify({"error": "Nessun modello selezionato."}), 400

    try:
        image = Image.open(io.BytesIO(request.files["image"].read()))
        image.load()
    except (UnidentifiedImageError, OSError):
        return jsonify({"error": "Il file non e' un'immagine leggibile. Usa JPG, PNG o BMP."}), 400

    try:
        predictor = get_predictor(model_name)
        pred = predictor.predict_image(image)
    except FileNotFoundError:
        return jsonify({"error": f"Checkpoint non trovato: {model_name}"}), 404
    except Exception:                                         # noqa: BLE001
        traceback.print_exc()
        return jsonify({"error": "Errore durante la classificazione. "
                                 "Il dettaglio e' nel terminale."}), 500

    return jsonify({
        "label": pred.label,
        "prob_malignant": pred.prob_malignant,
        "confidence": pred.confidence,
        "abstained": pred.abstained,
        "threshold": pred.threshold,
        "abstention_threshold": config.ABSTENTION_THRESHOLD,
        "model": {"file": predictor.checkpoint_path.name, "arch": predictor.arch,
                  "img_size": predictor.img_size, "device": predictor.device},
    })


def main() -> None:
    p = argparse.ArgumentParser(description="Interfaccia web del classificatore")
    # 7860 e non 5000: su macOS la 5000 e' occupata dal Ricevitore AirPlay.
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--host", default="127.0.0.1", help="127.0.0.1 = solo questo computer")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    print("=" * 62)
    print("  AI Skin Dermatologist - interfaccia web")
    print("=" * 62)
    models = available_models()
    if models:
        print(f"  Modelli disponibili ({len(models)}):")
        for m in models:
            acc = (f"bal-acc val {m['val_balanced_accuracy']:.3f}"
                   if m["val_balanced_accuracy"] else "")
            print(f"    - {m['file']}  [{m['arch']}, {m['img_size']}px, "
                  f"soglia {m['threshold']}]  {acc}")
    else:
        print("  ATTENZIONE: nessun checkpoint in Model/")
    print()
    print(f"  Apri nel browser:  http://{args.host}:{args.port}")
    print("  Per fermare: Ctrl+C")
    print("=" * 62)

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()

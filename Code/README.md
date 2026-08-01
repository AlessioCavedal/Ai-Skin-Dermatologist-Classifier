# Pipeline

Dettagli implementativi. Per installazione e uso generale vedi il
[README principale](../README.md).

## File

| File | Ruolo |
|---|---|
| `config.py` | percorsi, costanti e ricette di addestramento |
| `dataset.py` | split group-aware a tre vie, augmentation, normalizzazione |
| `dedup.py` | rilevamento delle immagini quasi identiche |
| `model.py` | le due architetture |
| `engine.py` | cicli di training, metriche, taratura della soglia |
| `train.py` | addestramento — non legge mai il test set |
| `evaluate.py` | valutazione finale sul test |
| `predict.py` | inferenza su singola immagine |
| `webapp.py` | interfaccia web (Flask) + `templates/` |
| `audit_dataset.py` | integrità, bilanciamento, duplicati, leakage |

## Comandi

```bash
python audit_dataset.py                       # 1. verifiche sul dataset
python dedup.py                               # 2. gruppi di near-duplicate
python train.py --arch cnn_v1                 # 3. addestramento
python evaluate.py --checkpoint cnn_v1.pth    # 4. valutazione finale
python webapp.py                              # interfaccia web

train.py     --arch {cnn_v1,resnet18}  --epochs N  --seed N  --workers N  --tag NOME
evaluate.py  --checkpoint NOME.pth  [--clean-test]
predict.py   IMMAGINE...  [--checkpoint NOME.pth]
webapp.py    [--port N]  [--host IP]
```

I passi 1 e 2 vanno eseguiti una volta sola: producono in `Results/dataset/` i file
che gli altri script rileggono. Senza il file dei gruppi, `dataset.py` avverte e
ripiega su uno split puramente casuale.

## I due modelli

**`cnn_v1`** — cinque blocchi Conv-BatchNorm-ReLU-MaxPool (da 32 a 512 canali) seguiti
da un classificatore denso. Accetta solo ingressi 128×128, perché il primo layer lineare
è dimensionato a mano su 512·4·4. Dei 5.831.298 parametri, il 72% sta in quel singolo
layer: è lì che si concentra il rischio di overfitting, ed è il motivo del dropout più
aggressivo (0.5) rispetto al secondo strato nascosto (0.3).

**`resnet18`** — pesi ImageNet, testa di classificazione sostituita, poi fine-tuning
completo con learning rate ridotto (1e-4). Nessun parametro viene congelato.

Ogni architettura porta la propria ricetta in `config.RECIPES` — risoluzione, batch,
learning rate, epoche, scheduler, augmentation, normalizzazione — quindi il solo
`--arch` riproduce la run corrispondente. Il seed è fisso e le run sono deterministiche.

## Garanzie metodologiche

| Garanzia | Come è ottenuta |
|---|---|
| Il test set non influenza nulla | solo `evaluate.py` lo legge; early stopping, scheduler e soglia usano il validation |
| Nessun gemello a cavallo dello split | `dedup.py` raggruppa le immagini quasi identiche, `dataset.py` assegna ogni gruppo a un lato solo |
| Run riproducibili | seed fisso propagato a `random`, `numpy`, `torch` e ai worker del DataLoader |
| Tracciabilità | ogni run scrive le metriche di ogni epoca in CSV; il checkpoint contiene architettura, soglia, seed e normalizzazione |
| Metriche adeguate al dominio | sensibilità, specificità, balanced accuracy, AUROC, matrice di confusione e ROC |
| Soglia reale | tarata sul validation, con banda di astensione. Con due classi e softmax una "soglia al 50%" non scarterebbe mai nulla, perché l'argmax ha sempre p ≥ 0,5 |

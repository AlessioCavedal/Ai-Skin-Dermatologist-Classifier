# AI Skin Dermatologist

Progetto universitario di Computer Vision: classificazione di lesioni cutanee in
**benigne** o **maligne** a partire da immagini dermatoscopiche.

Il progetto confronta due approcci: una **CNN progettata da zero** e una **ResNet18
pre-addestrata su ImageNet** sottoposta a fine-tuning.

![Interfaccia](docs/interfaccia.png)

## Risultati

| Modello | Accuratezza | AUROC | Sensibilità | Specificità |
|---|---:|---:|---:|---:|
| `cnn_v1` — CNN progettata da zero | 86,2% | 0,942 | 86,3% | 86,1% |
| `resnet18` — fine-tuning da ImageNet | **90,0%** | **0,967** | 90,3% | 89,7% |

Misurati sulle 660 immagini di test, con modello e soglia scelti esclusivamente sul
validation set.

## Requisiti

Python 3.9 o superiore. Le dipendenze sono in [`Code/requirements.txt`](Code/requirements.txt):
PyTorch, torchvision, Flask, Pillow, NumPy, Matplotlib.

```bash
git clone https://github.com/AlessioCav/Ai-Skin-Dermatologist-Classifier.git
cd Ai-Skin-Dermatologist-Classifier
python3 -m venv .venv
source .venv/bin/activate          # su Windows: .venv\Scripts\activate
pip install -r Code/requirements.txt
```

## Provare subito il classificatore

I due modelli addestrati non sono nel repository — pesano 65 MB — ma sono allegati
alla [Release più recente](https://github.com/AlessioCav/Ai-Skin-Dermatologist-Classifier/releases/latest).
Vanno scaricati in `Model/`, e a quel punto **non serve il dataset** per classificare
immagini proprie:

```bash
# dalla radice del progetto
curl -L -o Model/cnn_v1.pth   https://github.com/AlessioCav/Ai-Skin-Dermatologist-Classifier/releases/download/v1.0/cnn_v1.pth
curl -L -o Model/resnet18.pth https://github.com/AlessioCav/Ai-Skin-Dermatologist-Classifier/releases/download/v1.0/resnet18.pth

cd Code
python webapp.py                   # interfaccia web su http://127.0.0.1:7860
python predict.py immagine.jpg     # da riga di comando
```

Su macOS basta anche il doppio clic su `Avvia interfaccia.command`, che avvia il server
e apre il browser. Al primo avvio il sistema potrebbe bloccarlo perché scaricato da
internet: in quel caso *Impostazioni di Sistema → Privacy e Sicurezza → Apri comunque*.

## Riprodurre l'addestramento

Il dataset **non è incluso** nel repository: pesa 170 MB ed è materiale di terze parti.
Va scaricato da Kaggle:

**[fanconic/skin-cancer-malignant-vs-benign](https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign)**

Una volta scaricato, va estratto nella cartella `Skin cancer/` con questa struttura:

```
Skin cancer/
├── train/
│   ├── benign/        1440 immagini
│   └── malignant/     1197 immagini
└── test/
    ├── benign/         360 immagini
    └── malignant/      300 immagini
```

Poi:

```bash
cd Code
python audit_dataset.py            # verifiche di integrità e leakage
python dedup.py                    # gruppi di immagini quasi identiche
python train.py --arch cnn_v1      # oppure --arch resnet18
python evaluate.py --checkpoint cnn_v1.pth
```

Ogni architettura porta la propria ricetta di addestramento in `config.RECIPES`, quindi
il solo `--arch` è sufficiente a riprodurre esattamente i modelli inclusi: il seed è
fisso e le run sono deterministiche.

## Struttura

```
├── Code/          la pipeline: dataset, modelli, addestramento, valutazione, interfaccia
├── Model/         i due checkpoint addestrati (da scaricare dalle Release)
├── Results/       curve, matrici di confusione, metriche e audit del dataset
└── Skin cancer/   dataset (da scaricare separatamente)
```

Dettagli sulle scelte implementative: [`Code/README.md`](Code/README.md).

## Note metodologiche

Il test set non influenza in alcun modo l'addestramento: early stopping, scheduler e
taratura della soglia usano un validation set ritagliato dal train, e il test viene
letto una sola volta alla fine.

Lo split è **group-aware**. Il dataset contiene la stessa lesione fotografata più volte,
e uno split casuale ne separerebbe i gemelli fra train e validation, gonfiando le
metriche: `dedup.py` li raggruppa e assegna ogni gruppo interamente a un lato solo.

L'audit ha rilevato **57 coppie near-duplicate fra train e test**, che coinvolgono 26
immagini di test: il problema è presente all'origine nel dataset. `evaluate.py --clean-test`
ricalcola le metriche escludendole, e l'effetto misurato è di circa 0,4 punti.

Per uno screening oncologico l'accuratezza da sola non basta, quindi vengono sempre
riportate anche sensibilità, specificità, balanced accuracy e AUROC: un falso negativo
non è paragonabile a un falso positivo.

## Avvertenza

Strumento realizzato a scopo didattico. **Non è un dispositivo medico e non deve essere
usato per decisioni cliniche.** Non sostituisce in alcun modo la valutazione di un
dermatologo, e un esito benigno non esclude la necessità di un controllo specialistico.

## Licenza e attribuzione

Il codice è rilasciato sotto licenza [MIT](LICENSE). Sei libero di usarlo, modificarlo
e ridistribuirlo, anche in progetti tuoi, **a una condizione: devi citare l'autore**.

In concreto, la licenza MIT richiede che la nota di copyright e il testo della licenza
restino inclusi in ogni copia o parte sostanziale del codice riutilizzato. Se usi questo
lavoro — per intero o in parte, così com'è o modificato — mantieni il file `LICENSE` e
indica la fonte, ad esempio:

> Basato su *AI Skin Dermatologist* di Alessio Cavedal — https://github.com/AlessioCav/Ai-Skin-Dermatologist-Classifier

Il dataset è distribuito separatamente dai rispettivi autori e non è coperto da questa
licenza.

# Checkpoint addestrati

I pesi dei due modelli non sono versionati nel repository: sono file binari da 65 MB
complessivi, che appesantirebbero ogni clone senza essere diffabili.

Sono allegati alla [Release più recente](https://github.com/AlessioCavedal/Ai-Skin-Dermatologist-Classifier/releases/latest)
e vanno scaricati in questa cartella:

```bash
curl -L -o cnn_v1.pth   https://github.com/AlessioCavedal/Ai-Skin-Dermatologist-Classifier/releases/download/v1.0/cnn_v1.pth
curl -L -o resnet18.pth https://github.com/AlessioCavedal/Ai-Skin-Dermatologist-Classifier/releases/download/v1.0/resnet18.pth
```

In alternativa si rigenerano da zero con `python Code/train.py --arch cnn_v1`
(oppure `--arch resnet18`): il seed è fisso e le run sono deterministiche, quindi
il risultato è identico ai file pubblicati.

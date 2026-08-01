#!/bin/bash
# Doppio clic per avviare l'interfaccia web e aprirla nel browser.
# Per fermare il server: torna in questa finestra e premi Ctrl+C.

cd "$(dirname "$0")/Code" || exit 1
PORTA=7860

VENV="../.venv/bin/python"

if [ ! -x "$VENV" ]; then
  echo "Ambiente Python non trovato. Crealo con:"
  echo "  python3 -m venv .venv"
  echo "  .venv/bin/pip install -r Code/requirements.txt"
  read -r -p "Premi Invio per chiudere..."
  exit 1
fi

if ! ls ../Model/*.pth >/dev/null 2>&1; then
  echo "Nessun modello in Model/. Addestrane uno con:  python train.py"
  read -r -p "Premi Invio per chiudere..."
  exit 1
fi

while lsof -i ":$PORTA" >/dev/null 2>&1; do
  echo "Porta $PORTA occupata, provo la successiva..."
  PORTA=$((PORTA + 1))
done

( sleep 3; open "http://127.0.0.1:$PORTA" ) &
"$VENV" webapp.py --port "$PORTA"

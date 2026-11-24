#!/bin/bash

cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
    echo "Criando ambiente virtual..."
    python3 -m venv .venv
fi

if [ ! -f ".venv/.installed" ]; then
    echo "Instalando dependências..."
    .venv/bin/pip install -r requirements.txt
    touch .venv/.installed
fi

echo "Executando projeto..."
.venv/bin/python fixed_scripts/main.py

#!/bin/bash

cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

if [ ! -f ".venv/.installed" ]; then
    .venv/bin/pip install -r requirements.txt
    touch .venv/.installed
fi

.venv/bin/python scripts/main.py

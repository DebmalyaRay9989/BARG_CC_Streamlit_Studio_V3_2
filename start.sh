#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
python -m streamlit run streamlit_app.py --runner.magicEnabled=false --server.address "${HOST:-127.0.0.1}" --server.port "${PORT:-8501}"

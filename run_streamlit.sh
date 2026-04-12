#!/usr/bin/env bash
set -e

# Change to the script directory (repo root)
cd "$(cd "$(dirname "$0")" && pwd)"

# Ensure Python can import the local `app` package
export PYTHONPATH="$(pwd)"

echo "PWD: $(pwd)"
echo "PYTHONPATH: $PYTHONPATH"

# Use the project's virtualenv python to run streamlit
/Users/christine/Documents/PythonProjects/.venv/bin/python -m streamlit run app/streamlit_app.py

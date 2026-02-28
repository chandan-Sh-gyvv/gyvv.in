#!/bin/bash
set -e

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Installing SHSRS engine..."
# SHSRS engine is in the parent directory when developing locally.
# On Render, it must be in the repo. Check if it exists as a subdir.
if [ -d "shsrs" ]; then
    pip install -e .
    echo "SHSRS installed from local shsrs/ directory"
else
    echo "WARNING: shsrs/ directory not found."
    echo "Copy your shsrs/ engine folder into shsrs_rag_api/ before deploying."
fi

echo "Build complete."

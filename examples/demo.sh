#!/usr/bin/env bash
# Runs the four scenarios in RESULTS.md. About 90 seconds on 2 CPU cores.
set -e
cd "$(dirname "$0")/.."

echo "### 1. census a 100-node mixed consumer fleet"
python -m mesh.cli census --fleet 100 --out examples/census_demo100.json

echo "### 2. what can it train?"
python -m mesh.cli plan --file examples/census_demo100.json \
    --model tiny-0.5B --params 0.5 --tokens 20 --mode full
python -m mesh.cli plan --file examples/census_demo100.json \
    --model qwen-8B-lora --params 8 --tokens 2 --mode lora

echo "### 3. train across 4 processes"
python -m mesh.cli run --workers 4 --rounds 26 --H 10 --lr 3e-3

echo "### 4. one worker turns malicious"
python -m mesh.cli run --workers 4 --rounds 12 --H 10 --lr 3e-3 \
    --byzantine 1 --byzantine-kind noise

echo "### 5. one worker dies mid-run"
python -m mesh.cli run --workers 4 --rounds 20 --H 10 --lr 3e-3 --churn-after 6

echo "### 6. fleet dashboard -- serve, seed, then open http://localhost:8080/"
echo "    (run these two in separate terminals)"
echo "      mesh serve --port 8080"
echo "      mesh seed --nodes 240"

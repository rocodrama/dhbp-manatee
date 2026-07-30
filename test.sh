#!/usr/bin/env bash
set -euo pipefail

python -m py_compile \
  main/config.py \
  main/fixed/data_adapter.py \
  main/fixed/store.py \
  main/agent/prompts.py \
  main/agent/llm.py \
  main/agent/perturbation.py \
  main/agent/runner.py \
  main/run.py \
  main/app/app.py \
  main/gui.py

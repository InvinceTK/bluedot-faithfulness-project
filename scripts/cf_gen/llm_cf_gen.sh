#!/usr/bin/env bash

set -euo pipefail

GPU=0
MODEL=Qwen/Qwen3-1.7B
DATASET=zebra_logic
IN=data/natural_counterfactuals/compas_counterfactual_dataset_balanced_10.parquet
OUT=parquet/tests/compas_cf_80.parquet

CUDA_VISIBLE_DEVICES=$GPU python -m src.counterfactual_generation.llm_counterfactual_generation.generate_counterfactuals \
    "$IN" \
    --output-parquet "$OUT" \
    --model "$MODEL" \
    --dataset-name "$DATASET"

# how to run bash scripts/cf_gen/llm_cf_gen.sh


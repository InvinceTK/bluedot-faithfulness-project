#!/usr/bin/env bash

set -euo pipefail

GPU=1
MODEL=openai/gpt-oss-20b
IN=parquet/llm_gen_experiment/1.7B_reg_heart_disease_1200_reference_answers.parquet
OUT=parquet/llm_gen_experiment/1.7B_reg_heart_disease_1200_predictor_answers.parquet

CUDA_VISIBLE_DEVICES=$GPU python -m src.prediction_generation.generate_predictor_answers \
    "$IN" \
    --output-parquet "$OUT" \
    --model "$MODEL" \

# how to run bash scripts/cf_gen/llm_cf_gen.sh
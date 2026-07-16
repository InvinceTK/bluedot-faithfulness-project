#!/usr/bin/env bash
set -euo pipefail

GPU=0
MODEL=Qwen/Qwen3-1.7B
DATASET=zebra_logic
IN=parquet/zebra/experiment3/1.7b_llm_zebra_logic_5.parquet
OUT=parquet/zebra/experiment3/1.7b_llm_zebra_logic_5.parquet

CUDA_VISIBLE_DEVICES=$GPU python -m src.counterfactual_generation.reference_answer_generation.generate_reference_answers \
    "$IN" \
    --output-parquet "$OUT" \
    --model "$MODEL" \
    --dataset-name "$DATASET"


    CUDA_VISIBLE_DEVICES=$GPU python -m src.counterfactual_generation.reference_answer_generation.generate_reference_answers \
    "parquet/llm_gen_experiment2/llm_gen/1.7B_combined_counterfactuals_6000.parquet" \
    --output-parquet "$OUT" \
    --model "$MODEL" \
    --dataset-name "$DATASET"
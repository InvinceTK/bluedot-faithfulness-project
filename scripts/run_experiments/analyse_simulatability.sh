#!/usr/bin/env bash
set -euo pipefail

LLM_GEN_USED="$1"  

if [ "$LLM_GEN_USED" = "true" ]; then
  GEN_METHOD="llm"
else
  GEN_METHOD="reg"
fi

IN=parquet/llm_gen_experiment/data/${GEN_METHOD}_gen
OUT=parquet/llm_gen_experiment/simulatability_analysis/1.7B_${GEN_METHOD}_heart_disease_1200_simulatability_analysis

python -m analysis_scripts.analyze_simulatability \
    "${IN}" \
    --output "${OUT}"\
	--normalized \
	--multi-predictor

# command = bash scripts/run_experiments/analyse_simulatability.sh "false"
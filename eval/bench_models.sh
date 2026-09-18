#!/bin/zsh
# Run the German test set against several models, one process per model.
#   ./eval/bench_models.sh mlx-community/Qwen3.5-9B-4bit mlx-community/Qwen3.6-35B-A3B-4bit
# Each argument is a local path or a Hugging Face id; the tag is its last path component.
cd "${0:A:h}/.."
for model in "$@"; do
  tag=${${model:t}:l}
  echo "=== $tag  $(date +%H:%M:%S)"
  python eval/run_eval.py --local "$model" --tag "$tag"
done

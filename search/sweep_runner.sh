#!/bin/bash
# Shared WandB-sweep launcher. Source this from a run_<algo>_sweeps.bash after
# defining:
#
#   SCRIPT          python launcher path, relative to repo root
#   PROJECT_PREFIX  WandB project prefix; per-task project is PREFIX-<TASK>
#   ALGO            short tag used in log filenames
#   AGENTS_PER_GPU  agents launched per task
#   TASKS=( ... )   tasks to sweep (one sweep created per task)
#   GPUS=( ... )    optional, parallel to TASKS; empty => no CUDA pinning
#   EXTRA_ARGS      optional, extra args forwarded to every invocation
#
# One sweep is created per task and AGENTS_PER_GPU agents are launched for it in
# the background. Run from anywhere: this cd's to the repo root, where the
# config/ paths used by get_args() are resolved.

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

EXTRA_ARGS="${EXTRA_ARGS:-}"
ALGO_UP="$(echo "$ALGO" | tr '[:lower:]' '[:upper:]')"

echo "=========================================================="
echo "   Launching ${ALGO_UP} Hyperparameter Sweeps in Background"
echo "=========================================================="

for idx in "${!TASKS[@]}"; do
    task="${TASKS[$idx]}"
    gpu="${GPUS[$idx]:-}"
    project="${PROJECT_PREFIX}-$(echo "$task" | tr '[:lower:]' '[:upper:]')"

    echo "Initializing WandB Sweep for '$task'..."
    out=$(python3 "$SCRIPT" --count 0 --project "$project" $EXTRA_ARGS 2>&1)
    id=$(echo "$out" | grep "Created NEW wandb sweep with ID:" | awk '{print $NF}')
    if [ -z "$id" ]; then
        echo "❌ Failed to create sweep for '$task'. Output:"
        echo "$out"
        exit 1
    fi
    echo "✅ '$task' sweep created: $id"

    label="gpu${gpu:-X}"
    echo "🚀 Launching $AGENTS_PER_GPU agents for '$task' ${gpu:+(GPU $gpu)}..."
    for ((i=1; i<=AGENTS_PER_GPU; i++)); do
        log="log_sweep_${ALGO}_${task}_${label}_$i.txt"
        if [ -n "$gpu" ]; then
            CUDA_VISIBLE_DEVICES="$gpu" python3 "$SCRIPT" --sweep_id "$id" \
                --project "$project" --task "$task" $EXTRA_ARGS > "$log" 2>&1 &
        else
            python3 "$SCRIPT" --sweep_id "$id" \
                --project "$project" --task "$task" $EXTRA_ARGS > "$log" 2>&1 &
        fi
    done
    echo ""
done

echo "=========================================================="
echo "🎉 All agents deployed to the background!"
echo "   Logs: log_sweep_${ALGO}_<task>_gpu<gpu>_<i>.txt"
echo "=========================================================="

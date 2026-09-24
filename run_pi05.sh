#!/usr/bin/env bash
set -Eeuo pipefail

# Defaults for the current Pi0.5 deployment. Override with environment variables
# when using the launcher on another server.
PROJECT_DIR="${PI05_PROJECT_DIR:-/root/pi05_b601/src/rebot_lerobot}"
CONDA_SH="${PI05_CONDA_SH:-/root/miniforge3/etc/profile.d/conda.sh}"
CONDA_ENV="${PI05_CONDA_ENV:-lerobot}"
TMUX_SESSION="${PI05_TMUX_SESSION:-pi05_run}"
LOG_DIR="${PI05_LOG_DIR:-$PROJECT_DIR/logs}"
MODEL_DIR="${PI05_MODEL_DIR:-/root/pi05_b601/models/pi05_base}"
DATASET_DIR="${PI05_DATASET_DIR:-/root/pi05_b601/datasets/orange_rack_test_tubes_848}"
ALLOW_PARALLEL="${PI05_ALLOW_PARALLEL:-1}"

usage() {
    printf '%s\n' \
        "Usage: $0 [PYTHON_SCRIPT] [SCRIPT_ARGS...]" \
        "Example: $0 test_pi05.py" \
        "Optional environment variables: PI05_PROJECT_DIR, PI05_CONDA_ENV, PI05_TMUX_SESSION, PI05_LOG_DIR, PI05_MODEL_DIR, PI05_DATASET_DIR"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

inside=0
if [[ "${1:-}" == "--inside" ]]; then
    inside=1
    shift
fi

PYTHON_SCRIPT="${1:-test_pi05.py}"
if [[ $# -gt 0 ]]; then
    shift
fi

# If started outside tmux, create a dedicated detached session. The command
# remains alive after SSH disconnects and can be inspected later.
if [[ -z "${TMUX:-}" && "$inside" -eq 0 ]]; then
    if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
        printf 'tmux session already exists: %s\n' "$TMUX_SESSION" >&2
        printf 'Inspect it with: tmux attach -t %s\n' "$TMUX_SESSION" >&2
        exit 2
    fi

    script_path="$(readlink -f "$0")"
    command_parts=("$script_path" "--inside" "$PYTHON_SCRIPT" "$@")
    printf -v inner_command '%q ' "${command_parts[@]}"
    tmux new-session -d -s "$TMUX_SESSION" "cd $(printf '%q' "$PROJECT_DIR") && $inner_command"
    printf 'Started tmux session: %s\n' "$TMUX_SESSION"
    printf 'Enter it with: tmux attach -t %s\n' "$TMUX_SESSION"
    exit 0
fi

if [[ ! -d "$PROJECT_DIR" ]]; then
    printf 'Project directory does not exist: %s\n' "$PROJECT_DIR" >&2
    exit 1
fi
if [[ ! -f "$CONDA_SH" ]]; then
    printf 'Conda shell integration not found: %s\n' "$CONDA_SH" >&2
    exit 1
fi

if [[ "$PYTHON_SCRIPT" = /* ]]; then
    SCRIPT_PATH="$PYTHON_SCRIPT"
else
    SCRIPT_PATH="$PROJECT_DIR/$PYTHON_SCRIPT"
fi
if [[ ! -f "$SCRIPT_PATH" ]]; then
    printf 'Python script does not exist: %s\n' "$SCRIPT_PATH" >&2
    exit 1
fi

mkdir -p "$LOG_DIR"
timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="$LOG_DIR/${timestamp}_${PYTHON_SCRIPT##*/}.log"

# Persist every line from this point onward. Use tail -f on this file to watch
# the run without depending on the terminal connection.
exec >>"$log_file" 2>&1

printf '=== Pi0.5 launcher started: %s ===\n' "$(date -Is)"
printf 'project: %s\nmodel: %s\ndataset: %s\nscript: %s\n' \
    "$PROJECT_DIR" "$MODEL_DIR" "$DATASET_DIR" "$SCRIPT_PATH"

source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$PROJECT_DIR"

printf '\n=== Read-only resource check ===\n'
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
else
    printf 'nvidia-smi is not available\n'
fi
printf '\n'
free -h
printf '\n'
protected_processes="$(pgrep -af 'vllm|omni' || true)"
if [[ -n "$protected_processes" ]]; then
    printf '\nProtected vLLM/Omni process detected. Parallel mode is enabled.\n'
    printf '%s\n' "$protected_processes"
    printf 'This launcher will not modify, stop, restart, or kill that process.\n'
    printf 'The Pi0.5 job will continue only as a separate process; monitor memory and the log.\n'
    if [[ "$ALLOW_PARALLEL" != "1" ]]; then
        printf 'PI05_ALLOW_PARALLEL is not 1, so the job was intentionally not started.\n'
        exit 3
    fi
fi

printf '\n=== Environment ===\n'
printf 'python: '
which python
python -V

printf '\n=== Starting Python job ===\n'
printf 'command: python -u %q' "$SCRIPT_PATH"
printf ' %q' "$@"
printf '\nlog: %s\n' "$log_file"
if python -u "$SCRIPT_PATH" "$@"; then
    status=0
else
    status=$?
fi

printf '\n=== Python job finished with exit code %s: %s ===\n' "$status" "$(date -Is)"
exit "$status"

---
name: remote-ml-server-ops
description: "Use when operating a remote machine-learning server over SSH for LeRobot, Pi0/Pi0.5, GPU inference or training, dataset work, model downloads, tmux sessions, or shared vLLM services."
---

# Remote ML Server Operations

## Purpose

Use this skill for remote ML work where SSH disconnects, large model downloads, shared GPU resources, or long-running Python programs can cause loss of progress or affect another user's service.

## Known operating context

The primary target is a shared multi-user NVIDIA DGX Spark environment:

- architecture: `aarch64`;
- robot: reBot B601;
- Conda environment: `lerobot`;
- Pi0/Pi0.5 model path: `/root/pi05_b601/models/pi05_base`;
- project path: `/root/pi05_b601/src/rebot_lerobot`.

Treat these paths as defaults from the user's current deployment. Verify them when the active server or project may differ; never invent credentials, ports, or paths.

## Hard safety rules

These rules are mandatory for every operation on the shared server:

1. Never generate or recommend commands intended to kill, stop, restart, or otherwise disrupt another user's GPU process. This includes `kill`, `pkill`, `killall`, `systemctl stop`, container-stop commands, and equivalent code when the target is not proven to belong to the user.
2. Never assume that low instantaneous GPU utilization means a service is unused. A resident process and its allocated memory remain protected.
3. Use `nvidia-smi` only as a read-only inspection command for GPU memory, utilization, and process ownership. Do not use it to modify, reset, or terminate processes.
4. Every remote download, model-loading, rollout, inference, training, and data-processing job must write stdout and stderr to a log file with `> RUN.log 2>&1`; long-running jobs must also run inside `tmux`. One-shot read-only inspections such as `nvidia-smi`, `free -h`, and `tmux ls` may print directly for immediate status checks.
5. Before starting GPU work, inspect the current processes and available resources. If the requested work could affect a protected service, stop at inspection and request an explicit maintenance window or a confirmed resource allocation.

## Non-negotiable response format

For every executable command, explain immediately before or after it:

- what the command means;
- what it changes or reads;
- whether the user should remember it;
- the expected output or success condition;
- what to do if the result differs.

Never provide a bare command block for a deployment, training, inference, download, tmux, SSH, Conda, GPU, or dataset workflow.

## Shared-server safety

- Treat existing services such as `vLLM-Omni` as protected. Never stop, kill, restart, reconfigure, or upgrade them without explicit authorization.
- Do not provide a kill/stop workaround for a busy GPU. The correct response is to identify ownership, report the resource state, and wait for explicit authorization from the service owner or an agreed maintenance window.
- Before GPU model loading, training, or inference, inspect both the process list and live GPU/system memory. CPU utilization alone is insufficient.
- On unified-memory systems such as NVIDIA GB10/DGX Spark, do not claim that `CUDA_VISIBLE_DEVICES` isolates memory. A second process may still compete for the same physical memory.
- Distinguish a resident but idle service (`GPU-Util: 0%`) from a stopped service. A process listed in `nvidia-smi` is still running.
- If the user requires zero impact on another service, limit work to disk reads, dataset metadata, CPU-only inspection, or ask for an explicit maintenance window before GPU execution.

## SSH and tmux baseline

Use SSH keepalives for unstable client networks. Replace placeholders with the user's actual host and port; never invent credentials:

```bash
ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -p PORT USER@HOST
```

For downloads, training, and inference, run inside a named tmux session:

```bash
tmux ls
tmux attach -t SESSION_NAME
tmux new -s SESSION_NAME
```

Explain that `tmux ls` lists sessions, `attach` resumes an existing session, and `new` creates one. If a session exists, attach instead of creating a duplicate. To detach while preserving the process, use `Ctrl+B`, then `D`. Do not tell the user to type `:q` in Bash; that is a Vim command.

## Environment and resource checks

After reconnecting, restore the environment only when needed:

```bash
source /root/miniforge3/etc/profile.d/conda.sh
conda activate lerobot
```

Explain that the first command loads Conda shell integration and the second selects the project environment. Verify the active interpreter when ambiguity matters:

```bash
which python
```

Before GPU work, use:

```bash
nvidia-smi
free -h
pgrep -af 'vllm|omni'
```

Explain that `nvidia-smi` is read-only here and shows GPU processes, memory, and utilization; `free -h` shows system memory with `available` as the key field; and `pgrep` finds likely vLLM/Omni processes. Use `watch -n 2 'free -h; echo; nvidia-smi'` only when live monitoring is useful, and explain that `Ctrl+C` stops only the monitor. Never append a reset, terminate, or process-modifying option to `nvidia-smi` in this shared-server workflow.

## Downloads

- Prefer resumable download tools and preserve the existing cache and destination directory after a network interruption.
- Do not delete partial model files or lock files unless there is evidence no downloader is running and the lock is stale.
- Before retrying, check the destination size and active process. Re-run the same command and directory when the tool supports resume.
- Keep model downloads in the project-owned path, for example `/root/pi05_b601/models/`, and keep dataset caches separate.
- Put long downloads in tmux before starting them.

## Python execution and durable logs

Never recommend bare `python script.py` for a remote long-running task. Use unbuffered output and redirect both streams:

```bash
python -u SCRIPT.py > RUN.log 2>&1
```

Explain:

- `-u` makes output appear promptly;
- `>` writes standard output to the log;
- `2>&1` sends error output to the same log.

Use these viewing commands:

```bash
cat RUN.log
tail -f RUN.log
```

Explain that `cat` prints the complete saved log and `tail -f` follows new lines live; `Ctrl+C` stops only the viewer, not the logged process. When the process must survive SSH loss, combine the command with tmux or launch it in tmux before detaching.

## LeRobot/Pi0.5 boundary

- Verify the local model directory and dataset metadata before loading weights.
- Treat a downloaded `model.safetensors` as file completeness, not proof that the policy loaded or that inference works.
- Separate dataset inspection, model-load testing, single-sample inference, data collection, training, and hardware control into distinct steps.
- Do not start robot motion or a follower policy until cameras, robot port/CAN adapter, action dimensions, and the policy's expected observation keys are verified.
- Use `batch_size=1`, conservative workers, and explicit logging for the first inference test. Do not start training while a protected shared GPU service is active unless the user explicitly accepts the resource impact.

## Completion evidence

Do not call a task complete based only on a command being accepted or a file existing. Require evidence appropriate to the task:

- download: completion message plus expected weight/config files;
- environment: import/version/CUDA check;
- inference: model-load success and a real input producing an output;
- dataset: metadata and sample/shape inspection;
- robot: controlled connection/read-only check before motion.

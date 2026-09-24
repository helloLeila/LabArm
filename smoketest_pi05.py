import json
import time
import traceback
from pathlib import Path

import torch

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.policies.pi05 import PI05Policy
from lerobot.policies.pi05.configuration_pi05 import PI05Config

MODEL_DIR = Path('/root/pi05_b601/models/pi05_base')
DATASET_DIR = Path('/root/pi05_b601/datasets/orange_rack_test_tubes_848')
DATASET_ID = 'nyancos/orange_rack_test_tubes_848'


def main() -> None:
    started = time.time()
    print('=== Pi0.5 smoke test started ===', flush=True)
    print('torch:', torch.__version__, flush=True)
    print('torch CUDA:', torch.version.cuda, flush=True)
    print('CUDA available:', torch.cuda.is_available(), flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available')
    print('device:', torch.cuda.get_device_name(0), flush=True)
    print('model dir:', MODEL_DIR, flush=True)
    print('dataset dir:', DATASET_DIR, flush=True)
    if not (MODEL_DIR / 'config.json').is_file():
        raise FileNotFoundError(MODEL_DIR / 'config.json')
    if not (MODEL_DIR / 'model.safetensors').is_file():
        raise FileNotFoundError(MODEL_DIR / 'model.safetensors')

    print('=== Loading local policy config ===', flush=True)
    config = PI05Config.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        device='cuda',
        dtype='bfloat16',
    )
    config.device = 'cuda'
    config.dtype = 'bfloat16'
    print('config device:', config.device, flush=True)
    print('config dtype:', config.dtype, flush=True)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA device does not support bfloat16')
    print('input features:', sorted(config.input_features), flush=True)
    print('output features:', sorted(config.output_features), flush=True)

    print('=== Loading Pi0.5 weights ===', flush=True)
    policy = PI05Policy.from_pretrained(
        MODEL_DIR,
        config=config,
        local_files_only=True,
        strict=True,
    )
    policy.eval()
    parameter_count = sum(p.numel() for p in policy.parameters())
    print('policy load: OK', flush=True)
    print('parameters:', parameter_count, flush=True)
    dtypes = {}
    for parameter in policy.parameters():
        dtypes[str(parameter.dtype)] = dtypes.get(str(parameter.dtype), 0) + parameter.numel()
    print('parameter dtype counts:', dtypes, flush=True)
    if dtypes.get('torch.bfloat16', 0) == 0:
        raise RuntimeError('No bfloat16 parameters were instantiated')
    print('parameter device:', next(policy.parameters()).device, flush=True)
    print('CUDA allocated GiB:', round(torch.cuda.memory_allocated() / 1024**3, 2), flush=True)
    print('CUDA reserved GiB:', round(torch.cuda.memory_reserved() / 1024**3, 2), flush=True)

    print('=== Loading local dataset metadata ===', flush=True)
    metadata = LeRobotDatasetMetadata(DATASET_ID, root=DATASET_DIR)
    print('dataset metadata: OK', flush=True)
    print('dataset revision:', metadata.revision, flush=True)
    print('total episodes:', metadata.total_episodes, flush=True)
    print('total frames:', metadata.total_frames, flush=True)
    print('fps:', metadata.fps, flush=True)
    print('video keys:', sorted(metadata.video_keys), flush=True)
    print('feature keys:', sorted(metadata.features), flush=True)

    print('=== RESULT: PI05_SMOKETEST_SUCCESS ===', flush=True)
    print('elapsed seconds:', round(time.time() - started, 2), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        print('=== RESULT: PI05_SMOKETEST_FAILED ===', flush=True)
        traceback.print_exc()
        raise

import time
import traceback
from pathlib import Path

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pi05 import PI05Policy
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

MODEL_DIR = Path('/root/pi05_b601/models/pi05_base')
DATASET_DIR = Path('/root/pi05_b601/datasets/orange_rack_test_tubes_848')
DATASET_ID = 'nyancos/orange_rack_test_tubes_848'


def main() -> None:
    started = time.time()
    print('=== Pi0.5 image-to-action compute smoke test ===', flush=True)
    print('NOTE: Using real dataset images and a placeholder BOS token because the gated tokenizer is unavailable.', flush=True)
    print('This validates model computation, not semantic language-conditioned inference.', flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 unsupported by this CUDA device')

    config = PI05Config.from_pretrained(MODEL_DIR, local_files_only=True)
    config.device = 'cuda'
    config.dtype = 'bfloat16'
    policy = PI05Policy.from_pretrained(
        MODEL_DIR,
        config=config,
        local_files_only=True,
        strict=True,
    ).eval()

    dataset = LeRobotDataset(
        DATASET_ID,
        root=DATASET_DIR,
        episodes=[0],
        download_videos=False,
        video_backend='pyav',
    )
    frame = dataset[0]
    print('task from sample:', frame.get('task'), flush=True)
    for key in ('observation.images.front', 'observation.images.wrist'):
        image = frame[key]
        print(f'{key}: shape={tuple(image.shape)} dtype={image.dtype} range=({image.min().item():.4f},{image.max().item():.4f})', flush=True)

    device = torch.device('cuda')
    batch = {
        'observation.images.base_0_rgb': frame['observation.images.front'].unsqueeze(0).to(device),
        'observation.images.left_wrist_0_rgb': frame['observation.images.wrist'].unsqueeze(0).to(device),
        OBS_LANGUAGE_TOKENS: torch.tensor([[2]], dtype=torch.long, device=device),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones((1, 1), dtype=torch.bool, device=device),
    }
    print('unprovided third camera will use the policy configured empty-camera padding.', flush=True)
    print('=== Running policy.predict_action_chunk ===', flush=True)
    with torch.inference_mode():
        actions = policy.predict_action_chunk(batch)
    expected_action_dim = config.output_features['action'].shape[0]
    if actions.ndim != 3 or actions.shape[0] != 1 or actions.shape[2] != expected_action_dim:
        raise RuntimeError(f'Unexpected action chunk shape: {tuple(actions.shape)}; expected 1x{config.chunk_size}x{expected_action_dim}')
    if not torch.isfinite(actions).all():
        raise RuntimeError('Predicted actions contain non-finite values')
    print('action chunk shape:', tuple(actions.shape), flush=True)
    print('action dtype:', actions.dtype, flush=True)
    print('first action:', actions[0, 0].float().cpu().tolist(), flush=True)
    print('action values finite: True', flush=True)
    print('cuda allocated GiB:', round(torch.cuda.memory_allocated() / 1024**3, 2), flush=True)
    print('=== RESULT: PI05_COMPUTE_SMOKETEST_SUCCESS ===', flush=True)
    print('elapsed seconds:', round(time.time() - started, 2), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        print('=== RESULT: PI05_COMPUTE_SMOKETEST_FAILED ===', flush=True)
        traceback.print_exc()
        raise

import traceback
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoTokenizer

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pi05 import PI05Policy
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK

BASE_DIR = '/root/pi05_b601/models/pi05_base'
ADAPTER_DIR = '/root/pi05_b601/outputs/pi05_tubes_lora_first100_fast_1000/checkpoints/last/pretrained_model'
DATASET_DIR = '/root/pi05_b601/datasets/orange_rack_test_tubes_848'
DATASET_ID = 'nyancos/orange_rack_test_tubes_848'


def main():
    print('=== 1. 加载基础 pi0.5 模型 ===', flush=True)
    config = PI05Config.from_pretrained(BASE_DIR, local_files_only=True)
    config.device = 'cuda'
    config.dtype = 'bfloat16'
    base = PI05Policy.from_pretrained(BASE_DIR, config=config, local_files_only=True, strict=True)
    print('基础模型加载成功', flush=True)

    print('=== 2. 套上 LoRA 微调适配器 ===', flush=True)
    policy = PeftModel.from_pretrained(base, ADAPTER_DIR)
    policy.eval()
    print('LoRA 适配器加载成功', flush=True)

    print('=== 3. 加载打包好的 tokenizer ===', flush=True)
    tok = AutoTokenizer.from_pretrained(f'{ADAPTER_DIR}/tokenizer')
    print('tokenizer 加载成功', flush=True)

    print('=== 4. 取数据集样本 + 真实语言指令 ===', flush=True)
    dataset = LeRobotDataset(DATASET_ID, root=DATASET_DIR, episodes=[0],
                            download_videos=False, video_backend='pyav')
    frame = dataset[0]
    task = frame['task']
    if isinstance(task, (list, tuple)):
        task = task[0]
    print('任务指令:', task, flush=True)

    print('=== 5. 用真实指令分词（不再用占位 [[2]]）===', flush=True)
    prompt = f"Task: {task.strip().replace('_', ' ').replace(chr(10), ' ')};\nAction: "
    print('prompt:', repr(prompt), flush=True)
    enc = tok(prompt, return_tensors='pt', padding='max_length', max_length=200, truncation=True)
    tokens = enc['input_ids'].to('cuda')
    mask = enc['attention_mask'].bool().to('cuda')
    print('tokens shape:', tuple(tokens.shape), flush=True)

    print('=== 6. 推理 ===', flush=True)
    batch = {
        'observation.images.base_0_rgb': frame['observation.images.front'].unsqueeze(0).to('cuda'),
        'observation.images.left_wrist_0_rgb': frame['observation.images.wrist'].unsqueeze(0).to('cuda'),
        OBS_LANGUAGE_TOKENS: tokens,
        OBS_LANGUAGE_ATTENTION_MASK: mask,
    }
    with torch.inference_mode():
        actions = policy.base_model.predict_action_chunk(batch)
    print('action chunk shape:', tuple(actions.shape), flush=True)
    print('第一个动作 (7维):', [round(x, 4) for x in actions[0, 0].float().cpu().tolist()], flush=True)
    print('=== 成功 ===', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
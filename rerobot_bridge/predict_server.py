#!/usr/bin/env python3
"""π0.5 真机推理服务端：加载微调模型，提供 /predict 接口1。

用法（在 GPU 服务器上）：
    conda activate lerobot
    uvicorn predict_server:app --host 127.0.0.1 --port 8080
"""
import base64
import io
import json

import numpy as np
import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoTokenizer

from lerobot.policies.pi05 import PI05Policy
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK

# ===================== 配置（按需改） =====================
BASE_DIR = '/root/pi05_b601/models/pi05_base'
ADAPTER_DIR = '/root/pi05_b601/outputs/pi05_tubes_lora_100_200_fast_4000/checkpoints/005000/pretrained_model'
STATS_PATH = '/root/pi05_b601/datasets/orange_rack_test_tubes_848/meta/stats.json'
TASK = 'place two test tubes into the orange rack'

# 方向翻转：你的机器 joint_directions 与训练数据正好相反（7 个关节全反），
# 所以喂模型前把状态取负。若干跑验证后发现方向反了，把这里改成 1.0 即可关掉。
STATE_FLIP = -1.0

# ===================== 加载模型 =====================
print('[1/3] 加载基础 pi0.5 模型 ...', flush=True)
_config = PI05Config.from_pretrained(BASE_DIR, local_files_only=True)
_config.device = 'cuda'
_config.dtype = 'bfloat16'
# 诊断：打印模型真正期望的输入/输出特征名（冒烟测试若报「image features missing」就看这里）
print(f'[config] input_features  = {list(_config.input_features.keys())}', flush=True)
print(f'[config] image_features  = {list(_config.image_features.keys())}', flush=True)
print(f'[config] output_features = {list(_config.output_features.keys())}', flush=True)
_base = PI05Policy.from_pretrained(BASE_DIR, config=_config, local_files_only=True, strict=True)

print('[2/3] 套 LoRA 适配器 ...', flush=True)
policy = PeftModel.from_pretrained(_base, ADAPTER_DIR)
policy.eval()

print('[3/3] 加载 tokenizer + stats ...', flush=True)
tok = AutoTokenizer.from_pretrained(f'{ADAPTER_DIR}/tokenizer')

with open(STATS_PATH) as f:
    _stats = json.load(f)
_state_q01 = np.array(_stats['observation.state']['q01'], dtype=np.float32)
_state_q99 = np.array(_stats['observation.state']['q99'], dtype=np.float32)
_action_q01 = np.array(_stats['action']['q01'], dtype=np.float32)
_action_q99 = np.array(_stats['action']['q99'], dtype=np.float32)

print('模型就绪。', flush=True)


# ===================== 状态 -> prompt =====================
def state_to_prompt(state_deg: np.ndarray):
    """把 7 维关节角（度，你的机器原始值）转成带离散状态的语言 token。"""
    # 1. 方向翻转（训练约定 = -你的机器）
    s = STATE_FLIP * state_deg.astype(np.float32)
    # 2. 分位数归一化到 [-1,1]
    s_norm = 2.0 * (s - _state_q01) / (_state_q99 - _state_q01) - 1.0
    # 3. 离散化 256 桶
    s_disc = np.digitize(s_norm, np.linspace(-1, 1, 257)[:-1]) - 1
    s_disc = np.clip(s_disc, 0, 255).astype(int)
    # 4. 拼 prompt（和 processor_pi05.py 完全一致）
    state_str = " ".join(map(str, s_disc.tolist()))
    cleaned = TASK.strip().replace('_', ' ').replace('\n', ' ')
    prompt = f"Task: {cleaned}, State: {state_str};\nAction: "
    enc = tok(prompt, return_tensors='pt', padding='max_length', max_length=200, truncation=True)
    return enc['input_ids'].to('cuda'), enc['attention_mask'].bool().to('cuda'), s_disc.tolist(), prompt


# ===================== 图像解码 =====================
def decode_image(b64: str) -> torch.Tensor:
    data = base64.b64decode(b64)
    img = Image.open(io.BytesIO(data)).convert('RGB')
    arr = np.array(img, dtype=np.float32) / 255.0          # [H,W,3] in [0,1]
    arr = arr.transpose(2, 0, 1)                            # [3,H,W]
    return torch.from_numpy(arr).unsqueeze(0).to('cuda')    # [1,3,H,W]


# ===================== 推理 =====================
def predict(state_deg, front_b64, wrist_b64):
    tokens, mask, disc_state, prompt = state_to_prompt(np.array(state_deg, dtype=np.float32))
    front = decode_image(front_b64)
    wrist = decode_image(wrist_b64)
    batch = {
        'observation.images.base_0_rgb': front,
        'observation.images.left_wrist_0_rgb': wrist,
        OBS_LANGUAGE_TOKENS: tokens,
        OBS_LANGUAGE_ATTENTION_MASK: mask,
    }
    with torch.inference_mode():
        actions = policy.base_model.predict_action_chunk(batch)   # (1, 50, 7) 归一化
    actions = actions[0, :, :7].float().cpu().numpy()             # (50, 7) 前 7 维才是真实动作，模型输出是 32 维
    # 动作反归一化到度（动作方向不用翻转，动作空间训练和你这边一致）
    actions_deg = (actions + 1.0) / 2.0 * (_action_q99 - _action_q01) + _action_q01
    return actions_deg.tolist(), disc_state, prompt


# ===================== HTTP 服务 =====================
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class PredictRequest(BaseModel):
    state: list[float]   # 7 关节角（度，你的机器原始值）
    front: str           # 俯瞰相机 JPEG base64
    wrist: str           # 腕部相机 JPEG base64


@app.get('/health')
def health():
    return {'ok': True}


@app.post('/predict')
def predict_endpoint(req: PredictRequest):
    action_chunk, disc_state, prompt = predict(req.state, req.front, req.wrist)
    return {'action_chunk': action_chunk, 'state_discretized': disc_state, 'prompt': prompt}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8080)
import traceback
import numpy as np
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
EVAL_EPISODES = list(range(100, 130))
MAX_FRAMES = 300

JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex',
        'wrist_yaw', 'wrist_roll', 'gripper']


def to_np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def qnorm(x, q01, q99):
    return 2.0 * (x - q01) / (q99 - q01 + 1e-8) - 1.0


def main():
    print('=== 1. 加载模型 ===', flush=True)
    config = PI05Config.from_pretrained(BASE_DIR, local_files_only=True)
    config.device = 'cuda'
    config.dtype = 'bfloat16'
    base = PI05Policy.from_pretrained(BASE_DIR, config=config, local_files_only=True, strict=True)
    policy = PeftModel.from_pretrained(base, ADAPTER_DIR)
    policy.eval()
    tok = AutoTokenizer.from_pretrained(f'{ADAPTER_DIR}/tokenizer')

    print('=== 2. 加载验证集 ===', flush=True)
    dataset = LeRobotDataset(DATASET_ID, root=DATASET_DIR, episodes=EVAL_EPISODES,
                            download_videos=False, video_backend='pyav')
    print('验证集总帧数:', len(dataset), flush=True)

    # 直接从 stats.json 读分位数（QuantileNormalizer 用 q01/q99）
    import json as _json
    _stats = _json.load(open(f'{DATASET_DIR}/meta/stats.json'))
    act_q01 = np.asarray(_stats['action']['q01'], dtype=np.float32)
    act_q99 = np.asarray(_stats['action']['q99'], dtype=np.float32)
    st_q01 = np.asarray(_stats['observation.state']['q01'], dtype=np.float32)
    st_q99 = np.asarray(_stats['observation.state']['q99'], dtype=np.float32)
    print('动作 q01/q99 形状:', act_q01.shape, flush=True)

    print('=== 3. 逐帧预测 + 对比 ===', flush=True)
    errs = []
    n = 0
    for i in range(len(dataset)):
        frame = dataset[i]

        def prep_img(t):
            t = t.float()
            if t.max() > 1.5:
                t = t / 255.0
            return t
        front = prep_img(frame['observation.images.front']).unsqueeze(0).to('cuda')
        wrist = prep_img(frame['observation.images.wrist']).unsqueeze(0).to('cuda')

        # 状态归一化 + 离散化 256 bins + 拼进 prompt（和 processor 完全一致）
        state = to_np(frame['observation.state']).reshape(-1)[:7]
        state_norm = qnorm(state, st_q01, st_q99)
        state_disc = np.digitize(state_norm, np.linspace(-1, 1, 257)[:-1]) - 1
        state_str = ' '.join(str(int(s)) for s in state_disc)

        task = frame['task']
        if isinstance(task, (list, tuple)):
            task = task[0]
        clean = task.strip().replace('_', ' ').replace('\n', ' ')
        prompt = f"Task: {clean}, State: {state_str};\nAction: "

        enc = tok(prompt, return_tensors='pt', padding='max_length', max_length=200, truncation=True)
        tokens = enc['input_ids'].to('cuda')
        mask = enc['attention_mask'].bool().to('cuda')

        batch = {
            'observation.images.base_0_rgb': front,
            'observation.images.left_wrist_0_rgb': wrist,
            OBS_LANGUAGE_TOKENS: tokens,
            OBS_LANGUAGE_ATTENTION_MASK: mask,
        }
        with torch.inference_mode():
            actions = policy.base_model.predict_action_chunk(batch)   # (1,50,7) 归一化
        pred = actions[0, 0].float().cpu().numpy()

        gt = to_np(frame['action']).reshape(-1)[:7]
        gt_norm = qnorm(gt, act_q01, act_q99)

        errs.append(np.abs(pred - gt_norm))
        n += 1
        if n >= MAX_FRAMES:
            break

    errs = np.stack(errs)
    print(f'\n评估帧数: {n}', flush=True)
    print('每个关节平均绝对误差 (归一化空间，越小越好):', flush=True)
    for j, name in enumerate(JOINTS):
        rad = errs[:, j].mean() * (act_q99[j] - act_q01[j]) / 2.0
        print(f'  {name:15s}: {errs[:, j].mean():.4f}  (约 {rad:.4f} 弧度 = {np.degrees(rad):.1f}°)', flush=True)
    print(f'总平均归一化误差: {errs.mean():.4f}', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
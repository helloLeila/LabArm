#!/usr/bin/env python3
"""服务器 /predict 冒烟测试：不连机械臂，发一组假状态+图，看模型能不能跑通。

用法（机器人机器，隧道已开、服务器已起）：
    python test_predict.py
    python test_predict.py --state "5,-40,-30,10,0,15,20"   # 自定义 7 关节角
"""
import argparse
import base64
import os

import cv2
import numpy as np
import requests

SERVER = 'http://localhost:8080/predict'


def img_b64(path, size):
    """读图转 base64；图不存在就发一张纯灰图兜底。"""
    if path and os.path.exists(path):
        img = cv2.imread(path)  # BGR
        if img is None:
            img = np.zeros((size[1], size[0], 3), np.uint8)
    else:
        img = np.zeros((size[1], size[0], 3), np.uint8)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ok, buf = cv2.imencode('.jpg', rgb)
    return base64.b64encode(buf.tobytes()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--state', default='0,0,0,0,0,0,0', help='7 个关节角（度），逗号分隔')
    ap.add_argument('--front', default=os.path.expanduser('~/桌面/cam6.png'))
    ap.add_argument('--wrist', default=os.path.expanduser('~/桌面/cam4.png'))
    args = ap.parse_args()

    state = [float(x) for x in args.state.split(',')]
    assert len(state) == 7, 'state 要 7 个数，逗号分隔'

    print('发 state:', state)
    print('front:', args.front, '  wrist:', args.wrist)

    r = requests.post(SERVER, json={
        'state': state,
        'front': img_b64(args.front, (848, 480)),
        'wrist': img_b64(args.wrist, (640, 480)),
    }, timeout=300)

    print('HTTP', r.status_code)
    if r.status_code != 200:
        print('错误内容：\n', r.text[:1500])
        return

    data = r.json()
    if 'state_discretized' in data:
        print('离散化状态(256 桶，应约 0~255):', data['state_discretized'])
    if 'prompt' in data:
        print('prompt:', data['prompt'])

    chunk = np.array(data['action_chunk'])
    print('action_chunk 形状:', chunk.shape)
    print('前 3 步动作(度):')
    for row in chunk[:3]:
        print('  ', [round(x, 1) for x in row])
    print('整段 min:', [round(x, 1) for x in chunk.min(0)])
    print('整段 max:', [round(x, 1) for x in chunk.max(0)])


if __name__ == '__main__':
    main()

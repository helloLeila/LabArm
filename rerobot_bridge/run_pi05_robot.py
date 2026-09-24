#!/usr/bin/env python3
"""π0.5 真机控制：读机器人状态+相机 → 调服务器 /predict → 发动作。

用法（在机器人机器上，先开好 SSH 隧道）：
    ssh -L 8080:localhost:8080 -p 6022 root@43.155.204.215   # 先开隧道
    conda activate lerobot
    python run_pi05_robot.py --dry-run      # 第一步：干跑，只看不动
    python run_pi05_robot.py                # 第二步：真机（按回车才动）
"""
import argparse
import base64
import time

import cv2
import numpy as np
import requests

from lerobot_robot_seeed_b601.seeed_b601_dm_follower import SeeedB601DMFollower
from lerobot_robot_seeed_b601.config_seeed_b601_dm_follower import SeeedB601DMFollowerConfig

# 相机（已确认：video6=俯瞰 front，video4=腕部 wrist）
FRONT_IDX = 6
WRIST_IDX = 4

# 7 关节顺序（必须和驱动里一致）
JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow_flex',
          'wrist_flex', 'wrist_yaw', 'wrist_roll', 'gripper']

SERVER_URL = 'http://localhost:8080/predict'


def encode_jpeg(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    ok, buf = cv2.imencode('.jpg', rgb, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf.tobytes()).decode('utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true', help='只打印预测动作，不发机械臂')
    ap.add_argument('--steps', type=int, default=1, help='跑几步（每步拿一次 50 帧动作并执行）')
    args = ap.parse_args()

    # 相机
    front_cap = cv2.VideoCapture(FRONT_IDX)
    wrist_cap = cv2.VideoCapture(WRIST_IDX)
    front_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 848)
    front_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    wrist_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    wrist_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 机器人
    config = SeeedB601DMFollowerConfig(port='can0', cameras={})
    robot = SeeedB601DMFollower(config)
    robot.connect(calibrate=True)

    if args.dry_run:
        print('【干跑模式】只预测不动。')
    else:
        print('【真机模式】即将控制机械臂！')
        input('按回车开始（Ctrl+C 随时急停）...')

    try:
        for step in range(args.steps):
            obs = robot.get_observation()
            state = np.array([obs[f'{j}.pos'] for j in JOINTS], dtype=np.float32)

            ok1, front = front_cap.read()
            ok2, wrist = wrist_cap.read()
            if not (ok1 and ok2):
                print('相机读取失败，跳过本步'); continue

            resp = requests.post(SERVER_URL, json={
                'state': state.tolist(),
                'front': encode_jpeg(front),
                'wrist': encode_jpeg(wrist),
            }, timeout=60)
            chunk = resp.json()['action_chunk']   # (50, 7) 度

            print(f'\n[step {step}] 当前状态: {[round(x,1) for x in state.tolist()]}')
            print(f'            预测首动作: {[round(x,1) for x in chunk[0]]}')

            if args.dry_run:
                continue

            # 执行整段 50 帧动作（~1.7s，30fps），随时 Ctrl+C 急停
            for row in chunk:
                action = {f'{j}.pos': float(row[i]) for i, j in enumerate(JOINTS)}
                robot.send_action(action)
                time.sleep(0.033)
    except KeyboardInterrupt:
        print('\n【急停】收到 Ctrl+C，已停止。')
    finally:
        robot.disconnect()
        front_cap.release()
        wrist_cap.release()
        print('已断开机器人。')


if __name__ == '__main__':
    main()
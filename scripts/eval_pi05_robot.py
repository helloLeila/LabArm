#!/usr/bin/env python3
"""π0.5 真机 N 次评估：循环跑 episode，每个之间人工复位 + 判定成功，最后出成功率。

用法（先开隧道 + 服务器已起）：
    python eval_pi05_robot.py                            # 默认 50 个 episode
    python eval_pi05_robot.py --episodes 10              # 只跑 10 个
    python eval_pi05_robot.py --steps-per-episode 3      # 每个 episode 跑 3 个 50 帧 chunk(~5s)

流程：每个 episode 前按回车（先手动复位场景+机械臂）→ 自动跑完 → 输入 y/n 判定。
全程随时 Ctrl+C 急停。结果同时追加写到 eval_results.log。
"""
import argparse
import base64
import time

import cv2
import numpy as np
import requests

from lerobot_robot_seeed_b601.seeed_b601_dm_follower import SeeedB601DMFollower
from lerobot_robot_seeed_b601.config_seeed_b601_dm_follower import SeeedB601DMFollowerConfig

FRONT_IDX = 6
WRIST_IDX = 4
JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow_flex',
          'wrist_flex', 'wrist_yaw', 'wrist_roll', 'gripper']
SERVER_URL = 'http://localhost:8080/predict'
LOG = 'eval_results.log'


def encode_jpeg(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    ok, buf = cv2.imencode('.jpg', rgb, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf.tobytes()).decode('utf-8')


def open_cam(idx, w, h):
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    for _ in range(6):
        cap.read()
        time.sleep(0.05)
    return cap


def read_frame(cap, n=5):
    for _ in range(n):
        ok, frame = cap.read()
        if ok and frame is not None:
            return frame
        time.sleep(0.05)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=50, help='评估 episode 数')
    ap.add_argument('--steps-per-episode', type=int, default=1, help='每个 episode 跑几个 50 帧 chunk')
    args = ap.parse_args()

    front_cap = open_cam(FRONT_IDX, 848, 480)
    wrist_cap = open_cam(WRIST_IDX, 640, 480)
    if not (front_cap.isOpened() and wrist_cap.isOpened()):
        print('相机打不开，检查 USB / 编号'); return

    config = SeeedB601DMFollowerConfig(port='can0', cameras={})
    robot = SeeedB601DMFollower(config)
    robot.connect(calibrate=True)

    results = []
    try:
        for ep in range(1, args.episodes + 1):
            resp = input(f'\n[episode {ep}/{args.episodes}] 复位场景+机械臂后按回车开始（q 退出）...')
            if resp.strip().lower() == 'q':
                break

            for step in range(args.steps_per_episode):
                obs = robot.get_observation()
                state = np.array([obs[f'{j}.pos'] for j in JOINTS], dtype=np.float32)
                front = read_frame(front_cap)
                wrist = read_frame(wrist_cap)
                if front is None or wrist is None:
                    print('  相机读取失败，跳过本步'); continue
                r = requests.post(SERVER_URL, json={
                    'state': state.tolist(),
                    'front': encode_jpeg(front),
                    'wrist': encode_jpeg(wrist),
                }, timeout=60)
                if r.status_code != 200:
                    print(f'  服务器返回 {r.status_code}: {r.text[:300]}'); break
                chunk = r.json()['action_chunk']
                for row in chunk:
                    action = {f'{j}.pos': float(row[i]) for i, j in enumerate(JOINTS)}
                    robot.send_action(action)
                    time.sleep(0.033)

            ans = input('  成功? [y/n/q]: ').strip().lower()
            if ans == 'q':
                break
            results.append(ans == 'y')
            succ = sum(results)
            print(f'  当前成功率: {succ}/{len(results)} = {succ / len(results) * 100:.1f}%')
            with open(LOG, 'a') as f:
                f.write(f'episode {ep}: {"success" if ans == "y" else "fail"}\n')
    except KeyboardInterrupt:
        print('\n【急停】已停止。')
    finally:
        robot.disconnect()
        front_cap.release()
        wrist_cap.release()
        print('已断开机器人。')

    if results:
        succ = sum(results)
        print(f'\n=== 评估完成: {succ}/{len(results)} 成功 = {succ / len(results) * 100:.1f}% ===')


if __name__ == '__main__':
    main()

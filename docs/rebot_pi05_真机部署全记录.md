# reBot B601 + π0.5 真机部署全记录

> 从硬件接线 → 遥操作录制 → 权重微调 → 真机推理桥接的完整技术记录。
> 用途：断点续接的参考手册。关键数值、路径、调试结论都在这里。

---

## 0. 两台机器（术语约定）

| 称呼 | 实际 | 用途 |
|---|---|---|
| **本机** | `user@user-virtual-machine`（Ubuntu，带 GUI 桌面） | 机械臂走 CAN、两个 USB 相机接这里；跑控制脚本、开 SSH 隧道 |
| **服务器** | `43.155.204.215:6022`（`root@gx10-a9b2`，NVIDIA DGX Spark） | GPU 跑模型推理（`predict_server.py`） |

数据流向：**本机（状态+相机图）→ 隧道 → 服务器（模型推理）→ 本机（发动作给机械臂）**。

---

## 1. 硬件

- **Follower（主臂）**：Seeed reBot B601-DM，达妙电机 `dm4340p`（6 个关节）+ `dm4310`（夹爪），共 7 轴。
- **Leader（遥操作从臂）**：reBot Arm 102，FashionStar 总线舵机，`/dev/ttyUSB0`。
  - ⚠️ 正确类型是 `rebot_arm_102_leader`（包 `lerobot_teleoperator_rebot_arm_102`），**不是**内置 `rebot_102_leader`（那个 joint_directions 带了 -1 翻转，会导致主从臂动作不一致）。
- **CAN**：PCAN-USB 转 SocketCAN，接口 `can0` @ 1 Mbps。
  - ⚠️ LeRobot 里必须用 `socketcan/can0`，**不是** `damiao/ttyACM0`。
- 电机 CAN ID 映射在 `config_seeed_b601_dm_follower.py` 的 `motor_can_ids`（7 电机 → `(send_can_id, recv_can_id)`）。

---

## 2. 关节约定（单一事实来源）

7 关节顺序（数据集/模型/驱动一致）：
```
shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_yaw, wrist_roll, gripper
```

**本机驱动 joint_directions**（`config_seeed_b601_dm_follower.py`）：
```python
{shoulder_pan: -1.0, shoulder_lift: -1.0, elbow_flex: 1.0,
 wrist_flex: 1.0, wrist_yaw: 1.0, wrist_roll: -1.0, gripper: -6.0}
```

**joint_limits（度）**：
```python
{shoulder_pan: (-145,145), shoulder_lift: (-170,0), elbow_flex: (-200,0),
 wrist_flex: (-80,90), wrist_yaw: (-90,90), wrist_roll: (-130,130), gripper: (-270,0)}
```

**单位**：全程用「度」。`get_observation()` 返回**原始编码器角度（度，未套方向）**；`send_action()` 收「度」、内部乘 joint_directions、clip 到 joint_limits、转弧度发电机。

---

## 3. 环境（本机 + 服务器都是）

- conda 环境名 `lerobot`：python 3.12、lerobot 0.6.2、torch 2.11.0 CUDA 13。
- 可编辑安装源码：`~/rebot_lerobot/lerobot/lerobot-robot-seeed-b601/`。
- **重装后必须重新打的补丁**：`seeed_b601_follower.py` 的 `configure()` 需要「寄存器读握手」——先 `motor.get_register_u32(8)` + `get_register_u32(7)` 再 `disable_all()`，否则 `ensure_mode` 报 "register 10 write ack not received"。
- 夹爪零点检测容差设成 15°（回程间隙 ~9.5°）。
- lerobot 0.6.2 用新 argparse：`--robot.type` / `--teleop.type` / `--dataset.repo_id` / `--dataset.num_episodes`。

---

## 4. 遥操作 + 录制（已完成）

- Follower 已校准（`seeed_b601_dm_follower`），Leader 已校准（`rebot_arm_102_leader`）。
- 相机：USB HD Camera = `/dev/video4`（MJPG 1280x720@30 / YUYV 640x480@30）。
- 相机参数用 YAML 字符串传：`--robot.cameras="{cam_top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}}"`（字段是 `index_or_path` 不是 `camera_index`）。
- 录制按键：`n`/右 = 下一段（保存+前进），`r`/左 = 重录（**丢弃**），`q`/Esc = 退出。每段要按两次 `n`（结束动作 → 结束复位），**别按 `r`**。
- 加 `--dataset.push_to_hub false` 跳过 Hub 上传（无网会报 httpx 错，无害）。
- 数据集落盘：`~/.cache/huggingface/lerobot/<repo_id>_<时间戳>/`（repo_id 会带时间戳后缀）。
- 已录 5 段（3546 帧）到 `my_b601/pick_place_20260923_164050`，但**本模型没用这份数据微调**。

---

## 5. 权重微调（服务器上，已完成）

- **基座**：π0.5（pi05，Physical Intelligence 的 VLA）。
- **LoRA 微调**：`PeftModel.from_pretrained(base, adapter_dir)`。
- 训练数据集是**外部数据集** `nyancos/orange_rack_test_tubes_848`（**不是**本机录的数据）。
- 关键路径（服务器）：
  - 基座：`/root/pi05_b601/models/pi05_base`
  - LoRA 适配器：`/root/pi05_b601/outputs/pi05_tubes_lora_first100_fast_1000/checkpoints/last/pretrained_model`
  - stats：`/root/pi05_b601/datasets/orange_rack_test_tubes_848/meta/stats.json`
  - 服务器桥接脚本：`~/pi05_b601/src/rebot_lerobot/rerobot_bridge/predict_server.py`
- **任务文字**：`place two test tubes into the orange rack`
- 训练配置要点：
  - 输入：`observation.state`(7) + `observation.images.front`(3×480×848) + `observation.images.wrist`(3×480×640)
  - 输出：`action`(7)，action_feature_names = 7 关节 `.pos`
  - `image_resolution [224,224]`、`chunk_size=50`、`use_proprioceptive_memory=false`
  - `normalization_mapping: {STATE: QUANTILES, ACTION: QUANTILES, VISUAL: IDENTITY}`

### stats.json 的 q01/q99（已硬编码进脚本，别丢）

```
action q01: [-26.745, -0.362, -104.246, -58.670, -31.971, -10.265, 0.412]
action q99: [ 37.953, 120.979,   0.141,  60.491,   7.598,  46.667, 38.006]
state q01: [-26.522,  0.042,   0.067, -61.172,  -7.291,  -9.955, 3.947]
state q99: [ 37.728, 124.516, 101.810,  56.063,  31.528,  46.463, 215.074]
```

---

## 6. π0.5 模型理解（推理要复刻的细节）

- 结构：PaliGemma 2B backbone + gemma_300m 动作专家 + SigLIP 视觉，bf16。
- **状态放在「语言 prompt」里**（不是单独输入），因为 `use_proprioceptive_memory=false`。
- 处理器流程（`processor_pi05.py`）：
  1. `rename_observations`：front→`base_0_rgb`、wrist→`left_wrist_0_rgb`
  2. `add_batch_dim` → `relative_step`(禁用) → `normalize`(QUANTILES)
  3. `Pi05PrepareStateTokenizerProcessorStep`：状态归一化→离散化→拼 prompt
  4. `TokenizerProcessorStep`（`google/paligemma-3b-pt-224`，max_length=200，右对齐 max_length padding）
  5. `to_device`
- 状态分词（逐字复刻）：
  ```python
  cleaned_text = task.strip().replace("_", " ").replace("\n", " ")
  discretized = np.digitize(state_norm, bins=np.linspace(-1, 1, 257)[:-1]) - 1
  state_str = " ".join(map(str, discretized))
  prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
  ```
- 分位数归一化/反归一化：
  ```python
  norm = 2*(x - q01)/(q99 - q01) - 1
  unnorm = (x + 1)/2*(q99 - q01) + q01
  ```
- 输出：`predict_action_chunk(batch)` → `(1, 50, 32)`，**只有前 7 维是真实动作**（后 25 维是填充）。

---

## 7. 方向约定（本项目的核心结论，已交叉验证）

**结论：state 全部取负喂模型；action 不翻转、直接发。**

推导依据：
- `get_observation()` 返回**原始编码器**（物理量），`send_action()` 的输入是**逻辑量**（内部套方向）。
- 训练数据集的 `joint_directions` = 本机驱动的**负值**（7 个关节都如此，记 `D_train = -D_user`）。
- 对比训练数据与用户数据的均值：state 符号相反（原始编码器相反）、action 符号相同（逻辑约定一致）。
- 交叉验证：拿训练 state/action 的 q01/q99 范围 vs 本机关节限位，套上「训练 = -本机」后全部吻合，同时确认了 7 关节顺序没错。
- 夹爪方向也是「state 取负、action 不翻转」，`-6` 的尺度由本机 `send_action` 内部消化，桥接层不用管。

> ⚠️ 这是**唯一必须真机才能 100% 确认**的项，靠明天干跑（看「当前状态 vs 预测动作」方向是否合理）定盘。若方向反了，改服务器脚本里 `STATE_FLIP = -1.0` → `1.0` 即可。

---

## 8. 桥接脚本（3 个，都在本机 `~/桌面/rebot_bridge/`）

| 文件 | 跑在哪 | 作用 |
|---|---|---|
| `predict_server.py` | 服务器 | 加载 base+LoRA+tokenizer+stats，开 `/predict` 接口（FastAPI） |
| `run_pi05_robot.py` | 本机 | 读机械臂状态+相机 → POST → 发动作。有 `--dry-run`（只预测不动）、`--steps` |
| `test_predict.py` | 本机 | 冒烟测试：不连机械臂，发假状态+图，验证推理链路 |

`predict_server.py` 关键配置：`STATE_FLIP=-1.0`、`TASK`、三条路径（BASE_DIR/ADAPTER_DIR/STATS_PATH）。
`run_pi05_robot.py` 关键配置：`FRONT_IDX=6`、`WRIST_IDX=4`、`JOINTS` 顺序、`SERVER_URL`。

---

## 9. 调试踩坑记录（按时间）

1. **SSH 免密失败**（Permission denied publickey,password）→ 服务器命令让用户自己跑、贴输出。
2. **ffmpeg 打开 /dev/video4/6 报 "Operation not permitted"** → 不是权限问题（ACL 有 rw），用 `v4l2-ctl --stream-mmap --stream-to` 代替。
3. **相机 select() timeout** → 加「预热读 + 重试 + 强制 MJPG fourcc」。
4. **SSH "Broken pipe"** → 服务器前台跑进程随 SSH 断开被杀；改成 `nohup ... > server.log 2>&1 &` 后台跑，隧道加 `-o ServerAliveInterval=30 -o ServerAliveCountMax=3`。
5. **服务器缺 fastapi** → `pip install fastapi uvicorn`。
6. **SSH 隧道写法**：`-L 8080:8080` 报 "Bad local forwarding specification" → 要四段式 `-L 8080:localhost:8080`。
7. **动作没切片 bug（已修）**：模型输出 `(1,50,32)`，要 `[0,:,:7]` 取前 7 维再反归一化，否则广播报错/算错。
8. **隧道验证**：服务器本地 `curl localhost:8080/health` 返回 ok 才算服务在跑；之前"连接被拒绝"是因为服务没起。

---

## 10. 当前进度 + 剩余计划

**已验证 ✅**：模型权重加载、tokenizer/stats 加载、HTTP 服务+隧道、机械臂 CAN 连接/断开、全部代码写完。

**未验证 ❌ / 待办**：
1. **推理链路**（分词+图像+模型+反归一化）——模型还没真正跑出过一个动作。**今晚冒烟测试验证**：
   ```bash
   # 本机传新文件 → 服务器 nohup 重启 → 本机连测两次（真图 vs 灰图）
   python ~/桌面/rebot_bridge/test_predict.py
   python ~/桌面/rebot_bridge/test_predict.py --front /nonexistent --wrist /nonexistent
   ```
   两次输出贴回来核对：①分词（离散化状态 0~255、prompt 正确）②动作度数是否落在 q01/q99 附近 ③两次动作是否有差异（判断视觉有没有起作用）。
2. **图像格式存疑**：`[0,1]` 还是 ImageNet 归一化？用上面的「真图 vs 灰图」差异判断；若两次动作几乎一样 → 视觉没读进去，要查格式。
3. **方向翻转实证**（明天，必须真机）：
   ```bash
   v4l2-ctl --list-devices   # 相机编号可能变了，重新对号
   python ~/桌面/rebot_bridge/run_pi05_robot.py --dry-run   # 看方向
   python ~/桌面/rebot_bridge/run_pi05_robot.py             # 真机，按回车才动
   ```

**整体评估**：基础设施 80%，核心实证 0%。剩下的全是「验证」不是「开发」，预计两个半天：今晚排掉推理链 80% 的不确定性，明天连臂完成方向确认 + 真机。

---

## 11. 服务器启动 / 隧道 / 测试 速查

```bash
# —— 服务器 ——
cd ~/pi05_b601/src/rebot_lerobot/rerobot_bridge
nohup python predict_server.py > server.log 2>&1 &
tail -f server.log          # 看到 "Uvicorn running" 后 Ctrl+C 关 tail

# —— 本机：传文件 ——
scp -P 6022 ~/桌面/rebot_bridge/predict_server.py root@43.155.204.215:~/pi05_b601/src/rebot_lerobot/rerobot_bridge/predict_server.py

# —— 本机：隧道（保持开）——
ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -L 8080:localhost:8080 -p 6022 root@43.155.204.215

# —— 本机：冒烟测试 ——
python ~/桌面/rebot_bridge/test_predict.py
```

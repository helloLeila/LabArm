from lerobot.datasets.lerobot_dataset import LeRobotDataset
from transformers import AutoTokenizer
import numpy as np

AD = '/root/pi05_b601/outputs/pi05_tubes_lora_first100_fast_1000/checkpoints/last/pretrained_model'
d = LeRobotDataset('nyancos/orange_rack_test_tubes_848', root='/root/pi05_b601/datasets/orange_rack_test_tubes_848', episodes=[0],
download_videos=False)
f = d[0]
print('=== frame keys ==='); print(list(f.keys()))
print('=== task ==='); print(repr(f.get('task')))
print('=== state ==='); print(np.round(np.array(f['observation.state']),2))
print('=== action ==='); print(np.round(np.array(f['action']),2))
lt = f.get('observation.language_tokens')
if lt is not None:
    tok = AutoTokenizer.from_pretrained(AD + '/tokenizer')
    t = np.array(lt); m = np.array(f['observation.language_attention_mask'], dtype=bool)
    print('=== 完整 prompt (状态已嵌入) ==='); print(repr(tok.decode(t[m].tolist())))
else:
    print('=== 无预计算 language_tokens，需查 tokenize 代码 ===')

#!/usr/bin/env python3
"""从臂舵机 ping 测试：试多个波特率，看哪个能通。"""
from motorbridge_smart_servo.fashionstar import FashionStarServo

PORT = '/dev/ttyUSB0'

for br in [1000000, 115200, 921600, 460800]:
    try:
        b = FashionStarServo(PORT, baudrate=br)
        res = [b.ping(i) for i in range(7)]
        alive = [i for i, ok in enumerate(res) if ok]
        print(f'{br}:  {res}  通 {len(alive)} 个 (id={alive})')
        b.close()
    except Exception as e:
        print(f'{br}:  ERR {e!r}')

# verify_joints.py
#  ./isaaclab.sh -p isaaclab_master/verify_joints.py --headless
# 必须在 import pxr 之前启动 AppLauncher 或 SimulationApp
import torch
from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# 这一行是关键：它会初始化整个 Omniverse 引擎并配置好所有的 Python 搜索路径
app_launcher = AppLauncher(args_cli) 

# 现在再 import pxr 就不会报错了
from pxr import Usd, UsdGeom, Sdf
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG
import os
import re

import env_manager
from env_manager import Go2RetrievalEnvCfg
import gymnasium as gym

cfg = Go2RetrievalEnvCfg()
cfg.scene.num_envs = 1
env = gym.make("Isaac-Go2-Retrieval-v0", cfg=cfg)

robot = env.unwrapped.scene["robot"]
print("Joint names resolved by IsaacLab:")
for i, name in enumerate(robot.joint_names):
    print(f"  [{i:2d}] {name}")

print(f"\nTotal: {len(robot.joint_names)} joints (expected 18)")
env.close()
app.close()
# Files Needed From The Previous Repository

This file collects the exact files, paths, and code fragments needed to continue the integration work in a new repository/chat.

## 1. PPO training file

Source path:

`gym_pybullet_drones/examples/learn.py`

This is the locally modified version with Colab-friendly flags:
- `--total_timesteps`
- `--pause`
- `--plot`

Relevant full code:

```python
"""Script demonstrating the use of `gym_pybullet_drones`'s Gymnasium interface.

Classes HoverAviary and MultiHoverAviary are used as learning envs for the PPO algorithm.
"""
import os
import time
from datetime import datetime
import argparse
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False
DEFAULT_PLOT = True
DEFAULT_TIMESTEPS = int(1e7)
DEFAULT_PAUSE = True

DEFAULT_OBS = ObservationType('kin')
DEFAULT_ACT = ActionType('one_d_rpm')
DEFAULT_AGENTS = 2
DEFAULT_MA = False

def run(multiagent=DEFAULT_MA,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        gui=DEFAULT_GUI,
        plot=DEFAULT_PLOT,
        colab=DEFAULT_COLAB,
        record_video=DEFAULT_RECORD_VIDEO,
        total_timesteps=DEFAULT_TIMESTEPS,
        pause=DEFAULT_PAUSE):

    filename = os.path.join(output_folder, 'save-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    if not os.path.exists(filename):
        os.makedirs(filename+'/')

    if not multiagent:
        train_env = make_vec_env(HoverAviary,
                                 env_kwargs=dict(obs=DEFAULT_OBS, act=DEFAULT_ACT),
                                 n_envs=1,
                                 seed=0)
        eval_env = HoverAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT)
    else:
        train_env = make_vec_env(MultiHoverAviary,
                                 env_kwargs=dict(num_drones=DEFAULT_AGENTS, obs=DEFAULT_OBS, act=DEFAULT_ACT),
                                 n_envs=1,
                                 seed=0)
        eval_env = MultiHoverAviary(num_drones=DEFAULT_AGENTS, obs=DEFAULT_OBS, act=DEFAULT_ACT)

    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)

    model = PPO('MlpPolicy', train_env, verbose=1)

    if DEFAULT_ACT == ActionType.ONE_D_RPM:
        target_reward = 474.15 if not multiagent else 949.5
    else:
        target_reward = 467. if not multiagent else 920.

    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=target_reward, verbose=1)
    eval_callback = EvalCallback(eval_env,
                                 callback_on_new_best=callback_on_best,
                                 verbose=1,
                                 best_model_save_path=filename+'/',
                                 log_path=filename+'/',
                                 eval_freq=int(1000),
                                 deterministic=True,
                                 render=False)

    model.learn(total_timesteps=total_timesteps,
                callback=eval_callback,
                log_interval=100)

    model.save(filename+'/final_model.zip')
    print(filename)

    with np.load(filename+'/evaluations.npz') as data:
        for j in range(data['timesteps'].shape[0]):
            print(str(data['timesteps'][j])+","+str(data['results'][j][0]))

    if pause:
        input("Press Enter to continue...")

    if os.path.isfile(filename+'/best_model.zip'):
        path = filename+'/best_model.zip'
    else:
        print("[ERROR]: no model under the specified path", filename)
    model = PPO.load(path)

    if not multiagent:
        test_env = HoverAviary(gui=gui, obs=DEFAULT_OBS, act=DEFAULT_ACT, record=record_video)
        test_env_nogui = HoverAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT)
    else:
        test_env = MultiHoverAviary(gui=gui, num_drones=DEFAULT_AGENTS, obs=DEFAULT_OBS, act=DEFAULT_ACT, record=record_video)
        test_env_nogui = MultiHoverAviary(num_drones=DEFAULT_AGENTS, obs=DEFAULT_OBS, act=DEFAULT_ACT)

    logger = Logger(logging_freq_hz=int(test_env.CTRL_FREQ),
                    num_drones=DEFAULT_AGENTS if multiagent else 1,
                    output_folder=output_folder,
                    colab=colab)

    mean_reward, std_reward = evaluate_policy(model, test_env_nogui, n_eval_episodes=10)
    print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")

    obs, info = test_env.reset(seed=42, options={})
    start = time.time()
    for i in range((test_env.EPISODE_LEN_SEC+2)*test_env.CTRL_FREQ):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        obs2 = obs.squeeze()
        act2 = action.squeeze()
        print("Obs:", obs, "\tAction", action, "\tReward:", reward, "\tTerminated:", terminated, "\tTruncated:", truncated)
        if DEFAULT_OBS == ObservationType.KIN:
            if not multiagent:
                logger.log(drone=0,
                           timestamp=i/test_env.CTRL_FREQ,
                           state=np.hstack([obs2[0:3], np.zeros(4), obs2[3:15], act2]),
                           control=np.zeros(12))
            else:
                for d in range(DEFAULT_AGENTS):
                    logger.log(drone=d,
                               timestamp=i/test_env.CTRL_FREQ,
                               state=np.hstack([obs2[d][0:3], np.zeros(4), obs2[d][3:15], act2[d]]),
                               control=np.zeros(12))
        test_env.render()
        sync(i, start, test_env.CTRL_TIMESTEP)
        if terminated:
            obs = test_env.reset(seed=42, options={})
    test_env.close()

    if plot and DEFAULT_OBS == ObservationType.KIN:
        logger.plot()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script')
    parser.add_argument('--multiagent', default=DEFAULT_MA, type=str2bool, metavar='')
    parser.add_argument('--gui', default=DEFAULT_GUI, type=str2bool, metavar='')
    parser.add_argument('--record_video', default=DEFAULT_RECORD_VIDEO, type=str2bool, metavar='')
    parser.add_argument('--output_folder', default=DEFAULT_OUTPUT_FOLDER, type=str, metavar='')
    parser.add_argument('--colab', default=DEFAULT_COLAB, type=bool, metavar='')
    parser.add_argument('--plot', default=DEFAULT_PLOT, type=str2bool, metavar='')
    parser.add_argument('--pause', default=DEFAULT_PAUSE, type=str2bool, metavar='')
    parser.add_argument('--total_timesteps', default=DEFAULT_TIMESTEPS, type=int, metavar='')
    ARGS = parser.parse_args()
    run(**vars(ARGS))
```

## 2. Model loading / evaluation path

There is no separate local evaluation script yet.

Use the evaluation section already embedded in:

`gym_pybullet_drones/examples/learn.py`

The key loading/evaluation fragment is:

```python
if os.path.isfile(filename+'/best_model.zip'):
    path = filename+'/best_model.zip'
else:
    print("[ERROR]: no model under the specified path", filename)
model = PPO.load(path)

test_env = HoverAviary(gui=gui, obs=DEFAULT_OBS, act=DEFAULT_ACT, record=record_video)
test_env_nogui = HoverAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT)

mean_reward, std_reward = evaluate_policy(model, test_env_nogui, n_eval_episodes=10)
print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")

obs, info = test_env.reset(seed=42, options={})
for i in range((test_env.EPISODE_LEN_SEC+2)*test_env.CTRL_FREQ):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = test_env.step(action)
```

## 3. Hover environment used for PPO training

Source path:

`gym_pybullet_drones/envs/HoverAviary.py`

Relevant code:

```python
import numpy as np

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class HoverAviary(BaseRLAviary):
    """Single agent RL problem: hover at position."""

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM):
        self.TARGET_POS = np.array([0,0,1])
        self.EPISODE_LEN_SEC = 8
        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act)

    def _computeReward(self):
        state = self._getDroneStateVector(0)
        ret = max(0, 2 - np.linalg.norm(self.TARGET_POS-state[0:3])**4)
        return ret

    def _computeTerminated(self):
        state = self._getDroneStateVector(0)
        if np.linalg.norm(self.TARGET_POS-state[0:3]) < .0001:
            return True
        else:
            return False

    def _computeTruncated(self):
        state = self._getDroneStateVector(0)
        if (abs(state[0]) > 1.5 or abs(state[1]) > 1.5 or state[2] > 2.0
             or abs(state[7]) > .4 or abs(state[8]) > .4):
            return True
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False
```

## 4. PID baseline example for future comparison

Source path:

`gym_pybullet_drones/examples/pid.py`

Relevant code:

```python
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 3
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48

env = CtrlAviary(drone_model=drone,
                 num_drones=num_drones,
                 initial_xyzs=INIT_XYZS,
                 initial_rpys=INIT_RPYS,
                 physics=physics,
                 neighbourhood_radius=10,
                 pyb_freq=simulation_freq_hz,
                 ctrl_freq=control_freq_hz,
                 gui=gui,
                 record=record_video,
                 obstacles=obstacles,
                 user_debug_gui=user_debug_gui)

if drone in [DroneModel.CF2X, DroneModel.CF2P]:
    ctrl = [DSLPIDControl(drone_model=drone) for i in range(num_drones)]

for i in range(0, int(duration_sec*env.CTRL_FREQ)):
    obs, reward, terminated, truncated, info = env.step(action)
    for j in range(num_drones):
        action[j, :], _, _ = ctrl[j].computeControlFromState(
            control_timestep=env.CTRL_TIMESTEP,
            state=obs[j],
            target_pos=np.hstack([TARGET_POS[wp_counters[j], 0:2], INIT_XYZS[j, 2]]),
            target_rpy=INIT_RPYS[j, :]
        )
```

Note:
- this is not yet a direct PID-vs-PPO hover benchmark script;
- it is the baseline reference for later adaptation.

## 5. Local Colab-related modifications already made

The local `learn.py` was modified to support easier Colab use:
- `--total_timesteps` to avoid editing source each run;
- `--pause false` to skip the blocking `input()` call;
- `--plot false` for headless notebook runs.

Recommended Colab command:

```bash
python gym_pybullet_drones/examples/learn.py \
    --gui false \
    --record_video false \
    --plot false \
    --pause false \
    --total_timesteps 200000
```

## 6. Dependency versions known from this repo

Source path:

`pyproject.toml`

Versions:

```toml
python = "^3.10"
pybullet = "^3.2.5"
gymnasium = "^0.28"
stable-baselines3 = "^2.0.0"
open3d = "^0.19.0"
numpy = "^1.24"
scipy = "^1.10"
matplotlib = "^3.7"
transforms3d = "^0.4.1"
```

Library metadata:

```toml
name = "gym-pybullet-drones"
version = "2.0.0"
```

## Short summary for the next step

Current practical setup:
- PPO training entrypoint exists in `gym_pybullet_drones/examples/learn.py`
- model loading/evaluation currently also lives in the same file
- hover RL environment is `gym_pybullet_drones/envs/HoverAviary.py`
- PID reference exists in `gym_pybullet_drones/examples/pid.py`
- no standalone `train_hover.py` or `evaluate_hover.py` exists yet
- next clean step is to move this logic into a new research repo that depends on `gym-pybullet-drones` as a library

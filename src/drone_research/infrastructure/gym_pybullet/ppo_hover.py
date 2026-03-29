import json
import pkgutil
import time
from datetime import datetime
from pathlib import Path
from zipimport import zipimporter

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

from drone_research.domain.config import HoverEvaluationConfig, HoverTrainConfig


# gym-pybullet-drones currently imports pkg_resources paths that still expect
# pkgutil.ImpImporter, which was removed in Python 3.12.
if not hasattr(pkgutil, "ImpImporter"):
    pkgutil.ImpImporter = zipimporter  # type: ignore[attr-defined]


def _build_run_dir(base_dir: Path) -> Path:
    run_dir = base_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_metrics(output_path: Path, metrics: dict[str, float]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def train_hover(config: HoverTrainConfig) -> Path:
    from gym_pybullet_drones.envs.HoverAviary import HoverAviary
    from gym_pybullet_drones.utils.enums import ActionType, ObservationType

    output_dir = _build_run_dir(config.output_dir)
    obs_type = ObservationType("kin")
    act_type = ActionType("one_d_rpm")

    train_env = make_vec_env(
        HoverAviary,
        env_kwargs={"obs": obs_type, "act": act_type},
        n_envs=1,
        seed=config.seed,
    )
    eval_env = HoverAviary(obs=obs_type, act=act_type)

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=config.learning_rate,
        seed=config.seed,
        tensorboard_log=str(output_dir / "tb"),
    )

    target_reward = 474.15
    callback_on_best = StopTrainingOnRewardThreshold(
        reward_threshold=target_reward,
        verbose=1,
    )
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=callback_on_best,
        verbose=1,
        best_model_save_path=str(output_dir),
        log_path=str(output_dir),
        eval_freq=1000,
        deterministic=True,
        render=False,
    )

    model.learn(
        total_timesteps=config.total_timesteps,
        callback=eval_callback,
        log_interval=100,
    )

    final_model_path = output_dir / "final_model.zip"
    model.save(str(final_model_path))

    if config.pause_before_eval:
        input("Press Enter to continue...")

    best_model_path = output_dir / "best_model.zip"
    model_path = best_model_path if best_model_path.exists() else final_model_path

    metrics = evaluate_hover(
        HoverEvaluationConfig(
            model_path=model_path,
            episodes=config.eval_episodes,
            render=config.gui,
            record_video=config.record_video,
            output_dir=output_dir,
            target_position=config.target_position,
        )
    )
    _write_metrics(output_dir / "training_summary.json", metrics)
    return model_path


def evaluate_hover(config: HoverEvaluationConfig) -> dict[str, float]:
    from gym_pybullet_drones.envs.HoverAviary import HoverAviary
    from gym_pybullet_drones.utils.enums import ActionType, ObservationType

    obs_type = ObservationType("kin")
    act_type = ActionType("one_d_rpm")
    model = PPO.load(str(config.model_path))

    test_env = HoverAviary(
        gui=config.render,
        obs=obs_type,
        act=act_type,
        record=config.record_video,
    )
    eval_env = HoverAviary(obs=obs_type, act=act_type)

    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=config.episodes,
    )

    obs, info = test_env.reset(seed=42, options={})
    episode_reward = 0.0
    steps = int((test_env.EPISODE_LEN_SEC + 2) * test_env.CTRL_FREQ)
    start = time.time()

    for i in range(steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        episode_reward += float(np.asarray(reward).squeeze())
        if config.render:
            test_env.render()
            # Keep real-time sync only when GUI is enabled.
            elapsed = time.time() - start
            target = (i + 1) * test_env.CTRL_TIMESTEP
            if elapsed < target:
                time.sleep(target - elapsed)
        if terminated or truncated:
            break

    test_env.close()
    eval_env.close()

    metrics = {
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward),
        "rollout_reward": float(episode_reward),
        "steps_executed": float(i + 1),
    }
    if config.output_dir is not None:
        _write_metrics(config.output_dir / "evaluation_metrics.json", metrics)
    return metrics

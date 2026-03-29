from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class HoverTrainConfig:
    total_timesteps: int
    output_dir: Path
    target_position: tuple[float, float, float] = (0.0, 0.0, 1.0)
    seed: int = 42
    learning_rate: float = 3e-4
    gui: bool = False
    record_video: bool = False
    plot: bool = False
    pause_before_eval: bool = False
    eval_episodes: int = 10


@dataclass(slots=True)
class HoverEvaluationConfig:
    model_path: Path
    episodes: int
    render: bool = False
    record_video: bool = False
    output_dir: Path | None = None
    target_position: tuple[float, float, float] = (0.0, 0.0, 1.0)

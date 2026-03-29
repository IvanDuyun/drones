from pathlib import Path

from drone_research.domain.config import HoverTrainConfig
from drone_research.infrastructure.gym_pybullet.ppo_hover import train_hover


class TrainHoverUseCase:
    def execute(self, config: HoverTrainConfig) -> Path:
        return train_hover(config)

from drone_research.domain.config import HoverEvaluationConfig
from drone_research.infrastructure.gym_pybullet.ppo_hover import evaluate_hover


class EvaluateHoverUseCase:
    def execute(self, config: HoverEvaluationConfig) -> dict[str, float]:
        return evaluate_hover(config)

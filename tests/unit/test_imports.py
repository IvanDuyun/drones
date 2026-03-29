from drone_research.application.use_cases.evaluate_hover import EvaluateHoverUseCase
from drone_research.application.use_cases.train_hover import TrainHoverUseCase


def test_use_cases_are_importable() -> None:
    assert EvaluateHoverUseCase() is not None
    assert TrainHoverUseCase() is not None

import argparse
from pathlib import Path

from drone_research.application.use_cases.train_hover import TrainHoverUseCase
from drone_research.domain.config import HoverTrainConfig
from drone_research.infrastructure.settings import ProjectPaths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PPO for drone hover control.")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/models/ppo_hover"))
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--pause-before-eval", action="store_true")
    parser.add_argument("--eval-episodes", type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = ProjectPaths.from_root(Path.cwd())
    paths.ensure()

    config = HoverTrainConfig(
        total_timesteps=args.timesteps,
        output_dir=args.output_dir,
        seed=args.seed,
        learning_rate=args.learning_rate,
        gui=args.gui,
        record_video=args.record_video,
        plot=args.plot,
        pause_before_eval=args.pause_before_eval,
        eval_episodes=args.eval_episodes,
    )
    use_case = TrainHoverUseCase()
    model_path = use_case.execute(config)
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()

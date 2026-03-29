import argparse
from pathlib import Path

from drone_research.application.use_cases.evaluate_hover import EvaluateHoverUseCase
from drone_research.domain.config import HoverEvaluationConfig
from drone_research.infrastructure.settings import ProjectPaths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained hover policy.")
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/reports/hover_eval"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = ProjectPaths.from_root(Path.cwd())
    paths.ensure()

    config = HoverEvaluationConfig(
        model_path=args.model_path,
        episodes=args.episodes,
        render=args.render,
        record_video=args.record_video,
        output_dir=args.output_dir,
    )
    use_case = EvaluateHoverUseCase()
    metrics = use_case.execute(config)
    print(metrics)


if __name__ == "__main__":
    main()

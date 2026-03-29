from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class ProjectPaths:
    root: Path
    artifacts: Path
    models: Path
    logs: Path
    reports: Path

    @classmethod
    def from_root(cls, root: Path) -> "ProjectPaths":
        artifacts = root / "artifacts"
        return cls(
            root=root,
            artifacts=artifacts,
            models=artifacts / "models",
            logs=artifacts / "logs",
            reports=artifacts / "reports",
        )

    def ensure(self) -> None:
        for path in (self.artifacts, self.models, self.logs, self.reports):
            path.mkdir(parents=True, exist_ok=True)

# Drones Research

Исследовательский проект по управлению БАС с элементами ИИ.

Текущий целевой сценарий:
- обучение `PPO` для задачи зависания квадрокоптера;
- последующая оценка модели;
- сравнение `PID` и `PPO` в одинаковых сценариях.

## Структура

- `src/drone_research/domain` - предметные сущности и конфигурации.
- `src/drone_research/application` - use case-слой.
- `src/drone_research/infrastructure` - адаптеры к симулятору, ML-библиотекам и файловой системе.
- `src/drone_research/interfaces` - CLI-входы.
- `configs` - шаблоны конфигураций экспериментов.
- `notebooks` - ноутбуки для Colab и локального анализа.
- `artifacts` - модели, логи, результаты запусков.
- `tests` - unit/integration tests.

## Быстрый старт

```bash
poetry install
poetry run train-hover --help
poetry run evaluate-hover --help
```

## Что уже подключено

- базовый `PPO`-pipeline для `HoverAviary`;
- отдельный CLI для обучения и оценки;
- сохранение моделей и JSON-метрик в `artifacts`.

## Что еще нужно

- `PID`-сценарий для последующего сравнения;
- ноутбук под `Colab`;
- экспериментальные сценарии с возмущениями и случайной инициализацией.

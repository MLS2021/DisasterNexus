"""Temperature-scaled multi-task resampling and stage-aware index generation."""

from __future__ import annotations

import math
import random
from typing import Iterator, Sequence

from torch.utils.data import Sampler

from disasternexus.data.dataset import LONGFORM_TASKS, REFSEG_TASK


class TemperatureTaskSampler(Sampler[int]):
    """Sample indices across tasks with configurable temperature or stage mixtures."""

    def __init__(
        self,
        task_types: Sequence[str],
        *,
        stage: str = "main",
        temperature: float = 0.5,
        num_samples: int | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.task_types = list(task_types)
        self.stage = stage
        self.temperature = temperature
        self.num_samples = num_samples if num_samples is not None else len(self.task_types)
        self.seed = seed
        self.epoch = 0

        self.by_task: dict[str, list[int]] = {}
        for idx, t in enumerate(self.task_types):
            self.by_task.setdefault(t, []).append(idx)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _task_probabilities(self) -> tuple[list[str], list[float]]:
        tasks = sorted(self.by_task)
        if self.stage == "prefix":
            # Enrich referring segmentation (80%) during the warmup prefix.
            other_tasks = [t for t in tasks if t != REFSEG_TASK]
            probs: list[float] = []
            for t in tasks:
                if t == REFSEG_TASK:
                    probs.append(0.80)
                else:
                    probs.append(0.20 / max(len(other_tasks), 1))
            return tasks, probs

        if self.stage == "continuation":
            # Focus supervision exclusively on long-form report/advice tasks.
            lf_tasks = [t for t in tasks if t in LONGFORM_TASKS]
            if not lf_tasks:
                return tasks, [1.0 / len(tasks)] * len(tasks)
            return lf_tasks, [1.0 / len(lf_tasks)] * len(lf_tasks)

        # Main joint phase: temperature-scaled resampling p_t \propto N_t^\tau
        weights = [math.pow(len(self.by_task[t]), self.temperature) for t in tasks]
        total = sum(weights)
        return tasks, [w / total for w in weights]

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)
        tasks, probs = self._task_probabilities()
        chosen_tasks = rng.choices(tasks, weights=probs, k=self.num_samples)
        indices = [rng.choice(self.by_task[t]) for t in chosen_tasks]
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

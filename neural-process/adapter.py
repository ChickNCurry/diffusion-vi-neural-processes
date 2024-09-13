from typing import Tuple

from metalearning_benchmarks import MetaLearningBenchmark  # type: ignore
from torch import Tensor
from torch.utils.data import Dataset


class MetaLearningBenchmarkDatasetAdapter(Dataset):  # type: ignore
    def __init__(
        self,
        benchmark: MetaLearningBenchmark,
    ) -> None:
        self.benchmark = benchmark

    def __len__(self) -> int:
        return self.benchmark.n_task  # type: ignore

    def __getitem__(self, task_idx: int) -> Tuple[Tensor, Tensor]:
        task = self.benchmark.get_task_by_index(task_index=task_idx)
        return Tensor(task.x), Tensor(task.y)

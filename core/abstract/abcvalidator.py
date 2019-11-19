from abc import ABC, abstractmethod

from torch.utils.data import DataLoader

from core.tags import VALID
from .abctrainer import concat_dataset


class ABCValidator(ABC):
    @abstractmethod
    def __init__(self, config, model, writer):
        super(ABCValidator, self).__init__()
        self.config = config
        self.model = model

        self.data_loader = DataLoader(
            concat_dataset(self.config.dataset, VALID, self.config.dataset_param),
            batch_size=self.config.n_batch,
            num_workers=self.config.n_worker,
            shuffle=False,
            pin_memory=True,
        )

        self.writer = writer

    @abstractmethod
    def step(self, epoch, step):
        pass

from abc import ABC, abstractmethod

from torch.utils.data import DataLoader, ConcatDataset

from core.tags import TRAIN


class ABCTrainer(ABC):
    @abstractmethod
    def __init__(self, config, model, optim, writer):
        super(ABCTrainer, self).__init__()
        self.config = config
        self.model = model
        self.optim = optim

        self.data_loader = DataLoader(
            concat_dataset(self.config.dataset, TRAIN, self.config.dataset_param),
            batch_size=self.config.n_batch,
            num_workers=self.config.n_worker,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

        self.writer = writer

    @abstractmethod
    def step(self, epoch, step):
        pass


def concat_dataset(dataset, task, dataset_param):
    dataset_list = []
    for ds in dataset:
        param = {
            **dataset_param,
        }
        dataset_list.append(ds(**param))

    if len(dataset_list) > 1:
        return ConcatDataset(dataset_list)
    else:
        return dataset_list[0]

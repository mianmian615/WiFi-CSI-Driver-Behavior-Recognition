# fast_data_loader.py
# 主要功能：为领域泛化/多域实验自定义高效数据加载器，适配训练和评估不同需求。
# 
# 主要内容：
# 1. _InfiniteSampler：将普通Sampler包装为无限循环采样器，适合持续训练。
# 2. InfiniteDataLoader：无限训练数据加载器，支持加权采样和多线程，适合训练集。
# 3. InfiniteDataLoaderWithoutReplacement：不放回采样的无限加载器，适合epoch级遍历。
# 4. FastDataLoader：高效评估/测试数据加载器，避免worker频繁重启，适合验证集。
# 
# 优点：
# - 训练集可无限采样，适合大循环训练。
# - 评估集加载快，worker只初始化一次。
# - 支持num_workers、pin_memory等高效参数。
# 
# 适用场景：
# - 多域/领域泛化实验，训练和评估分开高效加载。
# - 数据量适中时全部加载到内存，极大数据集可结合懒加载。

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch

class _InfiniteSampler(torch.utils.data.Sampler):
    """Wraps another Sampler to yield an infinite stream."""
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch

class InfiniteDataLoader:
    def __init__(self, dataset, weights, batch_size, num_workers, pin_memory=False, prefetch_factor=2, persistent_workers=False):
        super().__init__()

        if weights is not None:
            sampler = torch.utils.data.WeightedRandomSampler(weights,
                replacement=True,
                num_samples=batch_size)
        else:
            sampler = torch.utils.data.RandomSampler(dataset,
                replacement=True)

        if weights == None:
            weights = torch.ones(len(dataset))

        batch_sampler = torch.utils.data.BatchSampler(
            sampler,
            batch_size=batch_size,
            drop_last=True)

        self._infinite_iterator = iter(torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            batch_sampler=_InfiniteSampler(batch_sampler)
        ))

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self):
        raise ValueError


class InfiniteDataLoaderWithoutReplacement:
    def __init__(self, dataset, weights, batch_size, num_workers):
        super().__init__()

        if weights is not None:
            sampler = torch.utils.data.WeightedRandomSampler(weights,
                replacement=False,
                num_samples=batch_size)
        else:
            sampler = torch.utils.data.RandomSampler(dataset,
                replacement=False)

        if weights == None:
            weights = torch.ones(len(dataset))

        batch_sampler = torch.utils.data.BatchSampler(
            sampler,
            batch_size=batch_size,
            drop_last=True)

        self._infinite_iterator = iter(torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler)
        ))

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self):
        raise ValueError



class FastDataLoader:
    """DataLoader wrapper with slightly improved speed by not respawning worker
    processes at every epoch."""
    def __init__(self, dataset, batch_size, num_workers, pin_memory=False, prefetch_factor=2, persistent_workers=False):
        super().__init__()

        batch_sampler = torch.utils.data.BatchSampler(
            torch.utils.data.RandomSampler(dataset, replacement=False),
            batch_size=batch_size,
            drop_last=False
        )

        self._infinite_iterator = iter(torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            batch_sampler=_InfiniteSampler(batch_sampler)
        ))

        self._length = len(batch_sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self._infinite_iterator)

    def __len__(self):
        return self._length

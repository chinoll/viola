"""Sharding for ordinary map-style PyTorch DataLoaders."""

from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler


def prepare_dataloader(loader, *, rank, world_size):
    if isinstance(loader.sampler, DistributedSampler):
        assert loader.sampler.rank == rank and loader.sampler.num_replicas == world_size
        return loader
    if world_size == 1:
        return loader
    assert loader.batch_size is not None and isinstance(loader.sampler, (RandomSampler, SequentialSampler)), \
        "Custom batch samplers and iterable datasets should be sharded by the training script."
    sampler = DistributedSampler(loader.dataset, num_replicas=world_size, rank=rank,
                                 shuffle=isinstance(loader.sampler, RandomSampler), drop_last=loader.drop_last)
    kwargs = dict(dataset=loader.dataset, batch_size=loader.batch_size, sampler=sampler,
                  num_workers=loader.num_workers, collate_fn=loader.collate_fn, pin_memory=loader.pin_memory,
                  drop_last=loader.drop_last, timeout=loader.timeout, worker_init_fn=loader.worker_init_fn,
                  generator=loader.generator, persistent_workers=loader.persistent_workers)
    if loader.num_workers:
        kwargs.update(prefetch_factor=loader.prefetch_factor, multiprocessing_context=loader.multiprocessing_context)
    return DataLoader(**kwargs)

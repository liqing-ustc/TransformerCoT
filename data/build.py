from fvcore.common.registry import Registry
from torch.utils.data import DataLoader

DATASET_REGISTRY = Registry("dataset")
DATASETWRAPPER_REGISTRY = Registry("dataset_wrapper")

def build_dataset(cfg, split='train'):
    return DATASET_REGISTRY.get(cfg.dataset.name)(cfg, split=split)

def build_dataloader(cfg, split='train'):
    dataset = build_dataset(cfg, split)
    dataset = DATASETWRAPPER_REGISTRY.get(cfg.dataset_wrapper)(cfg, dataset)
    batch_size = cfg.dataloader.batch_size
    if split != 'train' and hasattr(cfg.dataloader, 'eval_batch_size'):
        batch_size = cfg.dataloader.eval_batch_size
    return DataLoader(dataset,
                        batch_size=batch_size,
                        num_workers=cfg.dataloader.num_workers,
                        collate_fn=dataset.collate_fn,
                        pin_memory=True, # TODO: Test speed
                        shuffle= split == 'train',
                        drop_last= split == 'train')

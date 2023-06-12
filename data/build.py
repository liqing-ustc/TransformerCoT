from fvcore.common.registry import Registry
from torch.utils.data import DataLoader

DATASET_REGISTRY = Registry("dataset")

def build_dataset(cfg, split='train'):
    return DATASET_REGISTRY.get(cfg.dataset.name)(cfg, split=split)

def build_dataloader(cfg, split='train'):
    dataset = build_dataset(cfg, split)
    return DataLoader(dataset,
                        batch_size=cfg.dataloader.batchsize,
                        num_workers=cfg.dataloader.num_workers,
                        collate_fn=dataset.collate,
                        pin_memory=True, # TODO: Test speed
                        shuffle= split == 'train',
                        drop_last= split == 'train')

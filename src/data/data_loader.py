from torch.utils.data import DataLoader

from .ds_wireframe import (
    LineDataset,
    LineDataset_collate_fn,
)


class LineDataloader:
    @classmethod
    def get_dataloader(cls, cfg, is_train: bool = True):
        ds = LineDataset(cfg=cfg, is_train=is_train)
        if is_train:
            dloader = DataLoader(
                ds,
                batch_size=cfg.train.batch_size,
                shuffle=True,
                num_workers=cfg.sys.num_workers,
                drop_last=True,
                collate_fn=LineDataset_collate_fn,
            )
        else:
            dloader = DataLoader(
                ds,
                batch_size=cfg.val.batch_size,
                shuffle=True,
                num_workers=cfg.sys.num_workers,
                drop_last=True,
                collate_fn=LineDataset_collate_fn,
            )
        return dloader

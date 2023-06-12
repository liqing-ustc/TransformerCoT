import torch.optim as optim

from common.type_utils import cfg2dict

from .loss import Loss
from .scheduler import get_scheduler


def build_optim(cfg, params, total_steps):
    loss = Loss(cfg)
    optimizer = getattr(optim, cfg.solver.optim.name)(params, **cfg2dict(cfg.solver.optim.args))
    scheduler = get_scheduler(cfg, optimizer, total_steps)
    return loss, optimizer, scheduler
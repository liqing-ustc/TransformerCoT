import math
from torch.optim.lr_scheduler import LambdaLR


def warmup_cosine(step, **kwargs):
    warmup_steps = kwargs["warmup_steps"]
    total_steps = kwargs["total_steps"]
    if step <= warmup_steps:
        return step / warmup_steps
    return max(0.5 * (1 + math.cos((step - warmup_steps) / (total_steps - warmup_steps) * math.pi)), 1e-5)


def warmup_exp(step, **kwargs):
    warmup_steps = kwargs["warmup_steps"]
    total_steps = kwargs["total_steps"]
    if step <= warmup_steps:
        return step / warmup_steps
    return kwargs["gamma"] ** (step * 1. / (total_steps - warmup_steps))

def constant(step, **kwargs):
    return 1.0

def get_scheduler(cfg, optimizer, total_steps):
    args = dict(cfg.solver.sched.get('args', {}))
    args['total_steps'] = total_steps
    lambda_func = lambda step: globals()[cfg.solver.sched.name](step, **args)
    return LambdaLR(optimizer=optimizer, lr_lambda=lambda_func)
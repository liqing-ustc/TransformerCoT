from fvcore.common.registry import Registry

EVALUATOR_REGISTRY = Registry("evaluator")

@EVALUATOR_REGISTRY.register()
class BaseEvaluator():
    def __init__(self, cfg, accelerator):
        pass

    def batch_metrics(self, data_dict):
        raise NotImplementedError("Per batch metrics calculation is required for evaluation")

    def update(self, data_dict):
        raise NotImplementedError("Update function for aggregating per-batch statistics")

    def record(self):
        raise NotImplementedError("Record results after an entire epoch")

    def reset(self):
        raise NotImplementedError("Reset all metrics and statistics for new evaluation")

def build_eval(cfg, accelerator):
    return EVALUATOR_REGISTRY.get(cfg.eval.name)(cfg, accelerator)
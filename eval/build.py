import torch
import numpy as np
from fvcore.common.registry import Registry

EVALUATOR_REGISTRY = Registry("evaluator")

def compute_accuracy(preds, targets, ignore_index=-1):
    return ((preds == targets).float() * (targets != ignore_index).float()).sum() / (targets != ignore_index).float().sum()

@EVALUATOR_REGISTRY.register()
class BaseEvaluator():
    def __init__(self, cfg, accelerator):
        self.target_metric = cfg.eval.get('target_metric', 'acc')
        self.total_count = 0
        self.eval_dict = {'acc': [], 'token_acc': []}
        self.best_result = -np.inf

    def batch_metrics(self, data_dict):
        return {}

    def batch_metrics_for_generation(self, data_dict):
        output_ids = data_dict['output_ids']
        output_masks = data_dict['output_masks']

        preds = data_dict['preds']
        if preds.shape[1] > output_ids.shape[1]:
            preds = preds[:, :output_ids.shape[1]]
        else: # padding preds to the same length as output_ids
            preds = torch.cat([preds, torch.zeros_like(output_ids)[:, preds.shape[1]:]], dim=1)

        token_result = (preds == output_ids) | (output_masks == 0)
        token_acc = token_result[output_masks == 1].float().mean()
        acc = token_result.all(dim=1).float().mean()
        return {'acc': acc.item(), 'token_acc': token_acc.item(), 'count': len(output_ids)}

    def update(self, data_dict):
        metrics = self.batch_metrics_for_generation(data_dict)
        self.total_count += metrics['count']
        for key in self.eval_dict.keys():
            self.eval_dict[key].append(float(metrics[key]) * metrics['count'])

    def record(self):
        for k, v in self.eval_dict.items():
            self.eval_dict[k] = sum(v) / self.total_count
        if self.eval_dict[self.target_metric] > self.best_result:
            is_best = True
            self.best_result = self.eval_dict[self.target_metric]
        else:
            is_best = False
        return is_best, self.eval_dict

    def reset(self):
        for key in self.eval_dict.keys():
            self.eval_dict[key] = []
        self.total_count = 0

def build_eval(cfg, accelerator):
    return EVALUATOR_REGISTRY.get(cfg.eval.name)(cfg, accelerator)
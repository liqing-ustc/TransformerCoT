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
        self.eval_dict = {self.target_metric: []}
        self.best_result = -np.inf

    def batch_metrics(self, data_dict):
        logits = data_dict['logits']
        preds = torch.argmax(logits, dim=-1)
        output_ids = data_dict['output_ids']
        acc = compute_accuracy(preds, output_ids, ignore_index=-1)
        return {'acc': acc.item(), 'count': len(output_ids)}

    def batch_metrics_for_generation(self, data_dict):
        preds = data_dict['preds']
        output_ids = data_dict['output_ids_for_generation']
        acc = compute_accuracy(torch.cat(preds), torch.cat(output_ids))
        return {'acc': acc.item(), 'count': len(output_ids)}

    def update(self, data_dict):
        if 'preds' in data_dict.keys(): # evaluating generation
            metrics = self.batch_metrics_for_generation(data_dict)
        else:
            metrics = self.batch_metrics(data_dict)
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
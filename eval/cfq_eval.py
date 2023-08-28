import numpy as np

from .build import EVALUATOR_REGISTRY, BaseEvaluator


@EVALUATOR_REGISTRY.register()
class CFQEvaluator(BaseEvaluator):
    def __init__(self, cfg, accelerator, tokenizer):
        super().__init__(cfg, accelerator, tokenizer)

    def batch_metrics_for_generation(self, data_dict):
        metrics = super().batch_metrics_for_generation(data_dict)
        preds = data_dict['preds']
        output_ids = data_dict['output_ids']
        output_ids[output_ids == -100 ] = self.tokenizer.pad_token_id
        preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        gts = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        decode_acc = np.mean([pred == gt for pred, gt in zip(preds, gts)])
        metrics.update({'decode_acc': decode_acc})
        return metrics



"""GPT2 model from HuggingFace Transformers."""
from transformers import GPT2LMHeadModel, GPT2Config

from .build import MODEL_REGISTRY, BaseModel

@MODEL_REGISTRY.register()
class GPT2(BaseModel):
    def __init__(self, config):
        super().__init__(config)

        config = config.model
        self.config = config
        self.model = GPT2LMHeadModel(GPT2Config(**config))

    def get_opt_params(self):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.model.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        
        return optim_groups
    
    def forward(self, data_dict):
        ids, masks = data_dict['concat_ids'], data_dict['concat_masks']
        labels = ids.clone()
        labels[masks == 0] = -100 # ignore loss for padding tokens
        outputs = self.model(ids, attention_mask=masks, labels=labels)
        data_dict['loss'] = outputs.loss
        data_dict['logits'] = outputs.logits
        return data_dict

    def generate(self, data_dict):
        ids, masks = data_dict['input_ids'], data_dict['input_masks']
        outputs = self.model.generate(input_ids=ids, attention_mask=masks, max_length=self.config.max_length)
        data_dict['preds'] = outputs[:, ids.shape[1]:] # remove input
        return data_dict




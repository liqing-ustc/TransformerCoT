"""
This module contains models from HuggingFace's transformers library.
https://huggingface.co/docs/transformers/model_doc
"""
from .build import MODEL_REGISTRY, BaseModel

class HFModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)

        config = config.model
        self.config = config
        # self.model = self.model_class(self.config_class(**config))

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


from transformers import GPT2LMHeadModel, GPT2Config
@MODEL_REGISTRY.register()
class GPT2(HFModel):
    """
    GPT2: https://huggingface.co/docs/transformers/model_doc/gpt2
    """
    def __init__(self, config):
        super().__init__(config)
        self.model = GPT2LMHeadModel(GPT2Config(**self.config))


from transformers import RoFormerForCausalLM, RoFormerConfig
@MODEL_REGISTRY.register()
class RoFormer(HFModel):
    """
    RoFormer is a BERT-like autoencoding model with rotary position embeddings. 
    Rotary position embeddings have shown improved performance on classification tasks with long texts.
    https://huggingface.co/docs/transformers/model_doc/roformer
    """
    def __init__(self, config):
        super().__init__(config)
        self.model = RoFormerForCausalLM(RoFormerConfig(**self.config))


from transformers import GPTNeoXForCausalLM, GPTNeoXConfig
@MODEL_REGISTRY.register()
class GPTNeoX(HFModel):
    """
    GPT NeoX: https://huggingface.co/docs/transformers/model_doc/gpt_neox
    """
    def __init__(self, config):
        super().__init__(config)
        self.model = GPTNeoXForCausalLM(GPTNeoXConfig(**self.config))


from transformers import TransfoXLLMHeadModel, TransfoXLConfig
@MODEL_REGISTRY.register()
class TransfoXL(HFModel):
    """
    Transformer XL (relative sinusoidal positional embeddings)
    https://huggingface.co/docs/transformers/model_doc/transfo-xl
    """
    def __init__(self, config):
        super().__init__(config)
        self.model = TransfoXLLMHeadModel(TransfoXLConfig(**self.config))

    def forward(self, data_dict):
        ids, masks = data_dict['concat_ids'], data_dict['concat_masks']
        labels = ids.clone()
        labels[masks == 0] = -100 # ignore loss for padding tokens
        outputs = self.model(ids, labels=labels)
        data_dict['loss'] = outputs.loss
        data_dict['logits'] = outputs.logits
        return data_dict

    def generate(self, data_dict):
        ids, masks = data_dict['input_ids'], data_dict['input_masks']
        outputs = self.model.generate(input_ids=ids, max_length=self.config.max_length)
        data_dict['preds'] = outputs[:, ids.shape[1]:] # remove input
        return data_dict

from transformers import RobertaForCausalLM, RobertaConfig
@MODEL_REGISTRY.register()
class Roberta(HFModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = RobertaForCausalLM(RobertaConfig(**self.config))

from transformers import RobertaPreLayerNormForCausalLM, RobertaPreLayerNormConfig
@MODEL_REGISTRY.register()
class RobertaPreLayerNorm(HFModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = RobertaPreLayerNormForCausalLM(RobertaPreLayerNormConfig(**self.config))
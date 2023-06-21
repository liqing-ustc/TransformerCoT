"""
Rewrite the Roberta model to input correct position_ids and it only affects the absolute position embedding.
See https://github.com/huggingface/transformers/blob/v4.30.0/src/transformers/models/roberta/modeling_roberta.py#L1575 for the original implementation.
The original implementation is problematic when using the model for generation.
"""""
import torch
from . import modeling_roberta

class RobertaForCausalLM(modeling_roberta.RobertaForCausalLM):

    def forward(self, **kwargs):
        position_ids = kwargs.get("position_ids", None)
        input_ids = kwargs.get("input_ids", None)
        inputs_embeds = kwargs.get("inputs_embeds", None)
        past_key_values = kwargs.get("past_key_values", None)

        input_shape = input_ids.size() if input_ids is not None else inputs_embeds.size()[:-1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        past_length = 0 if past_key_values is None else past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
        
        kwargs["position_ids"] = position_ids
        return super().forward(**kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape

        position_ids = model_kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past_key_values, 'position_ids': position_ids}

from copy import deepcopy
import numpy as np
import torch
from torch.utils.data import Dataset

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from .build import DATASETWRAPPER_REGISTRY

class BaseTokenizer:
    def __init__(self, vocab, st_token='<s>', end_token='</s>', sep_token='<###>', pad_token='<pad>'):
        self.vocab = [pad_token, st_token, end_token, sep_token] + vocab
        self.w2i = {w: i for i, w in enumerate(self.vocab)}
        self.st_token = st_token
        self.end_token = end_token
        self.sep_token = sep_token
        self.pad_token = pad_token
        tokenizer = Tokenizer(WordLevel(vocab=self.w2i)) 
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.add_special_tokens([pad_token, st_token, end_token, sep_token])
        self.tokenizer = tokenizer
    
    def encode(self, text, pad_direction='right'):
        if isinstance(text, str):
            text = [text]
        self.tokenizer.enable_padding(direction=pad_direction, pad_id=self.vocab.index(self.pad_token), pad_token=self.pad_token)
        outputs = self.tokenizer.encode_batch(text)
        ids = torch.LongTensor([output.ids for output in outputs])
        masks = torch.LongTensor([output.attention_mask for output in outputs])
        max_len = len(outputs[0])
        return max_len, ids, masks 

    def decode(self, ids):
        pass


@DATASETWRAPPER_REGISTRY.register()
class GPTWrapper(Dataset):
    def __init__(self, cfg, dataset):
        self.dataset = dataset
        self.tokenizer = BaseTokenizer(dataset.vocab_input + dataset.vocab_output)
        self.use_cot = cfg.use_cot

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def collate_fn(self, batch):
        st_token, end_token, sep_token = self.tokenizer.st_token, self.tokenizer.end_token, self.tokenizer.sep_token

        def concat_sentence(sample):
            sentence = [st_token] + sample['input'] 
            if self.use_cot:
                for thought in sample['cot']:
                    sentence += [sep_token] + thought
            sentence += [sep_token] + sample['output'] + [end_token]
            return ' '.join(sentence)
        
        _, concat_ids, concat_masks = self.tokenizer.encode([concat_sentence(sample) for sample in batch], pad_direction='right')
        _, input_ids, input_masks = self.tokenizer.encode([' '.join([st_token] + sample['input'] + [sep_token]) 
                                                for sample in batch], pad_direction='left') # padding left to avoid the padding token in the middle of the sequence for generation.
        _, output_ids, output_masks = self.tokenizer.encode([' '.join(sample['output'] + [end_token]) 
                                                for sample in batch], pad_direction='right')


        new_batch = {}
        for key in batch[0].keys():
            new_batch[key] = [sample[key] for sample in batch]

        new_batch['input_ids'] = input_ids
        new_batch['input_masks'] = input_masks
        new_batch['output_ids'] = output_ids
        new_batch['output_masks'] = output_masks
        new_batch['concat_ids'] = concat_ids
        new_batch['concat_masks'] = concat_masks
        return new_batch
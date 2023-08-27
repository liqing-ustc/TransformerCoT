from copy import deepcopy
import numpy as np
import torch
from torch.utils.data import Dataset

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import T5Tokenizer

from .build import DATASETWRAPPER_REGISTRY

class BaseTokenizer:
    def __init__(self, vocab, bos_token='<s>', eos_token='</s>', sep_token='<###>', pad_token='<pad>'):
        self.vocab = [pad_token, bos_token, eos_token, sep_token] + vocab
        self.w2i = {w: i for i, w in enumerate(self.vocab)}
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.sep_token = sep_token
        self.pad_token = pad_token
        tokenizer = Tokenizer(WordLevel(vocab=self.w2i)) 
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.add_special_tokens([pad_token, bos_token, eos_token, sep_token])
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
        bos_token, eos_token, sep_token = self.tokenizer.bos_token, self.tokenizer.eos_token, self.tokenizer.sep_token

        def concat_sentence(sample):
            sentence = [bos_token] + sample['input'] 
            if self.use_cot:
                for thought in sample['cot']:
                    sentence += [sep_token] + thought
            sentence += [sep_token] + sample['output'] + [eos_token]
            return ' '.join(sentence)
        
        _, concat_ids, concat_masks = self.tokenizer.encode([concat_sentence(sample) for sample in batch], pad_direction='right')
        _, input_ids, input_masks = self.tokenizer.encode([' '.join([bos_token] + sample['input'] + [sep_token]) 
                                                for sample in batch], pad_direction='left') # padding left to avoid the padding token in the middle of the sequence for generation.
        _, output_ids, output_masks = self.tokenizer.encode([' '.join(sample['output'] + [eos_token]) 
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


@DATASETWRAPPER_REGISTRY.register()
class T5Wrapper(Dataset):
    def __init__(self, cfg, dataset):
        self.dataset = dataset
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
        self.input_types = getattr(cfg, 'input_types', 'input').split('-')
        self.output_types = getattr(cfg, 'output_types', 'output').split('-')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def collate_fn(self, batch):

        def tokenize_batch(types, delimiter=' . '):
            sequences = []
            for sample in batch:
                seq = ''
                for tp in types:
                    if tp == 'input':
                        seq += 'input: ' + (sample['input'] if isinstance(sample['input'], str) else ' '.join(sample['input']))
                    elif tp == 'tree':
                        seq += 'tree: ' + sample['tree']
                    elif tp == 'steps':
                        seq += 'steps: ' + ' , '.join([f'{i} = {o}' for i, o in sample['steps']])
                    elif tp == 'results':
                        seq += 'results: ' + ' , '.join([f'{i} = {o}' for i, o in sample['results']])
                    elif tp == 'rir':
                        seq += 'intermediate: ' + sample['rir']
                    elif tp == 'output':
                        seq += 'output: ' + (sample['output'] if isinstance(sample['output'], str) else ' '.join(sample['output']))
                    else:
                        assert False, f'Unknown type: {tp}'
                    seq += delimiter
                seq.replace('{', '(').replace('}', ')') # T5 tokenzier does not support { and }
                sequences.append(seq)
            encoding = self.tokenizer(
                sequences,
                padding="longest",
                truncation=False,
                max_length=1024,
                return_tensors="pt",
            )
                
            return encoding.input_ids, encoding.attention_mask

        input_ids, input_masks = tokenize_batch(self.input_types)
        output_ids, output_masks = tokenize_batch(self.output_types)
        output_ids[output_ids == self.tokenizer.pad_token_id] = -100

        new_batch = {}
        for key in batch[0].keys():
            new_batch[key] = [sample[key] for sample in batch]

        new_batch['input_ids'] = input_ids
        new_batch['input_masks'] = input_masks
        new_batch['output_ids'] = output_ids
        new_batch['output_masks'] = output_masks
        return new_batch


from torch.utils.data import Dataset
from copy import deepcopy
import numpy as np
import torch

from .build import DATASETWRAPPER_REGISTRY

@DATASETWRAPPER_REGISTRY.register()
class GPTWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.st_token = '<s>'
        self.end_token = '</s>'
        self.pad_token = '<pad>'
        self.sep_token = '<###>'
        self.vocab = dataset.vocab + dataset.vocab_output + [self.st_token, self.end_token, self.sep_token, self.pad_token]
        self.i2w = self.vocab
        self.w2i = {w: i for i, w in enumerate(self.vocab)}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = deepcopy(self.dataset[index])
        input_output_concat = [self.st_token] + sample['input'] + [self.sep_token] + sample['output'] + [self.end_token]
        input_ids = [self.w2i[w] for w in input_output_concat]
        sample['input_ids'] = input_ids
        return sample

    def collate_fn(self, batch):
        input_max_len = max([len(sample['input_ids']) for sample in batch])
        for sample in batch:
            sample['output_ids'] = sample['input_ids'][1:] + [-1] * (input_max_len - len(sample['input_ids'])) # -1 for not calculating loss
            sample['input_ids'] = sample['input_ids'][:-1] + [self.w2i[self.pad_token]] * (input_max_len - len(sample['input_ids']))

        new_batch = {}
        for key in batch[0].keys():
            new_batch[key] = [sample[key] for sample in batch]
        
        new_batch['input_ids'] = torch.from_numpy(np.array(new_batch['input_ids'])).long()
        new_batch['output_ids'] = torch.from_numpy(np.array(new_batch['output_ids'])).long()
        return new_batch
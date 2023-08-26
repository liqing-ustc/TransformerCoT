""" domain knowledge for SCAN
"""
from collections import OrderedDict
import os
import json
import nltk
from tqdm import tqdm
import random
from torch.utils.data import Dataset

from .build import DATASET_REGISTRY
from .helper import Program, generate_cot, index_tree, tree2postfix

@DATASET_REGISTRY.register()
class SCAN(Dataset):
    # Domain knowledge and rule-based parser for SCAN dataset.
    name = 'SCAN'
    action_word = ['turn', 'walk', 'look', 'run', 'jump']
    dir_word = ['left', 'right']
    turn_times_word = ['opposite', 'around']
    times_word = ['twice', 'thrice']
    connect_word = ['and', 'after']

    vocab_input = action_word + dir_word + turn_times_word + times_word + connect_word
    vocab_output = ['', 'I_WALK', 'I_LOOK', 'I_RUN', 'I_JUMP', 'I_TURN_LEFT', 'I_TURN_RIGHT']

    input2output = {'left': 'I_TURN_LEFT', 'right': 'I_TURN_RIGHT', 
    'turn': 'I_TURN', 'walk': 'I_WALK', 'look': 'I_LOOK', 'run': 'I_RUN', 'jump': 'I_JUMP'}

    grammar = nltk.CFG.fromstring('''
    S -> VP | VP 'and' VP | VP 'after' VP
    VP -> VP 'twice' | VP 'thrice'
    VP -> V | V N | V 'around' N | V 'opposite' N
    V -> 'turn' | 'walk' | 'look' | 'run' | 'jump'
    N -> 'left' | 'right'
    ''')
    parser = nltk.ChartParser(grammar)


    @classmethod
    def parse(cls, input):
        if not isinstance(input, list):
            input = input.split()
        tree = cls.parser.parse(input)
        tree = list(tree)[0]
        tree = index_tree(tree)
        return tree

    @classmethod
    def generate_steps(cls, tree):
        """
        Given a tree, return a list of reasoning steps.
        """
        steps = []

        def _steps(tree):
            if not isinstance(tree, nltk.Tree):
                return

            for child in tree:
                _steps(child)

            node_type = tree.label().split('_')[0]
            if node_type in ['V', 'N']:
                result = cls.input2output[tree[0]]
            elif node_type == 'VP':
                if len(tree) == 1: # VP -> V
                    result = tree[0].label()
                elif tree[1] == 'twice': # VP -> VP 'twice'
                    result = ' '.join([tree[0].label()]*2)
                elif tree[1] == 'thrice': # VP -> VP 'thrice'
                    result = ' '.join([tree[0].label()]*3)
                elif tree[1] == 'around': # VP -> V 'around' N
                    result = ' '.join([tree[2].label(), tree[0].label()]*4)
                elif tree[1] == 'opposite': # VP -> V 'opposite' N
                    result = ' '.join([tree[2].label(), tree[2].label(), tree[0].label()])
                else: # VP -> V N
                    result = ' '.join([tree[1].label(), tree[0].label()])
            elif node_type == 'S':
                if len(tree) == 1: # S -> VP:
                    result = tree[0].label()
                elif tree[1] == 'and': # S -> VP 'and' VP
                    result = ' '.join([tree[0].label(), tree[2].label()])
                elif tree[1] == 'after': # S -> VP 'after' VP
                    result = ' '.join([tree[2].label(), tree[0].label()])
            
            step = (tree.label(), result)
            steps.append(step)
        
        _steps(tree)
        return steps
    
    @classmethod
    def generate_results(cls, steps):
        """
        Given a list of reasoning steps, return a list of reasoning results.
        """
        results = OrderedDict()
        for input, output in steps:
            for tok in output.split():
                if tok in results:
                    output = output.replace(tok, f'( {results[tok]} )')
            results[input] = output
        return list(results.items())

    @classmethod
    def load_data(cls, filename):
        processed_dataset_file = filename + '.processed.json'
        if os.path.exists(processed_dataset_file):
            dataset = json.load(open(processed_dataset_file, 'r'))
            return dataset

        with open(filename, 'r') as f:
            lines = f.readlines()
        dataset = []
        for line in tqdm(lines):
            _, left, right = line.split(':')
            left = left.strip().split()[:-1]
            right = right.strip().split()
            data = {'input': left, 'output': right}
            tree = cls.parse(left)
            steps = cls.generate_steps(tree)
            results = cls.generate_results(steps)
            assert [x for x in results[-1][1].split() if x not in ['(', ')', cls.input2output['turn']]] == right, "The last reasoning result is not equal to the output!"
            data.update({'tree': tree2postfix(tree), 'steps': steps, 'results': results})
            rir = results[-1][1] # reversible intermediate representation
            data.update({'rir': rir})
            dataset.append(data)
        json.dump(dataset, open(processed_dataset_file, 'w'))
        return dataset

    def __init__(self, cfg, split='train'):

        subset = getattr(cfg.dataset, 'subset', 'length')
        n_sample = getattr(cfg.dataset, 'n_sample', None)
        assert split in ['train', 'val', 'test']
        split = 'test' if split == 'val' else split # there is no val split for scan
        if subset in ['simple', 'length']:
            filename = f'{cfg.dataset.data_dir}/{subset}_split/tasks_{split}_{subset}.txt'
        elif subset == 'addprim_jump':
            filename = f'{cfg.dataset.data_dir}/add_prim_split/tasks_{split}_{subset}.txt'
        elif subset == 'addprim_turn_left':
            filename = f'{cfg.dataset.data_dir}/add_prim_split/tasks_{split}_{subset}.txt'
        elif subset == 'template_around_right':
            filename = f'{cfg.dataset.data_dir}/template_split/tasks_{split}_{subset}.txt'
        else:
            assert False, f'Unknown split for SCAN: {subset}'
        
        dataset = self.load_data(filename)

        if n_sample:
            if n_sample <= 1: # it is percentage
                n_sample = int(len(dataset) * n_sample)
            random.shuffle(dataset)
            dataset = dataset[:n_sample]
            print(f'{split}: randomly select {n_sample} samples.')

        for sample in dataset:
            sample['len'] = len(sample['input'])
        
        self.dataset = dataset
        self.valid_ids = list(range(len(dataset)))

    def __getitem__(self, index):
        index = self.valid_ids[index]
        sample = self.dataset[index]
        
        return sample
    
    def __len__(self):
        return len(self.valid_ids)

    def filter_by_len(self, min_len=None, max_len=None):
        if min_len is None: min_len = -1
        if max_len is None: max_len = float('inf')
        self.valid_ids = [i for i, x in enumerate(self.dataset) if x['len'] <= max_len and x['len'] >= min_len]


if __name__ == '__main__':
    pass

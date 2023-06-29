""" domain knowledge for SCAN
"""
import random
from torch.utils.data import Dataset

from .build import DATASET_REGISTRY
from .helper import Program, generate_cot

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

    op2precedence = {}
    op2precedence.update({x: 1 for x in connect_word})
    op2precedence.update({x: 2 for x in times_word})
    op2precedence.update({x: 3 for x in turn_times_word})
    op2precedence.update({x: 4 for x in dir_word})

    op2precedence = {}
    op2precedence.update({x: 1 for x in connect_word})
    op2precedence.update({x: 2 for x in times_word})
    op2precedence.update({x: 3 for x in turn_times_word})
    op2precedence.update({x: 4 for x in dir_word})

    
    sym2prog = {
        'turn': Program(lambda: ()),
        'walk': Program(lambda: (1,)),
        'look': Program(lambda: (2,)),
        'run': Program(lambda: (3,)),
        'jump': Program(lambda: (4,)),

        'left': Program(lambda x: (5,) + x),
        'right': Program(lambda x: (6,) + x),

        'opposite': Program(lambda x: (x[0],) + x),
        'around': Program(lambda x: x * 4),

        'twice': Program(lambda x: x * 2),
        'thrice': Program(lambda x: x * 3),

        'and': Program(lambda x, y: x + y),
        'after': Program(lambda x, y: y + x),
    }
    
    sym2arity = {k: v.arity for k, v in sym2prog.items()}


    @classmethod
    def parse(cls, input):
        sym2arity = cls.sym2arity
        op2precedence = cls.op2precedence
        values = []
        operators = []
        
        head = [-1] * len(input)
        for (i,sym) in enumerate(input):
            if sym2arity[sym] == 0:
                values.append(i)
            else:
                while len(operators) > 0 and op2precedence[input[operators[-1]]] >= op2precedence[sym]:
                    op = operators.pop()
                    for _ in range(sym2arity[input[op]]):
                        head[values.pop()] = op
                    values.append(op)
                operators.append(i)

        while len(operators) > 0:
            op = operators.pop()
            for _ in range(sym2arity[input[op]]):
                head[values.pop()] = op
            values.append(op)

        root_op = values.pop()
        head[root_op] = -1
        assert len(values) == 0

        return head
    
    @classmethod
    def load_data(cls, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
        dataset = []
        for line in lines:
            _, left, right = line.split(':')
            left = left.strip().split()[:-1]
            head = cls.parse(left)
            right = right.strip().split()
            data = {'input': left, 'head': head, 'output': right}
            dataset.append(data)
        return dataset
    @classmethod
    def split_datasets_by_length(cls,filetrain, filetest, split_length):
        # load dataset
        def read_dataset(file_path):
            with open(file_path, 'r') as file:
                lines = file.readlines()

            dataset = []
            for line in lines:
                in_sentence, out_sentence = line.split(' OUT: ')
                in_sentence = in_sentence[4:]  # remove "IN: "

                out_length = len(out_sentence.split())
                dataset.append((in_sentence.strip(), out_sentence.strip(), out_length))

            return dataset

        def write_dataset(file_path, dataset):
            with open(file_path, 'w') as file:
                for item in dataset:
                    file.write(f'IN: {item[0]} OUT: {item[1]}\n')

        train_dataset = read_dataset(filetrain)
        test_dataset = read_dataset(filetest)
        all_data = train_dataset + test_dataset

        # sort the data according to the length of the out
        all_data.sort(key=lambda x: x[2])

        # divide training sets and test sets according to the length of the specified OUT
        new_train_dataset = [item for item in all_data if item[2] <= split_length]
        new_test_dataset = [item for item in all_data if item[2] > split_length]

        write_dataset(filetrain, new_train_dataset)
        write_dataset(filetest, new_test_dataset)

    def __init__(self, cfg, split='train'):

        subset = getattr(cfg.dataset, 'subset', 'length')
        n_sample = getattr(cfg.dataset, 'n_sample', None)
        if subset == 'length' and split=='train':
            self.split_datasets_by_length(f'{cfg.dataset.data_dir}/{subset}_split/tasks_train_{subset}.txt', f'{cfg.dataset.data_dir}/{subset}_split/tasks_test_{subset}.txt', cfg.dataset.length_split)
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

        if cfg.use_cot:
            dataset = generate_cot(dataset, self.sym2prog, self.vocab_output)

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

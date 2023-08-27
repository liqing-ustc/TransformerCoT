""" dataset loader for CFQ
"""
from collections import OrderedDict, defaultdict
import os
import json
import nltk
from tqdm import tqdm
import random
from torch.utils.data import Dataset

from .build import DATASET_REGISTRY
from .helper import compress_tree, index_tree, tree2postfix

def convert_grammar_rules(original_grammar_rules):
    new_grammar_rules = []
    lhs_all, rhs_all = [], []
    for rule in original_grammar_rules:
        rule = rule[:rule.rfind('_')]
        # Split the rule into the left-hand side and right-hand side
        lhs, rhs = rule.split('=')
        lhs_all.append(lhs)
        rhs_all.append(rhs)
    
    nonterminal_mapping = {}
    for i, lhs in enumerate(lhs_all):
        if '_' in lhs:
            new_lhs = lhs.replace('_', '-')
            nonterminal_mapping[lhs] = new_lhs
            lhs_all[i] = new_lhs
    
    terminal_mapping = {}
    for i, rhs in enumerate(rhs_all):
        for key, value in nonterminal_mapping.items():
            rhs = rhs.replace(key, value)
        new_rhs = []
        for tok in rhs.split('_'):
            if tok not in lhs_all[i+1:] or (tok == 'GENDER' and lhs_all[i] == 'NP'): # terminal node
                tok = tok.lower()
                if ' ' in tok:
                    new_tok = tok.replace(' ', '-')
                    terminal_mapping[tok] = new_tok
                else:
                    new_tok = tok
                new_tok = f'"{new_tok.lower()}"'
            else:
                new_tok = tok
            new_rhs.append(new_tok)
        new_rhs = ' '.join(new_rhs)
        rhs_all[i] = new_rhs

    for lhs, rhs in zip(lhs_all, rhs_all):
        new_grammar_rules.append(f'{lhs} -> {rhs}')

    # print(nonterminal_mapping)
    # print(terminal_mapping)
    return new_grammar_rules, nonterminal_mapping, terminal_mapping

def parse(sentence, grammar_rules):
    grammar_rules, _, terminal_mapping = convert_grammar_rules(grammar_rules)
    # for rule in grammar_rules:
    #     print(rule)

    sentence = sentence.lower()
    for key, value in terminal_mapping.items():
        sentence = sentence.replace(key, value)

    terminal_nodes = sentence.split(' ')
    rules = grammar_rules[:]
    def generate_tree(symbol):
        if symbol.startswith('"'): # terminal node
            symbol = symbol[1:-1] # remove the quotes
            next_terminal_node = terminal_nodes.pop(0)
            if symbol == '[entity]': # replace [entity] with the actual entity name
                assert next_terminal_node.startswith('m'), next_terminal_node
                symbol = next_terminal_node
            else:
                assert symbol == next_terminal_node, (symbol, next_terminal_node)
            return symbol

        rule = rules.pop(0)
        lhs, rhs = rule.split(' -> ')
        assert lhs == symbol
        rhs = rhs.split(' ')
        children = []
        for i, child in enumerate(rhs):
            child_tree = generate_tree(child)
            children.append(child_tree)

        tree = nltk.Tree(symbol, children)
        return tree

    tree = generate_tree('S')
    assert len(rules) == 0, rules
    assert len(terminal_nodes) == 0, terminal_nodes
    return tree, sentence
    

def load_predicates():
    with open('data/cfq_predicates.txt', 'r') as f:
        # adapted from Drozdov et al., 2023.
        raw_text = f.read()
    lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
    trimmed_predicates = {line.split(" ")[0]:line.split(" ")[1] for line in lines}
    return trimmed_predicates

trimmed_predicates = load_predicates()

class SPARQL():
    def __init__(self, query, prefix, conditions):
        self.query = query
        self.prefix = prefix
        self.conditions = conditions
        self.merged_conditions = self.merge_conditions(conditions)

    @classmethod
    def trim_predicate(cls, query):
        for predicate in trimmed_predicates:
            if predicate in query:
                query = query.replace(predicate, trimmed_predicates[predicate])
        return query

    @classmethod
    def parse(cls, query):
        """Parses a SPARQL query into a prefix and conditions."""
        query = cls.trim_predicate(query)
        query = query.lower().replace("\n", " ")
        query = query.replace('{', 'lb').replace('}', 'rb').replace('^', '#') # Some tokenzier (T5) does not support { } ^, replace them with lb rb #.
        # Remove the closing bracket and split on opening bracket.
        assert query.endswith(" rb"), f"Missing closing bracket: {query}"
        parts = query.replace(" rb", "").split(" lb ")
        assert len(parts) == 2, f"Invalid query: {query}" 
        prefix = parts[0].strip()
        conditions = [c.split() for c in parts[1].split(" . ") if not c.startswith("filter")]
        # (Drozdov et al., 2023) We strip FILTER statements because they always appear with the “sibling of” and “married to” properties, and would be trivial to add to the output after prediction (i.e., they can be considered to be part of the abbreviated property). For example, if “?x0 married to M0” appeared then so would “FILTER(?x0 != M0)”.   Essentially,  one can not be married to themselves, nor a sibling of themselves.
        return cls(query, prefix, conditions)

    @classmethod
    def merge_conditions(cls, conditions):
        # adapted from https://github.com/google-research/language/blob/master/language/compir/dataset_parsers/cfq_parser.py 
        def get_subj_rel_to_objects(conditions):
            """Merges conditions that share the same subject and relation."""
            subj_rel_to_objects = OrderedDict()
            for condition in conditions:
                subj, rel, obj = condition
                subj_rel = (subj, rel)
                if subj_rel not in subj_rel_to_objects:
                    subj_rel_to_objects[subj_rel] = []
                subj_rel_to_objects[subj_rel].append(obj)
            return subj_rel_to_objects

        def get_merged_conditions(subj_rel_to_objects, subj_objs_to_rels):
            """ merge conditions into the form of (s , (r_1, r_2 ...), (o_1 , o_2 ...))."""
            merged_conditions = []
            added_subj_objs = []
            for subj_rel, objects in subj_rel_to_objects.items():
                subj, _ = subj_rel
                objects_tup = tuple(objects)
                if (objects_tup, subj) in added_subj_objs:
                    # Already handled the conditions with this subject and objects list.
                    continue
                else:
                    added_subj_objs.append((objects_tup, subj))
                condition_merged = [subj, subj_objs_to_rels[(objects_tup, subj)], objects]
                merged_conditions.append(condition_merged)
            return merged_conditions
        
        # Prepare subject-relation to objects map.
        subj_rel_to_objects = get_subj_rel_to_objects(conditions)

        # Prepare subject-objects to relations map.
        subj_objs_to_rels = defaultdict(list)
        for subj_rel, objects in subj_rel_to_objects.items():
            if objects is not None:
                objects_tuple = tuple(objects)
                subj, rel = subj_rel
                key = (objects_tuple, subj)
                subj_objs_to_rels[key].append(rel)

        merged_conditions = get_merged_conditions(subj_rel_to_objects, subj_objs_to_rels)
        return merged_conditions

    def to_prompt(self):
        prompt = self.prefix + ' # prefix\n'
        prompt += '{\n'
        for i, condition in enumerate(self.conditions):
            prompt += ' '.join(condition)
            prompt += f' # condition_{i} \n'
        prompt += '}'
        return prompt

    def to_rir(self):
        prefix = self.prefix

        # (s (r_1,r_2 ...) (o_1,o_2...) )
        conditions = []
        for condition in self.merged_conditions:
            subj, rel, obj = condition
            subj = ','.join(subj) if isinstance(subj, list) else subj
            rel = ','.join(rel) if isinstance(rel, list) else rel
            obj = ','.join(obj) if isinstance(obj, list) else obj
            condition = f'(({subj}) ({rel}) ({obj}))'
            conditions.append(condition)
        conditions.sort()
        conditions = ' ; '.join(conditions)

        rir = f'({prefix}) ({conditions})'
        return rir


@DATASET_REGISTRY.register()
class CFQ(Dataset):

    subsets = ['mcd1', 'mcd2', 'mcd3']

    @classmethod
    def load_data(cls, filename):
        processed_dataset_file = filename + '.processed.json'
        if os.path.exists(processed_dataset_file):
            dataset = json.load(open(processed_dataset_file, 'r'))
            return dataset

        dataset = []
        print('Loading and processing the raw dataset...')
        raw_dataset = json.load(open(filename, 'r'))
        for sample in tqdm(raw_dataset):
            question = sample['questionPatternModEntities']
            grammar_rules = [x['stringValue'] for x in sample['ruleIds'] if x['type'] == 'GRAMMAR_RULE']
            query = sample['sparqlPatternModEntities']
            tree, question = parse(question, grammar_rules)
            tree = compress_tree(tree)
            tree = index_tree(tree)
            query = SPARQL.parse(query)

            data = {'input': question, 'output': query.query,
                    'tree': tree2postfix(tree), 'rir': query.to_rir()}
            dataset.append(data)

        json.dump(dataset, open(processed_dataset_file, 'w'))
        return dataset

    def __init__(self, cfg, split='train'):
        filename = f'{cfg.dataset.data_dir}/dataset.json'
        full_dataset = self.load_data(filename)

        subset = getattr(cfg.dataset, 'subset', self.subsets[0])
        assert subset in self.subsets, f'Invalid subset: {subset}'

        if split == 'val':
            split = 'dev'
        assert split in ['train', 'dev', 'test']
        splits_path = f'{cfg.dataset.data_dir}/splits/{subset}.json'
        splits = json.load(open(splits_path, 'r'))
        split_ids = splits[split + 'Idxs']
        dataset = [full_dataset[i] for i in split_ids]

        n_sample = getattr(cfg.dataset, 'n_sample', None)
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

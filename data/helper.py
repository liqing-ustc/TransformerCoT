from inspect import signature
from copy import deepcopy
from tqdm import tqdm

EMPTY_VALUE = -1
MISSING_VALUE = -2

class Program():
    def __init__(self, fn=None):
        self.fn = fn
        self.arity = len(signature(fn).parameters) if fn is not None else 0

    def __call__(self, inputs):
        y = self.fn(*inputs)
        return y

class Node:
    def __init__(self, index, symbol, smt, thoughts = None, replaces = None):
        self.index = index
        self.symbol = symbol
        self.smt = smt
        self.children = []
        self._res = None
        self._ids = None # ids of nodes in the subtree rooted at this node
        self.thoughts = thoughts
        self.replaces = replaces

    def res(self):
        if self._res is not None:
            return self._res

        self._res = self.smt(self.inputs())
        for _ in range(len(self.children)):
            self.replaces.pop()
        self.replaces.append((self.ids(), self._res))
        self.thoughts.append(deepcopy(self.replaces))
        return self._res

    def ids(self):
        if self._ids is not None:
            return self._ids

        self._ids = [self.index] + [x for c in self.children for x in c.ids()]
        return self._ids

    def inputs(self):
        return tuple([x.res() for x in self.children])

class AST: # Abstract Syntax Tree
    def __init__(self, sentence, head, semantics):
        self.sentence = sentence
        self.head = head
        self.semantics = semantics
        self.thoughts = []
        self.replaces = []

        nodes = [Node(i, s, semantics[s], self.thoughts, self.replaces) for i, s in enumerate(sentence)]

        for node, h in zip(nodes, head):
            if h == -1:
                self.root_node = node
                continue
            nodes[h].children.append(node)
        self.nodes = nodes

        self.root_node.res()
    
    def res(self): return self.root_node.res()
    
    def res_all(self): return [nd._res for nd in self.nodes]

    def cot(self): 
        cot = []
        for replace in self.thoughts:
            sentence = deepcopy(self.sentence)
            for ids, res in replace:
                # replace the root node with the result
                sentence[ids[0]] = res # the first id is the id of the root node
                # remove the other nodes
                for i in ids[1:]:
                    sentence[i] = None
            sentence = [x for x in sentence if x is not None]
            cot.append(sentence)
        return cot

def decode_thought(thought, vocab_output):
    sentence = []
    for w in thought:
        if isinstance(w, str):
            sentence.append(w)
        else:
            for i in w:
                sentence.append(vocab_output[i])
    return sentence

def generate_cot(dataset, sym2prog, vocab_output):
    print('Generating CoT...')
    for sample in tqdm(dataset):
        left = sample['input']
        head = sample['head']
        ast = AST(left, head, sym2prog)
        cot = [decode_thought(t, vocab_output) for t in ast.cot()]
        assert cot[-1] == sample['output'], "The last thought is not equal to the output!"
        cot = cot[:-1] # remove the last one
        sample['cot'] = cot
    return dataset
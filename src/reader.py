import json
import collections
import numpy as np
import torch
import torch.utils.data.dataset as Dataset
from tqdm import tqdm
import random
def _sample_set(s, k):
    if len(s) <= k:
        return s
    return set(random.sample(list(s), k))

class Vocabulary(object):
    def __init__(self, vocab_file, num_relations, num_entities):
        self.vocab = self.load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.num_relations = num_relations
        self.num_entities = num_entities
        assert len(self.vocab) == self.num_relations + self.num_entities + 2

    def load_vocab(self, vocab_file):
        """
        Load a vocabulary file into a dictionary.
        """
        vocab = collections.OrderedDict()
        fin = open(vocab_file, encoding='utf-8')
        for num, line in enumerate(fin):
            items = line.strip().split("\t")
            if len(items) > 2:
                break
            token = items[0]
            index = items[1] if len(items) == 2 else num
            token = token.strip()
            vocab[token] = int(index)
        return vocab
    
    def add_inverse_relations(self):
        """
        Add inverse relations to the vocabulary.
        """
        specials = [ self.inv_vocab[i] for i in range(2) ]
        R = self.num_relations
        orig_rels = [ self.inv_vocab[i] for i in range(2, 2+R) ]
        ents = [ self.inv_vocab[i] for i in range(2+R, 2+R+self.num_entities) ]

        new_vocab = collections.OrderedDict()
        for tok in specials:
            new_vocab[tok] = len(new_vocab)
        for rel in orig_rels:
            new_vocab[rel] = len(new_vocab)
        for rel in orig_rels:
            inv = rel + "_inv"
            new_vocab[inv] = len(new_vocab)
        for ent in ents:
            new_vocab[ent] = len(new_vocab)
        
        self.vocab = new_vocab
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.num_relations = R * 2
        
        assert len(self.vocab) == 2 + self.num_relations + self.num_entities
         
    def convert_by_vocab(self, vocab, items):
        """
        Convert a sequence of [tokens|ids] using the vocab.
        """
        output = []
        for item in items:
            output.append(vocab[item])
        return output
    
    def convert_tokens_to_ids(self, tokens):
        return self.convert_by_vocab(self.vocab, tokens)
    
    def convert_ids_to_tokens(self, ids):
        return self.convert_by_vocab(self.inv_vocab, ids)
    
    def convert_token_to_id(self, token):
        return self.vocab[token]
    
    def __len__(self):
        return len(self.vocab)

class NaryExample(object):
    def __init__(self,
                 arity,
                 head,
                 relation,
                 tail,
                 auxiliary_info=None):
        self.arity = arity
        self.head = head
        self.relation = relation
        self.tail = tail
        self.auxiliary_info = auxiliary_info

class NaryFeature(object):
    def __init__(self,
                 feature_id,
                 example_id,
                 input_tokens,
                 input_ids,
                 input_mask,
                 mask_position,
                 mask_label,
                 mask_type,
                 arity,
                 subgraph_metadata
    ):
        """
        Construct NaryFeature.
        Args:
            feature_id: unique feature id
            example_id: corresponding example id
            input_tokens: input sequence of tokens
            input_ids: input sequence of ids
            input_mask: input sequence mask
            mask_position: position of masked token
            mask_label: label of masked token
            mask_type: type of masked token,
                1 for entities (values) and -1 for relations (attributes)
            arity: arity of the corresponding example
        """
        self.feature_id = feature_id        
        self.example_id = example_id        
        self.input_tokens = input_tokens
        self.input_ids = input_ids          
        self.input_mask = input_mask
        self.mask_position = mask_position
        self.mask_label = mask_label
        self.mask_type = mask_type
        self.arity = arity
        self.subgraph_metadata = subgraph_metadata

def read_examples(input_file, max_arity):
    """
    Read a n-ary json file into a list of NaryExample.
    """
    examples, total_instance = [], 0

    with open(input_file, "r", encoding='utf-8') as fr:
        for line in fr.readlines():
            obj = json.loads(line.strip())
            arity = obj["N"]
            relation = obj["relation"]
            head = obj["subject"]
            tail = obj["object"]

            auxiliary_info = None
            if arity > 2:
                auxiliary_info = collections.OrderedDict()
                for attribute in sorted(obj.keys()):
                    if attribute in ("N", "relation", "subject", "object"):
                        continue
                    auxiliary_info[attribute] = sorted(obj[attribute])

            if arity <= max_arity:
                example = NaryExample(
                    arity=arity,
                    head=head,
                    relation=relation,
                    tail=tail,
                    auxiliary_info=auxiliary_info)
                examples.append(example)
                total_instance += (2 * (arity - 2) + 3)

    return examples, total_instance

def generate_token_id_dict(examples, vocabulary, max_arity, max_seq_length):
    token_exs_dict = collections.defaultdict(list)  # {token_id: [example_id]}
    ex_tokens_dict = collections.defaultdict(list)  # {example_id: [token_ids]}
    
    for (example_id, example) in tqdm(enumerate(examples), total=len(examples)):
        hrt = [example.head, example.relation, example.tail]
        aux_q = []
        if example.auxiliary_info is not None:
            for attr in example.auxiliary_info:
                for val in example.auxiliary_info[attr]:
                    aux_q.append(attr)
                    aux_q.append(val)
        while len(aux_q) < (max_arity - 2) * 2:
            aux_q.append("[PAD]")
            aux_q.append("[PAD]")
        
        orig_input_tokens = hrt + aux_q
        tokens_ids = vocabulary.convert_tokens_to_ids([t for t in orig_input_tokens])
        ex_tokens_dict[example_id] = tokens_ids
        
        for tok_id in tokens_ids:
            token_exs_dict[tok_id].append(example_id)
    
    return token_exs_dict, ex_tokens_dict

def convert_examples_to_features(examples, vocabulary, max_arity, max_seq_length, max_per_type, max_neighbors_num, is_train, train_examples=None):
    """
    Convert a set of NaryExample into a set of NaryFeature. Each single
    NaryExample is converted into (2*(n-2)+3) NaryFeature, where n is
    the arity of the given example.
    """
    max_aux = max_arity - 2
    assert max_seq_length == 2 * max_aux + 3, \
        "Each input sequence contains relation, head, tail, and max_aux attribute-value pairs."

    pad_id = vocabulary.convert_token_to_id("[PAD]")
    mask_id = vocabulary.convert_token_to_id("[MASK]")
    value_positions = [i for i in range(3, max_seq_length) if i % 2 == 0]

    features = []
    feature_id = 0

    if is_train:
        token_exs_dict, ex_tokens_dict = generate_token_id_dict(examples, vocabulary, max_arity, max_seq_length)
    else:
        token_exs_dict, ex_tokens_dict = generate_token_id_dict(train_examples, vocabulary, max_arity, max_seq_length)

    for (example_id, example) in tqdm(enumerate(examples), total=len(examples)):
        hrt = [example.head, example.relation, example.tail]
        hrt_mask = [1, 1, 1]

        aux_q = []
        aux_q_mask = []
        if example.auxiliary_info is not None:
            for attribute in example.auxiliary_info.keys():
                for value in example.auxiliary_info[attribute]:
                    aux_q.append(attribute)
                    aux_q.append(value)
                    aux_q_mask.append(1)
                    aux_q_mask.append(1)
        while len(aux_q) < max_aux * 2:
            aux_q.append("[PAD]")
            aux_q.append("[PAD]")
            aux_q_mask.append(0)
            aux_q_mask.append(0)

        orig_input_tokens = hrt + aux_q
        orig_input_mask = hrt_mask + aux_q_mask
        assert len(orig_input_tokens) == max_seq_length and len(orig_input_mask) == max_seq_length

        base_input_ids = vocabulary.convert_tokens_to_ids(orig_input_tokens)

        subj_id = base_input_ids[0]
        obj_id = base_input_ids[2]
        h_full = set(token_exs_dict.get(subj_id, []))
        t_full = set(token_exs_dict.get(obj_id, []))
        v_full = set()
        for i in value_positions:
            val_id = base_input_ids[i]
            if val_id != pad_id:
                v_full.update(token_exs_dict.get(val_id, []))

        h_neighbor_set = _sample_set(h_full, max_per_type)
        t_neighbor_set = _sample_set(t_full, max_per_type)
        v_neighbor_set = _sample_set(v_full, max_per_type)
        neighbor_list = []

        for mask_position in range(max_seq_length):
            if orig_input_tokens[mask_position] == "[PAD]":
                continue
            mask_label = base_input_ids[mask_position]
            mask_type = 1 if mask_position % 2 != 1 else -1

            input_tokens = orig_input_tokens[:]
            input_tokens[mask_position] = "[MASK]"
            input_ids = base_input_ids[:]
            input_ids[mask_position] = mask_id
            assert len(input_tokens) == max_seq_length and len(input_ids) == max_seq_length

            if mask_type == 1:              # entity/value
                if mask_position == 0:      # subject
                    neighbor_set = t_neighbor_set.union(v_neighbor_set)
                elif mask_position == 2:    # object
                    neighbor_set = h_neighbor_set.union(v_neighbor_set)
                else:                       # value
                    val_id = mask_label
                    v_full_mask = v_full - set(token_exs_dict.get(val_id, []))
                    v_neighbor_set = _sample_set(v_full_mask, max_per_type)
                    neighbor_set = h_neighbor_set.union(t_neighbor_set).union(v_neighbor_set)
            else:                           # relation/attribute
                neighbor_set = h_neighbor_set.union(t_neighbor_set).union(v_neighbor_set)
            
            neighbor_list = get_example_neighbor_list(neighbor_set, example_id, ex_tokens_dict, remove_self=is_train)

            if len(neighbor_list) > max_neighbors_num:
                sampled_neighbors = random.sample(list(neighbor_list), max_neighbors_num)
            else:
                sampled_neighbors = neighbor_list

            neighbor_token_list = []
            neighbor_token_list.append(input_ids)
            neighbor_token_list.extend(ex_tokens_dict[neighbor] for neighbor in sampled_neighbors)
            
            while len(neighbor_token_list) < 1 + max_neighbors_num:
                neighbor_token_list.append([pad_id] * max_seq_length)

            fact_rel_ids, fact_ent_ids, fact_entity_roles, fact_rel_roles = build_graph_metadata_from_token_id_sequences(
                neighbor_token_list, max_aux, pad_id
            )

            feature = NaryFeature(
                feature_id=feature_id,
                example_id=example_id,
                input_tokens=input_tokens,
                input_ids=input_ids,
                input_mask=orig_input_mask,
                mask_position=mask_position,
                mask_label=mask_label,
                mask_type=mask_type,
                arity=example.arity,
                subgraph_metadata=[fact_rel_ids, fact_ent_ids, fact_entity_roles, fact_rel_roles],
            )
            features.append(feature)
            feature_id += 1

    return features

def convert_examples_to_feature_headers(examples, vocabulary, max_arity, max_seq_length):
    max_aux = max_arity - 2
    pad_id = vocabulary.convert_token_to_id("[PAD]")
    mask_id = vocabulary.convert_token_to_id("[MASK]")

    features = []
    feature_id = 0

    for (example_id, example) in tqdm(enumerate(examples), total=len(examples)):
        hrt = [example.head, example.relation, example.tail]
        hrt_mask = [1, 1, 1]
        aux_q, aux_q_mask = [], []
        if example.auxiliary_info is not None:
            for attribute in example.auxiliary_info.keys():
                for value in example.auxiliary_info[attribute]:
                    aux_q += [attribute, value]
                    aux_q_mask += [1, 1]
        while len(aux_q) < (max_aux * 2):
            aux_q += ["[PAD]", "[PAD]"]
            aux_q_mask += [0, 0]
        orig_input_tokens = hrt + aux_q
        orig_input_mask = hrt_mask + aux_q_mask
        base_input_ids = vocabulary.convert_tokens_to_ids(orig_input_tokens)

        for mask_position in range(max_seq_length):
            if orig_input_tokens[mask_position] == "[PAD]":
                continue
            mask_label = base_input_ids[mask_position]
            mask_type = 1 if mask_position % 2 != 1 else -1
            input_tokens = orig_input_tokens[:]
            input_tokens[mask_position] = "[MASK]"
            input_ids = base_input_ids[:]
            input_ids[mask_position] = mask_id

            feature = NaryFeature(
                feature_id=feature_id,
                example_id=example_id,
                input_tokens=input_tokens,
                input_ids=input_ids,
                input_mask=orig_input_mask,
                mask_position=mask_position,
                mask_label=mask_label,
                mask_type=mask_type,
                arity=example.arity,
                subgraph_metadata=None,
            )
            features.append(feature)
            feature_id += 1

    return features

def resample_subgraph_metadata(feature: NaryFeature,
                               vocabulary: Vocabulary,
                               max_arity: int,
                               max_seq_length: int,
                               max_per_type: int,
                               max_neighbors_num: int,
                               is_train: bool,
                               token_exs_dict,
                               ex_tokens_dict,
                               ):
    max_aux = max_arity - 2
    pad_id = vocabulary.convert_token_to_id("[PAD]")
    mask_id = vocabulary.convert_token_to_id("[MASK]")

    # recover base_input_ids (undo mask)
    base_input_ids = feature.input_ids[:]
    base_input_ids[feature.mask_position] = feature.mask_label

    value_positions = [i for i in range(3, max_seq_length) if i % 2 == 0]

    subj_id = base_input_ids[0]
    obj_id = base_input_ids[2]

    h_full = set(token_exs_dict.get(subj_id, []))
    t_full = set(token_exs_dict.get(obj_id, []))
    v_full = set()
    for i in value_positions:
        val_id = base_input_ids[i]
        if val_id != pad_id:
            v_full.update(token_exs_dict.get(val_id, []))

    # generic neighbor pools
    h_neighbor_set = _sample_set(h_full, max_per_type)
    t_neighbor_set = _sample_set(t_full, max_per_type)
    v_neighbor_set = _sample_set(v_full, max_per_type)

    # choose neighbor_set by mask position/type
    if feature.mask_type == 1:              # entity/value
        if feature.mask_position == 0:      # subject
            neighbor_set = t_neighbor_set.union(v_neighbor_set)
        elif feature.mask_position == 2:    # object
            neighbor_set = h_neighbor_set.union(v_neighbor_set)
        else:                               # value
            val_id = feature.mask_label
            v_full_mask = v_full - set(token_exs_dict.get(val_id, []))
            v_neighbor_set2 = _sample_set(v_full_mask, max_per_type)
            neighbor_set = h_neighbor_set.union(t_neighbor_set).union(v_neighbor_set2)
    else:                                   # relation/attribute
        neighbor_set = h_neighbor_set.union(t_neighbor_set).union(v_neighbor_set)

    # ids only; do not remove self in test path
    neighbor_list = get_example_neighbor_list(neighbor_set, feature.example_id, ex_tokens_dict, remove_self=is_train)

    if len(neighbor_list) > max_neighbors_num:
        sampled_neighbors = random.sample(list(neighbor_list), max_neighbors_num)
    else:
        sampled_neighbors = neighbor_list

    neighbor_token_list = [feature.input_ids]
    neighbor_token_list.extend(ex_tokens_dict[nbr] for nbr in sampled_neighbors)
    while len(neighbor_token_list) < 1 + max_neighbors_num:
        neighbor_token_list.append([pad_id] * max_seq_length)

    return build_graph_metadata_from_token_id_sequences(neighbor_token_list, max_aux, pad_id)

def get_example_neighbor_list(neighbor_set, example_id, ex_tokens_dict, remove_self=True):
    if not neighbor_set:
        return list(neighbor_set)
    if remove_self:
        return list(neighbor_set - {example_id})
    return list(neighbor_set)

def build_graph_metadata_from_token_id_sequences(
    sequences,
    max_aux,
    pad_id
):
    """
    각 sequence는 [head, relation, tail, attr1, val1, attr2, val2, ..., padding...] 형태의 token_id 리스트.
    Masking이 적용되어 있다면 masked token도 그 위치에 해당하는 mask_id로 들어와 있음.

    Returns:
      fact_rel_ids: list[list[int]]     # relation token idxs (including masked if relation was masked)
      fact_ent_ids: list[list[int]]     # entity/value token idxs (masked reflected if applicable)
      fact_entity_roles: list[list[int]]# 0=subject,1=object,2=value
      fact_rel_roles: list[list[int]]   # 0=main-relation,1=attribute
    """
    all_pairs = []
    for seq in sequences:
        pairs = []
        s_id = seq[0]
        r_id = seq[1]
        o_id = seq[2]
        pairs.append((r_id, s_id, 0, 0))  # subject=0, main-rel=0
        pairs.append((r_id, o_id, 1, 0))  # object=1, main-rel=0

        # qualifiers: attr/value pairs
        start = 3
        for i in range(start, start + 2 * max_aux, 2):
            a_id = seq[i]
            v_id = seq[i + 1]
            if a_id == pad_id and v_id == pad_id:
                continue
            if a_id != pad_id and v_id != pad_id:
                pairs.append((a_id, v_id, 2, 1))  # value=2, attribute=1

        all_pairs.append(pairs)

    num_h = len(all_pairs)
    max_p = 2 + max_aux

    fact_rel_ids      = [[-1] * max_p for _ in range(num_h)]
    fact_ent_ids      = [[-1] * max_p for _ in range(num_h)]
    fact_entity_roles = [[-1] * max_p for _ in range(num_h)]
    fact_rel_roles    = [[-1] * max_p for _ in range(num_h)]

    for h, pairs in enumerate(all_pairs):
        for i, (rel_id, ent_id, er, rr) in enumerate(pairs):
            fact_rel_ids[h][i]      = rel_id
            fact_ent_ids[h][i]      = ent_id
            fact_entity_roles[h][i] = er
            fact_rel_roles[h][i]    = rr

    return fact_rel_ids, fact_ent_ids, fact_entity_roles, fact_rel_roles

class MultiDataset(Dataset.Dataset):
    def __init__(self, vocabulary: Vocabulary, examples, max_arity=2, max_seq_length=3, max_per_type=20, max_neighbors_num=15, precomputed_features=None, is_train=True, train_examples=None, dynamic_sampling=False):
        self.examples = examples
        self.vocabulary = vocabulary
        self.max_arity = max_arity
        self.max_seq_length = max_seq_length
        self.max_per_type = max_per_type
        self.max_neighbors_num = max_neighbors_num
        self.is_train = is_train
        self.dynamic_sampling = dynamic_sampling

        if is_train:
            self.token_exs_dict, self.ex_tokens_dict = generate_token_id_dict(self.examples, self.vocabulary, self.max_arity, self.max_seq_length)
        else:
            assert train_examples is not None, "train_examples must be provided for test/eval to avoid leakage"
            self.token_exs_dict, self.ex_tokens_dict = generate_token_id_dict(train_examples, self.vocabulary, self.max_arity, self.max_seq_length)

        if precomputed_features is not None:
            self.features = precomputed_features
        else:
            if dynamic_sampling:
                print("Building feature headers (dynamic neighbor sampling)...")
                self.features = convert_examples_to_feature_headers(
                    examples=self.examples,
                    vocabulary=self.vocabulary,
                    max_arity=self.max_arity,
                    max_seq_length=self.max_seq_length,
                )
            else:
                if is_train:
                    print("Converting training examples to features...")
                    self.features = convert_examples_to_features(
                        examples=self.examples,
                        vocabulary=self.vocabulary,
                        max_arity=self.max_arity,
                        max_seq_length=self.max_seq_length,
                        max_per_type=self.max_per_type,
                        max_neighbors_num=self.max_neighbors_num,
                        is_train=is_train,
                    )
                else:
                    print("Converting test examples to features...")
                    self.features = convert_examples_to_features(
                        examples=self.examples,
                        vocabulary=self.vocabulary,
                        max_arity=self.max_arity,
                        max_seq_length=self.max_seq_length,
                        max_per_type=self.max_per_type,
                        max_neighbors_num=self.max_neighbors_num,
                        is_train=is_train,
                        train_examples=train_examples,
                    )
        self.multidataset = []
        if not self.dynamic_sampling:
            for feature in self.features:
                feature_out = [feature.input_ids] + [feature.input_mask] + \
                    [feature.mask_position] + [feature.mask_label] + [feature.mask_type] + [feature.subgraph_metadata]
                self.multidataset.append(feature_out)

    def __len__(self):
        if self.dynamic_sampling:
            return len(self.features)
        return len(self.multidataset)
    
    def __getitem__(self, index):
        if self.dynamic_sampling:
            feature: NaryFeature = self.features[index]
            subgraph_metadata = resample_subgraph_metadata(
                feature,
                self.vocabulary,
                self.max_arity,
                self.max_seq_length,
                self.max_per_type,
                self.max_neighbors_num,
                self.is_train,
                self.token_exs_dict,
                self.ex_tokens_dict,
            )
            inst = [feature.input_ids, feature.input_mask, feature.mask_position, feature.mask_label, feature.mask_type, subgraph_metadata]
            return prepare_batch_data(inst, self.vocabulary, self.max_arity, self.max_seq_length)
        else:
            x = self.multidataset[index]
            batch_data = prepare_batch_data(x, self.vocabulary, self.max_arity, self.max_seq_length)
            return batch_data
        
def prepare_batch_data(inst, vocabulary: Vocabulary, max_arity, max_seq_length):
    input_ids = torch.tensor(inst[0], dtype=torch.long)
    raw_input_mask = torch.tensor(inst[1], dtype=torch.long)
    mask_position = torch.tensor(inst[2], dtype=torch.long)
    mask_label = torch.tensor(inst[3], dtype=torch.long)
    query_type = torch.tensor(inst[4], dtype=torch.long)
    subgraph_metadata = inst[5]  # [fact_rel_ids, fact_ent_ids, fact_entity_roles, fact_rel_roles]

    input_mask = (raw_input_mask.unsqueeze(0) * raw_input_mask.unsqueeze(1)).bool()
    
    mask_output = np.zeros(len(vocabulary.vocab)).astype("bool")

    if query_type == -1:
        mask_output[2:2+vocabulary.num_relations] = True
    else:
        mask_output[2+vocabulary.num_relations:] = True

    fact_rel_ids, fact_ent_ids, fact_entity_roles, fact_rel_roles = subgraph_metadata
    fact_rel_ids = torch.as_tensor(fact_rel_ids, dtype=torch.long)
    fact_ent_ids = torch.as_tensor(fact_ent_ids, dtype=torch.long)
    fact_entity_roles = torch.as_tensor(fact_entity_roles, dtype=torch.long)
    fact_rel_roles = torch.as_tensor(fact_rel_roles, dtype=torch.long)

    fact_pair_padding = (fact_rel_ids != -1)

    neighbor_graph = {
        "fact_rel_ids": fact_rel_ids,
        "fact_ent_ids": fact_ent_ids,
        "fact_entity_roles": fact_entity_roles,
        "fact_rel_roles": fact_rel_roles,
        "fact_pair_mask": fact_pair_padding,
    }

    return input_ids, input_mask, mask_position, mask_label, mask_output, query_type, neighbor_graph

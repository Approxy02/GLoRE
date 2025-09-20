from scipy.sparse import csr_matrix
import numpy as np
import math
import igraph
from reader import NaryExample
from collections import Counter
import torch


def get_relation_graph(vocabulary, facts, B, use_inverse=False):
    num_ent = vocabulary.num_entities
    num_rel = vocabulary.num_relations

    rows_r, cols_r, data_r = [], [], []
    rows_a, cols_a, data_a = [], [], []

    for fact in facts:
        r_gl = vocabulary.vocab[fact.relation]
        r = r_gl - 2
        h_gl = vocabulary.vocab[fact.head]
        h = h_gl - 2
        t_gl = vocabulary.vocab[fact.tail]
        t = t_gl - 2
        
        rows_r.append(h); cols_r.append(r); data_r.append(1)
        rows_r.append(t); cols_r.append(r); data_r.append(1)

        # qualifier
        if fact.auxiliary_info:
            for att, val_list in fact.auxiliary_info.items():
                a_gl = vocabulary.vocab[att]
                a = a_gl - 2
                rows_a.append(r); cols_a.append(a); data_a.append(1)
                rows_r.append(a); cols_r.append(r); data_r.append(1)
                    
                for val in val_list:
                    v_gl = vocabulary.vocab[val]
                    v = v_gl - 2
                    rows_a.append(v); cols_a.append(a); data_a.append(1)


            att_nodes = [vocabulary.vocab[att] - 2 for att in fact.auxiliary_info.keys()]
            for i in range(len(att_nodes)):
                for j in range(i + 1, len(att_nodes)):
                    ai, aj = att_nodes[i], att_nodes[j]
                    rows_a.append(ai); cols_a.append(aj); data_a.append(1)
                    rows_a.append(aj); cols_a.append(ai); data_a.append(1)

    E_r = csr_matrix((data_r, (rows_r, cols_r)), shape=(num_rel + num_ent, num_rel))
    E_a = csr_matrix((data_a, (rows_a, cols_a)), shape=(num_rel + num_ent, num_rel))

    def affinity(E):
        deg = np.asarray(E.sum(axis=1)).ravel()
        inv = np.zeros_like(deg, dtype=float)       
        nz = deg != 0
        inv[nz] = 1.0 / (deg[nz]**2)
        D_inv = csr_matrix((inv, (np.arange(num_rel + num_ent), np.arange(num_rel + num_ent))), shape=(num_rel + num_ent, num_rel + num_ent))
        return E.T.dot(D_inv).dot(E)

    rel_A = affinity(E_r) + affinity(E_a)

    return rel_A

def get_relation_triplets(G_rel, B):
    rel_triplets = []
    for tup in G_rel.get_edgelist():
        h,t = tup
        tupid = G_rel.get_eid(h,t)
        w = G_rel.es[tupid]["weight"]
        rel_triplets.append((int(h), int(t), float(w)))		
    rel_triplets = np.array(rel_triplets)

    nnz = len(rel_triplets)
    temp = (-rel_triplets[:,2]).argsort()
    weight_ranks = np.empty_like(temp)
    weight_ranks[temp] = np.arange(nnz) + 1

    relation_triplets = []
    for idx,triplet in enumerate(rel_triplets):
        h,t,w = triplet
        rk = int(math.ceil(weight_ranks[idx]/nnz*B))-1
        relation_triplets.append([int(h), int(t), rk])
        assert rk >= 0
        assert rk < B
	
    return np.array(relation_triplets)

def generate_relation_triplets(vocabulary, facts, B, use_inverse=False):
    """
    Generate relation and entity triplets from the vocabulary and facts.
    """
    rel_A = get_relation_graph(vocabulary, facts, use_inverse)
    G_rel = igraph.Graph.Weighted_Adjacency(rel_A)

    relation_triplets = get_relation_triplets(G_rel, B)

    return relation_triplets
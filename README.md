# GLoRE

Official resources of **"Bridging Global and Local Representations via Relation-Entity Pairs in a Hyper-Relational KG"**. **XX** \[[paper]()\]. 

## Overview


## Introduction

This is the [Pytorch](https://pytorch.org/) implementation of GLoRE, a novel framework that bridges Global and Local representations through a Relation-Entity pair perspective.

This repository contains the code and data, as well as the optimal configurations to reproduce the reported results.

## Implementation Details.

All the experiments are conducted on a single 48G NVIDIA RTX 6000 Ada.

## How to Run

### Step 1. Download raw data
We consider three representative n-ary relational datasets, and the datasets can be downloaded from:
- [JF17K](https://www.dropbox.com/sh/ryxohj363ujqhvq/AAAoGzAElmNnhXrWEj16UiUga?dl=0)
- [WikiPeople](https://github.com/gsp2014/WikiPeople)
- [WD50K](https://zenodo.org/record/4036498#.Yx06qoi-uNz)

### Step 2. Preprocess data
Then we convert the raw data into the required format for training and evaluation. The new data is organized into a directory named `data`, with a sub-directory for each dataset. In general, a sub-directory contains:
- `train.json`: train set
- `valid.json`: dev set
- `train+valid.json`: train set + dev set
- `test.json`: test set
- `all.json`: combination of train/dev/test sets, used only for *filtered* evaluation
- `vocab.txt`: vocabulary consisting of entities, relations, and special tokens like [MASK] and [PAD]

> Note: JF17K is the only one that provides no dev set.

### Step 3. Training & Evaluation

To train and evaluate the XXX, please run:

```
python -u ./src/run.py --name [TEST_NAME] --device [GPU_ID] -vocab_size [VOCAB_SIZE] --vocab_file [VOCAB_FILE] \
                       --train_file [TRAIN_FILE] --test_file [TEST_FILE] --ground_truth_file [GROUND_TRUTH_FILE] \
                       --num_workers [NUM_WORKERS] --num_relations [NUM_RELATIONS] \
                       --max_seq_len [MAX_SEQ_LEN] --max_arity [MAX_ARITY]
```

Here you should first create two directories to store the parameters and results of HAHE respectively, then you can set parameters of one dataset according to its statisitcs.
`[TEST_NAME]` is the unique name identifying one Training & Evaluation,  `[GPU_ID]` is the GPU ID you want to use.
`[VOCAB_SIZE]` is the number of vocab of the dataset.
`[VOCAB_FILE]` & `[TRAIN_FILE]` & `[TEST_FILE]` & `[GROUND_TRUTH_FILE]` are the paths storing the vocab file("vocab.txt"), train file("train.json"), test file("test.json") and ground truth file("all.json").
`[NUM_WORKERS]` is the number of workers when reading the data.
`[NUM_RELATIONS]` is the number of relations of the dataset.
`[MAX_ARITY]` is the maximum arity of N-arys in the datast, `[MAX_SEQ_LEN]` is the maximum length of N-ary sequences, which is equal to (2 * [MAX_ARITY] - 1).

Please modify those hyperparametes according to your needs and characteristics of different datasets.


For JF17K, to train and evalute on this dataset using default hyperparametes, please run:

```
python -u ./src/run.py --dataset "jf17k" --device "0" --vocab_size 29148 --vocab_file "./data/jf17k/vocab.txt" --train_file "./data/jf17k/train.json" --test_file "./data/jf17k/test.json" --ground_truth_file "./data/jf17k/all.json" --num_workers 1 --num_relations 501 --max_seq_len 11 --max_arity 6 --hidden_dim 256 --sync_layers 3 --global_activation elu --fusion_layers 1 --trans_layers 6 --local_dropout 0.2 --local_heads 4 --decoder_activation "gelu" --batch_size 1024 --lr 5e-4 --weight_decay 0.002 --entity_soft 0.9 --relation_soft 0.9 --epoch 300 --warmup_proportion 0.2 --num_layer_rel 2 --early_stop 25 --use_relgraph True --use_global True --max_per_type 40 --max_neighbors_num 70 --dynamic_sampling True --use_uncertainty True --global_dropout 0.1
```

For Wikipeople, to train and evalute on this dataset using default hyperparametes, please run:

```
python -u ./src/run.py --dataset "wikipeople" --device "0" --vocab_size 35005 --vocab_file "./data/wikipeople/vocab.txt" --train_file "./data/wikipeople/train+valid.json" --test_file "./data/wikipeople/test.json" --ground_truth_file "./data/wikipeople/all.json" --num_workers 1 --num_relations 178 --max_seq_len 13 --max_arity 7 --hidden_dim 256 --sync_layers 4 --global_activation elu --fusion_layers 1 --trans_layers 6 --local_dropout 0.1 --local_heads 4 --decoder_activation "gelu" --batch_size 1024 --lr 5e-4 --weight_decay 0.01 --entity_soft 0.2 --relation_soft 0.1 --epoch 100 --warmup_proportion 0.4 --num_layer_rel 2 --early_stop 10 --use_relgraph True --use_global True --max_per_type 25 --max_neighbors_num 25 --dynamic_sampling True --use_uncertainty True --global_dropout 0.4
```

For WD50K, to train and evalute on this dataset using default hyperparametes, please run:

```
python -u ./src/run.py --dataset "wd50k" --device "0" --vocab_size 47688 --vocab_file "./data/wd50k/vocab.txt" --train_file "./data/wd50k/train+valid.json" --test_file "./data/wd50k/test.json" --ground_truth_file "./data/wd50k/all.json" --num_workers 1 --num_relations 531 --max_seq_len 19 --max_arity 10 --hidden_dim 256 --sync_layers 4 --global_activation elu --fusion_layers 1 --trans_layers 6 --local_dropout 0.1 --local_heads 4 --decoder_activation "gelu" --batch_size 1024 --lr 5e-4 --weight_decay 0.01 --entity_soft 0.2 --relation_soft 0.1 --epoch 150 --warmup_proportion 0.15 --num_layer_rel 2 --early_stop 25 --use_relgraph True --use_global True --max_per_type 40 --max_neighbors_num 40 --dynamic_sampling True --use_uncertainty True --global_dropout 0.4
```

```
python -u ./src/run.py --dataset "wd50k_33" --device "0" --vocab_size 38599 --vocab_file "./data/wd50k_33/vocab.txt" --train_file "./data/wd50k_33/train+valid.json" --test_file "./data/wd50k_33/test.json" --ground_truth_file "./data/wd50k_33/all.json" --num_workers 1 --num_relations 474 --max_seq_len 19 --max_arity 10 --hidden_dim 256 --sync_layers 4 --global_activation elu --fusion_layers 1 --trans_layers 6 --local_dropout 0.1 --local_heads 4 --decoder_activation "gelu" --batch_size 1024 --lr 5e-4 --weight_decay 0.01 --entity_soft 0.2 --relation_soft 0.1 --epoch 150 --warmup_proportion 0.15 --num_layer_rel 2 --early_stop 25 --use_relgraph True --use_global True --max_per_type 40 --max_neighbors_num 40 --dynamic_sampling True --use_uncertainty True --global_dropout 0.4
```

```
python -u ./src/run.py --dataset "wd50k_66" --device "0" --vocab_size 27751 --vocab_file "./data/wd50k_66/vocab.txt" --train_file "./data/wd50k_66/train+valid.json" --test_file "./data/wd50k_66/test.json" --ground_truth_file "./data/wd50k_66/all.json" --num_workers 1 --num_relations 403 --max_seq_len 19 --max_arity 10 --hidden_dim 256 --sync_layers 4 --global_activation elu --fusion_layers 1 --trans_layers 6 --local_dropout 0.1 --local_heads 4 --decoder_activation "gelu" --batch_size 1024 --lr 5e-4 --weight_decay 0.01 --entity_soft 0.2 --relation_soft 0.1 --epoch 150 --warmup_proportion 0.15 --num_layer_rel 2 --early_stop 25 --use_relgraph True --use_global True --max_per_type 40 --max_neighbors_num 40 --dynamic_sampling True --use_uncertainty True --global_dropout 0.4
```

```
python -u ./src/run.py --dataset "wd50k_100" --device "0" --vocab_size 19071 --vocab_file "./data/wd50k_100/vocab.txt" --train_file "./data/wd50k_100/train+valid.json" --test_file "./data/wd50k_100/test.json" --ground_truth_file "./data/wd50k_100/all.json" --num_workers 1 --num_relations 278 --max_seq_len 19 --max_arity 10 --hidden_dim 256 --sync_layers 4 --global_activation elu --fusion_layers 1 --trans_layers 6 --local_dropout 0.1 --local_heads 4 --decoder_activation "gelu" --batch_size 1024 --lr 5e-4 --weight_decay 0.01 --entity_soft 0.2 --relation_soft 0.1 --epoch 150 --warmup_proportion 0.15 --num_layer_rel 2 --early_stop 25 --use_relgraph True --use_global True --max_per_type 40 --max_neighbors_num 40 --dynamic_sampling True --use_uncertainty True --global_dropout 0.4
```
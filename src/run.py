import argparse
import logging
import torch
import torch.nn as nn
import torch.optim
import os
import sys
import time
import hashlib
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
from reader import *
from evaluation import *
from model import *
from graph import *
import numpy as np
import random
from datetime import datetime

# def set_seed(seed=0):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True

# set_seed(119072237)


def _make_cache_path(prefix, args):
    key = f"{args.dataset}_maxpertype{args.max_per_type}_maxneighbors{args.max_neighbors_num}_vocab{os.path.basename(args.vocab_file)}"
    digest = hashlib.md5(key.encode()).hexdigest()[:8]
    cache_dir = f"./data/{args.dataset}/processed"
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{prefix}_{digest}.pth")

parser = argparse.ArgumentParser()

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

# Dataset information
parser.add_argument("--dataset", type=str, default="wd50k") #"jf17k", 
parser.add_argument("--vocab_size", type=int, default=47688) #29148
parser.add_argument("--vocab_file", type=str, default="./data/wd50k/vocab.txt") #"./data/jf17k/vocab.txt"
parser.add_argument("--train_file", type=str, default="./data/wd50k/train+valid.json") #"./data/jf17k/train+valid.json"
parser.add_argument("--test_file", type=str, default="./data/wd50k/test.json") #"./data/jf17k/test.json"
parser.add_argument("--ground_truth_file", type=str, default="./data/wd50k/all.json") #"./data/jf17k/all.json"
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--num_relations", type=int, default=531) #501
parser.add_argument("--max_seq_len", type=int, default=63) #11
parser.add_argument("--max_arity", type=int, default=32) #6

# Hyperparameters
parser.add_argument("--device", type=str, default="0123") # {0123}^n,1<=n<=4,the first cuda is used as master device and others are used for data parallel
parser.add_argument("--batch_size", type=int, default=1024) # 1024
parser.add_argument("--lr", type=float, default=5e-4) # 5e-4
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--hidden_dim", type=int, default=256)
parser.add_argument("--fusion_layers", type=int, default=12) # 12
parser.add_argument("--trans_layers", type=int, default=4) # 4
parser.add_argument("--global_dropout", type=float, default=0.1) # 0.1
parser.add_argument("--local_dropout", type=float, default=0.1) # 0.2
parser.add_argument("--decoder_activation", type=str, default="gelu")
parser.add_argument("--local_heads", type=int, default=4) # 4
parser.add_argument("--entity_soft", type=float, default=0.2) # 0.9
parser.add_argument("--relation_soft", type=float, default=0.1) # 0.9
parser.add_argument("--use_uncertainty", type=str2bool, default=True, help="use uncertainty-based weighting (True/False)")
parser.add_argument("--remove_mask", type=str2bool, default=False) # wheather to use extra mask
parser.add_argument("--max_per_type", type=int, default=40) # 40
parser.add_argument("--max_neighbors_num", type=int, default=40) # 70
parser.add_argument("--sync_layers", type=int, default=2) # 2
parser.add_argument("--global_activation", type=str, default="elu") # elu
parser.add_argument("--num_bin", type=int, default=10) # 10
parser.add_argument("--dim_rel", type=int, default=256)
parser.add_argument("--hid_dim_ratio_rel", type=int, default=2) # 2
parser.add_argument("--num_layer_rel", type=int, default=2) # 2
parser.add_argument("--num_head", type=int, default=8) # 8

# ablation study
parser.add_argument("--use_relgraph", type=str2bool, default=True) # whether to use relation graph
parser.add_argument("--use_global", type=str2bool, default=True) # whether to use global context
parser.add_argument("--dynamic_sampling", type=str2bool, default=False, help="whether to use dynamic sampling")

# others for training
parser.add_argument("--epoch", type=int, default=300) # 200
parser.add_argument("--warmup_proportion", type=float, default=0.2)
parser.add_argument("--early_stop", type=int, default=300, help="early stopping patience, 안하려면 300으로") # 5

# directory position settings
parser.add_argument("--result_save_dir", type=str, default="results")
parser.add_argument("--ckpt_save_dir", type=str, default="ckpts")

args = parser.parse_args()

args.num_entities = args.vocab_size - args.num_relations - 2
if not os.path.exists(args.result_save_dir):
    os.mkdir(args.result_save_dir)
if not os.path.exists(args.ckpt_save_dir):
    os.mkdir(args.ckpt_save_dir)
dir_name = os.path.join(args.result_save_dir,args.dataset)
if not os.path.exists(dir_name):
    os.mkdir(dir_name)

date_str = datetime.now().strftime("%Y%m%d")
time_str = datetime.now().strftime("%H%M%S")
dir_date = os.path.join(dir_name, date_str)
if not os.path.exists(dir_date):
    os.makedirs(dir_date, exist_ok=True)
log_path = os.path.join(dir_date, f"train_{time_str}.log")

logging.basicConfig(
    format='%(asctime)s  %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    filename=log_path,
    filemode="w",
    level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))

def main(limit=1e9):
    device = torch.device(f"cuda:{args.device[0]}")
    devices = []
    for i in range(len(args.device)):
        devices.append(torch.device(f"cuda:{args.device[i]}"))
    
    vocabulary = Vocabulary(
        vocab_file=args.vocab_file,
        num_relations=args.num_relations,
        num_entities=args.num_entities)
    
    train_examples, _ = read_examples(args.train_file, args.max_arity)
    test_examples, _ = read_examples(args.test_file, args.max_arity)

    train_cache = _make_cache_path("train_dataset", args)
    test_cache = _make_cache_path("test_dataset", args)

    logger.info(f"--------------- Vocabulary info ----------------")
    logger.info(f"Total vocab size: {len(vocabulary.vocab)}")
    logger.info(f"Number of relations: {vocabulary.num_relations}")
    logger.info(f"Number of entities: {vocabulary.num_entities}")

    logger.info('------------------------------------------------')
    logger.info(f"---------Train/Test Data Preprocessing ---------")

    if args.dynamic_sampling:
        logger.info("Dynamic sampling is ON → skip training dataset cache (build headers only and resample per step)")
        train_dataset = MultiDataset(
            vocabulary, train_examples, args.max_arity, args.max_seq_len, args.max_per_type, args.max_neighbors_num,
            is_train=True, dynamic_sampling=True
        )
        if os.path.exists(test_cache):
            loaded = torch.load(test_cache, map_location="cpu", weights_only=False)
            test_dataset = MultiDataset(
                vocabulary, test_examples, args.max_arity, args.max_seq_len, args.max_per_type, args.max_neighbors_num,
                precomputed_features=loaded["features"], is_train=False, train_examples=train_examples, dynamic_sampling=False
            )
            logger.info(f"Loaded cached test dataset from {test_cache}")
        else:
            test_dataset = MultiDataset(
                vocabulary, test_examples, args.max_arity, args.max_seq_len, args.max_per_type, args.max_neighbors_num,
                is_train=False, train_examples=train_examples, dynamic_sampling=False
            )
            torch.save({"features": test_dataset.features}, test_cache)
            logger.info(f"Saved test dataset to cache {test_cache}")
    else:
        if os.path.exists(train_cache):
            loaded = torch.load(train_cache, map_location="cpu", weights_only=False)
            train_dataset = MultiDataset(
                vocabulary, train_examples, args.max_arity, args.max_seq_len, args.max_per_type, args.max_neighbors_num,
                precomputed_features=loaded["features"], is_train=True, dynamic_sampling=False
            )
            logger.info(f"Loaded cached train dataset from {train_cache}")
        else:
            train_dataset = MultiDataset(
                vocabulary, train_examples, args.max_arity, args.max_seq_len, args.max_per_type, args.max_neighbors_num,
                is_train=True, dynamic_sampling=False
            )
            torch.save({"features": train_dataset.features}, train_cache)
            logger.info(f"Saved train dataset to cache {train_cache}")

        if os.path.exists(test_cache):
            loaded = torch.load(test_cache, map_location="cpu", weights_only=False)
            test_dataset = MultiDataset(
                vocabulary, test_examples, args.max_arity, args.max_seq_len, args.max_per_type, args.max_neighbors_num,
                precomputed_features=loaded["features"], is_train=False, train_examples=train_examples, dynamic_sampling=False
            )
            logger.info(f"Loaded cached test dataset from {test_cache}")
        else:
            test_dataset = MultiDataset(
                vocabulary, test_examples, args.max_arity, args.max_seq_len, args.max_per_type, args.max_neighbors_num,
                is_train=False, train_examples=train_examples, dynamic_sampling=False
            )
            torch.save({"features": test_dataset.features}, test_cache)
            logger.info(f"Saved test dataset to cache {test_cache}")
    logger.info('------------------------------------------------')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)

    max_train_steps = args.epoch * len(train_loader)

    rel_triplets = None
    
    if args.use_relgraph:
        rel_triplets = generate_relation_triplets(vocabulary, train_examples, args.num_bin)
        rel_triplets = torch.tensor(rel_triplets, dtype=torch.long, device=device)

    logger.info(f'rel_triplets shape: {rel_triplets.shape if rel_triplets is not None else "None"}')
    logger.info('------------------------------------------------')

    if len(args.device) > 1:
        model = torch.nn.DataParallel(Model(len(vocabulary.vocab), vocabulary.num_relations, args.num_entities, args.max_arity, args.sync_layers, args.fusion_layers,
                                            args.trans_layers, args.hidden_dim, args.local_heads, args.global_dropout, args.local_dropout, args.global_activation, args.decoder_activation,
                                            args.num_head, args.remove_mask, args.dim_rel, args.hid_dim_ratio_rel,
                                            args.num_bin, args.num_layer_rel, args.use_relgraph, args.use_global, rel_triplets), device_ids=devices)
        model.to(device)
    else:
        model = Model(len(vocabulary.vocab), vocabulary.num_relations, args.num_entities, args.max_arity, args.sync_layers, args.fusion_layers,
                      args.trans_layers, args.hidden_dim, args.local_heads, args.global_dropout, args.local_dropout, args.global_activation, args.decoder_activation,
                      args.num_head, args.remove_mask, args.dim_rel, args.hid_dim_ratio_rel,
                      args.num_bin, args.num_layer_rel, args.use_relgraph, args.use_global, rel_triplets).to(device)

    if args.use_uncertainty:
        log_var_entity = torch.nn.Parameter(torch.zeros(1, device=device), requires_grad=True)
        log_var_relation = torch.nn.Parameter(torch.zeros(1, device=device), requires_grad=True)
        optimizer = torch.optim.AdamW([
            {"params": model.parameters(), "weight_decay": args.weight_decay},
            {"params": [log_var_entity, log_var_relation], "weight_decay": 0.0},
        ])
    else:
        log_var_entity = None
        log_var_relation = None
        optimizer = torch.optim.AdamW([
            {"params": model.parameters(), "weight_decay": args.weight_decay},
        ])
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, total_steps=max_train_steps, 
                            pct_start=args.warmup_proportion, anneal_strategy="linear", cycle_momentum=False)
    limit = min(args.epoch, limit)

    dataset_ckpt_dir = os.path.join(args.ckpt_save_dir, args.dataset, date_str)
    os.makedirs(dataset_ckpt_dir, exist_ok=True)

    best_mrr = 0.0
    best_epoch = 0
    best_ckpt_path = os.path.join(
                    dataset_ckpt_dir,
                    f"{time_str}_" + ".ckpt"
                )
    training_time = int(0)
    early_stop = 0

    logger.info('------------------------------------------------')

    for epoch in range(limit):
        time_start_epoch = time.time()
        epoch_loss = 0.0
        entity_loss_sum = 0.0
        relation_loss_sum = 0.0
        num_entity = 0
        num_relation = 0

        for item in tqdm(train_loader):
            model.train()
            input_ids, input_mask, mask_position, mask_label, mask_output, query_type, neighbor_graph = item

            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            mask_position = mask_position.to(device)
            mask_label = mask_label.to(device)
            mask_output = mask_output.to(device)
            # Assertion for mask_output after moving to device
            assert mask_output.any(dim=1).all().item(), "Found a sample with zero allowed candidates."
            query_type = query_type.to(device)
            neighbor_graph = {k: v.to(device) for k, v in neighbor_graph.items()}

            result = model(input_ids, input_mask, mask_position, mask_output, neighbor_graph)

            entities = (query_type == 1)
            relations = (query_type == -1)

            if entities.any().item():
                mask_e   = mask_output[entities]
                labels_e = mask_label[entities]

                K_e = mask_e.sum(dim=1, keepdim=True).float()
                eps_e = args.entity_soft
                
                eps_vec_e = torch.full_like(K_e, eps_e)
                eps_vec_e = torch.where(K_e > 1, eps_vec_e, torch.zeros_like(eps_vec_e))

                base_e   = (eps_vec_e / torch.clamp(K_e - 1, min=1.0)) * mask_e.float()
                target_e = base_e
                rows_e   = torch.arange(labels_e.size(0), device=device)
                target_e[rows_e, labels_e] = 1.0 - eps_vec_e.squeeze(1)

                loss1_vec  = nn.functional.cross_entropy(result[entities], target_e, reduction='none')
                entity_loss_sum += loss1_vec.sum().item()
                num_entity += loss1_vec.numel()
                loss1_mean = loss1_vec.mean()
            else:
                loss1_vec  = torch.tensor([], device=device)
                loss1_mean = torch.tensor(0.0, device=device)


            if relations.any().item():
                mask_r   = mask_output[relations]
                labels_r = mask_label[relations]

                K_r = mask_r.sum(dim=1, keepdim=True).float()
                eps_r = args.relation_soft
                eps_vec_r = torch.full_like(K_r, eps_r)
                eps_vec_r = torch.where(K_r > 1, eps_vec_r, torch.zeros_like(eps_vec_r))

                base_r   = (eps_vec_r / torch.clamp(K_r - 1, min=1.0)) * mask_r.float()
                target_r = base_r
                rows_r   = torch.arange(labels_r.size(0), device=device)
                target_r[rows_r, labels_r] = 1.0 - eps_vec_r.squeeze(1)
                loss2_vec  = nn.functional.cross_entropy(result[relations], target_r, reduction='none')
                relation_loss_sum += loss2_vec.sum().item()
                num_relation += loss2_vec.numel()
                loss2_mean = loss2_vec.mean()
            else:
                loss2_vec  = torch.tensor([], device=device)
                loss2_mean = torch.tensor(0.0, device=device)

            if args.use_uncertainty:
                loss = (
                    torch.exp(-log_var_entity) * loss1_mean + 0.5 * log_var_entity + # type: ignore
                    torch.exp(-log_var_relation) * loss2_mean + 0.5 * log_var_relation # type: ignore
                )
            else:
                loss = 0.5 * loss1_mean + 0.5 * loss2_mean

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        avg_entity_loss = entity_loss_sum / max(num_entity, 1)
        avg_relation_loss = relation_loss_sum / max(num_relation, 1)
        time_end_epoch = time.time()
        training_time += round(time_end_epoch - time_start_epoch)
        hours, minutes, seconds = calculate_training_time(training_time)

        logger.info(f"epoch: {epoch}\tlr: {scheduler.get_last_lr()[0]:.6f}\ttrain time: {hours:02d}:{minutes:02d}:{seconds:02d}\tloss: {avg_loss:.4f}")
        logger.info(f"\t\tloss details\tentity loss: {avg_entity_loss:.4f}\trelation loss: {avg_relation_loss:.4f}")

        if args.use_uncertainty:
            logger.info(f"\t\tlog_var_entity: {log_var_entity.item():.4f}\tlog_var_relation: {log_var_relation.item():.4f}") # type: ignore
        else:
            logger.info("\t\tlog_var_entity: N/A\tlog_var_relation: N/A")

        if epoch % 5 == 0 or epoch == limit - 1:
            eval_performance = predict(
                model=model,
                test_loader=test_loader,
                all_features=test_dataset.features,
                vocabulary=vocabulary,
                device=device)
            show_perforamance(eval_performance)

            current_mrr = eval_performance['entity']['mrr']
            if current_mrr > best_mrr:
                early_stop = 0
                logger.info(f"New best model found at epoch {epoch} with MRR {current_mrr:.4f}")
                best_mrr = current_mrr
                best_epoch = epoch
                torch.save(model.state_dict(), best_ckpt_path)
            else:
                early_stop += 1
                logger.info(f"Early stopping counter: {early_stop}")
            
            if args.early_stop != 0 and early_stop >= args.early_stop:
                logger.info(f"Early stopping at epoch {epoch} with best MRR {best_mrr:.4f} at epoch {best_epoch}")
                break

    f_hours, f_minutes, f_seconds = calculate_training_time(training_time)

    logger.info(f"Best model found at epoch {best_epoch} with MRR {best_mrr:.4f}\ttotal train time: {f_hours:02d}:{f_minutes:02d}:{f_seconds:02d}")

    model.load_state_dict(torch.load(best_ckpt_path, weights_only=True))
   
def calculate_training_time(training_time: int):
    minutes, seconds = divmod(training_time, 60)
    hours, minutes = divmod(minutes, 60)
    return hours, minutes, seconds

def predict(model, test_loader, all_features, vocabulary, device):
    eval_result_file = os.path.join(dir_name, "eval_result.json")

    gt_dict = generate_ground_truth(
        ground_truth_path=args.ground_truth_file,
        vocabulary=vocabulary,
        max_arity=args.max_arity,
        max_seq_length=args.max_seq_len)

    step = 0
    global_idx = 0
    ent_lst = []
    rel_lst = []
    _2_r_lst = []
    _2_ht_lst = []
    _n_r_lst = []
    _n_ht_lst = []
    _n_a_lst = []
    _n_v_lst = []
    model.eval()
    with torch.no_grad():
        for item in tqdm(test_loader):
            input_ids, input_mask, mask_position, mask_label, mask_output, query_type, neighbor_graph = item

            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            mask_position = mask_position.to(device)
            mask_label = mask_label.to(device)
            mask_output = mask_output.to(device)
            query_type = query_type.to(device)
            neighbor_graph = {k: v.to(device) for k, v in neighbor_graph.items()}

            output = model(input_ids, input_mask, mask_position, mask_output, neighbor_graph)

            batch_results = output.cpu().numpy()
            ent_ranks, rel_ranks, _2_r_ranks, _2_ht_ranks, \
            _n_r_ranks, _n_ht_ranks, _n_a_ranks, _n_v_ranks = batch_evaluation(global_idx, batch_results, all_features, gt_dict)
            ent_lst.extend(ent_ranks)
            rel_lst.extend(rel_ranks)
            _2_r_lst.extend(_2_r_ranks)
            _2_ht_lst.extend(_2_ht_ranks)
            _n_r_lst.extend(_n_r_ranks)
            _n_ht_lst.extend(_n_ht_ranks)
            _n_a_lst.extend(_n_a_ranks)
            _n_v_lst.extend(_n_v_ranks)
            step += 1
            global_idx += output.size(0)

    eval_result = compute_metrics(
        ent_lst=ent_lst,
        rel_lst=rel_lst,
        _2_r_lst=_2_r_lst,
        _2_ht_lst=_2_ht_lst,
        _n_r_lst=_n_r_lst,
        _n_ht_lst=_n_ht_lst,
        _n_a_lst=_n_a_lst,
        _n_v_lst=_n_v_lst,
        eval_result_file=eval_result_file
    )

    return eval_result


def show_perforamance(eval_performance):
    def pad(x):
        return x + (10 - len(x)) * ' '
    all_entity = f"{pad('ENTITY')}\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f" % (
        eval_performance['entity']['mrr'],
        eval_performance['entity']['hits1'],
        eval_performance['entity']['hits3'],
        eval_performance['entity']['hits5'],
        eval_performance['entity']['hits10'])

    all_relation = f"{pad('RELATION')}\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f" % (
        eval_performance['relation']['mrr'],
        eval_performance['relation']['hits1'],
        eval_performance['relation']['hits3'],
        eval_performance['relation']['hits5'],
        eval_performance['relation']['hits10'])

    all_ht = f"{pad('HEAD/TAIL')}\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f" % (
        eval_performance['ht']['mrr'],
        eval_performance['ht']['hits1'],
        eval_performance['ht']['hits3'],
        eval_performance['ht']['hits5'],
        eval_performance['ht']['hits10'])

    all_r = f"{pad('PRIMARY_R')}\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f" % (
        eval_performance['r']['mrr'],
        eval_performance['r']['hits1'],
        eval_performance['r']['hits3'],
        eval_performance['r']['hits5'],
        eval_performance['r']['hits10'])

    logger.info("\n-------- Evaluation Performance --------\n%s\n%s\n%s\n%s\n%s" % (
        "\t".join([pad("TASK"), "  MRR  ", "Hits@1", "Hits@3", "Hits@5", "Hits@10"]),
        all_ht, all_r, all_entity, all_relation))

if __name__ == '__main__':
    logger.info('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).items()):
        logger.info('%s: %s' % (arg, value))
    logger.info('------------------------------------------------')
    main()
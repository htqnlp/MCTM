import argparse
from functools import reduce
import importlib
from typing import Any
from warnings import catch_warnings

import torch
import torch.nn as nn
from texar.torch.run import *
from pathlib import Path
import misc_utils
from misc_utils import init_logger, logger

import texar.torch as tx

from model import SWCC
import data_utils
import numpy as np
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances, \
    nan_euclidean_distances

import pickle
from tqdm import tqdm
import csv

parser = argparse.ArgumentParser()
parser.add_argument(
    '--config-model', type=str, default="config_model",
    help="The model config.")
parser.add_argument(
    '--config-data', type=str, default="config_data",
    help="The dataset config.")
parser.add_argument(
    "--do-train", action="store_true", help="Whether to run training.")
parser.add_argument(
    "--do-eval", action="store_true",
    help="Whether to run eval on the dev set.")
parser.add_argument(
    "--output-dir", type=str, default="./outputs1/",
    help="Path to save the trained model and logs.")
parser.add_argument(
    "--log-file", type=str, default="exp.log",
    help="Path to save the trained model and logs.")

parser.add_argument(
    '--checkpoint', type=str, default=None,
    help="Model checkpoint to load model weights from.")
args = parser.parse_args()

config_model: Any = importlib.import_module(args.config_model)
config_data: Any = importlib.import_module(args.config_data)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

make_deterministic(config_model.random_seed)

train_data = data_utils.TrainData(config_data.train_hparams, device=device)
model = SWCC(config_model=config_model, config_data=config_data)

if args.checkpoint:
    print(f"loading checkpoint {args.checkpoint}...")
    model.load_state_dict(torch.load(args.checkpoint)['model'])
model.to(device)

from test_1 import mcnc_data

tokenizer = tx.data.BERTTokenizer(pretrained_model_name="bert-base-uncased")
evt_chain = mcnc_data()


def get_events(raw_list):
    all_events = []
    for ex in raw_list:
        event = []
        selected_id = [3, 0, 4, 5]
        for i in selected_id:
            if ex[i] is not None:
                event.append(ex[i].replace('+', ' '))

        all_events.append(event)
    return all_events


def convert_events_to_embeddings(events):
    events = [data_utils.map_evt_to_tokens_for_text(evt) for evt in events]
    evt_ids = [tokenizer.map_text_to_id(evt) for evt in events]
    evt_ids, evt_lengths = tx.data.padded_batch(
        evt_ids, pad_value=tokenizer.map_token_to_id(tokenizer.pad_token))

    evt_ids = torch.from_numpy(evt_ids).to(device)
    evt_lengths = torch.tensor(evt_lengths).to(device)
    # evt_emb = model.encoder_q.get_glove_embedding(evt_ids,evt_lengths)
    evt_emb = model.encoder_q(evt_ids, evt_lengths)
    return evt_emb


def map_evt_to_tokens(evt):
    # evt ='<cls> ' + ' <sep> '.join(evt) + ' <sep>'
    evt = [evt[1], evt[0], evt[2]]
    evt = ' '.join(evt)
    evt = evt.replace("''", "").lower().split()
    evt = ' '.join([tokenizer.cls_token] + evt + [tokenizer.sep_token])
    return evt


# evt_chain_holdout = []
# evt_chain_gold = []
# evt_chain_candi = []
# for j in range(len(evt_chain)):
#     evt_chain_holdout = evt_chain[j]['HOLDOUT']
#     evt_chain_gold = evt_chain[j]['NEG']
#     # evt_chain_candi = evt_chain[j]['DOC'] + evt_chain_gold
#     evt_chain_candi = evt_chain_gold + evt_chain[j]['DOC']
#     # yield evt_chain_holdout, evt_chain_gold, evt_chain_candi

model.eval()

# evt_chain_holdout_split = []
# evt_chain_holdout_ids = []
# for line in evt_chain_holdout:
#     line = line.strip().split('|')
#     line = map_evt_to_tokens(line)
#     evt_chain_holdout_split.append(line)
#     evt_chain_holdout_ids.append(tokenizer.map_text_to_id(line))
#
# evt_chain_candi_split = []
# evt_chain_candi_ids = []
# for line in evt_chain_candi:
#     line = line.strip().split('|')
#     line = map_evt_to_tokens(line)
#     evt_chain_candi_split.append(line)
#     evt_chain_candi_ids.append(tokenizer.map_text_to_id(line))
score_ = []
for example in tqdm(evt_chain):
    evt_chain_holdout = example['HOLDOUT']
    # evt_chain_candi = example['DOC'] + example['NEG']
    evt_chain_candi = example['NEG'] + example['DOC']

    evt_chain_holdout_split = []
    evt_chain_holdout_ids = []
    for line in evt_chain_holdout:
        line = line.strip().split('|')
        line = map_evt_to_tokens(line)
        evt_chain_holdout_split.append(line)
        evt_chain_holdout_ids.append(tokenizer.map_text_to_id(line))

    evt_chain_candi_split = []
    evt_chain_candi_ids = []
    for line in evt_chain_candi:
        line = line.strip().split('|')
        line = map_evt_to_tokens(line)
        evt_chain_candi_split.append(line)
        evt_chain_candi_ids.append(tokenizer.map_text_to_id(line))

    evt_chain_holdout_ids, evt_chain_holdout_ids_lengths = tx.data.padded_batch(
        evt_chain_holdout_ids, pad_value=tokenizer.map_token_to_id(tokenizer.pad_token))
    evt_chain_holdout_ids = torch.from_numpy(evt_chain_holdout_ids).to(device)
    evt_chain_holdout_ids_lengths = torch.tensor(evt_chain_holdout_ids_lengths).to(device)
    evt_chain_holdout_emb = model.encoder_q(evt_chain_holdout_ids, evt_chain_holdout_ids_lengths)

    # for i in range(len(evt_chain_candi_ids)):
    #     example1 = torch.tensor(evt_chain_candi_ids[i])
    #     example1 = example1.unsqueeze(1)
    #     example1_lengths = example1.shape
    #     example1_emb = model.encoder_q(example1, example1_lengths)
    #
    #     print()

    evt_chain_candi_ids, evt_chain_candi_ids_lengths = tx.data.padded_batch(
        evt_chain_candi_ids, pad_value=tokenizer.map_token_to_id(tokenizer.pad_token))
    evt_chain_candi_ids = torch.from_numpy(evt_chain_candi_ids).to(device)
    evt_chain_candi_ids_lengths = torch.tensor(evt_chain_candi_ids_lengths).to(device)
    evt_chain_candi_ids_emb = model.encoder_q(evt_chain_candi_ids, evt_chain_candi_ids_lengths)

    score = cosine_similarity(evt_chain_holdout_emb.tolist(), evt_chain_candi_ids_emb.tolist())
    value = np.sum(score, axis=0)
    value0 = max(value)
    if value[0] == max(value):
        score_.append(1)
    else:
        score_.append(0)
    find = []
    for i in range(len(score_)):
        if score_[i] == 1:
            find.append(1)
    final = len(find) / len(score_)

print(final)

from logging import BASIC_FORMAT
from typing import List, Optional, Tuple, Iterator, Dict

import numpy as np
import torch

import texar.torch as tx
from texar.torch.hyperparams import HParams
import random
import json
from tqdm import tqdm
from pathlib import Path
import pickle
from typing import Any
import argparse
import importlib

# from nltk import WordNetLemmatizer
Example = Tuple[np.ndarray, np.ndarray]

tokenizer = tx.data.BERTTokenizer(pretrained_model_name="bert-base-uncased")
pad_token_id = tokenizer.map_token_to_id(tokenizer.pad_token)


def freq_norm(val):
    max_val = 38354
    min_val = 3
    return (val - min_val) / (max_val - min_val)


# iter 需要迭代
class SpanData0Source(tx.data.TextLineDataSource):

    def __iter__(self) -> Iterator[List[str]]:
        path = Path(self._file_paths[0])

        with self._open_file(path) as f:
            for line in f:
                ex = json.loads(line.strip())
                evt_q = ex['evt_q'].split('\t')
                evt_q_sub = evt_q[0]
                evt_q_ver = evt_q[1]
                evt_q_obj = evt_q[2]
                print(evt_q_ver)

                sampled_evt = random.choice(ex['evt_k'])
                sampled_evt_str = sampled_evt[0]
                sampled_evt_freq = freq_norm(sampled_evt[1])
                sampled_evt_list = sampled_evt_str.split('\t')

                sampled_evt_sub = sampled_evt_list[0]
                sampled_evt_ver = sampled_evt_list[1]
                sampled_evt_obj = sampled_evt_list[2]

                yield evt_q, evt_q_sub, evt_q_ver, evt_q_obj, sampled_evt, sampled_evt_sub, sampled_evt_ver, sampled_evt_obj, sampled_evt_freq


class SpanData0(tx.data.DatasetBase[Example, Example]):

    # transfer hparams and device
    def __init__(self, hparams=None, device: Optional[torch.device] = None):
        self._hparams = HParams(hparams, self.default_hparams())
        data_source = SpanData0Source(
            self._hparams.dataset.files,
            compression_type=self._hparams.dataset.compression_type)
        # self._vocab = Vocab(self._hparams.dataset.vocab_file)
        self.vocab_size = tokenizer.vocab_size

        super().__init__(data_source, hparams, device=device)

    @staticmethod
    # default_hparams() includes allowed hyperparameters and their default values
    def default_hparams():
        return {
            **tx.data.DatasetBase.default_hparams(),
            'dataset': {
                'files': 'data.txt',
                'compression_type': None,
                'vocab_file': 'vocab.txt'
            },
        }

    def process(self, raw_example):
        evt_q = raw_example[0]
        evt_q_sub = raw_example[1]
        evt_q_ver = raw_example[2]
        evt_q_obj = raw_example[3]
        sampled_evt = raw_example[4]
        sampled_evt_sub = raw_example[5]
        sampled_evt_ver = raw_example[6]
        sampled_evt_obj = raw_example[7]
        sampled_evt_freq = raw_example[8]

        return {
            "evt_q": evt_q,
            "evt_q_sub": evt_q_sub,
            "evt_q_ver": evt_q_ver,
            "evt_q_obj": evt_q_obj,
            "sampled_evt": sampled_evt,
            "sampled_evt_sub": sampled_evt_sub,
            "sampled_evt_ver": sampled_evt_ver,
            "sampled_evt_obj": sampled_evt_obj,
            "sampled_evt_freq": sampled_evt_freq,
        }

    # examples is a list of objects returned from the 'process' method. These data should be collated into a batch
    # examples: A list of processed examples in a batch.
    # collate method must be implemented
    def collate(self, examples: List[Example]) -> tx.data.Batch:
        # evt_q = [ex["evt_q"] for ex in examples]
        # Pad a batch of integer lists (or numpy arrays) to the same length, and stack them together.
        evt_q_ids, evt_q_lengths = tx.data.padded_batch(
            [ex["evt_q_ids"] for ex in examples], pad_value=pad_token_id)

        evt_k = [ex["evt_k"] for ex in examples]
        evt_k_ids, evt_k_lengths = tx.data.padded_batch(
            [ex["evt_k_ids"] for ex in examples], pad_value=pad_token_id)

        evt_p_ids, evt_p_lengths = tx.data.padded_batch(
            [ex["evt_p_ids"] for ex in examples], pad_value=pad_token_id)

        evt_freq = [ex["evt_freq"] for ex in examples]
        mask_id = [ex["mask_id"] for ex in examples]
        mask_pos = [ex["mask_pos"] for ex in examples]

        return tx.data.Batch(
            len(examples),
            # evt_q=evt_q,
            evt_q_ids=torch.from_numpy(evt_q_ids),
            evt_q_lengths=torch.tensor(evt_q_lengths),
            evt_k=evt_k,
            evt_k_ids=torch.from_numpy(evt_k_ids),
            evt_k_lengths=torch.tensor(evt_k_lengths),
            evt_p_ids=torch.from_numpy(evt_p_ids),
            evt_p_lengths=torch.tensor(evt_p_lengths),
            evt_freq=torch.tensor(evt_freq),
            mask_id=torch.tensor(mask_id),
            mask_pos=torch.tensor(mask_pos),
        )

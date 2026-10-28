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

# from nltk import WordNetLemmatizer
Example = Tuple[np.ndarray, np.ndarray]

tokenizer = tx.data.BERTTokenizer(pretrained_model_name="bert-base-uncased")
pad_token_id = tokenizer.map_token_to_id(tokenizer.pad_token)


def freq_norm(val):
    max_val = 38354
    min_val = 3
    return (val - min_val) / (max_val - min_val)


def map_evt_to_tokens(evt):
    # evt ='<cls> ' + ' <sep> '.join(evt) + ' <sep>'
    evt = [evt[1], evt[0], evt[2]]
    evt = ' '.join(evt)
    evt = evt.replace("''", "").lower().split()
    evt = ' '.join([tokenizer.cls_token] + evt + [tokenizer.sep_token])
    return evt


class SpanDataSource(tx.data.TextLineDataSource):

    def __iter__(self) -> Iterator[List[str]]:
        path = Path(self._file_paths[0])

        with self._open_file(path) as f:
            for line in f:
                # I think evt_q is represented as like central event and evt_k is like some similar semantic events..
                ex = json.loads(line.strip())
                evt_q = ex['evt_q'].split('\t')
                evt_q_sub = evt_q[0]
                evt_q_ver = evt_q[1]
                evt_q_obj = evt_q[2]
                evt_q_ver_and_obj = evt_q_ver + ' ' + evt_q_obj

                evt_k_sub = []
                evt_k_ver = []
                evt_k_obj = []
                evt_k_ver_and_obj = []
                evt_k = ex['evt_k']
                for i in evt_k:
                    evt_k_0 = i[0]
                    evt_k_0_freq = i[1]
                    evt_k_0_freq_norm = freq_norm(evt_k_0_freq)
                    evt_k_0_list = evt_k_0.split('\t')
                    if len(evt_k_0_list) == 3:

                        evt_k_0_sub = evt_k_0_list[0]
                        evt_k_0_ver = evt_k_0_list[1]
                        evt_k_0_obj = evt_k_0_list[2]
                        evt_k_0_ver_and_obj = evt_k_0_ver + ' ' + evt_k_0_obj

                        evt_k_sub.append(evt_k_0_sub)
                        evt_k_ver.append(evt_k_0_ver)
                        evt_k_obj.append(evt_k_0_obj)
                        evt_k_ver_and_obj.append([evt_k_0_ver_and_obj, evt_k_0_freq_norm])
                    else:
                        continue

                evt_k_ver_and_obj_sampled_0 = random.choice(evt_k_ver_and_obj)
                evt_k_ver_and_obj_sampled = evt_k_ver_and_obj_sampled_0[0]
                evt_k_freq = freq_norm(evt_k_ver_and_obj_sampled_0[1])
                # print()
                # print()

                # yield
                yield evt_q, evt_q_ver_and_obj, evt_k_ver_and_obj_sampled, evt_k_freq


class SpanData(tx.data.DatasetBase[Example, Example]):

    # transfer hparams and device
    def __init__(self, hparams=None, device: Optional[torch.device] = None):  # introduct hparams and device
        # introduct new hparams to default_hparams
        self._hparams = HParams(hparams, self.default_hparams())
        # data_source has been defined here
        # the transfered mention is hparams 's datafiles(the train dataset 's position) and compression_type
        data_source = SpanDataSource(
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

    # raw_example is a data sample read from self.source
    # and in this case,a line of tokenized text, represented as a list of 'str'
    def process(self, raw):
        evt_q, evt_q_ver_and_obj, evt_k_ver_and_obj_sampled, evt_k_freq = raw[0], raw[
            1], raw[2], raw[3]

        evt_k_ver_and_obj_sampled_0 = evt_k_ver_and_obj_sampled
        # evt_k_ver_and_obj_freq = freq_norm(evt_k_ver_and_obj_sampled[1])

        evt_k_ver_and_obj_sampled_0 = evt_k_ver_and_obj_sampled_0.replace("''", "").lower().split()
        evt_k_ver_and_obj_sampled_0 = ' '.join(
            [tokenizer.cls_token] + evt_k_ver_and_obj_sampled_0 + [tokenizer.sep_token])

        evt_k_ver_and_obj_sampled_0_ids = tokenizer.map_text_to_id(evt_k_ver_and_obj_sampled_0)

        evt_q_ver_and_obj = evt_q_ver_and_obj.replace("''", "").lower().split()
        evt_q_ver_and_obj = ' '.join(
            [tokenizer.cls_token] + evt_q_ver_and_obj + [tokenizer.sep_token])

        evt_q_ver_and_obj_ids = tokenizer.map_text_to_id(evt_q_ver_and_obj)

        if len(evt_q_ver_and_obj_ids) > 3:
            mask_pos = random.randint(1, len(evt_q_ver_and_obj_ids) - 2)
        else:
            mask_pos = 1
        mask_id = evt_q_ver_and_obj_ids[mask_pos]
        evt_q_ver_and_obj_ids[mask_pos] = tokenizer.map_token_to_id(tokenizer.mask_token)

        # print()
        return {
            "evt_q_ver_and_obj_ids": evt_q_ver_and_obj_ids,
            "evt_k_ver_and_obj_sampled_0_ids": evt_k_ver_and_obj_sampled_0_ids,
            "evt_k_ver_and_obj_freq": evt_k_freq,
            "evt_k_freq": evt_k_freq,
            "mask_pos": mask_pos,
            "mask_id": mask_id,
            # "evt_k": ' '.join(evt_k_str),
            # # "evt_p": ' '.join(evt_p),
            # # "evt_q_test": evt_q_test,
            # "evt_q_ids": evt_q_ids,
            # # "evt_k_test": evt_k_test,
            # "evt_k_ids": evt_k_ids,
            # # "evt_p_test": evt_p_test,
            # "evt_p_ids": evt_p_ids,
            # "mask_pos": mask_pos,
            # "mask_id": mask_id,
            # "evt_freq": evt_freq,
        }

    # examples is a list of objects returned from the 'process' method. These data should be collated into a batch
    # examples: A list of processed examples in a batch.
    # collate method must be implemented
    def collate(self, examples: List[Example]) -> tx.data.Batch:
        # evt_q = [ex["evt_q"] for ex in examples]
        # Pad a batch of integer lists (or numpy arrays) to the same length, and stack them together.

        evt_q_ver_and_obj_ids, evt_q_ver_and_obj_ids_lengths = tx.data.padded_batch(
            [ex["evt_q_ver_and_obj_ids"] for ex in examples], pad_value=pad_token_id
        )

        evt_k_ver_and_obj_sampled_0_ids, evt_k_ver_and_obj_sampled_0_ids_lengths = tx.data.padded_batch(
            [ex["evt_k_ver_and_obj_sampled_0_ids"] for ex in examples], pad_value=pad_token_id
        )

        evt_k_freq = [ex["evt_k_freq"] for ex in examples]
        mask_id = [ex["mask_id"] for ex in examples]
        mask_pos = [ex["mask_pos"] for ex in examples]

        # print()

        return tx.data.Batch(
            len(examples),
            evt_q_ver_and_obj_ids=torch.from_numpy(evt_q_ver_and_obj_ids),
            evt_q_ver_and_obj_ids_lengths=torch.tensor(evt_q_ver_and_obj_ids_lengths),
            evt_k_ver_and_obj_sampled_0_ids=torch.from_numpy(evt_k_ver_and_obj_sampled_0_ids),
            evt_k_ver_and_obj_sampled_0_ids_lengths=torch.tensor(evt_k_ver_and_obj_sampled_0_ids_lengths),
            evt_k_freq=evt_k_freq,
            mask_id=torch.tensor(mask_id),
            mask_pos=torch.tensor(mask_pos),
        )

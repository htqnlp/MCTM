import argparse
import importlib
from logging import BASIC_FORMAT
from typing import Any
from warnings import catch_warnings

import torch
import torch.nn as nn
from texar.torch.run import *
from pathlib import Path
import misc_utils
from misc_utils import init_logger, logger

import texar.torch as tx
from model import SWCC, Memory_Bank
# from model import SWCC_Span
import data_utils
import numpy as np
import torch.nn.functional as F
import numpy as np
import torch.distributed as dist

# import span_test
import span_utils
from model_moe import MoE

parser = argparse.ArgumentParser()  # chuang jian jie xi qi
parser.add_argument('--config-model',  # model
                    type=str,
                    default="config_model",
                    help="The model config.")
parser.add_argument('--config-data',  # data
                    type=str,
                    default="config_data",
                    help="The dataset config.")
parser.add_argument("--do-train",
                    action="store_false",
                    help="Whether to run training.")
parser.add_argument("--do-eval",
                    action="store_true",
                    help="Whether to run eval on the dev set.")
parser.add_argument("--do-tsne",
                    action="store_true",
                    help="Output tsne metadata.")
parser.add_argument("--output-dir",
                    type=str,
                    default="./models/",
                    help="Path to save the trained model and logs.")
parser.add_argument("--log-file",
                    type=str,
                    default="exp.log",
                    help="Path to save the trained model and logs.")

parser.add_argument('--checkpoint',
                    type=str,
                    default="./models/checkpoint1.pt",
                    help="Model checkpoint to load model weights from.")
parser.add_argument('--baseline',
                    type=str,
                    default=None,
                    help="Model checkpoint to load model weights from.")

args = parser.parse_args()  # jie xi can shu

config_model: Any = importlib.import_module(args.config_model)  # dong tai dao ru mo kuai
config_data: Any = importlib.import_module(args.config_data)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

make_deterministic(config_model.random_seed)

output_dir = Path(args.output_dir)
tx.utils.maybe_create_dir(output_dir)

init_logger(output_dir / args.log_file)


def safe_log(val):
    eps = 1e-7
    return torch.log(val + eps)


# Sinkhorn-Knopp algorithm
@torch.no_grad()
def distributed_sinkhorn(out):
    Q = torch.exp(out / config_model.epsilon).t(
    )  # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1]  # number of samples to assign
    K = Q.shape[0]  # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    Q /= sum_Q

    for it in range(config_model.sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B  # the colomns must sum to 1 so that Q is an assignment
    return Q.t()


def main() -> None:
    # First, we introduce few datasets, such as train_data, valid_data, hard_data, hardext_data and trans_data.
    # introduce train dataset
    # config_data.train_hparams  train datas' settings
    # data_utils.TrainData

    # span_data = span_test.SpanData0Source(config_data.train_hparams)
    # for batch in span_data:
    #     print()
    # span_data1 = span_data

    # span_data = span_utils.SpanData(config_data.train_hparams, device=device)
    # span_data_iterator = tx.data.DataIterator(span_data)

    # mcnc_data = data_utils.MCNCData(config_data.train_hparams, device=device)
    # mcnc_data_iterator = tx.data.DataIterator(mcnc_data)

    # model1 = MoE(config_model, config_data, 3, k=4, noisy_gating=True)

    train_data = data_utils.TrainData(config_data.train_hparams, device=device)
    # Data iterator that switches and iterates through multiple datasets.(tx.data.DataIterator)
    train_data_iterator = tx.data.DataIterator(train_data)

    valid_data = data_utils.ValidData(config_data.valid_hparams, device=device)
    valid_data_iterator = tx.data.DataIterator(valid_data)

    hard_data = data_utils.HardData(config_data.hard_hparams, device=device)
    hard_data_iterator = tx.data.DataIterator(hard_data)

    hardext_data = data_utils.HardData(config_data.hardext_hparams,
                                       device=device)
    hardext_data_iterator = tx.data.DataIterator(hardext_data)

    trans_data = data_utils.TransData(config_data.trans_hparams, device=device)
    trans_data_iterator = tx.data.DataIterator(trans_data)

    # introduce model (SWCC)
    # model = SWCC(config_model=config_model, config_data=config_data)
    model = MoE(config_model, config_data, 10, 1, noisy_gating=True)
    # model1 = SWCC_Span(config_model=config_model, config_data=config_data)
    # if args.checkpoint:
    #     logger.info(f"loading checkpoint {args.checkpoint}...")
    #     model.load_state_dict(torch.load(args.checkpoint))
    model.to(device)
    # model1.to(device)

    # introduce memory_bank
    memory_bank = Memory_Bank(config_model.bank_size,
                              dim=config_model.hidden_dim)
    memory_bank.to(device)

    # ce_loss_fn = nn.CrossEntropyLoss()
    # ce_loss_fn.to(device)
    # realize Adam algorithm (We trained our model with a batch size of 256 using an Adam optimizer.P4.1)
    # parameters() :Returns an iterator over module parameters. And this is typically passed to an optimizer.
    optim = torch.optim.Adam(model.parameters(),
                             lr=config_model.lr,  # Ir : Learning Rate (float, default = 1e-3)
                             # betas: computing the gradient and **gradient, the second is always be set as closed to 1
                             betas=(0.9, 0.997),
                             # eps: Increase the stability of numerical calculation
                             eps=1e-9)
    # memory_bank optimizer
    mem_optim = torch.optim.Adam(memory_bank.parameters(),
                                 lr=config_model.mem_lr,
                                 betas=(0.9, 0.997),
                                 eps=1e-9)

    def get_pos_neg_mask(logits):
        mask = torch.ones(logits.shape[0])
        mask = torch.diag_embed(mask)
        # mask = torch.cat([mask, torch.zeros((logits.shape[0],config_model.bank_size))],dim=-1)
        return mask.to(device), (1 - mask).to(device)

    def _calculate_loss(batch):
        # model_batch_test = model(batch)
        q, k1, k2, p, mlm_logits, mlm_logits_sub_and_ver, mlm_logits_ver_and_obj, experts_loss1, experts_loss2, experts_loss3 = model(
            batch)
        # What is memory_bank? Normalize and matrix 's multiplication
        # k1_top
        k1_topic_logits = memory_bank(k1)
        k2_topic_logits = memory_bank(k2)
        k1_topic_logits = k1_topic_logits / config_model.moco_t
        k2_topic_logits = k2_topic_logits / config_model.moco_t
        # torch.einsum() tsumari : Matrix operations
        l_pos1 = torch.einsum('nc,ck->nk', [q, k1.T])
        l_pos2 = torch.einsum('nc,ck->nk', [q, k2.T])
        l_pos3 = torch.einsum('nc,nc->n', [q, p])

        # torch.arrange() : Creating a one-dimensional tensor
        labels = torch.arange(0, l_pos1.shape[0], dtype=torch.long).to(device)
        l_pos1 = l_pos1 / config_model.moco_t
        l_pos2 = l_pos2 / config_model.moco_t
        l_pos3 = l_pos3 / config_model.moco_t

        # F.cross_entropy() : Computing the Cross Entropy.
        pos1_loss = F.cross_entropy(l_pos1, labels)
        pos2_loss = F.cross_entropy(l_pos2, labels)
        # pos3_loss = F.cross_entropy(l_pos3, labels, reduction='none')

        pos_mask, neg_mask = get_pos_neg_mask(l_pos1)

        neg = l_pos1.masked_select(neg_mask.bool())

        neg = torch.reshape(neg, (l_pos1.shape[0], l_pos1.shape[1] - 1))
        neg = torch.exp(neg)
        pos3 = torch.exp(l_pos3)

        # pos = 0.5*pos1+0.5*pos2
        # reweight = neg/neg.mean(dim=1,keepdim=True)
        # Ng = (reweight*neg).sum(dim = -1)
        Ng = neg.sum(dim=-1)
        # self_sup_loss = 0.5*(- safe_log(pos1 / (pos1 + Ng) )).mean() + 0.5*(- safe_log(pos2 / (pos2 + Ng) )).mean()
        self_sup_loss = (pos1_loss + pos2_loss) / 2
        weakly_sup_loss = (batch.evt_freq * (-safe_log(pos3 /
                                                       (pos3 + Ng)))).mean()
        # weakly_sup_loss = (batch.evt_freq*pos3_loss).mean()
        nce_loss = self_sup_loss + weakly_sup_loss

        sub_cluster_loss1 = distributed_sinkhorn(
            k1_topic_logits) * torch.log_softmax(k2_topic_logits, dim=-1)
        sub_cluster_loss1 = sub_cluster_loss1.mean()

        sub_cluster_loss2 = distributed_sinkhorn(
            k2_topic_logits) * torch.log_softmax(k1_topic_logits, dim=-1)
        sub_cluster_loss2 = sub_cluster_loss2.mean()

        sub_cluster_ent1 = torch.softmax(
            k1_topic_logits, dim=-1) * torch.log_softmax(k1_topic_logits,
                                                         dim=-1)
        sub_cluster_ent1 = sub_cluster_ent1.mean()
        sub_cluster_ent2 = torch.softmax(
            k2_topic_logits, dim=-1) * torch.log_softmax(k2_topic_logits,
                                                         dim=-1)
        sub_cluster_ent2 = sub_cluster_ent2.mean()

        cluster_loss = -(sub_cluster_loss1 + sub_cluster_loss2) / 2
        cluster_ent = -(sub_cluster_ent1 + sub_cluster_ent2) / 2
        mlm_loss = F.cross_entropy(mlm_logits, batch.mask_id)

        mlm_sub_and_ver_loss = F.cross_entropy(mlm_logits_sub_and_ver.view(-1, 30522),
                                               batch.mask_id_sub_and_ver.view(-1, ))
        # mlm_sub_and_ver_loss = F.cross_entropy(mlm_logits_sub_and_ver, batch.mask_id_sub_and_ver)
        mlm_ver_and_obj_loss = F.cross_entropy(mlm_logits_ver_and_obj.view(-1, 30522),
                                               batch.mask_id_ver_and_obj.view(-1, ))

        # span_loss = F.cross_entropy(mlm_logits_span, batch.mask_id_span)

        loss = nce_loss + experts_loss1 + experts_loss2 + experts_loss3 + mlm_loss + \
               mlm_sub_and_ver_loss * 0.01 + mlm_ver_and_obj_loss * 0.45 + 0.3 * cluster_loss + config_model.epsilon * cluster_ent

        # span_loss = F.cross_entropy(mlm_logits_span, batch.mask_id_span)

        return {
            'nce_loss': nce_loss.item(),
            'mlm_loss': mlm_loss.item(),
            'cluster_loss': cluster_loss.item(),
            'mlm_sub_and_ver_loss': mlm_sub_and_ver_loss.item(),
            'mlm_ver_and_obj_loss': mlm_ver_and_obj_loss.item(),
            'experts_loss1': experts_loss1.item(),
            'experts_loss2': experts_loss2.item(),
            'experts_loss3': experts_loss3.item(),
            # 'softmax1': softmax1.item(),
            # 'softmax2': softmax2.item(),
            # 'softmax3': softmax3.item(),
            # 'span_loss1': span_loss1.item(),
            # 'span_loss': span_loss.item(),
        }, loss

    # update the gradients
    def _update(batch):
        return_dict, loss = _calculate_loss(batch)
        # zero_grad() : Zero the gradients before running the backward pass.
        optim.zero_grad()
        mem_optim.zero_grad()
        loss.backward()
        # step()
        mem_optim.step()
        optim.step()
        return return_dict

    def _save_epoch(epoch):
        logger.info(f"saving model...epoch {epoch}")
        torch.save(model.state_dict(), output_dir / f"checkpoint{epoch}.pt")

    def _save_step(step):
        logger.info(f"saving model... step {step}")
        ckpt = {
            "model": model.state_dict(),
            "memory_bank": memory_bank.state_dict(),
        }
        torch.save(ckpt, output_dir / f"checkpoint{step}.pt")

    # evaluation
    def _eval_model():
        model.eval()
        ###### Valid data
        avg_rec = tx.utils.AverageRecorder()
        for batch in valid_data_iterator:
            return_dict, _ = _calculate_loss(batch)
            avg_rec.add(return_dict)

        valid_loss = avg_rec.avg('nce_loss')
        logger.info(f"valid loss: {valid_loss}")
        ###### Test data

        cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)

        hard_results = []
        for batch in hard_data_iterator:
            evt_a = model.encoder_q(batch.evt_a_ids, batch.evt_a_lengths)
            evt_b = model.encoder_q(batch.evt_b_ids, batch.evt_b_lengths)
            evt_c = model.encoder_q(batch.evt_c_ids, batch.evt_c_lengths)
            evt_d = model.encoder_q(batch.evt_d_ids, batch.evt_d_lengths)
            ab_sim = cosine_similarity(evt_a, evt_b)
            cd_sim = cosine_similarity(evt_c, evt_d)

            # print(memory_bank(F.normalize(evt_a,dim=-1))[:10])

            ret = ab_sim > cd_sim
            hard_results += ret.tolist()
        hard_results = np.array(hard_results)

        hardext_results = []
        for batch in hardext_data_iterator:
            evt_a = model.encoder_q(batch.evt_a_ids, batch.evt_a_lengths)
            evt_b = model.encoder_q(batch.evt_b_ids, batch.evt_b_lengths)
            evt_c = model.encoder_q(batch.evt_c_ids, batch.evt_c_lengths)
            evt_d = model.encoder_q(batch.evt_d_ids, batch.evt_d_lengths)
            ab_sim = cosine_similarity(evt_a, evt_b)
            cd_sim = cosine_similarity(evt_c, evt_d)
            ret = ab_sim > cd_sim
            hardext_results += ret.tolist()
        hardext_results = np.array(hardext_results)
        trans_results = []
        human_scores = []
        for batch in trans_data_iterator:
            evt_a = model.encoder_q(batch.evt_a_ids, batch.evt_a_lengths)
            evt_b = model.encoder_q(batch.evt_b_ids, batch.evt_b_lengths)
            ab_sim = cosine_similarity(evt_a, evt_b)
            trans_results += ab_sim.tolist()
            human_scores += batch.score
        trans_corr, _ = misc_utils.spearmanr(trans_results, human_scores)

        all_results = np.array([valid_loss])
        logger.info(
            f"Hard: {hard_results.mean():.4f} | Hard Ext: {hardext_results.mean():.4f} | Transitive: {trans_corr:.4f} ({len(trans_results)})"
        )
        return all_results

    # def train1(epoch):
    #     model1.train()

    # train steps
    def _train_epoch(epoch):
        # model.load_state_dict(torch.load(output_dir / f"checkpoint_all.pt"))
        model.train()  # Sets the module in training mode.
        step = 0
        best_eval_results = np.array([0.0])
        avg_rec = tx.utils.AverageRecorder()

        for batch in train_data_iterator:
            return_dict = _update(batch)
            avg_rec.add(return_dict)
            # every 100 steps save ones epoch steps and losses
            if step % config_data.display_steps == 0:
                logger.info(
                    f"epoch: {epoch} | step: {step} | {avg_rec.to_str(precision=4, delimiter=' | ')}"
                )
                avg_rec.reset()

            # every 1000 steps eval ones and changes model to train()
            if step % config_data.eval_steps == 0:
                eval_results = _eval_model()
                eval_results = np.array(eval_results)
                model.train()

            # every 1000 steps save ones models
            if step % config_data.save_steps == 0:
                _save_step(step)
            step += 1

    # def _train_entity_level():
    #

    if args.do_train:
        logger.info(f"start training...")

        for epoch in range(config_data.max_train_epoch):
            # _train_span(epoch)
            # torch.cuda.empty_cache()
            # train1(epoch)
            _train_epoch(epoch)
            _save_epoch(epoch)

    elif args.do_eval:
        print(f"start testing...")
        model.load_state_dict(torch.load(args.checkpoint))
        model.eval()
        hard_results = []
        for batch in hard_data_iterator:
            evt_a = model.encoder_q(batch.evt_a_ids,
                                    batch.evt_a_lengths).tolist()[0]
            evt_b = model.encoder_q(batch.evt_b_ids,
                                    batch.evt_b_lengths).tolist()[0]
            evt_c = model.encoder_q(batch.evt_c_ids,
                                    batch.evt_c_lengths).tolist()[0]
            ab_sim = misc_utils.cosine_similarity(evt_a, evt_b)
            ac_sim = misc_utils.cosine_similarity(evt_a, evt_c)
            if ab_sim > ac_sim:
                hard_results.append(1)
            else:
                hard_results.append(0)
        hard_results = np.array(hard_results)
        hardext_results = []
        for batch in hardext_data_iterator:
            evt_a = model.encoder_q(batch.evt_a_ids,
                                    batch.evt_a_lengths).tolist()[0]
            evt_b = model.encoder_q(batch.evt_b_ids,
                                    batch.evt_b_lengths).tolist()[0]
            evt_c = model.encoder_q(batch.evt_c_ids,
                                    batch.evt_c_lengths).tolist()[0]
            ab_sim = misc_utils.cosine_similarity(evt_a, evt_b)
            ac_sim = misc_utils.cosine_similarity(evt_a, evt_c)
            if ab_sim > ac_sim:
                hardext_results.append(1)
            else:
                hardext_results.append(0)
        hardext_results = np.array(hardext_results)
        trans_results = []
        human_scores = []
        for batch in trans_data_iterator:
            evt_a = model.encoder_q(batch.evt_a_ids,
                                    batch.evt_a_lengths).tolist()[0]
            evt_b = model.encoder_q(batch.evt_b_ids,
                                    batch.evt_b_lengths).tolist()[0]
            ab_sim = misc_utils.cosine_similarity(evt_a, evt_b)
            trans_results.append(ab_sim)
            human_scores.append(batch.score)
        trans_corr, _ = misc_utils.spearmanr(trans_results, human_scores)
        print(
            f"Hard: {hard_results.mean():.4f} | Hard Ext: {hardext_results.mean():.4f} | Transitive: {trans_corr}"
        )


if __name__ == '__main__':
    # span_data = span_test.SpanData0Source(config_data.train_hparams)
    main()

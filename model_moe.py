from logging import BASIC_FORMAT
from numpy.core.numeric import base_repr
import torch
import torch.nn as nn
import texar.torch as tx
from texar.torch.utils.utils import sequence_mask
from texar.torch.data import embedding
import numpy as np
import torch.nn.functional as F
import math
from torch.distributions.normal import Normal
import model


# from model import EventEncoder


class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0).exp()

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        # add eps to all zero values in order to avoid nans when going back to log space
        combined[combined == 0] = np.finfo(float).eps
        # back to log space
        return combined.log()

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()
        self.soft = nn.Softmax(1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        # out = self.fc2(out)
        out = self.soft(out)
        return out


class MoE(nn.Module):
    def __init__(self, config_model, config_data, num_experts, k, noisy_gating=True):
        super(MoE, self).__init__()
        self.config_model = config_model
        self.vocab_size = config_model.encoder['vocab_size']
        self.encoder_q = model.EventEncoder(config_model, config_data)
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        # self.output_size = output_size
        self.output_size = config_model.encoder['vocab_size']
        # self.input_size = input_size
        self.input_size = config_model.hidden_dim
        # self.hidden_size = hidden_size
        self.k = k

        self.mask_fc = nn.Linear(config_model.hidden_dim, self.vocab_size)
        self.mask_sub_and_ver = nn.Linear(config_model.hidden_dim, self.vocab_size)
        self.mask_ver_and_obj = nn.Linear(config_model.hidden_dim, self.vocab_size)

        # instantiate experts
        self.experts = nn.ModuleList([MLP(self.input_size, self.output_size) for i in range(self.num_experts)])
        self.w_gate = nn.Parameter(torch.zeros(self.input_size, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(self.input_size, num_experts), requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert (self.k <= self.num_experts)

    def cv_squared(self, x):
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, loss_coef=1):
        x_all = x
        evt_k, evt_k_lengths = x.evt_k_ids, x.evt_k_lengths
        evt_p, evt_p_lengths = x.evt_p_ids, x.evt_p_lengths
        evt_q, evt_q_lengths = x.evt_q_ids, x.evt_q_lengths
        mask_pos, mask_id = x.mask_pos, x.mask_id
        m, m_outputs = self.encoder_q(evt_q, evt_q_lengths, is_train=True)
        mlm_logits = [ex[mask_pos[idx], :].unsqueeze(0) for idx, ex in enumerate(m_outputs)]
        mlm_logits = torch.cat(mlm_logits, dim=0)
        mlm_logits = self.mask_fc(mlm_logits)
        gates1, load1 = self.noisy_top_k_gating(m_outputs.view(-1, 768), self.training)
        importance1 = gates1.sum(0)
        experts_loss1 = self.cv_squared(importance1) + self.cv_squared(load1)
        experts_loss1 *= loss_coef

        evt_q_sub_and_ver, evt_q_sub_and_ver_lengths = x.evt_q_sub_and_ver_ids, x.evt_q_sub_and_ver_ids_lengths
        mask_pos_sub_and_ver = x.mask_pos_sub_and_ver
        m_sub_and_ver, m_sub_and_ver_outputs = self.encoder_q(evt_q_sub_and_ver, evt_q_sub_and_ver_lengths,
                                                              is_train=True)
        mlm_logits_sub_and_ver = [ex[mask_pos_sub_and_ver[idx], :].unsqueeze(0) for idx, ex in
                                  enumerate(m_sub_and_ver_outputs)]
        mlm_logits_sub_and_ver = torch.cat(mlm_logits_sub_and_ver, dim=0)
        mlm_logits_sub_and_ver = self.mask_sub_and_ver(mlm_logits_sub_and_ver)
        gates2, load2 = self.noisy_top_k_gating(m_sub_and_ver_outputs.view(-1, 768), self.training)
        importance2 = gates2.sum(0)
        experts_loss2 = self.cv_squared(importance2) + self.cv_squared(load2)
        experts_loss2 *= loss_coef

        evt_q_ver_and_obj, evt_q_ver_and_obj_lengths = x.evt_q_ver_and_obj_ids, x.evt_q_ver_and_obj_ids_lengths
        mask_pos_ver_and_obj = x.mask_pos_ver_and_obj
        m_ver_and_obj, m_ver_and_obj_outputs = self.encoder_q(evt_q_ver_and_obj, evt_q_ver_and_obj_lengths,
                                                              is_train=True)
        mlm_logits_ver_and_obj = [ex[mask_pos_ver_and_obj[idx], :].unsqueeze(0) for idx, ex in
                                  enumerate(m_ver_and_obj_outputs)]
        mlm_logits_ver_and_obj = torch.cat(mlm_logits_ver_and_obj, dim=0)
        mlm_logits_ver_and_obj = self.mask_ver_and_obj(mlm_logits_ver_and_obj)
        gates3, load3 = self.noisy_top_k_gating(m_ver_and_obj_outputs.view(-1, 768), self.training)
        importance3 = gates3.sum(0)
        experts_loss3 = self.cv_squared(importance3) + self.cv_squared(load3)
        experts_loss3 *= loss_coef

        # q: Anchor Event
        q, q_outputs = self.encoder_q(evt_k, evt_k_lengths, is_train=True)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # k1: za1
        k1, k1_outputs = self.encoder_q(evt_k, evt_k_lengths, is_train=True)
        k1 = nn.functional.normalize(k1, dim=1)

        # k2: za2
        k2, k2_outputs = self.encoder_q(evt_k, evt_k_lengths, is_train=True)
        k2 = nn.functional.normalize(k2, dim=1)

        # p: xp
        p, p_outputs = self.encoder_q(evt_p, evt_p_lengths, is_train=True)
        p = nn.functional.normalize(p, dim=1)

        # dispatcher = SparseDispatcher(self.num_experts, gates1)
        # expert_inputs = dispatcher.dispatch(m_outputs.view(-1, 768))
        # gates = dispatcher.expert_to_gates()
        # expert_outputs = [self.experts[i](expert_inputs) for i in range(self.num_experts)]
        # y = dispatcher.combine(expert_outputs)

        return q, k1, k2, p, mlm_logits, mlm_logits_sub_and_ver, mlm_logits_ver_and_obj, experts_loss1, experts_loss2, experts_loss3

import torch.nn
import torch.nn as nn

from module import encoder, common, initializer
from util import consts
from . import config
from ..model_meta import MetaType, model

import logging
logger = logging.getLogger(__name__)


@model("meta_gru4rec_ood", MetaType.ModelBuilder)
class GRU4Rec(nn.Module):
    def __init__(self, model_conf):
        super(GRU4Rec, self).__init__()

        assert isinstance(model_conf, config.ModelConfig)

        self._id_encoder = encoder.IDEncoder(
            model_conf.id_vocab,
            model_conf.id_dimension
        )

        self._target_trans = common.StackedDense(
            model_conf.id_dimension, [model_conf.id_dimension] * 2, [torch.nn.Tanh, None]
        )
        self._seq_trans = common.StackedDense(
            model_conf.id_dimension, [model_conf.id_dimension], [torch.nn.Tanh]
        )

        self._gru_cell = torch.nn.GRU(
            model_conf.id_dimension,
            model_conf.id_dimension,
            batch_first=True
        )
        initializer.default_weight_init(self._gru_cell.weight_hh_l0)
        initializer.default_weight_init(self._gru_cell.weight_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_hh_l0)

        self._mlp_trans = common.StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
        )
        # self._classifier = common.StackedDense(
        #     model_conf.id_dimension * 2,
        #     model_conf.classifier + [1],
        #     ([torch.nn.Tanh] * len(model_conf.classifier)) + [None]
        # )

        self._ood_classifier = common.StackedDense(
            model_conf.id_dimension * 2,
            model_conf.classifier + [1],
            ([torch.nn.Tanh] * len(model_conf.classifier)) + [None]
        )

        # self._meta_classifier_param_list = common.HyperNetwork_FC(
        self._meta_classifier_param_list = common.HyperNetwork_FC_ood(
            # model_conf.id_dimension * 2,
            model_conf.id_dimension,
            # model_conf.classifier + [1],
            [model_conf.id_dimension] * model_conf.mlp_layers,
            # ([torch.nn.Tanh] * len(model_conf.classifier)) + [None],
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None],
            batch=True,
            # trigger_sequence_len=10,
            model_conf=model_conf
        )

    def forward(self, features, pred=False, train_ood_threshold=0.99):
        # Encode target item

        trigger_embed = self._id_encoder(features[consts.FIELD_TRIGGER_SEQUENCE])
        trigger_embed = self._seq_trans(trigger_embed)

        # B * D
        target_embed = self._id_encoder(features[consts.FIELD_TARGET_ID])
        target_embed = self._target_trans(target_embed)

        # Encode user historical behaviors
        with torch.no_grad():
            mask = torch.not_equal(features[consts.FIELD_CLK_SEQUENCE], 0).to(dtype=torch.int32)
            # B
            seq_length = torch.maximum(torch.sum(mask, dim=1) - 1, torch.Tensor([0]).to(device=mask.device))
            seq_length = seq_length.to(torch.long)

            trigger_mask = torch.not_equal(features[consts.FIELD_TRIGGER_SEQUENCE], 0).to(dtype=torch.int32)
            # B
            trigger_seq_length = torch.maximum(torch.sum(trigger_mask, dim=1) - 1,
                                               torch.Tensor([0]).to(device=trigger_mask.device))
            trigger_seq_length = trigger_seq_length.to(torch.long)

        # B * L * D
        hist_embed = self._id_encoder(features[consts.FIELD_CLK_SEQUENCE])
        hist_embed = self._seq_trans(hist_embed)

        # Get embedding of last step
        user_state, _ = self._gru_cell(hist_embed)
        user_state = user_state[range(user_state.shape[0]), seq_length, :]
        trigger_user_state, _ = self._gru_cell(trigger_embed)
        trigger_user_state = trigger_user_state[range(trigger_user_state.shape[0]), trigger_seq_length, :]

        click_embedding = self._mlp_trans(user_state)
        trigger_embedding = self._mlp_trans(trigger_user_state)

        if not pred:
            user_embedding, trigger_embed_ = self._meta_classifier_param_list(user_state, hist_embed,
                                             user_state.size()[0],
                                             seq_length)
        elif pred:
            user_embedding, trigger_embed_ = self._meta_classifier_param_list(user_state, trigger_embed,
                                                                              user_state.size()[0],
                                                                              trigger_seq_length)
        # ood_pred = self._ood_classifier(torch.cat([trigger_embed_, atten_aggregated_embed], dim=1))
        # ood_pred = self._ood_classifier(torch.cat([trigger_embed_, user_state], dim=1))
        ood_pred = self._ood_classifier(torch.cat([trigger_embedding, click_embedding], dim=1))
        output = torch.sum(user_embedding * target_embed, dim=1, keepdim=True)
        if pred:
            total_num = ood_pred.size()[0]
            # request_index = torch.where(torch.sigmoid(ood_pred).detach().cpu() < 0.99, True, False).view(-1)
            request_index = torch.where(torch.sigmoid(ood_pred).detach().cpu() < train_ood_threshold, True, False).view(-1)
            request_num = torch.sum(torch.where(request_index, 1, 0))
            if request_num > 0:
                # trigger_embed = hist_embed
                user_embedding, trigger_embed_ = self._meta_classifier_param_list(user_state, hist_embed,
                                                                                  user_state.size()[0],
                                                                                  seq_length)
            output[request_index] = torch.sum(user_embedding[request_index] * target_embed[request_index], dim=1, keepdim=True)
            return output, ood_pred, request_num, total_num
        return output, ood_pred


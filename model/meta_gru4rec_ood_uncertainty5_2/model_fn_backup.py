import torch
import torch.nn as nn

from module import encoder, common, initializer
from util import consts
from util import uncertainty_utils
from . import config
from ..model_meta import MetaType, model
from torch.distributions import Normal, Independent, kl

import logging
logger = logging.getLogger(__name__)


# @model("meta_gru4rec_ood_uncertainty", MetaType.ModelBuilder)
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

        self._pred_decoder = nn.Linear(model_conf.id_dimension * 2, model_conf.id_dimension)

        self._meta_classifier_param_list = common.HyperNetwork_FC(
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

        self._mlp_trans = common.StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
        )

        self.prior_net = vae_space(in_dimension=model_conf.id_dimension,
                                   hidden_dimension=model_conf.id_dimension // 2,
                                   out_dimension=model_conf.id_dimension)

        self.posterior_net = vae_space(in_dimension=model_conf.id_dimension * 2,
                                   hidden_dimension=model_conf.id_dimension // 2,
                                   out_dimension=model_conf.id_dimension)

        self.mis_recommendation_layer1 = nn.Linear(model_conf.id_dimension * 2, 1)
        self.mis_recommendation_act1 = nn.Tanh()
        self.mis_recommendation_layer2 = nn.Linear(2, 1)
        self.sample_times = 10

    # def forward(self, features, pred=False, train_ood_threshold=0.99, use_uncertainty_net=False, use_duet_net=False, use_mis_recommendation=False):
    def forward(self, features, pred=False, train_ood_threshold=0.99, stage=0):

        sample_times = self.sample_times

        # Encode target item

        trigger_embed = self._id_encoder(features[consts.FIELD_TRIGGER_SEQUENCE])
        trigger_embed = self._seq_trans(trigger_embed)

        # B * D
        target_embed = self._id_encoder(features[consts.FIELD_TARGET_ID])
        target_embed = self._target_trans(target_embed)

        batch_size = target_embed.size()[0]

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

        # trigger_user_state, _ = self._gru_cell(trigger_embed)
        # trigger_user_state = trigger_user_state[range(trigger_user_state.shape[0]), trigger_seq_length, :]

        # user_embedding, trigger_embed_ = self._meta_classifier_param_list(user_state, trigger_embed,
        #                                  user_state.size()[0],
        #                                  trigger_seq_length)

        if not pred:
            # if use_duet_net:
            if stage == 1:
                # user_embedding = self._meta_classifier_param_list(user_state, trigger_embed,
                #                                                   user_state.size()[0],
                #                                                   trigger_seq_length)
                user_embedding = self._meta_classifier_param_list(user_state, hist_embed,
                                                                  user_state.size()[0],
                                                                  seq_length)

                output = torch.sum(user_embedding * target_embed, dim=1, keepdim=True)
                return output

            # if use_uncertainty_net:
            elif stage == 2:
                ### vae uncertainty module
                user_embedding = self._mlp_trans(user_state)
                # prior net
                # print(user_embedding.size())
                z, prior_latent_dist = self.prior_net.forward(user_embedding)
                z_posterior, posterior_latent_dist = self.posterior_net.forward(torch.cat([user_embedding, target_embed], dim=-1))
                # next_item_embedding_pred = self._pred_decoder(torch.cat([user_embedding, z]))
                next_item_embedding_pred = self._pred_decoder(torch.cat([user_embedding, z_posterior], dim=-1))
                # posterior net
                analytic_kl = True
                # kl_loss = self.kl_divergence(analytic=analytic_kl, calculate_posterior=False, z_posterior=z_posterior)
                kl_div = kl.kl_divergence(posterior_latent_dist, prior_latent_dist)
                label = features[consts.FIELD_LABEL]
                kl_coefficient = torch.where(label == 1, 1, -1)
                # beta = 10
                # beta = 5
                beta = 1
                # beta = 0.1
                # beta = 0.001
                # beta = 0
                # next_item_embedding = user_embedding_reconstruct
                # criterion = nn.BCEWithLogitsLoss()
                criterion = nn.MSELoss()
                loss_regression = criterion(next_item_embedding_pred, target_embed)
                # print(kl_div.size())
                # print(kl_coefficient.size())
                # print((kl_coefficient * beta * kl_div).size())
                # print(loss_regression)
                # print(loss_regression)
                # print(torch.mean(kl_coefficient * kl_div))
                # print(torch.mean(kl_coefficient * beta * kl_div))
                # cvae_loss = -(loss_regression + torch.mean(kl_coefficient * beta * kl_div))
                cvae_loss = loss_regression + torch.mean(kl_coefficient * beta * kl_div)
                # cvae_loss = loss_regression
                # if use_mis_recommendation:
                #     # 训练的时候没有uncertainty的值，是训练也多次采样分三个stage训？
                #     # 第一个stage是meta，
                #     # 第二个stage是用embedding和重复层(梯度置零)训vae，
                #     # 第三个stage训mis-recommendation，
                #     # forward的选项用stage1 2 3而不是现在这个use_uncertainty_net和use_duet_net
                #     trigger_user_state, _ = self._gru_cell(trigger_embed)
                #     trigger_user_state = trigger_user_state[range(trigger_user_state.shape[0]), trigger_seq_length, :]
                #     mis_rec_pred = self.mis_recommendation_act1(self.mis_recommendation_layer1(torch.cat([trigger_embed_, user_state], dim=1)))
                #     mis_rec_pred = self.mis_recommendation_layer2(torch.cat([mis_rec_pred, output], dim=1))
                return cvae_loss

            elif stage == 3:

                # user_embedding = self._meta_classifier_param_list(user_state, trigger_embed,
                #                                                   user_state.size()[0],
                #                                                   trigger_seq_length)
                #
                # output = torch.sum(user_embedding * target_embed, dim=1, keepdim=True)

                user_embedding = self._mlp_trans(user_state)

                # prior net
                user_embedding_ = user_embedding.unsqueeze(1)
                user_embedding_ = user_embedding_.repeat(1, sample_times, 1)
                z_, prior_latent_dist_ = self.prior_net.forward(user_embedding_)
                next_item_embedding_pred = self._pred_decoder(torch.cat([user_embedding_, z_], dim=-1))
                uncertainty = torch.sum(torch.var(next_item_embedding_pred, dim=-1), dim=-1)

                trigger_user_state, _ = self._gru_cell(trigger_embed)
                trigger_user_state = trigger_user_state[range(trigger_user_state.shape[0]), trigger_seq_length, :]
                trigger_embedding = self._mlp_trans(trigger_user_state)
                mis_rec_pred = self.mis_recommendation_act1(self.mis_recommendation_layer1(torch.cat([trigger_embedding, user_embedding], dim=1)))
                mis_rec_pred = self.mis_recommendation_layer2(torch.cat([mis_rec_pred, uncertainty], dim=1))

        else:
            # request_index = 0
            # request_num = 0
            if stage == 1:
                user_embedding = self._meta_classifier_param_list(user_state, hist_embed,
                                                                  user_state.size()[0],
                                                                  seq_length)
                # if use_duet_net:
                #     user_embedding = self._meta_classifier_param_list(user_state, trigger_embed,
                #                                                       user_state.size()[0],
                #                                                       trigger_seq_length)
                output = torch.sum(user_embedding * target_embed, dim=1, keepdim=True)
                return output

            # if use_uncertainty_net:
            elif stage == 2:
                ### vae uncertainty module
                user_embedding = self._mlp_trans(user_state)
                # sample_times = 10
                # prior net
                user_embedding_ = user_embedding.unsqueeze(1)
                user_embedding_ = user_embedding_.repeat(1, sample_times, 1)
                z_, prior_latent_dist_ = self.prior_net.forward(user_embedding_)
                # print("=" * 50)
                # print("{}test{}".format("-" * 30, "-" * 30))
                # print(user_embedding_.size())
                # print(z_.size())
                # print(torch.cat([user_embedding_, z_], dim=1).size())
                next_item_embedding_pred = self._pred_decoder(torch.cat([user_embedding_, z_], dim=-1))
                # print("next_item_embedding_pred.size() ", next_item_embedding_pred.size())
                uncertainty = torch.sum(torch.var(next_item_embedding_pred, dim=-1), dim=-1)
                # print("uncertainty.size() ", next_item_embedding_pred.size())
                criterion = nn.MSELoss()
                criterion_list = nn.MSELoss(reduction="none")
                # loss_regression = criterion(next_item_embedding_pred, target_embed)
                mse = criterion(next_item_embedding_pred, target_embed.unsqueeze(1).repeat(1, sample_times, 1))
                mse_list = torch.mean(torch.mean(criterion_list(next_item_embedding_pred,
                                                                target_embed.unsqueeze(1).repeat(1, sample_times, 1)), dim=1), dim=1)
                # print("mse_list.size() ", mse_list.size())
                # mse_list = torch.mean(mse_list, dim=1)

                # print("target_embed.unsqueeze(1).repeat(1, sample_times, 1).size() ", target_embed.unsqueeze(1).repeat(1, sample_times, 1).size())
                # request_index = torch.where(torch.sigmoid(uncertainty).detach().cpu() < train_ood_threshold, True, False).view(-1)
                request_index = torch.where(uncertainty.detach().cpu() > train_ood_threshold, True, False).view(-1)
                request_num = torch.sum(torch.where(request_index, 1, 0))
                # print(criterion(next_item_embedding_pred, target_embed.unsqueeze(1).repeat(1, sample_times, 1)))
                # print("uncertainty.size() ", uncertainty.size())
                # print("-" * 50)
                # print("mse ", mse)
                # print("torch.mean(mse_list) ", torch.mean(mse_list))
                # print("mse_list.size() ", mse_list.size())
                # print("max(mse_list) {}, min(mse_list) {}".format(max(mse_list), min(mse_list)))
                # print("max(uncertainty) {}, min(uncertainty) {}".format(max(uncertainty), min(uncertainty)))

                if use_duet_net:
                    user_embedding = self._meta_classifier_param_list(user_state, trigger_embed,
                                                                      user_state.size()[0],
                                                                      trigger_seq_length)

                    output = torch.sum(user_embedding * target_embed, dim=1, keepdim=True)

                    # request_index = torch.where(torch.sigmoid(ood_pred).detach().cpu() < train_ood_threshold, True, False).view(-1)
                    # request_num = torch.sum(torch.where(request_index, 1, 0))
                    if request_num > 0:
                        # trigger_embed = hist_embed
                        user_embedding = self._meta_classifier_param_list(user_state, hist_embed,
                                                                                          user_state.size()[0],
                                                                                          trigger_seq_length)

                    output[request_index] = torch.sum(user_embedding[request_index] * target_embed[request_index], dim=1, keepdim=True)
                    # return output, ood_pred, request_num, total_num
                    return output, request_num, total_num

                return mse_list, uncertainty, request_num

            elif stage == 3:
                user_embedding = self._meta_classifier_param_list(user_state, trigger_embed,
                                                                  user_state.size()[0],
                                                                  trigger_seq_length)
                output = torch.sum(user_embedding * target_embed, dim=1, keepdim=True)
                return output

        # ood_pred = self._ood_classifier(torch.cat([trigger_embed_, atten_aggregated_embed], dim=1))

        # ood_pred = self._ood_classifier(torch.cat([trigger_embed_, user_state], dim=1))
        # output = torch.sum(user_embedding * target_embed, dim=1, keepdim=True)
        # if pred:
        #     total_num = ood_pred.size()[0]
        #     # request_index = torch.where(torch.sigmoid(ood_pred).detach().cpu() < 0.99, True, False).view(-1)
        #     request_index = torch.where(torch.sigmoid(ood_pred).detach().cpu() < train_ood_threshold, True, False).view(-1)
        #     request_num = torch.sum(torch.where(request_index, 1, 0))
        #     if request_num > 0:
        #         # trigger_embed = hist_embed
        #         user_embedding, trigger_embed_ = self._meta_classifier_param_list(user_state, hist_embed,
        #                                                                           user_state.size()[0],
        #                                                                           trigger_seq_length)
        #     output[request_index] = torch.sum(user_embedding[request_index] * target_embed[request_index], dim=1, keepdim=True)
        #     return output, ood_pred, request_num, total_num
        # return output, ood_pred

class vae_space(nn.Module):
    def __init__(self, in_dimension, hidden_dimension, out_dimension):
        super(vae_space, self).__init__()

        # self._encoder = nn.Linear(model_conf.id_dimension, model_conf.id_dimension // 2)
        # self._encoder_act = nn.Tanh()
        # self._mean_linear = nn.Linear(model_conf.id_dimension // 2, model_conf.id_dimension // 2)
        # self._var_linear = nn.Linear(model_conf.id_dimension // 2, model_conf.id_dimension // 2)
        # self._decoder = nn.Linear(model_conf.id_dimension // 2, model_conf.id_dimension)
        # self._pred_decoder = nn.Linear(model_conf.id_dimension, model_conf.id_dimension)

        self._encoder = nn.Linear(in_dimension, hidden_dimension)
        self._encoder_act = nn.Tanh()
        self._mean_linear = nn.Linear(hidden_dimension, hidden_dimension)
        self._var_linear = nn.Linear(hidden_dimension, hidden_dimension)
        self._decoder = nn.Linear(hidden_dimension, out_dimension)
        # self._pred_decoder = nn.Linear(model_conf.id_dimension, model_conf.id_dimension)

    # def prior_net(self, z):
    #     mu, log_var = self.vae_encoder(z)
    #     dist = Independent(Normal(loc=mu, scale=torch.exp(log_var)), 1)
    #     z = self.vae_sampler(mu, log_var)
    #     z_reconstruct = self.vae_decoder(z)
    #     return z_reconstruct, dist
    #
    # def posterior_net(self, z):
    #     mu, log_var = self.posterior_vae_encoder(z)
    #     dist = Independent(Normal(loc=mu, scale=torch.exp(log_var)), 1)
    #     z = self.vae_sampler(mu, log_var)
    #     z_reconstruct = self.posterior_vae_decoder(z)
    #     return z_reconstruct, dist

    def vae_encoder(self, user_embedding):
        h = self._encoder_act(self._encoder(user_embedding))
        return self._mean_linear(h), self._var_linear(h)

    def vae_sampler(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.rand_like(std)
        return mu + eps * std

    def vae_decoder(self, z):
        h = self._decoder(z)
        return h

    def forward(self, user_embedding):
        # print("-" * 50)
        # print(user_embedding.size())
        mu, log_var = self.vae_encoder(user_embedding)

        # print(mu.size())
        # print(log_var.size())
        dist = Independent(Normal(loc=mu, scale=torch.exp(log_var)), 1)
        z = self.vae_sampler(mu, log_var)
        # print(z.size())
        z_reconstruct = self.vae_decoder(z)
        # print(z.size())
        return z_reconstruct, dist



if __name__ == '__main__':
    import sys
    # sys.path.append("../..")
    sys.path.extend("../..")
    sys.path.extend("../../")
    sys.path.extend(".")
    model_conf = {
        "id_dimension": 32,
        "id_vocab": 8000,
        "classifier": [128, 64],
        "mlp_layers": 2
    }
    model = GRU4Rec(model_conf)

import torch.nn
import torch.nn as nn

from module import attention, encoder, common
from util import consts
from . import config
from ..model_meta import MetaType, model
from torch.distributions import Normal, Independent, kl

import logging
logger = logging.getLogger(__name__)


@model("meta_din_ood_uncertainty5_2", MetaType.ModelBuilder)
class DeepInterestNetwork(nn.Module):
    def __init__(self, model_conf):
        super(DeepInterestNetwork, self).__init__()

        assert isinstance(model_conf, config.ModelConfig)
        self._id_encoder = encoder.IDEncoder(
            model_conf.id_vocab,
            model_conf.id_dimension,
        )
        self._target_trans = common.StackedDense(
            model_conf.id_dimension, [model_conf.id_dimension], [torch.nn.Tanh]
        )
        self._seq_trans = common.StackedDense(
            model_conf.id_dimension, [model_conf.id_dimension], [torch.nn.Tanh]
        )

        self._target_attention = attention.TargetAttention(
            key_dimension=model_conf.id_dimension,
            value_dimension=model_conf.id_dimension,
        )

        # self._classifier = common.StackedDense(
        #     model_conf.id_dimension * 2,
        #     model_conf.classifier + [1],
        #     ([torch.nn.Tanh] * len(model_conf.classifier)) + [None]
        # )

        self._meta_classifier_param_list = common.HyperNetwork_FC(
            # model_conf.id_dimension * 2,
            model_conf.id_dimension,
            model_conf.classifier + [1],
            ([torch.nn.Tanh] * len(model_conf.classifier)) + [None],
            batch=True,
            trigger_sequence_len=30,
            model_conf=model_conf,
            expand=True
        )

        self.stage2_mlp_trans = common.StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
        )

        self.stage2_prior_net = vae_space(in_dimension=model_conf.id_dimension,
                                          hidden_dimension=model_conf.id_dimension // 2,
                                          out_dimension=model_conf.id_dimension)

        self.stage2_posterior_net = vae_space(in_dimension=model_conf.id_dimension * 2,
                                              hidden_dimension=model_conf.id_dimension // 2,
                                              out_dimension=model_conf.id_dimension)

        self.stage2_pred_decoder = nn.Linear(model_conf.id_dimension * 2, model_conf.id_dimension)

        self.stage3_mis_recommendation_layer1 = nn.Linear(model_conf.id_dimension * 2, 1)
        self.stage3_mis_recommendation_act1 = nn.Tanh()
        self.stage3_mis_recommendation_layer2 = nn.Linear(2, 1)
        self.sample_times = 10

    def __setitem__(self, k, v):
        self.k = v

    # def forward(self, features):
    def forward(self, features, pred=False, uncertainty_threshold=0.99, mis_rec_threshold=0.2, stage=0, use_uncertainty_net=False, fig1=False):
        sample_times = self.sample_times

        # Encode target item
        # B * D
        # print("-" * 50)
        # print(features)
        trigger_embed = self._id_encoder(features[consts.FIELD_TRIGGER_SEQUENCE])
        trigger_embed = self._seq_trans(trigger_embed)

        target_embed = self._id_encoder(features[consts.FIELD_TARGET_ID])
        target_embed = self._target_trans(target_embed)

        batch_size = target_embed.size()[0]
        total_num = batch_size
        
        # Encode user historical behaviors
        with torch.no_grad():
            click_mask = torch.not_equal(features[consts.FIELD_CLK_SEQUENCE], 0).to(dtype=torch.int32)
            trigger_mask = torch.not_equal(features[consts.FIELD_TRIGGER_SEQUENCE], 0).to(dtype=torch.int32)
            # B
            seq_length = torch.maximum(torch.sum(click_mask, dim=1) - 1, torch.Tensor([0]).to(device=click_mask.device))
            seq_length = seq_length.to(torch.long)

            trigger_seq_length = torch.maximum(torch.sum(trigger_mask, dim=1) - 1,
                                               torch.Tensor([0]).to(device=trigger_mask.device))
            trigger_seq_length = trigger_seq_length.to(torch.long)

        # B * L * D
        hist_embed = self._id_encoder(features[consts.FIELD_CLK_SEQUENCE])
        hist_embed = self._seq_trans(hist_embed)

        # Target attention
        atten_aggregated_embed = self._target_attention(
            target_key=target_embed,
            item_keys=hist_embed,
            item_values=hist_embed,
            mask=click_mask
        )

        trigger_atten_aggregated_embed = self._target_attention(
            target_key=target_embed,
            item_keys=trigger_embed,
            item_values=trigger_embed,
            mask=trigger_mask
        )

        # if not pred:
        #     classifier_input = torch.cat([target_embed, atten_aggregated_embed], dim=1)
        #     # return self._classifier(classifier_input)
        #     # output, trigger_embed_ = self._meta_classifier_param_list(classifier_input, hist_embed,
        #     output = self._meta_classifier_param_list(classifier_input, hist_embed,
        #                                                               classifier_input.size()[0],
        #                                                               trigger_seq_length=seq_length
        #                                                               )
        # elif pred:
        #     classifier_input = torch.cat([target_embed, trigger_atten_aggregated_embed], dim=1)
        #     # output, trigger_embed_ = self._meta_classifier_param_list(classifier_input, trigger_embed,
        #     output = self._meta_classifier_param_list(classifier_input, trigger_embed,
        #                                                               classifier_input.size()[0],
        #                                                               trigger_seq_length=trigger_seq_length
        #                                                               )

        classifier_input = torch.cat([target_embed, atten_aggregated_embed], dim=1)
        trigger_classifier_input = torch.cat([target_embed, trigger_atten_aggregated_embed], dim=1)
        if not pred:
            # if use_duet_net:
            if stage == 1:
                # user_embedding = self._meta_classifier_param_list(atten_aggregated_embed, trigger_embed,
                #                                                   atten_aggregated_embed.size()[0],
                #                                                   trigger_seq_length)

                # user_embedding = self._meta_classifier_param_list(atten_aggregated_embed, hist_embed,
                #                                                   atten_aggregated_embed.size()[0],
                #                                                   seq_length)
                #
                # output = torch.sum(user_embedding * target_embed, dim=1, keepdim=True)
                output = self._meta_classifier_param_list(classifier_input, hist_embed,
                                                                  classifier_input.size()[0],
                                                                  seq_length)
                return output

            # if use_uncertainty_net:
            elif stage == 2:
                ### vae uncertainty module
                # user_embedding = self.stage2_mlp_trans(atten_aggregated_embed)
                user_embedding = torch.mean(self.stage2_mlp_trans(hist_embed), dim=1)
                # prior net
                # print(user_embedding.size())
                z, prior_latent_dist = self.stage2_prior_net.forward(user_embedding)
                z_posterior, posterior_latent_dist = self.stage2_posterior_net.forward(torch.cat([user_embedding, target_embed], dim=-1))
                # next_item_embedding_pred = self.stage2_pred_decoder(torch.cat([user_embedding, z]))
                next_item_embedding_pred = self.stage2_pred_decoder(torch.cat([user_embedding, z_posterior], dim=-1))
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
                # cvae_loss = loss_regression + torch.mean(kl_coefficient * beta * kl_div)
                cvae_loss = torch.mean(kl_coefficient * beta * kl_div)
                # cvae_loss = abs(torch.mean(kl_coefficient * beta * kl_div))
                
                # cvae_loss = loss_regression
                # if use_mis_recommendation:
                #     # 训练的时候没有uncertainty的值，是训练也多次采样分三个stage训？
                #     # 第一个stage是meta，
                #     # 第二个stage是用embedding和重复层(梯度置零)训vae，
                #     # 第三个stage训mis-recommendation，
                #     # forward的选项用stage1 2 3而不是现在这个use_uncertainty_net和use_duet_net
                #     trigger_atten_aggregated_embed, _ = self._gru_cell(trigger_embed)
                #     trigger_atten_aggregated_embed = trigger_atten_aggregated_embed[range(trigger_atten_aggregated_embed.shape[0]), trigger_seq_length, :]
                #     mis_rec_pred = self.stage3_mis_recommendation_act1(self.stage3_mis_recommendation_layer1(torch.cat([trigger_embed_, atten_aggregated_embed], dim=1)))
                #     mis_rec_pred = self.stage3_mis_recommendation_layer2(torch.cat([mis_rec_pred, output], dim=1))
                # return cvae_loss
                output = torch.sum(next_item_embedding_pred * target_embed, dim=1, keepdim=True)
                # print(output.size())
                return output, cvae_loss

            elif stage == 3 and use_uncertainty_net:

                # user_embedding = self._meta_classifier_param_list(atten_aggregated_embed, trigger_embed,
                #                                                   atten_aggregated_embed.size()[0],
                #                                                   trigger_seq_length)

                # output = torch.sum(user_embedding * target_embed, dim=1, keepdim=True)

                output = self._meta_classifier_param_list(classifier_input, trigger_embed,
                                                                  classifier_input.size()[0],
                                                                  trigger_seq_length)

                # output = self._meta_classifier_param_list(classifier_input, hist_embed,
                #                                           classifier_input.size()[0],
                #                                           seq_length)

                # user_embedding = self.stage2_mlp_trans(atten_aggregated_embed)
                user_embedding = torch.mean(self.stage2_mlp_trans(hist_embed), dim=1)
                trigger_user_embedding = torch.mean(self.stage2_mlp_trans(trigger_embed), dim=1)
                # prior net
                # user_embedding_ = user_embedding.unsqueeze(1)
                # user_embedding_ = user_embedding_.repeat(1, sample_times, 1)
                # z_, prior_latent_dist_ = self.stage2_prior_net.forward(user_embedding_)
                # next_item_embedding_pred = self.stage2_pred_decoder(torch.cat([user_embedding_, z_], dim=-1))
                # uncertainty = torch.sum(torch.var(next_item_embedding_pred, dim=-1), dim=-1)
                mis_rec_pred_list = []
                for i in range(sample_times):
                    z_, prior_latent_dist_ = self.stage2_prior_net.forward(user_embedding)
                    z_trigger, prior_latent_dist_trigger = self.stage2_prior_net.forward(trigger_user_embedding)
                    # next_item_embedding_pred = self.stage2_pred_decoder(torch.cat([user_embedding, z_], dim=-1))
                    # next_item_embedding_pred_list.append(next_item_embedding_pred)
                    mis_rec_pred = self.stage3_mis_recommendation_act1(
                        self.stage3_mis_recommendation_layer1(torch.cat([z_, z_trigger], dim=1))
                    )
                    # print(mis_rec_pred.size())
                    mis_rec_pred_list.append(mis_rec_pred)
                    # print(next_item_embedding_pred.size())
                # uncertainty = torch.sum(torch.var(next_item_embedding_pred, dim=-1), dim=-1)
                # print(torch.Tensor(next_item_embedding_pred_list).size())
                # print(len(next_item_embedding_pred_list))
                # print(torch.stack(next_item_embedding_pred_list, dim=1).size())
                # print(torch.cat(next_item_embedding_pred_list, dim=-1).size())
                # uncertainty = torch.sum(torch.var(torch.Tensor(next_item_embedding_pred_list), dim=-1), dim=-1)

                # next_item_embedding_pred = torch.stack(next_item_embedding_pred_list, dim=1)
                # uncertainty = torch.sum(torch.var(next_item_embedding_pred, dim=-1), dim=-1)

                mis_rec_pred = torch.stack(mis_rec_pred_list, dim=1)
                # print(mis_rec_pred.size())
                mis_rec_pred_mean = torch.mean(mis_rec_pred, dim=1)
                # print(mis_rec_pred_mean.size())
                uncertainty = torch.var(mis_rec_pred, dim=1)

                # # trigger_atten_aggregated_embed, _ = self._gru_cell(trigger_embed)
                # # trigger_atten_aggregated_embed = trigger_atten_aggregated_embed[range(trigger_atten_aggregated_embed.shape[0]), trigger_seq_length, :]
                # # trigger_embedding = self.stage2_mlp_trans(trigger_atten_aggregated_embed)
                # trigger_embedding = torch.mean(self.stage2_mlp_trans(trigger_embed), dim=1)
                # mis_rec_pred = self.stage3_mis_recommendation_act1(self.stage3_mis_recommendation_layer1(torch.cat([trigger_embedding, user_embedding], dim=1)))
                # # print(mis_rec_pred.size())
                # # print(uncertainty.view(-1, 1).size())
                # mis_rec_pred = self.stage3_mis_recommendation_layer2(torch.cat([mis_rec_pred, uncertainty.view(-1, 1)], dim=1))
                mis_rec_pred = self.stage3_mis_recommendation_layer2(torch.cat([mis_rec_pred_mean, uncertainty], dim=1))

                return output, mis_rec_pred

        else:
            # request_index = 0
            # request_num = 0
            if stage == 1:
                # user_embedding = self._meta_classifier_param_list(atten_aggregated_embed, hist_embed,
                #                                                   atten_aggregated_embed.size()[0],
                #                                                   seq_length)
                #
                # output = torch.sum(user_embedding * target_embed, dim=1, keepdim=True)
                output = self._meta_classifier_param_list(classifier_input, hist_embed,
                                                                  classifier_input.size()[0],
                                                                  seq_length)
                return output

            # if use_uncertainty_net:
            elif stage == 2:
                ### vae uncertainty module
                # user_embedding = self.stage2_mlp_trans(atten_aggregated_embed)
                user_embedding = torch.mean(self.stage2_mlp_trans(hist_embed), dim=1)
                # trigger_user_embedding = torch.mean(self.stage2_mlp_trans(trigger_user_state), dim=1)
                trigger_user_embedding = torch.mean(self.stage2_mlp_trans(trigger_embed), dim=1)
                # sample_times = 10
                # prior net
                # user_embedding_ = user_embedding.unsqueeze(1)
                # user_embedding_ = user_embedding_.repeat(1, sample_times, 1)
                # z_, prior_latent_dist_ = self.stage2_prior_net.forward(user_embedding_)
                # next_item_embedding_pred = self.stage2_pred_decoder(torch.cat([user_embedding_, z_], dim=-1))
                # uncertainty = torch.sum(torch.var(next_item_embedding_pred, dim=-1), dim=-1)
                next_item_embedding_pred_list = []
                # for i in range(sample_times):
                # for i in range(1):
                #     z_, prior_latent_dist_ = self.stage2_prior_net.forward(user_embedding)
                #     next_item_embedding_pred = self.stage2_pred_decoder(torch.cat([user_embedding, z_], dim=-1))
                #     next_item_embedding_pred_list.append(next_item_embedding_pred)
                #     # print(next_item_embedding_pred.size())
                z_, prior_latent_dist_ = self.stage2_prior_net.forward(user_embedding)
                # uncertainty = torch.sum(torch.var(next_item_embedding_pred, dim=-1), dim=-1)
                # print(torch.Tensor(next_item_embedding_pred_list).size())
                # print(len(next_item_embedding_pred_list))
                # print(torch.stack(next_item_embedding_pred_list, dim=1).size())
                # print(torch.cat(next_item_embedding_pred_list, dim=-1).size())
                # uncertainty = torch.sum(torch.var(torch.Tensor(next_item_embedding_pred_list), dim=-1), dim=-1)
                # next_item_embedding_pred = torch.stack(next_item_embedding_pred_list, dim=1)
                # uncertainty = torch.sum(torch.var(next_item_embedding_pred, dim=-1), dim=-1)
                next_item_embedding_pred = self.stage2_pred_decoder(torch.cat([user_embedding, z_], dim=-1))

                criterion = nn.MSELoss()
                criterion_list = nn.MSELoss(reduction="none")

                # mse = criterion(next_item_embedding_pred, target_embed.unsqueeze(1).repeat(1, sample_times, 1))
                # mse_list = torch.mean(torch.mean(criterion_list(next_item_embedding_pred,
                #                                                 target_embed.unsqueeze(1).repeat(1, sample_times, 1)), dim=1), dim=1)
                mse_list = torch.mean(criterion_list(next_item_embedding_pred,
                                                     target_embed), dim=1)

                # request_index = torch.where(uncertainty.detach().cpu() > uncertainty_threshold, True, False).view(-1)
                # request_num = torch.sum(torch.where(request_index, 1, 0))
                #
                # return mse_list, uncertainty, request_num
                next_item_embedding_pred = next_item_embedding_pred.squeeze(1)
                # print(next_item_embedding_pred.size())
                # output = torch.sum(user_embedding * target_embed, dim=1, keepdim=True)
                # print(output.size())
                # output = torch.sum(user_embedding * next_item_embedding_pred, dim=1, keepdim=True)
                output = torch.sum(next_item_embedding_pred * target_embed, dim=1, keepdim=True)
                # print(output.size())
                uncertainty = 0
                request_num = 0
                return output, mse_list, uncertainty, request_num

            elif stage == 3:
                ###
                # user_embedding = self._meta_classifier_param_list(atten_aggregated_embed, trigger_embed,
                #                                                   atten_aggregated_embed.size()[0],
                #                                                   trigger_seq_length)
                # output = torch.sum(user_embedding * target_embed, dim=1, keepdim=True)
                output = self._meta_classifier_param_list(classifier_input, trigger_embed,
                                                                  classifier_input.size()[0],
                                                                  trigger_seq_length)

                ### vae uncertainty module
                # user_embedding = self.stage2_mlp_trans(atten_aggregated_embed)
                user_embedding = torch.mean(self.stage2_mlp_trans(hist_embed), dim=1)
                trigger_embedding = torch.mean(self.stage2_mlp_trans(trigger_embed), dim=1)

                # sample_times = 10
                # prior net
                # user_embedding_ = user_embedding.unsqueeze(1)
                # user_embedding_ = user_embedding_.repeat(1, sample_times, 1)
                # z_, prior_latent_dist_ = self.stage2_prior_net.forward(user_embedding_)
                # next_item_embedding_pred = self.stage2_pred_decoder(torch.cat([user_embedding_, z_], dim=-1))
                # uncertainty = torch.sum(torch.var(next_item_embedding_pred, dim=-1), dim=-1)
                mis_rec_pred_list = []
                for i in range(sample_times):
                    z_, prior_latent_dist_ = self.stage2_prior_net.forward(user_embedding)
                    z_trigger, prior_latent_dist_trigger = self.stage2_prior_net.forward(trigger_embedding)
                    # next_item_embedding_pred = self.stage2_pred_decoder(torch.cat([user_embedding, z_], dim=-1))
                    # next_item_embedding_pred_list.append(next_item_embedding_pred)
                    mis_rec_pred = self.stage3_mis_recommendation_act1(
                        self.stage3_mis_recommendation_layer1(torch.cat([z_, z_trigger], dim=1))
                    )
                    mis_rec_pred_list.append(mis_rec_pred)

                mis_rec_pred = torch.stack(mis_rec_pred_list, dim=1)
                mis_rec_pred_mean = torch.mean(mis_rec_pred, dim=1)
                uncertainty = torch.var(mis_rec_pred, dim=1)
                # uncertainty = torch.sum(torch.var(next_item_embedding_pred, dim=-1), dim=-1)
                # print(torch.Tensor(next_item_embedding_pred_list).size())
                # print(len(next_item_embedding_pred_list))
                # print(torch.stack(next_item_embedding_pred_list, dim=1).size())
                # print(torch.cat(next_item_embedding_pred_list, dim=-1).size())
                # uncertainty = torch.sum(torch.var(torch.Tensor(next_item_embedding_pred_list), dim=-1), dim=-1)
                # next_item_embedding_pred = torch.stack(next_item_embedding_pred_list, dim=1)
                # uncertainty = torch.sum(torch.var(next_item_embedding_pred, dim=-1), dim=-1)

                # trigger_atten_aggregated_embed, _ = self._gru_cell(trigger_embed)
                # trigger_atten_aggregated_embed = trigger_atten_aggregated_embed[range(trigger_atten_aggregated_embed.shape[0]), trigger_seq_length, :]
                # trigger_embedding = self.stage2_mlp_trans(trigger_atten_aggregated_embed)
                # trigger_embedding = torch.mean(self.stage2_mlp_trans(trigger_embed), dim=1)

                if not use_uncertainty_net:
                    mis_rec_pred = uncertainty
                    # request_index = torch.where(mis_rec_pred.detach().cpu() > uncertainty_threshold, True, False).view(-1)
                    request_index = torch.where(mis_rec_pred.detach().cpu() > mis_rec_threshold, True, False).view(-1)
                    # request_index = torch.where(torch.sigmoid(mis_rec_pred).detach().cpu() > mis_rec_threshold, True, False).view(-1)
                elif use_uncertainty_net:
                    # mis_rec_pred = self.stage3_mis_recommendation_act1(self.stage3_mis_recommendation_layer1(torch.cat([trigger_embedding, user_embedding], dim=1)))
                    # mis_rec_pred = self.stage3_mis_recommendation_layer2(torch.cat([mis_rec_pred, uncertainty.view(-1, 1)], dim=1))
                    mis_rec_pred = self.stage3_mis_recommendation_layer2(torch.cat([mis_rec_pred_mean, uncertainty], dim=1))

                    request_index = torch.where(torch.sigmoid(mis_rec_pred).detach().cpu() < mis_rec_threshold, True, False).view(-1)

                request_num = torch.sum(torch.where(request_index, 1, 0))

                if request_num > 0:
                    # trigger_embed = hist_embed
                    # user_embedding = self._meta_classifier_param_list(atten_aggregated_embed, hist_embed,
                    #                                                   atten_aggregated_embed.size()[0],
                    #                                                   seq_length)
                    output[request_index] = \
                        self._meta_classifier_param_list(classifier_input[request_index], hist_embed[request_index],
                                                         classifier_input[request_index].size()[0],
                                                         trigger_seq_length=seq_length[request_index]
                                                         )
                # output[request_index] = torch.sum(user_embedding[request_index] * target_embed[request_index], dim=1, keepdim=True)
                if fig1:
                    output1 = self._meta_classifier_param_list(classifier_input, trigger_embed,
                                                               classifier_input.size()[0],
                                                               trigger_seq_length)
                    output = self._meta_classifier_param_list(classifier_input, hist_embed,
                                                              classifier_input.size()[0],
                                                              seq_length)
                    return output, mis_rec_pred, request_num, total_num, output1
                return output, mis_rec_pred, request_num, total_num


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





class TestModule(nn.Module):
    def __init__(self):
        super(TestModule, self).__init__()
        self.conv = nn.Conv2d(1, 20, 5)
        self.params = nn.ModuleDict({"conv": self.conv})
        self.params2 = nn.ModuleDict({"conv": self.conv})

    def forward(self, x):
        return self.conv(x)


if __name__ == '__main__':
    m = TestModule()
    for idx,p in enumerate(m.parameters()):
        print(idx, ":", p)

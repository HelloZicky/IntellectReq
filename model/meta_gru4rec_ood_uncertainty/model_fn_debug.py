import torch
import torch.nn as nn

import sys

sys.path.append("../..")
sys.path.append("../../..")
sys.path.append("./")
print(sys.path)
from module import encoder, common, initializer
from util import consts
from util import uncertainty_utils
import config
from model_meta import MetaType, model

import logging

logger = logging.getLogger(__name__)


@model("meta_gru4rec_ood_uncertainty_separate", MetaType.ModelBuilder)
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

        # self._encoder = common.StackedDense(
        #     model_conf.id_dimension,
        #     [model_conf.id_dimension // 2],
        #     [torch.nn.Tanh] + [None]
        # )
        self._encoder = nn.Linear(model_conf.id_dimension, model_conf.id_dimension // 2)
        self._encoder_act = nn.Tanh()
        self._mean_linear = nn.Linear(model_conf.id_dimension // 2, model_conf.id_dimension // 2)
        self._var_linear = nn.Linear(model_conf.id_dimension // 2, model_conf.id_dimension // 2)
        # self._decoder = common.StackedDense(
        #     model_conf.id_dimension // 2,
        #     [model_conf.id_dimension],
        #     [torch.nn.Tanh] + [None]
        # )

        # self._decoder = nn.Linear(model_conf.id_dimension // 2, model_conf.id_dimension)
        self._decoder = nn.Linear(model_conf.id_dimension, model_conf.id_dimension)

        # self._classifier = common.StackedDense(
        #     model_conf.id_dimension * 2,
        #     model_conf.classifier + [1],
        #     ([torch.nn.Tanh] * len(model_conf.classifier)) + [None]
        # )

        # self._ood_classifier = common.StackedDense(
        #     model_conf.id_dimension * 2,
        #     model_conf.classifier + [1],
        #     ([torch.nn.Tanh] * len(model_conf.classifier)) + [None]
        # )

        # # self._meta_classifier_param_list = common.HyperNetwork_FC(
        # self._meta_classifier_param_list = common.HyperNetwork_FC_ood(
        #     # model_conf.id_dimension * 2,
        #     model_conf.id_dimension,
        #     # model_conf.classifier + [1],
        #     [model_conf.id_dimension] * model_conf.mlp_layers,
        #     # ([torch.nn.Tanh] * len(model_conf.classifier)) + [None],
        #     ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None],
        #     batch=True,
        #     # trigger_sequence_len=10,
        #     model_conf=model_conf
        # )

    def prior_net(self, z):
        mu, log_var = self.vae_encoder(z)
        z = self.vae_sampler(mu, log_var)
        z_reconstruct = self.vae_decoder(z)
        return z_reconstruct

    def posterior_net(self, z):
        mu, log_var = self.vae_encoder(z)
        z = self.vae_sampler(mu, log_var)
        z_reconstruct = self.vae_decoder(z)
        return z_reconstruct

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

        user_embedding, trigger_embed_ = self._meta_classifier_param_list(user_state, trigger_embed,
                                                                          user_state.size()[0],
                                                                          trigger_seq_length)

        ### vae uncertainty module

        if not pred:
            # prior net
            z = self.prior_net(user_embedding)
            next_item_embedding_pred = self.vae_decoder(torch.concat([user_embedding, z]))
            # posterior net
            analytic_kl = True
            kl_loss = self.kl_divergence(analytic=analytic_kl, calculate_posterior=False, z_posterior=z_posterior)
            label = features[consts.FIELD_LABEL]
            kl_coefficient = torch.where(label == 1, 1, -1)
            # next_item_embedding = user_embedding_reconstruct
            # criterion = nn.BCEWithLogitsLoss()
            criterion = nn.MSELoss()
            loss_regression = criterion(next_item_embedding_pred, target_embed)
            cvae_loss = -(loss_regression + kl_coefficient * kl_loss)

            return cvae_loss

        else:
            sample_times = 10
            # prior net
            user_embedding_ = user_embedding.unsqueeze(1)
            user_embedding_ = user_embedding_.repeat(1, sample_times, 1)
            z_ = self.prior_net(user_embedding_)
            # uncertainty = torch.var(z, dim=1)
            request_index = torch.where(torch.sigmoid(ood_pred).detach().cpu() < train_ood_threshold, True, False).view(-1)
            next_item_embedding_pred = self.vae_decoder(torch.concat([user_embedding_, z_]))
            uncertainty = torch.var(next_item_embedding_pred, dim=1)
            return uncertainty

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


# class ProbabilisticUnet(nn.Module):
class ProbabilisticRec(nn.Module):
    """
    A probabilistic UNet (https://arxiv.org/abs/1806.05034) implementation.
    input_channels: the number of channels in the image (1 for greyscale and 3 for RGB)
    num_classes: the number of classes to predict
    num_filters: is a list consisint of the amount of filters layer
    latent_dim: dimension of the latent space
    no_cons_per_block: no convs per block in the (convolutional) encoder of prior and posterior
    """

    # def __init__(self, input_channels=1, num_classes=1, num_filters=[32, 64, 128, 192], latent_dim=6, no_convs_fcomb=4, beta=10.0):
    def __init__(self, input_channels=1, num_classes=1, num_filters=[32, 64, 128, 192], latent_dim=6, no_convs_fcomb=4, beta=10.0):
        super(ProbabilisticUnet, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.no_convs_per_block = 3
        self.no_convs_fcomb = no_convs_fcomb
        self.initializers = {'w': 'he_normal', 'b': 'normal'}
        self.beta = beta
        self.z_prior_sample = 0

        self.unet = Unet(self.input_channels, self.num_classes, self.num_filters, self.initializers, apply_last_layer=False, padding=True).to(device)
        self.prior = AxisAlignedConvGaussian(self.input_channels, self.num_filters, self.no_convs_per_block, self.latent_dim, self.initializers, ).to(device)
        self.posterior = AxisAlignedConvGaussian(self.input_channels, self.num_filters, self.no_convs_per_block, self.latent_dim, self.initializers, posterior=True).to(device)
        self.fcomb = Fcomb(self.num_filters, self.latent_dim, self.input_channels, self.num_classes, self.no_convs_fcomb, {'w': 'orthogonal', 'b': 'normal'}, use_tile=True).to(device)

    def forward(self, patch, segm, training=True):
        """
        Construct prior latent space for patch and run patch through UNet,
        in case training is True also construct posterior latent space
        """
        if training:
            self.posterior_latent_space = self.posterior.forward(patch, segm)
        self.prior_latent_space = self.prior.forward(patch)
        self.unet_features = self.unet.forward(patch, False)

    def sample(self, testing=False):
        """
        Sample a segmentation by reconstructing from a prior sample
        and combining this with UNet features
        """
        if testing == False:
            z_prior = self.prior_latent_space.rsample()
            self.z_prior_sample = z_prior
        else:
            # You can choose whether you mean a sample or the mean here. For the GED it is important to take a sample.
            # z_prior = self.prior_latent_space.base_dist.loc
            z_prior = self.prior_latent_space.sample()
            self.z_prior_sample = z_prior
        return self.fcomb.forward(self.unet_features, z_prior)

    def reconstruct(self, use_posterior_mean=False, calculate_posterior=False, z_posterior=None):
        """
        Reconstruct a segmentation from a posterior sample (decoding a posterior sample) and UNet feature map
        use_posterior_mean: use posterior_mean instead of sampling z_q
        calculate_posterior: use a provided sample or sample from posterior latent space
        """
        if use_posterior_mean:
            z_posterior = self.posterior_latent_space.loc
        else:
            if calculate_posterior:
                z_posterior = self.posterior_latent_space.rsample()
        return self.fcomb.forward(self.unet_features, z_posterior)

    def kl_divergence(self, analytic=True, calculate_posterior=False, z_posterior=None):
        """
        Calculate the KL divergence between the posterior and prior KL(Q||P)
        analytic: calculate KL analytically or via sampling from the posterior
        calculate_posterior: if we use samapling to approximate KL we can sample here or supply a sample
        """
        if analytic:
            # Neeed to add this to torch source code, see: https://github.com/pytorch/pytorch/issues/13545
            kl_div = kl.kl_divergence(self.posterior_latent_space, self.prior_latent_space)
        else:
            if calculate_posterior:
                z_posterior = self.posterior_latent_space.rsample()
            log_posterior_prob = self.posterior_latent_space.log_prob(z_posterior)
            log_prior_prob = self.prior_latent_space.log_prob(z_posterior)
            kl_div = log_posterior_prob - log_prior_prob
        return kl_div

    def elbo(self, segm, analytic_kl=True, reconstruct_posterior_mean=False):
        """
        Calculate the evidence lower bound of the log-likelihood of P(Y|X)
        """

        criterion = nn.BCEWithLogitsLoss(size_average=False, reduce=False, reduction=None)
        z_posterior = self.posterior_latent_space.rsample()

        self.kl = torch.mean(self.kl_divergence(analytic=analytic_kl, calculate_posterior=False, z_posterior=z_posterior))

        # Here we use the posterior sample sampled above
        self.reconstruction = self.reconstruct(use_posterior_mean=reconstruct_posterior_mean, calculate_posterior=False, z_posterior=z_posterior)

        reconstruction_loss = criterion(input=self.reconstruction, target=segm)
        self.reconstruction_loss = torch.sum(reconstruction_loss)
        self.mean_reconstruction_loss = torch.mean(reconstruction_loss)

        return -(self.reconstruction_loss + self.beta * self.kl)


if __name__ == '__main__':
    import json
    args = parse_args()
    ap.print_arguments(args)
    args.model = "meta_gru4rec_ood_uncertainty_separate"
    model_uncertainty = model.get_model_meta(args.model_uncertainty)  # type: model.ModelMeta
    model_uncertainty_conf, raw_model_uncertainty_conf = ap.parse_arch_config_from_args(model_uncertainty, args)
    model_uncertainty_obj = model_meta.model_builder(model_conf=model_uncertainty_conf)
    model_conf = {
        "id_dimension": 32,
        "id_vocab": 8000,
        "classifier": [128, 64],
        "mlp_layers": 2
    }
    model = GRU4Rec(model_conf)

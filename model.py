"""
Consolidated model module.
Combines: distributions, loss, base_layers, models
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod
from scipy.interpolate import interp1d
from typing import Union, Optional
import argparse


# =============================================================================
# DISTRIBUTIONS (from distributions.py)
# =============================================================================

class ParametrizedGaussian:
    """Gaussian distribution with parametrized mean and sigma."""
    
    def __init__(self, mu, rho=None, sigma=None, name=''):
        self.name = name
        self.mu = mu
        if sigma is not None:
            self.sigma = sigma
        elif rho is not None:
            self.rho = rho
            self.sigma = torch.log1p(torch.exp(rho))
        else:
            raise ValueError("Either sigma or rho must be provided")
            
    def sample(self, n_samples=1):
        rho = getattr(self, 'rho', None)
        if rho is not None:
            rho_clamped = torch.clamp(rho, min=-8.0, max=8.0)
            sigma = torch.log1p(torch.exp(rho_clamped))
        else:
            sigma = self.sigma
        epsilon = torch.randn(n_samples, *self.mu.shape, device=self.mu.device)
        samples = self.mu + sigma * epsilon
        if torch.isnan(samples).any():
            samples = torch.where(torch.isnan(samples), torch.randn_like(samples) * 0.01, samples)
        return samples
    
    def log_prob(self, value, sigma=None):
        if sigma is None:
            sigma = self.sigma
        if torch.isnan(value).any() or torch.isnan(self.mu).any() or torch.isnan(sigma).any():
            return torch.tensor(-1e10, device=value.device)
        log_prob = -0.5 * torch.log(2 * math.pi * sigma**2) - 0.5 * ((value - self.mu) / sigma)**2
        if torch.isnan(log_prob).any():
            log_prob = torch.where(torch.isnan(log_prob), torch.tensor(-1e10, device=log_prob.device), log_prob)
        return log_prob
        
    def entropy(self):
        return 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(self.sigma)


class ScaleMixtureGaussian:
    """Scale mixture of two Gaussians as prior."""
    
    def __init__(self, pi, sigma1, sigma2, name=''):
        self.name = name
        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.gaussian1 = torch.distributions.Normal(0, sigma1)
        self.gaussian2 = torch.distributions.Normal(0, sigma2)
        
    def log_prob(self, value, sigma=None):
        if torch.isnan(value).any():
            return torch.zeros_like(value) - 1e10
        log_prob1 = torch.clamp(self.gaussian1.log_prob(value), min=-1e10, max=1e10)
        log_prob2 = torch.clamp(self.gaussian2.log_prob(value), min=-1e10, max=1e10)
        log_mix = torch.log(self.pi * torch.exp(log_prob1) + (1 - self.pi) * torch.exp(log_prob2))
        if torch.isnan(log_mix).any():
            log_mix = torch.where(torch.isnan(log_mix), torch.tensor(-1e10, device=log_mix.device), log_mix)
        return log_mix


class InverseGamma:
    """Inverse Gamma distribution."""
    
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        
    def log_prob(self, value):
        return self.alpha * torch.log(self.beta) - torch.lgamma(torch.tensor(self.alpha)) - (self.alpha + 1) * torch.log(value) - self.beta / value
    
    def exp_inverse(self):
        return self.alpha / self.beta
    
    def exp_log(self):
        return torch.digamma(self.alpha) - torch.log(self.beta)
    
    def entropy(self):
        return self.alpha + torch.log(self.beta) + torch.lgamma(self.alpha) - (1 + self.alpha) * torch.digamma(self.alpha)
    
    def update(self, new_alpha, new_beta):
        self.alpha = new_alpha
        self.beta = new_beta


# =============================================================================
# LOSS FUNCTIONS (from loss.py)
# =============================================================================

def masked_logsumexp(x, mask, dim=-1):
    """Computes logsumexp over elements specified by mask."""
    max_val, _ = (x * mask).max(dim=dim)
    max_val = torch.clamp_min(max_val, 0)
    return torch.log(torch.sum(torch.exp((x - max_val.unsqueeze(dim)) * mask) * mask, dim=dim)) + max_val


def mtlr_nll(logits, target, model, C1, average=False):
    """Computes negative log-likelihood for MTLR."""
    device = logits.device
    
    if target.shape[1] > logits.shape[1]:
        if target.shape[1] == logits.shape[1] + 1:
            target = target[:, :-1]
        elif target.shape[1] == logits.shape[1] + 2:
            target = target[:, :-2]
        else:
            target = target[:, :logits.shape[1]]
    
    assert logits.shape == target.shape, f"Shape mismatch: logits {logits.shape} != target {target.shape}"
    
    censored = target.sum(dim=1) > 1
    nll_censored = masked_logsumexp(logits[censored], target[censored]).sum() if censored.any() else torch.tensor(0.0, device=device)
    nll_uncensored = (logits[~censored] * target[~censored]).sum() if (~censored).any() else torch.tensor(0.0, device=device)
    norm = torch.logsumexp(logits, dim=1).sum()
    nll_total = -(nll_censored + nll_uncensored - norm)

    if average:
        nll_total = nll_total / target.size(0)

    l2_reg = torch.tensor(0.0, device=device)
    for k, v in model.named_parameters():
        if "mtlr_weight" in k or "weight" in k:
            l2_reg += torch.sum(v ** 2)
    nll_total += (C1 / 2) * l2_reg
    return nll_total


def cox_nll(risk_pred, true_times, true_indicator, model, C1):
    """Computes negative log-likelihood for Cox model."""
    eps = 1e-20
    risk_pred = risk_pred.reshape(-1, 1)
    true_times = true_times.reshape(-1, 1)
    true_indicator = true_indicator.reshape(-1, 1)
    mask = torch.ones(true_times.shape[0], true_times.shape[0]).to(true_times.device)
    mask[(true_times.T - true_times) > 0] = 0
    max_risk = risk_pred.max()
    log_loss = torch.exp(risk_pred - max_risk) * mask
    log_loss = torch.sum(log_loss, dim=0)
    log_loss = torch.log(log_loss + eps).reshape(-1, 1) + max_risk
    neg_log_loss = -torch.sum((risk_pred - log_loss) * true_indicator) / torch.sum(true_indicator)

    for k, v in model.named_parameters():
        if "weight" in k:
            neg_log_loss += C1/2 * torch.norm(v, p=2)
    return neg_log_loss


# =============================================================================
# BASE LAYERS (from base_layers.py)
# =============================================================================

def exp_log_inverse_gamma(shape, exp_rate, exp_log_rate, exp_log_x, exp_x_inverse):
    exp_log = - torch.lgamma(shape) + shape * exp_log_rate - (shape + 1) * exp_log_x - exp_rate * exp_x_inverse
    return torch.sum(exp_log)


def exp_log_gaussian(mean, std):
    dim = mean.shape[0] * mean.shape[1]
    return - 0.5 * dim * (torch.log(torch.tensor(2 * math.pi))) - 0.5 * (torch.sum(mean ** 2) + torch.sum(std ** 2))


class BayesianLinear(nn.Module):
    """Bayesian linear layer with scale mixture prior."""

    def __init__(self, in_features, out_features, config=None, prior_pi=0.5, 
                 prior_sigma1=1.0, prior_sigma2=0.0025, bias=True, name=''):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        if config is not None:
            self.prior_pi = getattr(config, 'pi', prior_pi)
            self.prior_sigma1 = getattr(config, 'sigma1', prior_sigma1)
            self.prior_sigma2 = getattr(config, 'sigma2', prior_sigma2)
        else:
            self.prior_pi = prior_pi
            self.prior_sigma1 = prior_sigma1
            self.prior_sigma2 = prior_sigma2
        
        self.weight = ScaleMixtureGaussian(pi=self.prior_pi, sigma1=self.prior_sigma1, sigma2=self.prior_sigma2)
        if self.use_bias:
            self.bias = ScaleMixtureGaussian(pi=self.prior_pi, sigma1=self.prior_sigma1, sigma2=self.prior_sigma2)
        
        self.W_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.W_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        if self.use_bias:
            self.b_mu = nn.Parameter(torch.Tensor(out_features))
            self.b_rho = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('b_mu', None)
            self.register_parameter('b_rho', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.W_mu, mode='fan_in', nonlinearity='relu')
        self.W_rho.data.fill_(-4.0)
        if self.use_bias:
            nn.init.zeros_(self.b_mu)
            self.b_rho.data.fill_(-4.0)

    def forward(self, x, sample=True, n_samples=None):
        if n_samples is None:
            n_samples = 10
        
        if len(x.shape) == 2:
            batch_size = x.shape[0]
            if sample:
                x = x.unsqueeze(0).expand(n_samples, batch_size, -1)
        
        if sample:
            weight = self.sample_weights(n_samples)
            bias = self.sample_bias(n_samples) if self.use_bias else None
        else:
            weight = self.W_mu
            bias = self.b_mu if self.use_bias else None
        
        if torch.isnan(weight).any():
            weight = self.W_mu.unsqueeze(0).expand(n_samples, -1, -1)
        if self.use_bias and bias is not None and torch.isnan(bias).any():
            bias = self.b_mu.unsqueeze(0).expand(n_samples, -1)
        
        if sample:
            output = torch.bmm(weight, x.transpose(1, 2)).transpose(1, 2)
            if self.use_bias:
                output += bias.unsqueeze(1)
        else:
            output = F.linear(x, weight, bias)
        
        if torch.isnan(output).any():
            if sample:
                w_mean = self.W_mu.unsqueeze(0).expand(n_samples, -1, -1)
                b_mean = self.b_mu.unsqueeze(0).expand(n_samples, -1) if self.use_bias else None
                output = torch.bmm(w_mean, x.transpose(1, 2)).transpose(1, 2)
                if self.use_bias:
                    output += b_mean.unsqueeze(1)
            else:
                output = F.linear(x, self.W_mu, self.b_mu if self.use_bias else None)
        return output

    def sample_weights(self, n_samples=1):
        epsilon_w = torch.randn(n_samples, self.out_features, self.in_features, device=self.W_mu.device)
        W_rho = torch.clamp(self.W_rho, min=-8.0, max=8.0)
        W_sigma = torch.log1p(torch.exp(W_rho))
        return self.W_mu + W_sigma * epsilon_w

    def sample_bias(self, n_samples=1):
        if not self.use_bias:
            return None
        epsilon_b = torch.randn(n_samples, self.out_features, device=self.b_mu.device)
        b_rho = torch.clamp(self.b_rho, min=-8.0, max=8.0)
        b_sigma = torch.log1p(torch.exp(b_rho))
        return self.b_mu + b_sigma * epsilon_b

    @property
    def log_prior(self):
        log_prior_w = self.weight.log_prob(self.W_mu, torch.log1p(torch.exp(self.W_rho))).sum()
        if self.use_bias:
            log_prior_b = self.bias.log_prob(self.b_mu, torch.log1p(torch.exp(self.b_rho))).sum()
            return log_prior_w + log_prior_b
        return log_prior_w

    @property
    def log_variational_posterior(self):
        W_rho = torch.clamp(self.W_rho, min=-8.0, max=8.0)
        W_sigma = torch.log1p(torch.exp(W_rho))
        W_gaussian = ParametrizedGaussian(mu=self.W_mu, sigma=W_sigma)
        W_log_posterior = W_gaussian.log_prob(self.W_mu, W_sigma).sum()
        if self.use_bias:
            b_rho = torch.clamp(self.b_rho, min=-8.0, max=8.0)
            b_sigma = torch.log1p(torch.exp(b_rho))
            b_gaussian = ParametrizedGaussian(mu=self.b_mu, sigma=b_sigma)
            return W_log_posterior + b_gaussian.log_prob(self.b_mu, b_sigma).sum()
        return W_log_posterior


class BayesianElementwiseLinear(nn.Module):
    """Elementwise linear layer with mixture Gaussian prior."""

    def __init__(self, input_output_size, config):
        super().__init__()
        self.input_output_size = input_output_size
        self.config = config
        if self.config.mu_scale is None:
            self.config.mu_scale = 1. * np.sqrt(6. / input_output_size)

        self.weight_mu = nn.init.uniform_(nn.Parameter(torch.Tensor(input_output_size)),
                                          -self.config.mu_scale, self.config.mu_scale)
        self.weight_rho = nn.Parameter(torch.ones([input_output_size]) * self.config.rho_scale)
        self.weight = ParametrizedGaussian(self.weight_mu, self.weight_rho)
        self.weight_prior = ScaleMixtureGaussian(config.pi, config.sigma1, config.sigma2)
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, x, sample=True, n_samples=1):
        if self.training or sample:
            weight = self.weight.sample(n_samples=n_samples)
        else:
            weight = self.weight.mu.expand(n_samples, -1, -1)

        if self.training:
            self.log_prior = self.weight_prior.log_prob(weight)
            self.log_variational_posterior = self.weight.log_prob(weight)
        else:
            self.log_prior, self.log_variational_posterior = 0, 0

        weight = torch.einsum('bj, jk->bjk', weight,
                              torch.eye(weight.shape[1], dtype=weight.dtype, device=weight.device))
        x = x.expand(n_samples, -1, -1)
        return torch.einsum('bij,bjk->bik', x, weight)

    def reset_parameters(self):
        nn.init.uniform_(self.weight_mu, -self.config.mu_scale, self.config.mu_scale)
        nn.init.constant_(self.weight_rho, self.config.rho_scale)
        self.weight = ParametrizedGaussian(self.weight_mu, self.weight_rho)


class BayesianHorseshoeLayer(nn.Module):
    """Horseshoe prior linear layer."""

    def __init__(self, in_features, out_features, config):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config

        self.beta_mu = nn.init.xavier_uniform_(nn.Parameter(torch.Tensor(out_features, in_features)))
        self.beta_rho = nn.Parameter(torch.ones([out_features, in_features]) * config.rho_scale)
        self.beta = ParametrizedGaussian(self.beta_mu, self.beta_rho)
        self.bias_mu = nn.Parameter(torch.zeros(1, out_features))
        self.bias_rho = nn.Parameter(torch.ones([1, out_features]) * config.rho_scale)
        self.bias = ParametrizedGaussian(self.bias_mu, self.bias_rho)

        self.prior_tau_shape = torch.Tensor([0.5]).to(config.device)
        self.prior_lambda_shape = torch.Tensor([0.5]).to(config.device)
        self.prior_lambda_rate = torch.Tensor([1 / config.weight_cauchy_scale ** 2]).to(config.device)
        self.lambda_shape = self.prior_lambda_shape * torch.ones(in_features).to(config.device)
        self.lambda_rate = self.prior_lambda_rate * torch.ones(in_features).to(config.device)
        self.lambda_ = InverseGamma(self.lambda_shape, self.lambda_rate)

        distr = torch.distributions.HalfCauchy(1 / torch.sqrt(self.prior_lambda_rate))
        sample = distr.sample(torch.Size([in_features])).squeeze()
        self.log_tau_mean = nn.Parameter(torch.log(sample))
        self.log_tau_rho = nn.Parameter(torch.ones(in_features) * config.rho_scale)
        self.log_tau = ParametrizedGaussian(self.log_tau_mean, self.log_tau_rho)

        self.prior_v_shape = torch.Tensor([0.5]).to(config.device)
        self.prior_theta_shape = torch.Tensor([0.5]).to(config.device)
        self.prior_theta_rate = torch.Tensor([1 / config.global_cauchy_scale ** 2]).to(config.device)
        self.theta_shape = self.prior_theta_shape
        self.theta_rate = self.prior_theta_rate
        self.theta = InverseGamma(self.theta_shape, self.theta_rate)

        distr = torch.distributions.HalfCauchy(1 / torch.sqrt(self.prior_theta_rate))
        sample = distr.sample()
        self.log_v_mean = nn.Parameter(torch.log(sample))
        self.log_v_rho = nn.Parameter(torch.ones(1) * config.rho_scale)
        self.log_v = ParametrizedGaussian(self.log_v_mean, self.log_v_rho)

    @property
    def log_prior(self):
        shape = self.prior_tau_shape
        exp_lambda_inverse = self.lambda_.exp_inverse()
        exp_log_lambda = self.lambda_.exp_log()
        exp_log_tau = self.log_tau.mu
        exp_tau_inverse = torch.exp(- self.log_tau.mu + 0.5 * self.log_tau.sigma ** 2)
        log_inv_gammas_weight = exp_log_inverse_gamma(shape, exp_lambda_inverse, -exp_log_lambda, exp_log_tau, exp_tau_inverse)
        shape = self.prior_lambda_shape
        rate = self.prior_lambda_rate
        log_inv_gammas_weight += exp_log_inverse_gamma(shape, rate, torch.log(rate), exp_log_lambda, exp_lambda_inverse)
        shape = self.prior_v_shape
        exp_theta_inverse = self.theta.exp_inverse()
        exp_log_theta = self.theta.exp_log()
        exp_log_v = self.log_v.mu
        exp_v_inverse = torch.exp(- self.log_v.mu + 0.5 * self.log_v.sigma ** 2)
        log_inv_gammas_global = exp_log_inverse_gamma(shape, exp_theta_inverse, -exp_log_theta, exp_log_v, exp_v_inverse)
        shape = self.prior_theta_shape
        rate = self.prior_theta_rate
        log_inv_gammas_global += exp_log_inverse_gamma(shape, rate, torch.log(rate), exp_log_theta, exp_theta_inverse)
        log_inv_gammas = log_inv_gammas_weight + log_inv_gammas_global
        log_gaussian = exp_log_gaussian(self.beta.mu, self.beta.sigma) + exp_log_gaussian(self.bias.mu, self.bias.sigma)
        return log_gaussian + log_inv_gammas

    @property
    def log_variational_posterior(self):
        entropy = (self.beta.entropy() + self.log_tau.entropy() + torch.sum(self.log_tau.mu) +
                   self.lambda_.entropy() + self.bias.entropy() + self.log_v.entropy() + 
                   torch.sum(self.log_v.mu) + self.theta.entropy())
        return -entropy

    def forward(self, x, sample=True, n_samples=1):
        if self.training or sample:
            beta = torch.clamp(self.beta.sample(n_samples), min=-1e6, max=1e6)
            log_tau = torch.clamp(torch.unsqueeze(self.log_tau.sample(n_samples), 1), min=-10.0, max=10.0)
            log_v = torch.clamp(torch.unsqueeze(self.log_v.sample(n_samples), 1), min=-10.0, max=10.0)
            bias = self.bias.sample(n_samples)
        else:
            beta = self.beta.mu.expand(n_samples, -1, -1)
            log_tau = self.log_tau.mu.expand(n_samples, -1, -1)
            log_v = self.log_v.mu.expand(n_samples, -1, -1)
            bias = self.bias.mu.expand(n_samples, -1, -1)

        weight = beta * log_tau * log_v
        x = x.expand(n_samples, -1, -1)
        return torch.einsum('bij,bkj->bik', x, weight) + bias

    def fixed_point_update(self):
        new_shape = torch.Tensor([1]).to(self.config.device)
        new_lambda_rate = (torch.exp(- self.log_tau.mu + 0.5 * (self.log_tau.sigma ** 2)) + self.prior_lambda_rate).to(self.config.device)
        new_theta_rate = (torch.exp(- self.log_v.mu + 0.5 * (self.log_v.sigma ** 2)) + self.prior_theta_rate).to(self.config.device)
        self.lambda_.update(new_shape, new_lambda_rate)
        self.theta.update(new_shape, new_theta_rate)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.beta_mu)
        nn.init.constant_(self.beta_rho, self.config.rho_scale)
        self.beta = ParametrizedGaussian(self.beta_mu, self.beta_rho)
        nn.init.constant_(self.bias_mu, 0)
        nn.init.constant_(self.bias_rho, self.config.rho_scale)
        self.bias = ParametrizedGaussian(self.bias_mu, self.bias_rho)


# =============================================================================
# MODELS (from models.py)
# =============================================================================

class BayesianBaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def reset_parameters(self):
        pass

    @abstractmethod
    def log_prior(self):
        pass

    @abstractmethod
    def log_variational_posterior(self):
        pass

    def get_name(self):
        return self._get_name()


class mtlr(nn.Module):
    """Multi-task logistic regression for survival prediction."""

    def __init__(self, in_features, num_time_bins, config):
        super().__init__()
        if num_time_bins < 1:
            raise ValueError("The number of time bins must be at least 1")
        self.config = config
        self.in_features = in_features
        self.num_time_bins = num_time_bins + 1

        self.mtlr_weight = nn.Parameter(torch.Tensor(self.in_features, self.num_time_bins - 1))
        self.mtlr_bias = nn.Parameter(torch.Tensor(self.num_time_bins - 1))
        self.register_buffer("G", torch.tril(torch.ones(self.num_time_bins - 1, self.num_time_bins, requires_grad=True)))
        self.reset_parameters()

    def forward(self, x):
        out = torch.matmul(x, self.mtlr_weight) + self.mtlr_bias
        return torch.matmul(out, self.G)

    def reset_parameters(self):
        nn.init.xavier_normal_(self.mtlr_weight)
        nn.init.constant_(self.mtlr_bias, 0.)

    def get_name(self):
        return self._get_name()


class BayesLinMtlr(BayesianBaseModel):
    """Bayesian Linear MTLR model."""

    def __init__(self, in_features, num_time_bins, config):
        torch.manual_seed(42)
        np.random.seed(42)
        super().__init__()
        if num_time_bins < 1:
            raise ValueError("The number of time bins must be at least 1")
        self.config = config
        self.in_features = in_features
        self.num_time_bins = num_time_bins + 1
        self.l1 = BayesianLinear(self.in_features, self.num_time_bins - 1, config=config)
        self.register_buffer("G", torch.tril(torch.ones(self.num_time_bins - 1, self.num_time_bins, requires_grad=True)))

    def forward(self, x, sample=True, n_samples=1):
        outputs = self.l1(x, sample, n_samples)
        final_output = torch.einsum('bij,jk->bik', outputs, self.G)
        if not self.training and not sample and final_output.dim() == 3:
            final_output = final_output.mean(dim=0)
        return final_output

    def log_prior(self):
        return self.l1.log_prior

    def log_variational_posterior(self):
        return self.l1.log_variational_posterior

    def sample_elbo(self, x, y, dataset_size, n_samples=10, see1=1):
        x = x.clone().detach()
        y = y.clone().detach()
        num_batch = max(1, dataset_size / max(1, self.config.batch_size))
        outputs = self.forward(x, sample=True, n_samples=n_samples)
        log_prior = self.log_prior() / n_samples
        log_variational_posterior = self.log_variational_posterior() / n_samples
        mean_outputs = outputs.mean(0)
        negative_log_likelihood = mtlr_nll(mean_outputs, y, model=self, C1=see1, average=False)
        loss = (log_variational_posterior - log_prior) / num_batch + negative_log_likelihood
        return loss, log_prior, log_variational_posterior, negative_log_likelihood

    def reset_parameters(self):
        self.l1.reset_parameters()
        return self


class BayesHsLinMtlr(BayesLinMtlr):
    """Bayesian Horseshoe Linear MTLR."""
    def __init__(self, in_features, num_time_bins, config):
        super(BayesLinMtlr, self).__init__()
        self.config = config
        self.in_features = in_features
        self.hidden_size = config.hidden_size
        self.num_time_bins = num_time_bins + 1
        self.l1 = BayesianHorseshoeLayer(self.in_features, self.num_time_bins - 1, config)
        self.register_buffer("G", torch.tril(torch.ones(self.num_time_bins - 1, self.num_time_bins, requires_grad=True)))

    def fixed_point_update(self):
        return self.l1.fixed_point_update()


class BayesEleMtlr(BayesianBaseModel):
    """Bayesian Elementwise MTLR."""
    def __init__(self, in_features, num_time_bins, config):
        super().__init__()
        self.config = config
        self.in_features = in_features
        self.hidden_size = in_features
        self.num_time_bins = num_time_bins
        self.l1 = BayesianElementwiseLinear(self.in_features, config)
        self.l2 = BayesianLinear(self.in_features, self.num_time_bins, config=config)
        self.register_buffer("G", torch.tril(torch.ones(self.num_time_bins, self.num_time_bins + 1, requires_grad=True)))

    def forward(self, x, sample, n_samples):
        this_batch_size = x.shape[0]
        x = F.dropout(F.relu(self.l1(x, n_samples=n_samples)), p=self.config.dropout)
        outputs = self.l2(x, sample, n_samples)
        outputs = outputs.reshape(n_samples, this_batch_size, self.num_time_bins - 1)
        G_with_samples = self.G.expand(n_samples, -1, -1)
        return torch.einsum('bij,bjk->bik', outputs, G_with_samples)

    def log_prior(self):
        return self.l1.log_prior + self.l2.log_prior

    def log_variational_posterior(self):
        return self.l1.log_variational_posterior + self.l2.log_variational_posterior

    def sample_elbo(self, x, y, dataset_size, see1=0):
        num_batch = dataset_size / self.config.batch_size
        n_samples = self.config.n_samples_train
        outputs = self(x, sample=True, n_samples=n_samples)
        log_prior = self.log_prior() / n_samples
        log_variational_posterior = self.log_variational_posterior() / n_samples
        nll = mtlr_nll(outputs.mean(dim=0), y, model=self, C1=see1, average=False)
        loss = (log_variational_posterior - log_prior) / num_batch + nll
        return loss, log_prior, log_variational_posterior, nll

    def reset_parameters(self):
        self.l1.reset_parameters()
        self.l2.reset_parameters()
        return self


class BayesMtlr(BayesianBaseModel):
    """Bayesian MTLR model."""
    def __init__(self, in_features, num_time_bins, config):
        super().__init__()
        self.config = config
        self.in_features = in_features
        self.num_time_bins = num_time_bins
        self.l1 = BayesianLinear(self.in_features, self.num_time_bins, config=config)
        self.register_buffer("G", torch.tril(torch.ones(self.num_time_bins, self.num_time_bins, requires_grad=False)))

    def forward(self, x, sample=False, n_samples=1):
        logits = self.l1(x, sample=sample, n_samples=n_samples)
        if n_samples > 1:
            batch_size = x.shape[0] if x.dim() > 1 else 1
            logits = logits.view(n_samples, batch_size, self.num_time_bins)
            G_expanded = self.G.expand(n_samples, -1, -1)
            return torch.einsum('bij,bjk->bik', logits, G_expanded)
        return torch.matmul(logits, self.G)

    def log_prior(self):
        return self.l1.log_prior

    def log_variational_posterior(self):
        return self.l1.log_variational_posterior

    def sample_elbo(self, x, target, dataset_size, see1=0):
        num_batch = dataset_size / self.config.batch_size
        n_samples = self.config.n_samples_train
        outputs = self(x, sample=True, n_samples=n_samples)
        log_prior = self.log_prior() / n_samples
        log_variational_posterior = self.log_variational_posterior() / n_samples
        if n_samples > 1:
            mean_outputs = outputs.mean(dim=0)
            nll = mtlr_nll(mean_outputs, target, self, see1)
        else:
            nll = mtlr_nll(outputs, target, self, see1)
        loss = (log_variational_posterior - log_prior) / num_batch + nll
        return loss, log_prior, log_variational_posterior, nll

    def reset_parameters(self):
        self.l1.reset_parameters()
        return self


class BayesHsMtlr(BayesEleMtlr):
    """Bayesian Horseshoe MTLR."""
    def __init__(self, in_features, num_time_bins, config):
        super(BayesEleMtlr, self).__init__()
        self.config = config
        self.in_features = in_features
        self.hidden_size = config.hidden_size
        self.num_time_bins = num_time_bins
        self.l1 = BayesianHorseshoeLayer(self.in_features, self.config.hidden_size, config)
        self.l2 = BayesianLinear(self.config.hidden_size, self.num_time_bins - 1, config=config)
        self.register_buffer("G", torch.tril(torch.ones(self.num_time_bins - 1, self.num_time_bins, requires_grad=True)))

    def fixed_point_update(self):
        return self.l1.fixed_point_update()


class CoxPH(nn.Module):
    """Cox proportional hazard model."""

    def __init__(self, in_features, config):
        super().__init__()
        self.config = config
        self.in_features = in_features
        self.time_bins = None
        self.cum_baseline_hazard = None
        self.baseline_survival = None
        self.l1 = nn.Linear(self.in_features, 1)

    def forward(self, x):
        return self.l1(x)

    def calculate_baseline_survival(self, x, t, e):
        outputs = self.forward(x)
        self.time_bins, self.cum_baseline_hazard, self.baseline_survival = baseline_hazard(outputs, t, e)

    def reset_parameters(self):
        self.l1.reset_parameters()
        return self

    def get_name(self):
        return self._get_name()


class BayesLinCox(BayesianBaseModel):
    """Bayesian Linear Cox model."""
    def __init__(self, in_features, config):
        super().__init__()
        self.config = config
        self.in_features = in_features
        self.time_bins = None
        self.cum_baseline_hazard = None
        self.baseline_survival = None
        self.l1 = BayesianLinear(self.in_features, 1, config)

    def forward(self, x, sample, n_samples):
        return self.l1(x, sample, n_samples)

    def calculate_baseline_survival(self, x, t, e):
        outputs = self.forward(x, sample=True, n_samples=self.config.n_samples_train).mean(dim=0)
        self.time_bins, self.cum_baseline_hazard, self.baseline_survival = baseline_hazard(outputs, t, e)

    def log_prior(self):
        return self.l1.log_prior

    def log_variational_posterior(self):
        return self.l1.log_variational_posterior

    def sample_elbo(self, x, t, e, dataset_size):
        n_samples = self.config.n_samples_train
        outputs = self(x, sample=True, n_samples=n_samples)
        log_prior = self.log_prior() / n_samples
        log_variational_posterior = self.log_variational_posterior() / n_samples
        nll = cox_nll(outputs.mean(dim=0), t, e, model=self, C1=0)
        loss = (log_variational_posterior - log_prior) / dataset_size + nll
        return loss, log_prior, log_variational_posterior, nll

    def reset_parameters(self):
        self.l1.reset_parameters()
        return self


class BayesHsLinCox(BayesLinCox):
    def __init__(self, in_features, config):
        super(BayesLinCox, self).__init__()
        self.config = config
        self.in_features = in_features
        self.time_bins = None
        self.cum_baseline_hazard = None
        self.baseline_survival = None
        self.l1 = BayesianHorseshoeLayer(self.in_features, 1, config)

    def fixed_point_update(self):
        return self.l1.fixed_point_update()


class BayesEleCox(BayesianBaseModel):
    def __init__(self, in_features, config):
        super().__init__()
        self.config = config
        self.in_features = in_features
        self.hidden_size = in_features
        self.time_bins = None
        self.cum_baseline_hazard = None
        self.baseline_survival = None
        self.l1 = BayesianElementwiseLinear(self.in_features, config)
        self.l2 = BayesianLinear(self.in_features, 1, config)

    def forward(self, x, sample, n_samples):
        x = F.dropout(F.relu(self.l1(x, n_samples=n_samples)), p=self.config.dropout)
        outputs = self.l2(x, sample, n_samples)
        return outputs.squeeze(dim=-1)

    def calculate_baseline_survival(self, x, t, e):
        outputs = self(x, sample=True, n_samples=self.config.n_samples_train).mean(dim=0)
        self.time_bins, self.cum_baseline_hazard, self.baseline_survival = baseline_hazard(outputs, t, e)

    def log_prior(self):
        return self.l1.log_prior + self.l2.log_prior

    def log_variational_posterior(self):
        return self.l1.log_variational_posterior + self.l2.log_variational_posterior

    def sample_elbo(self, x, t, e, dataset_size):
        num_batch = dataset_size / self.config.batch_size
        n_samples = self.config.n_samples_train
        outputs = self(x, sample=True, n_samples=n_samples)
        log_prior = self.log_prior() / n_samples
        log_variational_posterior = self.log_variational_posterior() / n_samples
        nll = cox_nll(outputs.mean(dim=0), t, e, model=self, C1=0)
        loss = (log_variational_posterior - log_prior) / (32 * dataset_size) + nll
        return loss, log_prior, log_variational_posterior / dataset_size, nll

    def reset_parameters(self):
        self.l1.reset_parameters()
        self.l2.reset_parameters()
        return self


class BayesCox(BayesEleCox):
    def __init__(self, in_features, config):
        super(BayesEleCox, self).__init__()
        self.config = config
        self.in_features = in_features
        self.hidden_size = config.hidden_size
        self.time_bins = None
        self.cum_baseline_hazard = None
        self.baseline_survival = None
        self.l1 = BayesianLinear(self.in_features, self.hidden_size, config)
        self.l2 = BayesianLinear(self.hidden_size, 1, config)


class BayesHsCox(BayesEleCox):
    def __init__(self, in_features, config):
        super(BayesEleCox, self).__init__()
        self.config = config
        self.in_features = in_features
        self.hidden_size = config.hidden_size
        self.time_bins = None
        self.cum_baseline_hazard = None
        self.baseline_survival = None
        self.l1 = BayesianHorseshoeLayer(self.in_features, self.hidden_size, config)
        self.l2 = BayesianLinear(self.hidden_size, 1, config)

    def fixed_point_update(self):
        return self.l1.fixed_point_update()


# =============================================================================
# SURVIVAL FUNCTIONS
# =============================================================================

def cox_survival(baseline_survival, linear_predictor):
    """Calculate individual survival distributions for Cox model."""
    n_sample = linear_predictor.shape[0]
    n_data = linear_predictor.shape[1]
    risk_score = torch.exp(linear_predictor)
    survival_curves = torch.empty((n_sample, n_data, baseline_survival.shape[0]), dtype=torch.float).to(linear_predictor.device)
    for i in range(n_sample):
        for j in range(n_data):
            survival_curves[i, j, :] = torch.pow(baseline_survival, risk_score[i, j])
    return survival_curves


def baseline_hazard(logits, time, event):
    """Calculate baseline cumulative hazard and survival using Breslow estimator."""
    from utils import compute_unique_counts, make_monotonic
    
    risk_score = torch.exp(logits)
    order = torch.argsort(time)
    risk_score = risk_score[order]
    uniq_times, n_events, n_at_risk, _ = compute_unique_counts(event, time, order)

    divisor = torch.empty(n_at_risk.shape, dtype=torch.float, device=n_at_risk.device)
    value = torch.sum(risk_score)
    divisor[0] = value
    k = 0
    for i in range(1, len(n_at_risk)):
        d = n_at_risk[i - 1] - n_at_risk[i]
        value -= risk_score[k:(k + d)].sum()
        k += d
        divisor[i] = value

    hazard = n_events / divisor
    if 0 not in uniq_times:
        uniq_times = torch.cat([torch.tensor([0]).to(uniq_times.device), uniq_times], 0)
        hazard = torch.cat([torch.tensor([0]).to(hazard.device), hazard], 0)
    cum_baseline_hazard = torch.cumsum(hazard.cpu(), dim=0).to(hazard.device)
    baseline_survival = torch.exp(- cum_baseline_hazard)
    if baseline_survival.isinf().any():
        last_zero = torch.where(baseline_survival == 0)[0][-1].item()
        baseline_survival[last_zero + 1:] = 0
    baseline_survival = make_monotonic(baseline_survival)
    return uniq_times, cum_baseline_hazard, baseline_survival


def mtlr_survival(logits, with_sample=True):
    """Generates predicted survival curves from MTLR logits."""
    if with_sample:
        assert logits.dim() == 3
        G = torch.tril(torch.ones(logits.shape[2], logits.shape[2])).to(logits.device)
        density = torch.softmax(logits, dim=2)
        G_with_samples = G.expand(density.shape[0], -1, -1)
        return torch.einsum('bij,bjk->bik', density, G_with_samples)
    else:
        assert logits.dim() == 2
        G = torch.tril(torch.ones(logits.shape[1], logits.shape[1])).to(logits.device)
        density = torch.softmax(logits, dim=1)
        return torch.matmul(density, G)


def mtlr_survival_at_times(logits, train_times, pred_times):
    """Generates predicted survival curves at arbitrary timepoints."""
    train_times = np.pad(train_times, (1, 0))
    surv = mtlr_survival(logits).detach().cpu().numpy()
    interpolator = interp1d(train_times, surv)
    return interpolator(np.clip(pred_times, 0, train_times.max()))


def mtlr_hazard(logits):
    """Computes hazard function from MTLR predictions."""
    return torch.softmax(logits, dim=1)[:, :-1] / (mtlr_survival(logits) + 1e-15)[:, 1:]


def mtlr_risk(logits):
    """Computes overall risk of event from MTLR predictions."""
    hazard = mtlr_hazard(logits)
    return torch.sum(hazard.cumsum(1), dim=1)


# Helper function to initialize model by name
def initialize_bayesian_model(input_dim, num_time_bins, config):
    """Returns the correct Bayesian model based on config.model."""
    if config.model == "BayesianLinearMTLR":
        return BayesLinMtlr(in_features=input_dim, num_time_bins=num_time_bins, config=config)
    elif config.model == "BayesianMTLR":
        return BayesMtlr(in_features=input_dim, num_time_bins=num_time_bins, config=config)
    elif config.model == "BayesianHorseshoeLinearMTLR":
        return BayesHsLinMtlr(in_features=input_dim, num_time_bins=num_time_bins, config=config)
    elif config.model == "BayesianElementwiseMTLR":
        return BayesEleMtlr(in_features=input_dim, num_time_bins=num_time_bins, config=config)
    elif config.model == "BayesianHorseshoeMTLR":
        return BayesHsMtlr(in_features=input_dim, num_time_bins=num_time_bins, config=config)
    else:
        raise NotImplementedError(f"Model {config.model} not implemented")

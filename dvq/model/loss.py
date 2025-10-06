"""
VQVAE losses, used for the reconstruction term in the ELBO
"""

import math
import torch

# -----------------------------------------------------------------------------


class LogitLaplace:
    """the Logit Laplace distribution log likelihood from OpenAI's DALL-E paper"""

    logit_laplace_eps = 0.1

    @classmethod
    def inmap(cls, x):
        # map [0,1] range to [eps, 1-eps]
        return (1 - 2 * cls.logit_laplace_eps) * x + cls.logit_laplace_eps

    @classmethod
    def unmap(cls, x):
        # inverse map, from [eps, 1-eps] to [0,1], with clamping
        return torch.clamp(
            (x - cls.logit_laplace_eps) / (1 - 2 * cls.logit_laplace_eps), 0, 1
        )

    @classmethod
    def nll(cls, x, mu_logb):
        mu, logb = mu_logb.chunk(2, 1)
        b = torch.exp(logb).clamp(min=1e-6)

        # log likelihood of the laplace distribution
        # see https://en.wikipedia.org/wiki/Laplace_distribution
        ll = -torch.abs(x - mu) / b - logb - math.log(2)

        # now correct for the fact that we are actually modeling a discretized
        # and clipped version of the distribution, as per DALL-E paper
        bin_size = 1.0 / 256.0 / (1 - 2 * cls.logit_laplace_eps)
        cdf_plus = 0.5 + 0.5 * torch.sign(x + bin_size - mu) * (
            1 - torch.exp(-torch.abs(x + bin_size - mu) / b)
        )
        cdf_min = 0.5 + 0.5 * torch.sign(x - bin_size - mu) * (
            1 - torch.exp(-torch.abs(x - bin_size - mu) / b)
        )
        ll_discrete = torch.log((cdf_plus - cdf_min).clamp(min=1e-6))

        # use discrete log likelihood if it is lower (more negative) than the
        # continuous one, this should help with the fact that we have pixels
        # that are exactly 0 or 1 which the continuous distribution cannot model
        ll = torch.where(ll_discrete < ll, ll_discrete, ll)

        return -ll.mean()  # return negative log likelihood as a loss


class Normal:
    """
    simple normal distribution with fixed variance, as used by DeepMind in their VQVAE
    note that DeepMind's reconstruction loss (I think incorrectly?) misses a factor of 2,
    which I have added to the normalizer of the reconstruction loss in nll(), we'll report
    number that is half of what we expect in their jupyter notebook
    """

    data_variance = (
        0.06327039811675479  # cifar-10 data variance, from deepmind sonnet code
    )

    @classmethod
    def inmap(cls, x):
        return x - 0.5  # map [0,1] range to [-0.5, 0.5]

    @classmethod
    def unmap(cls, x):
        return torch.clamp(x + 0.5, 0, 1)

    @classmethod
    def nll(cls, x, mu):
        return ((x - mu) ** 2).mean() / (
            2 * cls.data_variance
        )  # + math.log(math.sqrt(2 * math.pi * cls.data_variance))

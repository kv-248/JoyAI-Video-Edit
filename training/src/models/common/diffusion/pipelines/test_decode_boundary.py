"""Chunk-boundary VAE handoff must not draw a new posterior sample."""

import unittest

import torch

from src.models.common.diffusion.pipelines.streaming import Pipeline


class _Posterior:
    def __init__(self, mean: torch.Tensor):
        self.mean = mean

    def sample(self, generator=None):
        return self.mean + torch.randn_like(self.mean)

    def mode(self):
        return self.mean.clone()


class _EncodeOutput:
    def __init__(self, posterior: _Posterior):
        self.latent_dist = posterior


class _VAE:
    def __init__(self, mean: torch.Tensor):
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.mean = mean
        self.decode_inputs: list[torch.Tensor] = []

    def encode(self, boundary: torch.Tensor) -> _EncodeOutput:
        return _EncodeOutput(_Posterior(self.mean))

    def decode(self, latents: torch.Tensor, return_dict: bool = False):
        self.decode_inputs.append(latents.detach().clone())
        mixed = latents.sum(dim=2, keepdim=True).expand_as(latents).contiguous()
        return (mixed,)


class _Host:
    def __init__(self, vae: _VAE):
        self.vae = vae
        self.args = type("Args", (), {"vae_precision": "fp32"})()
        self.vae_scale_factor_temporal = 4
        self._autocast = Pipeline._autocast


class DecodeBoundaryTest(unittest.TestCase):
    def _decode(self, host: _Host, previous: torch.Tensor | None):
        latents = torch.zeros(1, 1, 4, 1, 1, 1)
        return Pipeline._decode_chunk(host, latents, previous, False)

    def test_boundary_anchor_is_the_posterior_mean(self):
        mean = torch.tensor([1.0, -2.0, 3.5, 0.25]).reshape(1, 4, 1, 1, 1)
        host = _Host(_VAE(mean))
        previous = torch.zeros(1, 4, 1, 1, 1)

        self._decode(host, previous)
        self._decode(host, previous)

        first, second = host.vae.decode_inputs
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(first[:, :, :1], mean))

    def test_first_chunk_does_not_reencode_a_boundary(self):
        host = _Host(_VAE(torch.zeros(1, 4, 1, 1, 1)))
        self._decode(host, None)
        self.assertEqual(len(host.vae.decode_inputs), 1)
        self.assertEqual(host.vae.decode_inputs[0].shape[2], 1)


if __name__ == "__main__":
    unittest.main()

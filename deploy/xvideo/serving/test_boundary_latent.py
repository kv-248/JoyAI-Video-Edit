"""Streaming decode handoff must reuse the boundary frame's posterior mean."""

import unittest

import torch

import xvideo.serving.joyomni_streaming as streaming
from xvideo.serving.joyomni_streaming import JoyOmniV2VStreamingSession


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
        self.mean = mean

    def encode(self, pixels: torch.Tensor) -> _EncodeOutput:
        return _EncodeOutput(_Posterior(self.mean))


class _Compile:
    def maybe_setup_encode(self, vae) -> None:
        return None

    def prep_input(self, pixels: torch.Tensor) -> torch.Tensor:
        return pixels


class BoundaryLatentTest(unittest.TestCase):
    def test_pseudo_latent_is_the_posterior_mean(self):
        mean = torch.tensor([0.5, -1.5, 2.0, 4.0]).reshape(1, 4, 1, 1, 1)
        stored: list[tuple[int, torch.Tensor]] = []

        host = type("Host", (), {})()
        host.pseudo_encode_vae = _VAE(mean)
        host.vae_dtype = torch.float32
        host.vae_autocast_enabled = False
        host._set_debug_state = lambda *args, **kwargs: None
        host._store_decode_pseudo_latent = lambda chunk_idx, latent: stored.append(
            (chunk_idx, latent.detach().clone())
        )

        pixels = torch.zeros(1, 3, 4, 1, 1)
        original = streaming._vae_compile_module
        streaming._vae_compile_module = lambda: _Compile()
        try:
            JoyOmniV2VStreamingSession._encode_next_decode_pseudo_latent(
                host, pixels, chunk_idx=3,
            )
            JoyOmniV2VStreamingSession._encode_next_decode_pseudo_latent(
                host, pixels, chunk_idx=3,
            )
        finally:
            streaming._vae_compile_module = original

        self.assertEqual([item[0] for item in stored], [4, 4])
        self.assertTrue(torch.equal(stored[0][1], stored[1][1]))
        self.assertTrue(torch.equal(stored[0][1], mean))


if __name__ == "__main__":
    unittest.main()

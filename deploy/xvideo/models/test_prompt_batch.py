"""Prompt repeats must keep each embedding paired with its own mask."""

from pathlib import Path
import unittest

import torch

from xvideo.models.prompt_batch import expand_prompts_per_video


class ExpandPromptsPerVideoTest(unittest.TestCase):
    def test_batch_masks_follow_repeated_embeddings(self):
        embeds = torch.tensor(
            [
                [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
                [[4.0, 40.0], [5.0, 50.0], [6.0, 60.0]],
            ]
        )
        masks = torch.tensor(
            [
                [1, 1, 0],
                [1, 0, 0],
            ]
        )

        out_embeds, out_masks = expand_prompts_per_video(embeds, masks, 2)

        self.assertEqual(tuple(out_embeds.shape), (4, 3, 2))
        self.assertEqual(tuple(out_masks.shape), (4, 3))
        for row in range(4):
            source = row // 2
            self.assertTrue(torch.equal(out_embeds[row], embeds[source]))
            self.assertTrue(torch.equal(out_masks[row], masks[source]))

    def test_single_prompt_repeat_is_unchanged(self):
        embeds = torch.arange(6, dtype=torch.float32).reshape(1, 3, 2)
        masks = torch.tensor([[1, 1, 0]])
        out_embeds, out_masks = expand_prompts_per_video(embeds, masks, 3)
        self.assertTrue(torch.equal(out_embeds, embeds.repeat_interleave(3, dim=0)))
        self.assertTrue(torch.equal(out_masks, masks.repeat_interleave(3, dim=0)))

    def test_one_video_is_identity(self):
        embeds = torch.randn(2, 4, 3)
        masks = torch.ones(2, 4)
        out_embeds, out_masks = expand_prompts_per_video(embeds, masks, 1)
        self.assertTrue(torch.equal(out_embeds, embeds))
        self.assertTrue(torch.equal(out_masks, masks))

    def test_pipeline_delegates_batch_expansion(self):
        source = Path(__file__).with_name("pipeline.py").read_text(encoding="utf-8")
        self.assertIn("return expand_prompts_per_video(", source)
        self.assertNotIn("prompt_embeds_mask.repeat(", source)

    def test_rejects_non_positive_count(self):
        embeds = torch.zeros(1, 2, 1)
        masks = torch.ones(1, 2)
        with self.assertRaises(ValueError):
            expand_prompts_per_video(embeds, masks, 0)
        with self.assertRaises(ValueError):
            expand_prompts_per_video(embeds, masks, True)


if __name__ == "__main__":
    unittest.main()

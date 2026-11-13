"""Repeat encoded prompts when a caller asks for several videos each."""

from __future__ import annotations

import torch


def expand_prompts_per_video(
    prompt_embeds: torch.Tensor,
    prompt_embeds_mask: torch.Tensor,
    num_videos_per_prompt: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Repeat each prompt on the batch axis, embeddings and mask together.

    One video per prompt returns the inputs unchanged. Several videos repeat
    rows as ``[p0, p0, p1, p1]`` so row ``i`` of the mask still belongs to row
    ``i`` of the embeddings.
    """
    if type(num_videos_per_prompt) is not int or num_videos_per_prompt <= 0:
        raise ValueError(
            f"`num_videos_per_prompt` must be a positive integer, got {num_videos_per_prompt}."
        )
    if num_videos_per_prompt == 1:
        return prompt_embeds, prompt_embeds_mask
    return (
        prompt_embeds.repeat_interleave(num_videos_per_prompt, dim=0),
        prompt_embeds_mask.repeat_interleave(num_videos_per_prompt, dim=0),
    )

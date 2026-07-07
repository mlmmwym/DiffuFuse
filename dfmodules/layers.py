from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn


class TextConditionedDiffusionGenerator(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        diffusion_steps=50,
        beta_start=1e-4,
        beta_end=0.02,
        dropout_prob=0.1,
        eval_step=None,
    ):
        super(TextConditionedDiffusionGenerator, self).__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size {} must be divisible by num_heads {}.".format(hidden_size, num_heads))
        self.diffusion_steps = diffusion_steps
        self.eval_step = eval_step if eval_step is not None else diffusion_steps // 2

        betas = torch.linspace(beta_start, beta_end, diffusion_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))

        self.time_embedding = nn.Embedding(diffusion_steps, hidden_size)
        self.text_condition_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout_prob,
            batch_first=True,
        )
        self.noise_predictor = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def _masked_mse(preds, targets, mask):
        mask = mask.to(preds.dtype).unsqueeze(-1)
        squared_error = (preds - targets) ** 2
        denom = (mask.sum() * preds.size(-1)).clamp(min=1.0)
        return (squared_error * mask).sum() / denom

    def forward(self, clean_tokens, text_tokens, text_token_mask):
        batch_size = clean_tokens.size(0)
        device = clean_tokens.device
        if self.training:
            timesteps = torch.randint(0, self.diffusion_steps, (batch_size,), device=device)
            noise = torch.randn_like(clean_tokens)
        else:
            eval_step = max(0, min(self.diffusion_steps - 1, int(self.eval_step)))
            timesteps = torch.full((batch_size,), eval_step, dtype=torch.long, device=device)
            noise = torch.zeros_like(clean_tokens)

        sqrt_alpha_bar = self.sqrt_alpha_bars[timesteps].view(batch_size, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bars[timesteps].view(batch_size, 1, 1)
        noised_tokens = sqrt_alpha_bar * clean_tokens + sqrt_one_minus_alpha_bar * noise
        timestep_features = self.time_embedding(timesteps).unsqueeze(1)

        conditioned_tokens, _ = self.text_condition_attention(
            query=noised_tokens + timestep_features,
            key=text_tokens,
            value=text_tokens,
            key_padding_mask=~text_token_mask,
            need_weights=False,
        )
        predicted_noise = self.noise_predictor(noised_tokens + conditioned_tokens + timestep_features)
        shared_tokens = (noised_tokens - sqrt_one_minus_alpha_bar * predicted_noise) / sqrt_alpha_bar.clamp(min=1e-8)
        diffusion_loss = self._masked_mse(predicted_noise, noise, text_token_mask)
        return shared_tokens, diffusion_loss

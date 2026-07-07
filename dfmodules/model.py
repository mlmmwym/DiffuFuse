from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn

from dfmodules.backbone import load_backbone
from dfmodules.layers import TextConditionedDiffusionGenerator


class DiffFuseRegression(nn.Module):
    """DiffFuse multimodal sentiment regression model."""

    def __init__(
        self,
        visual_dim=64,
        acoustic_dim=64,
        av_lstm_hidden_size=256,
        av_lstm_num_layers=1,
        cam_num_heads=8,
        diffusion_steps=50,
        diffusion_beta_start=1e-4,
        diffusion_beta_end=0.02,
        diffusion_eval_step=None,
        shared_tensor_fusion_dim=16,
        dropout_prob=0.1,
        cache_dir=None,
        local_files_only=False,
        backbone_path=None,
    ):
        super(DiffFuseRegression, self).__init__()
        self.encoder = load_backbone(
            backbone_path=backbone_path,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        hidden_size = self.encoder.config.hidden_size
        if hidden_size % cam_num_heads != 0:
            raise ValueError("encoder hidden_size {} must be divisible by cam_num_heads {}.".format(hidden_size, cam_num_heads))
        lstm_dropout = dropout_prob if av_lstm_num_layers > 1 else 0.0
        self.visual_lstm = nn.LSTM(
            input_size=visual_dim,
            hidden_size=av_lstm_hidden_size,
            num_layers=av_lstm_num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.acoustic_lstm = nn.LSTM(
            input_size=acoustic_dim,
            hidden_size=av_lstm_hidden_size,
            num_layers=av_lstm_num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.visual_projection = nn.Linear(av_lstm_hidden_size, hidden_size)
        self.acoustic_projection = nn.Linear(av_lstm_hidden_size, hidden_size)
        self.visual_diffusion = TextConditionedDiffusionGenerator(
            hidden_size=hidden_size,
            num_heads=cam_num_heads,
            diffusion_steps=diffusion_steps,
            beta_start=diffusion_beta_start,
            beta_end=diffusion_beta_end,
            dropout_prob=dropout_prob,
            eval_step=diffusion_eval_step,
        )
        self.acoustic_diffusion = TextConditionedDiffusionGenerator(
            hidden_size=hidden_size,
            num_heads=cam_num_heads,
            diffusion_steps=diffusion_steps,
            beta_start=diffusion_beta_start,
            beta_end=diffusion_beta_end,
            dropout_prob=dropout_prob,
            eval_step=diffusion_eval_step,
        )
        self.shared_text_projection = nn.Linear(hidden_size, shared_tensor_fusion_dim)
        self.shared_visual_projection = nn.Linear(hidden_size, shared_tensor_fusion_dim)
        self.shared_acoustic_projection = nn.Linear(hidden_size, shared_tensor_fusion_dim)
        self.shared_tensor_mlp = nn.Sequential(
            nn.LayerNorm(shared_tensor_fusion_dim ** 3),
            nn.Linear(shared_tensor_fusion_dim ** 3, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, hidden_size),
        )
        self.private_text_visual_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=cam_num_heads,
            dropout=dropout_prob,
            batch_first=True,
        )
        self.private_text_acoustic_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=cam_num_heads,
            dropout=dropout_prob,
            batch_first=True,
        )
        self.regressor = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, 1),
        )

    @staticmethod
    def _encode_lstm_sequence(lstm, sequence, sequence_lengths=None):
        if sequence_lengths is None:
            outputs, _ = lstm(sequence)
            return outputs

        max_length = sequence.size(1)
        lengths = sequence_lengths.clamp(min=1, max=max_length).detach().cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            sequence,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        packed_outputs, _ = lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outputs,
            batch_first=True,
            total_length=max_length,
        )
        return outputs

    @staticmethod
    def _masked_mean_pool(sequence, mask):
        mask = mask.to(sequence.dtype).unsqueeze(-1)
        summed = (sequence * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return summed / denom

    @staticmethod
    def _cross_modal_mean_pool(attention, modality_tokens, text_tokens, text_token_mask):
        fused_tokens, _ = attention(
            query=modality_tokens,
            key=text_tokens,
            value=modality_tokens,
            key_padding_mask=~text_token_mask,
            need_weights=False,
        )
        return DiffFuseRegression._masked_mean_pool(fused_tokens, text_token_mask)

    def _shared_tensor_fusion(self, text_vector, visual_vector, acoustic_vector):
        text_vector = self.shared_text_projection(text_vector)
        visual_vector = self.shared_visual_projection(visual_vector)
        acoustic_vector = self.shared_acoustic_projection(acoustic_vector)
        tensor_product = torch.einsum("bi,bj,bk->bijk", text_vector, visual_vector, acoustic_vector)
        return self.shared_tensor_mlp(tensor_product.reshape(tensor_product.size(0), -1))

    @staticmethod
    def _orthogonal_component(local_vector, global_vector, eps=1e-8):
        dot_product = (local_vector * global_vector).sum(dim=-1, keepdim=True)
        global_norm_sq = (global_vector * global_vector).sum(dim=-1, keepdim=True).clamp(min=eps)
        projection = dot_product / global_norm_sq * global_vector
        return local_vector - projection

    def forward(self, input_ids, attention_mask=None, visual=None, acoustic=None, sequence_lengths=None, return_aux=False):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        cls_token = outputs.last_hidden_state[:, 0, :]
        text_tokens = outputs.last_hidden_state[:, 1:, :]
        if attention_mask is None:
            text_token_mask = torch.ones(text_tokens.size()[:2], dtype=torch.bool, device=text_tokens.device)
        else:
            text_token_mask = attention_mask[:, 1:].bool()

        visual_tokens = self._encode_lstm_sequence(self.visual_lstm, visual, sequence_lengths)
        acoustic_tokens = self._encode_lstm_sequence(self.acoustic_lstm, acoustic, sequence_lengths)
        visual_tokens = self.visual_projection(visual_tokens)
        acoustic_tokens = self.acoustic_projection(acoustic_tokens)

        visual_shared_tokens, visual_diffusion_loss = self.visual_diffusion(
            clean_tokens=visual_tokens,
            text_tokens=text_tokens,
            text_token_mask=text_token_mask,
        )
        acoustic_shared_tokens, acoustic_diffusion_loss = self.acoustic_diffusion(
            clean_tokens=acoustic_tokens,
            text_tokens=text_tokens,
            text_token_mask=text_token_mask,
        )
        visual_private_tokens = visual_tokens - visual_shared_tokens
        acoustic_private_tokens = acoustic_tokens - acoustic_shared_tokens

        text_pooled = self._masked_mean_pool(text_tokens, text_token_mask)
        shared_visual_pooled = self._masked_mean_pool(visual_shared_tokens, text_token_mask)
        shared_acoustic_pooled = self._masked_mean_pool(acoustic_shared_tokens, text_token_mask)
        shared_pooled = self._shared_tensor_fusion(text_pooled, shared_visual_pooled, shared_acoustic_pooled)
        private_visual_pooled = self._cross_modal_mean_pool(
            self.private_text_visual_attention,
            visual_private_tokens,
            text_tokens,
            text_token_mask,
        )
        private_acoustic_pooled = self._cross_modal_mean_pool(
            self.private_text_acoustic_attention,
            acoustic_private_tokens,
            text_tokens,
            text_token_mask,
        )

        private_pooled = private_visual_pooled + private_acoustic_pooled
        global_pooled = cls_token + private_pooled
        shared_orthogonal_pooled = self._orthogonal_component(shared_pooled, global_pooled)
        fused = global_pooled + shared_orthogonal_pooled
        logits = self.regressor(fused).squeeze(-1)
        if return_aux:
            return {
                "logits": logits,
                "visual_diffusion_loss": visual_diffusion_loss,
                "acoustic_diffusion_loss": acoustic_diffusion_loss,
                "visual_shared_tokens": visual_shared_tokens,
                "acoustic_shared_tokens": acoustic_shared_tokens,
                "visual_private_tokens": visual_private_tokens,
                "acoustic_private_tokens": acoustic_private_tokens,
                "shared_pooled": shared_pooled,
                "private_pooled": private_pooled,
                "shared_orthogonal_pooled": shared_orthogonal_pooled,
            }
        return logits

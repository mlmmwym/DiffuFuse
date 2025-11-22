from torch import nn
import logging
import math
import torch
import sys

from config import get_args
import torch.nn.functional as F
from dfmodules.position_embedding import SinusoidalPositionalEmbedding
from dfmodules.transformer_layer import TransformerEncoderLayer, Linear, LayerNorm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import torch  
import numpy as np  
import struct  

def append_tensor_to_file(tensor, file_path):  
    """将tensor追加到文件中，使用简单的格式"""  
    # 转换为numpy数组  
    tensor_np = tensor.detach().cpu().numpy()  
    
    # 打开文件以追加二进制数据  
    with open(file_path, 'ab') as f:  
        # 先写入数组形状和数据类型信息  
        shape = tensor_np.shape  
        dtype = tensor_np.dtype.str  
        
        # 写入维度数量  
        f.write(struct.pack('I', len(shape)))  
        
        # 写入每个维度的大小  
        for dim in shape:  
            f.write(struct.pack('I', dim))  
        
        # 写入dtype字符串长度和内容  
        dtype_bytes = dtype.encode('utf-8')  
        f.write(struct.pack('I', len(dtype_bytes)))  
        f.write(dtype_bytes)  
        
        # 写入实际数据  
        f.write(tensor_np.tobytes())


logger = logging.getLogger(__name__)
args = get_args()
class AdapterConfig:
    project_hidden_size: int = args.hidden_size
    hidden_act: str = "gelu"
    adapter_size: int = 64  # 64
    adapter_initializer_range: float = 0.001
    is_decoder: bool = False
    attention_probs_dropout_prob: float = 0.1
    hidden_dropout_prob: float = 0.1
    hidden_size: int = 544
    out_hidden_size: int = project_hidden_size
    initializer_range: float = 0.02
    intermediate_size: int = 3072
    layer_norm_eps: float = 1e-05
    max_position_embeddings: int = 514
    num_attention_heads: int = 12
    num_labels: int = 2
    output_attentions: bool = False
    output_hidden_states: bool = False
    torchscript: bool = False
    type_vocab_size: int = 1
    vocab_size: int = 50265

class FFN_Adapter(nn.Module):
    #### 考虑的是多模态的情况
    def __init__(self, hp):
        super(FFN_Adapter, self).__init__()
        self.adapter_config =  AdapterConfig()
        self.multi = hp.multi
        self.visualize = hp.visualize
        self.adapter_layer = hp.adapter_layer
        if hp.multi:
            in_dim = self.adapter_config.project_hidden_size + hp.d_vout + hp.d_aout
        else:
            in_dim = self.adapter_config.project_hidden_size

        self.adapter_down_project = nn.Linear(in_dim,self.adapter_config.adapter_size)
        self.adapter_up_project = nn.Linear(self.adapter_config.adapter_size,in_dim)
        self.adapter_down_project.weight = torch.nn.Parameter(torch.normal(mean=0.0, std=self.adapter_config.adapter_initializer_range,
                                                                   size=(self.adapter_config.adapter_size, in_dim,)))
        self.adapter_down_project.bias = torch.nn.Parameter(torch.zeros(self.adapter_config.adapter_size))

        self.adapter_up_project.weight = torch.nn.Parameter(torch.normal(mean=0.0, std=self.adapter_config.adapter_initializer_range,
                                                    size=(in_dim, self.adapter_config.adapter_size,)))
        self.adapter_up_project.bias = torch.nn.Parameter(torch.zeros(in_dim))
        self.adapter_linear = nn.Linear(in_dim,self.adapter_config.out_hidden_size)
####
    def forward(self, hidden_states, visual=None, acoustic=None, id=3):
        # print("FF",hidden_states.shape,visual.shape,acoustic.shape)
        # exit(0)
        ### visualization应该保存第几个adapter的可视化结果
        if self.multi:
            seq_len = hidden_states.size(1)
            if len(visual.shape) == 1:
                visual = visual.unsqueeze(dim=0)
            if len(acoustic.shape) == 1:
                acoustic = acoustic.unsqueeze(dim=0)
            visual = visual.unsqueeze(dim=1).expand(visual.size(0),seq_len,visual.size(1))
            acoustic = acoustic.unsqueeze(dim=1).expand(acoustic.size(0),seq_len,acoustic.size(1))
            hidden_states = torch.cat([hidden_states, visual, acoustic], dim=-1)
            
        down_output = self.adapter_down_project(hidden_states)
        down_output_nolinear = torch.sigmoid(down_output)
        up_output = self.adapter_up_project(down_output_nolinear)
        output = up_output + hidden_states
        output = self.adapter_linear(output)
            
        if self.visualize:
            ###先尝试只做一个batch size内的可视化
            pool_hidden_state = torch.mean(hidden_states,dim=1).cpu().detach().numpy()
            
            X_tsne = TSNE(n_components=2,random_state=33).fit_transform(pool_hidden_state)
            X_pca = PCA(n_components=2).fit_transform(pool_hidden_state)

            ckpt_dir="images"
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

            plt.figure(figsize=(10, 5))
            plt.subplot(121)
            plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=None,label="t-SNE")
            plt.legend()
            plt.subplot(122)
            plt.scatter(X_pca[:, 0], X_pca[:, 1], c=None,label="PCA")
            plt.legend()
            plt.savefig('images/orig_tsne-pca_{}.png'.format(str(id)), dpi=120)
            # plt.show()
            
            pool_fusion = torch.mean(output, dim=1).cpu().detach().numpy()
            
            X_tsne = TSNE(n_components=2,random_state=33).fit_transform(pool_fusion)
            X_pca = PCA(n_components=2).fit_transform(pool_fusion)


            plt.figure(figsize=(10, 5))
            plt.subplot(121)
            plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=None,label="t-SNE")
            plt.legend()
            plt.subplot(122)
            plt.scatter(X_pca[:, 0], X_pca[:, 1], c=None,label="PCA")
            plt.legend()
            plt.savefig('images/fusion_tsne-pca_{}.png'.format(str(id)), dpi=120)
            # plt.show()
            
            

        return output
# class MAG(nn.Module):
#     def __init__(self,args=None):
#         super(MAG, self).__init__()
#         print(
#             "Initializing MAG with beta_shift:{} hidden_prob:{}".format(
#                 args.beta_shift, args.dropout_prob
#             )
#         )
#         # print(args)
#
#         self.W_hv = nn.Linear(args.d_vh + args.hidden_size, args.hidden_size)
#         self.W_ha = nn.Linear(args.d_ah + args.hidden_size, args.hidden_size)
#
#         self.W_v = nn.Linear(args.d_vh, args.hidden_size)
#         self.W_a = nn.Linear(args.d_ah, args.hidden_size)
#         self.beta_shift = args.beta_shift
#
#         self.LayerNorm = nn.LayerNorm(args.hidden_size)
#         self.dropout = nn.Dropout(args.dropout_prob)
#
#     def forward(self, text_embedding, visual, acoustic):
#         # print(text_embedding.shape)
#         # print(visual.shape)
#         # print(acoustic.shape)
#
#         # print('B')
#         # exit(0)
#         eps = 1e-6
#         weight_v = F.relu(self.W_hv(torch.cat((visual, text_embedding), dim=-1)))
#         weight_a = F.relu(self.W_ha(torch.cat((acoustic, text_embedding), dim=-1)))
#
#         h_m = weight_v * self.W_v(visual) + weight_a * self.W_a(acoustic)
#
#         em_norm = text_embedding.norm(2, dim=-1)
#         hm_norm = h_m.norm(2, dim=-1)
#
#         hm_norm_ones = torch.ones(hm_norm.shape, requires_grad=True).to(text_embedding.device)
#         hm_norm = torch.where(hm_norm == 0, hm_norm_ones, hm_norm)
#
#         thresh_hold = (em_norm / (hm_norm + eps)) * self.beta_shift
#
#         ones = torch.ones(thresh_hold.shape, requires_grad=True).to(text_embedding.device)
#
#         alpha = torch.min(thresh_hold, ones)
#         alpha = alpha.unsqueeze(dim=-1)
#
#         acoustic_vis_embedding = alpha * h_m
#
#         embedding_output = self.dropout(
#             self.LayerNorm(acoustic_vis_embedding + text_embedding)
#         )
#
#         return embedding_output
from dfmodules.denoising_diffusion_pytorch_1d import Unet1D, GaussianDiffusion1D


class Diffusion(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=torch.nn.Linear(768,32)
        model = Unet1D(
            dim=64,
            dim_mults=(1, 2, 4, 8),
            channels=3
        )

        self.diffusion = GaussianDiffusion1D(
            model,
            seq_length=32,
            timesteps=20,
            objective='pred_v'
        )
    def forward(self, T,AV):
        T=self.linear(T)
        loss=0.0
        T=T.unsqueeze(1)
        AV=AV.unsqueeze(1)
        if self.training:
            loss = self.diffusion(T.detach(), text_inf=AV.detach())
        AV_sampled_seq = self.diffusion.sample(batch_size=T.shape[0], text_inf=T)
        AV_sampled_seq=AV_sampled_seq.squeeze(1)
        return loss, AV_sampled_seq,AV.squeeze(1)-AV_sampled_seq.detach()
import torch
import torch.nn as nn

class TripleInteractionModel(nn.Module):
    def __init__(self, lenA: int=32, lenV: int=32):
        """
        Args:
            lenA: 输入A的序列长度（第二个模态的维度）
            lenV: 输入V的序列长度（第三个模态的维度）
        """
        super().__init__()
        # 第一阶段全连接：将 (..., lenV) 压缩到 1
        self.fc_v = nn.Linear(lenV, 1)
        # 第二阶段全连接：将 (..., lenA) 压缩到 1
        self.fc_a = nn.Linear(lenA, 1)

    def forward(self, T: torch.Tensor, A: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        Args:
            T: (batch_size, lenT)
            A: (batch_size, lenA)
            V: (batch_size, lenV)
        Returns:
            output: (batch_size, lenT)
        """
        ####################################
        # 阶段1：三维外积计算
        ####################################
        # 维度扩展（广播机制）
        T_exp = T.unsqueeze(2).unsqueeze(3)  # (batch, lenT, 1, 1)
        A_exp = A.unsqueeze(1).unsqueeze(3)  # (batch, 1, lenA, 1)
        V_exp = V.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, lenV)

        # 三维外积（逐元素相乘）
        outer_product = T_exp * A_exp * V_exp  # (batch, lenT, lenA, lenV)

        ####################################
        # 阶段2：逐步降维
        ####################################
        # 第一次降维：消除lenV维度
        x = outer_product.view(-1, outer_product.size(-1))  # (batch*lenT*lenA, lenV)
        x = torch.relu(self.fc_v(x))  # (batch*lenT*lenA, 1)
        x = x.view(*outer_product.shape[:-1], 1)  # (batch, lenT, lenA, 1)

        # 第二次降维：消除lenA维度
        x = x.squeeze(-1)  # (batch, lenT, lenA)
        x = x.view(-1, x.size(-1))  # (batch*lenT, lenA)
        x = torch.relu(self.fc_a(x))  # (batch*lenT, 1)
        x = x.view(x.size(0)//T.size(1), T.size(1))  # (batch, lenT)

        return x
class BEM(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc_v = nn.Linear(32, 1)
        # 第二阶段全连接：将 (..., lenA) 压缩到 1
        self.fc_a = nn.Linear(32, 1)

        self.fc_aup_1=nn.Linear(32, 768)
        self.fc_aup_2=nn.Linear(32, 768)

        self.fc_vup_1=nn.Linear(32, 768)
        self.fc_vup_2=nn.Linear(32, 768)

    def forward(self, T: torch.Tensor, A: torch.Tensor, V: torch.Tensor):

        AKey=torch.tanh(self.fc_aup_1(A))
        AValue=torch.tan(self.fc_aup_2(A))

        VKey=torch.tanh(self.fc_vup_1(V))
        VValue=torch.tanh(self.fc_vup_2(V))


        T_exp = T.unsqueeze(3)  # (batch,seq, lenT, 1)
        A_exp = A.unsqueeze(2)  # (batch,seq ,1, lenA)
        V_exp = V.unsqueeze(2)  # (batch,seq, 1, lenV)

        TA=T_exp*A_exp  # (batch,seq, lenT, lenA)
        TV=T_exp*V_exp  # (batch,seq, lenT, lenV)

        TA=torch.tanh(self.fc_a(TA))
        TV=torch.tanh(self.fc_v(TV))

        TAQ=TA.squeeze()     # (batch,seq, lenT)
        TVQ=TV.squeeze()     # (batch,seq, lenT)

        TAQK=TAQ*VKey
        TVQK=TVQ*AKey

        TAQK=torch.sum(TAQK,dim=-1)
        TVQK=torch.sum(TVQK,dim=-1)

        TAQK=torch.softmax(TAQK,dim=-1)
        TVQK=torch.softmax(TVQK,dim=-1)

        TAQK=TAQK.unsqueeze(2)
        TVQK=TVQK.unsqueeze(2)

        AValue=AValue*TAQK
        VValue=VValue*TVQK

        return AValue,VValue
from functools import partial
from typing import Callable
from torchvision.models.vision_transformer import MLPBlock
class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int=12,
        hidden_dim: int=768,
        mlp_dim: int=768,
        dropout: float=0.5,
        attention_dropout: float=0.0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor,KV):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        KV=self.ln_1(KV)
        x, _ = self.self_attention(x, KV, KV, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y
def matrix_orthogonal_fusion(tensor1, tensor2):
    """
    对两个形状为 (length, batch_size, dim) 的张量进行矩阵正交融合。
    对于每个位置 (i, b)，进行：
    fused_vec = vec_g + (vec_l - ((vec_l · vec_g)/(vec_g · vec_g)) * vec_g)

    参数：
        tensor1: (length, batch_size, dim)
        tensor2: (length, batch_size, dim)
    返回：
        fused_tensor: (length, batch_size, dim)
    """
    tensor1=tensor1.permute(1,0,2)
    tensor2=tensor2.permute(1,0,2)
    # torch.
    # tensor1 对应 vec_g，tensor2 对应 vec_l
    # 计算内积 (vec_l · vec_g) 和 (vec_g · vec_g)
    # shape: (length, batch_size)
    dot_BA = torch.sum(tensor2 * tensor1, dim=-1)  # (l, b)
    dot_AA = torch.sum(tensor1 * tensor1, dim=-1)  # (l, b)

    # ratio = (vec_l · vec_g) / (vec_g · vec_g), shape: (l, b)
    ratio = dot_BA / dot_AA

    # 投影分量 proj_component = ratio * vec_g，shape: (l, b, 1) * (l, b, dim) => (l, b, dim)
    proj_component = ratio.unsqueeze(-1) * tensor1

    # 正交分量 orth_component = vec_l - proj_component
    orth_component = tensor2 - proj_component

    append_tensor_to_file(torch.sum(orth_component.permute(1,0,2)[:,:,:],dim=1),"/root/lxj/UniMSE/plot_bars/To1.pt")
    append_tensor_to_file(torch.sum(tensor1.permute(1,0,2)[:,:,:],dim=1),"/root/lxj/UniMSE/plot_bars/To2.pt")

    # 融合结果 fused = vec_g + orth_component
    fused_tensor = tensor1 + orth_component
    fused_tensor = fused_tensor.permute(1,0,2)
    return fused_tensor
class MAG_Adapter(nn.Module):
    #### 考虑的是多模态的情况
    def __init__(self, hp):
        super(MAG_Adapter, self).__init__()
        # self.mag=MAG(hp)
        self.ta_diffusion=Diffusion()
        self.tv_diffusion=Diffusion()
        self.tripleInteractionModel=TripleInteractionModel()
        self.BEM=BEM()
        self.EncoderBlockTA=EncoderBlock()
        self.EncoderBlockTV=EncoderBlock()
    def forward(self,text_embedding, visual, acoustic,IS_BAG_list,attention_mask):
        IS_BAG_list=IS_BAG_list.permute(1,0)

        attention_mask = attention_mask.squeeze()

        V=torch.zeros([text_embedding.shape[0],text_embedding.shape[1],visual.shape[-1]],device=text_embedding.device)
        V[torch.where(attention_mask==0)]=visual[torch.where(IS_BAG_list==1)]

        A=torch.zeros([text_embedding.shape[0],text_embedding.shape[1],acoustic.shape[-1]],device=text_embedding.device)
        A[torch.where(attention_mask==0)]=acoustic[torch.where(IS_BAG_list==1)]

        ta_sampled_c=torch.zeros_like(A)
        ta_sampled_u=torch.zeros_like(A)

        tv_sampled_u=torch.zeros_like(V)
        tv_sampled_c=torch.zeros_like(V)



        ta_loss, ta_sampled_c[torch.where(attention_mask==0)], ta_sampled_u[
            torch.where(attention_mask==0)]=self.ta_diffusion(text_embedding[torch.where(attention_mask==0)].detach(),
                                                              acoustic[torch.where(attention_mask==0)])
        tv_loss, tv_sampled_c[torch.where(attention_mask==0)], tv_sampled_u[
            torch.where(attention_mask==0)]=self.tv_diffusion(text_embedding[torch.where(attention_mask==0)].detach(),
                                                              visual[torch.where(attention_mask==0)])

        text_embedding[torch.where(attention_mask==0)]=self.tripleInteractionModel(text_embedding[torch.where(attention_mask==0)]
                                                                                   ,tv_sampled_c[torch.where(attention_mask==0)]
                                                                                   ,ta_sampled_c[torch.where(attention_mask==0)])


        ta_sampled_u, tv_sampled_u=self.BEM(text_embedding,ta_sampled_u,tv_sampled_u)

        tau=self.EncoderBlockTA(text_embedding,ta_sampled_u)
        tvu=self.EncoderBlockTV(text_embedding,tv_sampled_u)

        new_text_embedding=matrix_orthogonal_fusion(tau,tvu)

        return new_text_embedding,ta_loss+tv_loss
        # return text_embedding,0.0


    def init_weights(self):
        self.down_project.weight.data.normal_(mean=0.0, std=self.adapter_config.adapter_initializer_range)
        self.down_project.bias.data.zero_()
        self.up_project.weight.data.normal_(mean=0.0, std=self.adapter_config.adapter_initializer_range)
        self.up_project.bias.data.zero_()

class Parallel_Adapter(nn.Module):
    #### 考虑的是多模态的情况
    def __init__(self, hp):
        super(Parallel_Adapter, self).__init__()
        self.adapter_config =  AdapterConfig()
        self.multi = hp.multi
        if hp.multi:
            in_dim = self.adapter_config.project_hidden_size + hp.d_vout + hp.d_aout
        else:
            in_dim = self.adapter_config.project_hidden_size

        self.adapter_down_project = nn.Linear(in_dim,self.adapter_config.adapter_size)
        self.adapter_up_project = nn.Linear(self.adapter_config.adapter_size,in_dim)
        self.adapter_down_project.weight = torch.nn.Parameter(torch.normal(mean=0.0, std=self.adapter_config.adapter_initializer_range,
                                                                   size=(self.adapter_config.adapter_size, in_dim,)))
        self.adapter_down_project.bias = torch.nn.Parameter(torch.zeros(self.adapter_config.adapter_size))

        self.adapter_up_project.weight = torch.nn.Parameter(torch.normal(mean=0.0, std=self.adapter_config.adapter_initializer_range,
                                                    size=(in_dim, self.adapter_config.adapter_size,)))
        self.adapter_up_project.bias = torch.nn.Parameter(torch.zeros(in_dim))
        self.adapter_linear = nn.Linear(in_dim,self.adapter_config.out_hidden_size)
####
    def forward(self, x_states, hidden_states, visual=None, acoustic=None):
        ### x_states 表示FFN模块的输入，hidden_states表示FFN模块的输出
        if self.multi:
            seq_len = x_states.size(1)
            if len(visual.shape) == 1:
                visual = visual.unsqueeze(dim=0)
            if len(acoustic.shape) == 1:
                acoustic = acoustic.unsqueeze(dim=0)
            visual = visual.unsqueeze(dim=1).expand(visual.size(0),seq_len,visual.size(1))
            acoustic = acoustic.unsqueeze(dim=1).expand(acoustic.size(0),seq_len,acoustic.size(1))
            hidden_states_add = torch.cat([x_states, visual, acoustic], dim=-1)

            down_output = self.adapter_down_project(hidden_states_add)
            down_output_nolinear = torch.sigmoid(down_output)
            up_output = self.adapter_up_project(down_output_nolinear)
            output = up_output + torch.cat([hidden_states, visual, acoustic], dim=-1)
            output = self.adapter_linear(output)


        else:
            down_output = self.adapter_down_project(x_states)
            down_output_nolinear = torch.sigmoid(down_output)
            up_output = self.adapter_up_project(down_output_nolinear)
            output = up_output + hidden_states
            output = self.adapter_linear(output)

        return output


    def init_weights(self):
        self.down_project.weight.data.normal_(mean=0.0, std=self.adapter_config.adapter_initializer_range)
        self.down_project.bias.data.zero_()
        self.up_project.weight.data.normal_(mean=0.0, std=self.adapter_config.adapter_initializer_range)
        self.up_project.bias.data.zero_()

# class Attention_Adapter(nn.Module):
#     #### 考虑的是多模态的情况
#     def __init__(self, hp):
#         super(Attention_Adapter, self).__init__()
#         self.adapter_config =  AdapterConfig()
#         self.multi = hp.multi
#         if hp.multi:
#             in_dim = self.adapter_config.project_hidden_size + hp.d_vout + hp.d_aout
#         else:
#             in_dim = self.adapter_config.project_hidden_size
#
# ####
#     def forward(self, hidden_states, visual=None, acoustic=None):
#         ### hidden_states表示FFN模块的输出, (32, seq_len, t_dim) => (32, seq_len*t_dim)
#         if self.multi:
#             seq_len = hidden_states.size(1)
#             if len(visual.shape) == 1:
#                 visual = visual.unsqueeze(dim=0)
#             if len(acoustic.shape) == 1:
#                 acoustic = acoustic.unsqueeze(dim=0)
#             ##
#             visual = visual.unsqueeze(dim=1).expand(visual.size(0),seq_len,visual.size(1)) ## (32, seq_len, v_dim) => (32, seq_len*v_dim)
#             acoustic = acoustic.unsqueeze(dim=1).expand(acoustic.size(0),seq_len,acoustic.size(1)) ## (32, seq_len, a_dim) => (32, seq_len*a_dim)
#
#             hidden_states_add = torch.cat([hidden_states, visual, acoustic], dim=-1)
#
#         else:
#             down_output = self.adapter_down_project(hidden_states)
#             down_output_nolinear = torch.sigmoid(down_output)
#             up_output = self.adapter_up_project(down_output_nolinear)
#             output = up_output + hidden_states
#             output = self.adapter_linear(output)
#
#         return output
#
#
#     def init_weights(self):
#         self.down_project.weight.data.normal_(mean=0.0, std=self.adapter_config.adapter_initializer_range)
#         self.down_project.bias.data.zero_()
#         self.up_project.weight.data.normal_(mean=0.0, std=self.adapter_config.adapter_initializer_range)
#         self.up_project.bias.data.zero_()

class Sub_Networks(nn.Module):
    def __init__(self, hp, embed_dim):
        super(Sub_Networks, self).__init__()

        self.dropout = hp.embed_dropout  # Embedding dropout
        self.attn_dropout = hp.attn_dropout
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(self.embed_dim)
        self.embed_positions = SinusoidalPositionalEmbedding(self.embed_dim)

        self.attn_mask = hp.attn_mask
        self.num_heads = hp.num_heads
        self.relu_dropout = hp.relu_dropout
        self.res_dropout = hp.res_dropout

        self.multi = hp.multi
        self.normalize = True

        self.layers = nn.ModuleList([])
        for layer in range(hp.num_layers):
            new_layer = TransformerEncoderLayer(self.embed_dim,
                                                num_heads=self.num_heads,
                                                attn_dropout=self.attn_dropout,
                                                relu_dropout=self.relu_dropout,
                                                res_dropout=self.res_dropout,
                                                attn_mask=self.attn_mask)
            self.layers.append(new_layer)

    def forward(self,  x_in, x_in_k = None, x_in_v = None):
        if self.multi:
            x = self.embed_scale * x_in
            if self.embed_positions is not None:
                x += self.embed_positions(x_in.transpose(0, 1)[:, :, 0]).transpose(0, 1)  # Add positional embedding
            x = F.dropout(x, p=self.dropout, training=self.training)

            if x_in_k is not None and x_in_v is not None:
                # embed tokens and positions
                x_k = self.embed_scale * x_in_k
                x_v = self.embed_scale * x_in_v
                if self.embed_positions is not None:
                    x_k += self.embed_positions(x_in_k.transpose(0, 1)[:, :, 0]).transpose(0, 1)  # Add positional embedding
                    x_v += self.embed_positions(x_in_v.transpose(0, 1)[:, :, 0]).transpose(0, 1)  # Add positional embedding
                x_k = F.dropout(x_k, p=self.dropout, training=self.training)
                x_v = F.dropout(x_v, p=self.dropout, training=self.training)

            # encoder layers
            intermediates = [x]
            for layer in self.layers:
                if x_in_k is not None and x_in_v is not None:
                    x = layer(x, x_k, x_v)
                else:
                    x = layer(x)
                intermediates.append(x)

            # if self.normalize:
            #     x = self.layer_norm(x)

            return x

        else:
            down_output = self.adapter_down_project(x_in)
            down_output_nolinear = torch.sigmoid(down_output)
            up_output = self.adapter_up_project(down_output_nolinear)
            output = up_output + x_in
            output = self.adapter_linear(output)

        return output

class Cross_Attention_Adapter(nn.Module):
    #### 考虑的是多模态的情况
    def __init__(self, hp):
        super(Cross_Attention_Adapter, self).__init__()
        self.adapter_config =  AdapterConfig()
        self.multi = hp.multi
        if hp.multi:

            self.embed_dropout = hp.embed_dropout

            ### 将三种模态通过卷积操作卷到相同维度上
            self.orig_d_l = hp.hidden_size
            self.orig_d_a = hp.d_ah
            self.orig_d_v = hp.d_vh
            self.d_l, self.d_a, self.d_v = 30, 30, 30

            self.adapter_proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
            self.adapter_proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)
            self.adapter_proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)

            ### 只考虑将语音、视频信息注入到文本中来的情况

            # embed_dim, attn_dropout = self.d_l, hp.attn_dropout
            self.adapter_V2L_subnet = Sub_Networks(hp, self.d_l)
            self.adapter_A2L_subnet = Sub_Networks(hp, self.d_l)

            # self.adapter_mem = Sub_Networks(hp, self.d_l)

            trans_in_dim = self.orig_d_l + self.d_a + self.d_v

            self.adapter_trans_out = Linear(trans_in_dim, self.orig_d_l)
            self.normalize = True
            if self.normalize:
                self.layer_norm = LayerNorm(self.d_l)
        else:
            in_dim = self.adapter_config.project_hidden_size

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())

####
    def forward(self,x_l, x_a, x_v):
        ### x_in text embedding (batch, seq_len, t_d)
        ### x_in_k, x_in_v video/audio embedding (batch, seq_len, v_d/a_d)
        ### 需要考虑一个问题，融合应该发生在小粒度上还是应该在clause level上
        x_l_ = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_a_ = x_a.transpose(1, 2)
        x_v_ = x_v.transpose(1, 2)

        # Project the textual/visual/audio features
        proj_x_l = x_l_ if self.orig_d_l == self.d_l else self.adapter_proj_l(x_l_)
        proj_x_a = x_a_ if self.orig_d_a == self.d_a else self.adapter_proj_a(x_a_)
        proj_x_v = x_v_ if self.orig_d_v == self.d_v else self.adapter_proj_v(x_v_)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        proj_x_l = proj_x_l.permute(2, 0, 1)

        # (V,A) --> L
        h_l_with_as = self.adapter_A2L_subnet(proj_x_l, proj_x_a, proj_x_a)  # Dimension (L, N, d_l)
        h_l_with_vs = self.adapter_V2L_subnet(proj_x_l, proj_x_v, proj_x_v)  # Dimension (L, N, d_l)

        ## 将其嵌入到原始的text embedding中
        h_ls = torch.cat([x_l, h_l_with_as.transpose(0,1), h_l_with_vs.transpose(0,1)], dim=2)
        # h_ls = self.adapter_mem(h_ls)
        output = self.adapter_trans_out(h_ls)

        return output


    def init_weights(self):
        self.down_project.weight.data.normal_(mean=0.0, std=self.adapter_config.adapter_initializer_range)
        self.down_project.bias.data.zero_()
        self.up_project.weight.data.normal_(mean=0.0, std=self.adapter_config.adapter_initializer_range)
        self.up_project.bias.data.zero_()

class AdapterModel(nn.Module):
    def __init__(self, args, pretrained_model_config, n_rel):
        super(AdapterModel, self).__init__()
        self.config = pretrained_model_config
        self.args = args

    def forward(self, pretrained_model_outputs, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                labels=None, subj_special_start_id=None, obj_special_start_id=None):

        return

    def save_pretrained(self, save_directory):
        assert os.path.isdir(save_directory), "Saving path should be a directory where the model and configuration can be saved"
        # Only save the model it-self if we are using distributed training
        model_to_save = self.module if hasattr(self, 'module') else self
        # Save configuration file
        model_to_save.config.save_pretrained(save_directory)
        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(model_to_save.state_dict(), output_model_file)
        logger.info("Saving model checkpoint to %s", save_directory)
import torch.nn.functional as F
from torch import nn
from .modeling_t5_prefix import T5ForConditionalGeneration
from transformers import T5Config

def add_noise(x, intens=1e-7):
    return x + torch.rand(x.size()) * intens

class LanguageEmbeddingLayer(nn.Module):
    """Embed input text with "glove" or "Bert"
    """
    def __init__(self, hp):
        super(LanguageEmbeddingLayer, self).__init__()
        self.init_checkpoint = hp.init_checkpoint
        self.hp = hp

        model_path = '../t5-base'
        # model_path = '../t5-large'


        t5_config = T5Config.from_pretrained(model_path)

        # self.model = T5ForConditionalGeneration(t5_Config)
        self.t5_model = T5ForConditionalGeneration(hp, t5_config)

        self.load_checkpoint()

    def save_checkpoint(state, file_name):
        print('saving check_point')
        torch.save(state, file_name)

    # 第二个是加载模型
    def load_checkpoint(self):
        print('Load T5_model!')
        T5_dict = self.t5_model.state_dict()  # 取出自己网络的参数字典
        # print(T5_dict.keys())
        pretrained_dict = torch.load(self.init_checkpoint)  # 加载预训练网络的参数字典
        for k, v in pretrained_dict.items():
            if k in T5_dict.keys() and v.size() == T5_dict[k].size():
                T5_dict[k] = pretrained_dict[k]
        self.t5_model.load_state_dict(T5_dict)
        print('T5 model init....')

    def forward(self, sentences, t5_input_id, t5_att_mask, t5_labels, prompt_key_values=None, visual=None, acoustic=None,IS_BAG_list=None):

        if self.hp.use_prefix_p:
            output = self.t5_model(input_ids=t5_input_id, attention_mask=t5_att_mask, labels=t5_labels, prompt_key_values=prompt_key_values, visual=visual, acoustic=acoustic,IS_BAG_list=None)
        else:

            output = self.t5_model(input_ids=t5_input_id, attention_mask=t5_att_mask, labels=t5_labels, visual=visual, acoustic=acoustic,IS_BAG_list=IS_BAG_list)

        return output   # return head (sequence representation)
import torch
import torch.nn as nn
class BAG(nn.Module):
    def __init__(self,main_modal_dim=100,branch_modal_dim=100,beta_shift=1.0,dropout_prob=0.5):
        super(BAG, self).__init__()
        print(
            "Initializing BAG with args.beta_shift:{} hidden_prob:{}".format(
                beta_shift, dropout_prob
            )
        )

        self.W_mb = nn.Linear(branch_modal_dim + main_modal_dim, main_modal_dim)

        self.W_b = nn.Linear(branch_modal_dim, main_modal_dim)

        self.beta_shift = beta_shift

        self.LayerNorm = nn.LayerNorm(main_modal_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self,main_modal=None,branch_modal=None,is_bag_list=None):
        eps = 1e-6
        weight_b = F.relu(self.W_mb(torch.cat((main_modal,branch_modal), dim=-1)))

        h_m = weight_b * self.W_b(branch_modal)

        em_norm = main_modal.norm(2, dim=-1)
        hm_norm = h_m.norm(2, dim=-1)

        hm_norm_ones = torch.ones(hm_norm.shape, requires_grad=True).to(main_modal.device)
        hm_norm = torch.where(hm_norm == 0, hm_norm_ones, hm_norm)

        thresh_hold = (em_norm / (hm_norm + eps)) * self.beta_shift

        ones = torch.ones(thresh_hold.shape, requires_grad=True).to(main_modal.device)

        alpha = torch.min(thresh_hold, ones)
        alpha = alpha.unsqueeze(dim=-1)

        branch_embedding = alpha * h_m

        embedding_output = self.dropout(
            self.LayerNorm(branch_embedding + main_modal)
        )
        embedding_output = embedding_output * is_bag_list+ main_modal*(1.0-is_bag_list)

        return embedding_output


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Separate fully connected layers for each gate
        self.i_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.f_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.g_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.o_gate = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x, hidden,is_rnn_list):
        h_prev, c_prev = hidden
        
        # Combined input
        combined = torch.cat((x, h_prev), dim=1).to(x.device)
        
        # Compute gates using separate fully connected layers
        i = torch.sigmoid(self.i_gate(combined))
        f = torch.sigmoid(self.f_gate(combined))
        g = torch.tanh(self.g_gate(combined))
        o = torch.sigmoid(self.o_gate(combined))
        
        # New cell state
        c = f * c_prev + i * g

        # print(c,c.shape)
        # print(c_prev,c_prev.shape)
        # print(is_rnn_list,is_rnn_list.shape)
        # print(f,f.shape)
        
        c = c*is_rnn_list+ c_prev*(1.0-is_rnn_list)
        o = o*is_rnn_list+ (1.0-is_rnn_list)
        # New hidden state
        # h = o * torch.tanh(c)

        # return h, (h, c)
        return c,o

class BAG_LSTM(nn.Module):
    def __init__(self, a_input_size=0, a_hidden_size=0, v_input_size=0, v_hidden_size=0):
        super(BAG_LSTM, self).__init__()

        self.a_input_size = a_input_size
        self.a_hidden_size = a_hidden_size

        self.v_input_size = v_input_size
        self.v_hidden_size = v_hidden_size

        self.a_lstm=LSTMCell(input_size=a_input_size,hidden_size=a_hidden_size)
        self.v_lstm=LSTMCell(input_size=v_input_size,hidden_size=v_hidden_size)
        self.bag=BAG(main_modal_dim=a_hidden_size,branch_modal_dim=v_hidden_size,beta_shift=1.0,dropout_prob=0.5)

    def forward(self, a_x, a_hidden, v_x, v_hidden,aco_is_rnn_list=None,vis_is_rnn_list=None,is_bag_list=None):
        
        a_c,a_o=self.a_lstm(a_x, a_hidden,aco_is_rnn_list)
        v_c,v_o=self.v_lstm(v_x, v_hidden,vis_is_rnn_list)

        a_shift_c=self.bag(main_modal=a_c,branch_modal=v_c,is_bag_list=is_bag_list)
        v_shift_c=self.bag(main_modal=v_c,branch_modal=a_c,is_bag_list=is_bag_list)


        # New hidden state
        a_h = a_o * torch.tanh(a_shift_c)
        v_h = v_o * torch.tanh(v_shift_c)

        return a_h, (a_h, a_shift_c),    v_h, (v_h, v_shift_c)


class RNNEncoder(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
        super(RNNEncoder, self).__init__()
        print(in_size, hidden_size, out_size)
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.bag_lstm = BAG_LSTM(a_input_size=in_size, a_hidden_size=hidden_size, v_input_size=in_size, v_hidden_size=hidden_size)

        self.fc = nn.Linear(hidden_size, out_size)  # Final output layer

    def forward(self, acoustic, a_lengths, visual, v_lengths,IS_BAG_list=None,use_seq=False):
        # print("A",x.shape,lengths)

        v_x,VIS_IS_rnn_list=visual
        a_x,ACO_IS_rnn_list=acoustic

        a_x=a_x.permute(1,0,2)
        v_x=v_x.permute(1,0,2)
        VIS_IS_rnn_list=VIS_IS_rnn_list.permute(1,0)
        ACO_IS_rnn_list=ACO_IS_rnn_list.permute(1,0)
        IS_BAG_list=IS_BAG_list.permute(1,0)

        # print(a_x.shape)
        # print(v_x.shape)
        # print(VIS_IS_rnn_list.shape)
        # print(ACO_IS_rnn_list.shape)
        # print(IS_BAG_list)
        # exit(0)
        zeros_a=torch.zeros(a_x.shape[0],max(a_x.shape[1],v_x.shape[1]),a_x.shape[2],dtype=a_x.dtype).to(a_x.device)
        zeros_v=torch.zeros(v_x.shape[0],max(a_x.shape[1],v_x.shape[1]),v_x.shape[2],dtype=v_x.dtype).to(v_x.device)

        zeros_a[:,:a_x.shape[1],:]=a_x
        zeros_v[:,:v_x.shape[1],:]=v_x

        a_x=zeros_a
        v_x=zeros_v

        # print("A",x.shape,lengths)
        a_out=torch.zeros(a_x.shape[0],a_x.shape[1],self.hidden_size).to(a_x.device)
        a_y=torch.zeros(a_x.shape[0],self.hidden_size).to(a_x.device)
        a_h = torch.zeros(a_x.size(0), self.hidden_size).to(a_x.device)
        a_c = torch.zeros(a_x.size(0), self.hidden_size).to(a_x.device)

        v_out=torch.zeros(v_x.shape[0],v_x.shape[1],self.hidden_size).to(v_x.device)
        v_y=torch.zeros(v_x.shape[0],self.hidden_size).to(v_x.device)
        v_h = torch.zeros(v_x.size(0), self.hidden_size).to(v_x.device)
        v_c = torch.zeros(v_x.size(0), self.hidden_size).to(v_x.device)
        
        for i in range(a_x.size(1)):
            # print("A",h.shape)
            # h, (h, c) = self.lstm_cell(x[:, i], (h, c))
            a_h, (a_h, a_shift_c),v_h, (v_h, v_shift_c) = self.bag_lstm(a_x[:, i], (a_h, a_c) ,v_x[:, i], (v_h, v_c),ACO_IS_rnn_list[:,i].unsqueeze(1),VIS_IS_rnn_list[:,i].unsqueeze(1),IS_BAG_list[:,i].unsqueeze(1))
            a_out[:,i]=a_h
            v_out[:,i]=v_h
            # print("B",h.shape)
            # exit(0)
        # print("O",out)
        a_out[torch.where(IS_BAG_list==1)] = self.fc(a_out[torch.where(IS_BAG_list==1)])  # Output from the last time step
        v_out[torch.where(IS_BAG_list==1)] = self.fc(v_out[torch.where(IS_BAG_list==1)])  # Output from the last time step
        bag_index = torch.tensor([torch.where(IS_BAG_list[i,:]==1)[0].tolist()[-1]    for i in range(v_x.shape[0])])
        # print(torch.where(IS_BAG_list==1))
        # print(a_out[torch.where(IS_BAG_list==1)].shape)
        # print(v_out[torch.where(IS_BAG_list==1)].shape)
        a_index_range = torch.arange(a_out.size(0))
        a_y[:,:]=a_out[a_index_range,bag_index]

        v_index_range = torch.arange(v_out.size(0))
        v_y[:,:]=v_out[v_index_range,bag_index]
        # print(bag_index)    
        # print("O",out)
        # print("Y",y)        
        # print(torch.where(IS_BAG_list==1))
        # exit(0)
        return a_y,a_out,v_y,v_out


# class RNNEncoder(nn.Module):
#     def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
#         '''
#         Args:
#             in_size: input dimension
#             hidden_size: hidden layer dimension
#             num_layers: specify the number of layers of LSTMs.
#             dropout: dropout probability
#             bidirectional: specify usage of bidirectional LSTM
#         Output:
#             (return value in forward) a tensor of shape (batch_size, out_size)
#         '''
#         super().__init__()
#         self.bidirectional = bidirectional

#         self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=False)
#         self.dropout = nn.Dropout(dropout)
#         self.linear_1 = nn.Linear((2 if bidirectional else 1)*hidden_size, out_size)

#     def forward(self, x, lengths, use_seq=False):
#         '''
#         x: (batch_size, sequence_len, in_size)
#         '''
#         lengths = lengths.to(torch.int64)
#         bs = x.size(0)
#         # print('x_shape:{}'.format(x.shape))
#         # print('lengths_shape:{}'.format(lengths.shape))
#         print(x, lengths)
#         packed_sequence = pack_padded_sequence(x, lengths, enforce_sorted=False)
#         print(packed_sequence)
#         # print('x shape:{}'.format(x.shape))
#         # print('length shape:{}'.format(lengths.shape))
#         out_pack, final_states = self.rnn(packed_sequence)
#         # print('out_pack_data_shape:{}'.format(out_pack.data.shape))

#         if self.bidirectional:
#             h = self.dropout(torch.cat((final_states[0][0],final_states[0][1]),dim=-1))
#         else:
#             h = self.dropout(final_states[0].squeeze())
#         y_1 = self.linear_1(h)
#         # print('h_shape:{}'.format(h.shape))

#         if use_seq:
#             x_sort_idx = torch.argsort(-lengths)
#             x_unsort_idx = torch.argsort(x_sort_idx).long()
#             # print('out_pack_shape:{}'.format(out_pack.shape))
#             out = torch.nn.utils.rnn.pad_packed_sequence(out_pack, batch_first=True)  # (sequence, lengths)
#             out = out[0]  #
#             out = out[x_unsort_idx]
#             return y_1, out
#         else:
#             return y_1, None


from dfmodules.encoders import *
from dfmodules.prefix_encoder import *
from dfmodules.attention import *
from config import DEVICE

class Model(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.hp = hp
        self.add_va = hp.add_va
        self.multi = hp.multi
        hp.d_tout = hp.d_tin
        
        ###prompt

        if hp.use_prefix_p:
            self.n_layer = 1
            self.n_head = 1
            self.n_embd = hp.prompt_hidden_size // self.n_head
            self.dropout = torch.nn.Dropout(hp.hidden_dropout_prob)
            self.pre_seq_len = hp.pre_seq_len
            self.prefix_tokens = torch.arange(self.pre_seq_len).long()
            self.prefix_encoder = PrefixEncoder(hp)

        ###prompt

        self.T5_encoder = LanguageEmbeddingLayer(hp)

        if hp.multi:
            # self.visual_enc = RNNEncoder(
            #     in_size=hp.d_vin,
            #     hidden_size=hp.d_vh,
            #     out_size=hp.d_vout,
            #     num_layers=hp.n_layer,
            #     dropout=hp.dropout_v if hp.n_layer > 1 else 0.3,
            #     bidirectional=hp.bidirectional
            # )
            # self.acoustic_enc = RNNEncoder(
            #     in_size=hp.d_ain,
            #     hidden_size=hp.d_ah,
            #     out_size=hp.d_aout,
            #     num_layers=hp.n_layer,
            #     dropout=hp.dropout_a if hp.n_layer > 1 else 0.3,
            #     bidirectional=hp.bidirectional
            # )
            self.acoustic_visual_enc=RNNEncoder(
                in_size=hp.d_ain,
                hidden_size=hp.d_ah,
                out_size=hp.d_aout,
                num_layers=hp.n_layer,
                dropout=hp.dropout_a if hp.n_layer > 1 else 0.3,
                bidirectional=hp.bidirectional
            )
            self.acoustic_attention=UnimodelAttention(target_len=32)
            self.visual_attention=UnimodelAttention(target_len=32)
            self.bimodal_attention=BimodalAttention()
            
    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(DEVICE)
        past_key_values = self.prefix_encoder(prefix_tokens)
        # bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2, 
            self.n_head,
            self.n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def generate(self, t5_input_id, t5_att_mask, visual=None, acoustic=None, v_len=None, a_len=None):

        if self.multi:
            if self.hp.adapter_name == 'cross-atten':
                acoustic, acoustic_seq = self.acoustic_enc(acoustic, a_len, use_seq=True) ## bs, seq_len, a_dim
                visual, visual_seq = self.visual_enc(visual, v_len, use_seq=True) ## bs, seq_len, v_dim
                outputs =self.T5_encoder.t5_model.generate(t5_input_id, attention_mask=t5_att_mask,visual=(visual,visual_seq), acoustic=(acoustic,acoustic_seq))
            else:
                if self.hp.info_nce:
                    acoustic, acoustic_seq = self.acoustic_enc(acoustic, a_len, use_seq=True)  ## bs, a_dim
                    visual, visual_seq = self.visual_enc(visual, v_len, use_seq=True)  ## bs, a_dim
                else:
                    acoustic, acoustic_seq = self.acoustic_enc(acoustic, a_len)  ## bs, a_dim
                    visual, visual_seq = self.visual_enc(visual, v_len)  ## bs, a_dim
                
                if self.hp.use_prefix_p:
                    batch_size = t5_input_id.shape[0]
                    prompt_key_values = self.get_prompt(batch_size=batch_size)
                    prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(DEVICE)
                    attention_mask = torch.cat((prefix_attention_mask, t5_att_mask), dim=1)
                    outputs =self.T5_encoder.t5_model.generate(t5_input_id, attention_mask=attention_mask,visual=(visual,visual_seq), acoustic=(acoustic,acoustic_seq),prompt_key_values=prompt_key_values)
                else:
                    outputs =self.T5_encoder.t5_model.generate(t5_input_id, attention_mask=t5_att_mask,visual=(visual,visual_seq), acoustic=(acoustic,acoustic_seq))

        else:
            outputs =self.T5_encoder.t5_model.generate(t5_input_id, attention_mask=t5_att_mask)

        return outputs

    def forward(self, sentences,t5_input_id, t5_att_mask, t5_labels, ids,  visual=None, acoustic=None, v_len=None, a_len=None,IS_BAG_list=None):
        if self.multi:
            if self.hp.adapter_name == 'cross-atten':
                acoustic, acoustic_seq = self.acoustic_enc(acoustic, a_len, use_seq=True)
                visual, visual_seq = self.visual_enc(visual, v_len, use_seq=True)
                enc_output = self.T5_encoder(sentences, t5_input_id, t5_att_mask, t5_labels, visual=(visual,visual_seq), acoustic=(acoustic,acoustic_seq)) 
            else:
                # acoustic = self.acoustic_enc(acoustic, a_len)  ## bs, a_dim
                # visual = self.visual_enc(visual, v_len)  ## bs, a_dim
                    
                if self.hp.info_nce:
                    # acoustic, acoustic_seq = self.acoustic_enc(acoustic, a_len, use_seq=True)  ## bs, a_dim
                    # visual, visual_seq = self.visual_enc(visual, v_len, use_seq=True)  ## bs, a_dim

                    acoustic=self.acoustic_attention(acoustic,a_len)
                    visual=self.visual_attention(visual,v_len)
                    acoustic, acoustic_seq,visual, visual_seq = self.acoustic_visual_enc(acoustic, a_len,visual, v_len,IS_BAG_list, use_seq=True)
                    # print(acoustic,acoustic.shape, acoustic_seq, acoustic_seq.shape)
                    acoustic_seq,visual_seq=self.bimodal_attention(acoustic_seq,visual_seq,IS_BAG_list.permute(1,0))
                    # print("A")
                    # print("A")
                    # print(visual,visual.shape, visual_seq, visual_seq.shape)
                    # print("B")
                    # exit(0)
                else:
                    acoustic, acoustic_seq = self.acoustic_enc(acoustic, a_len)  ## bs, a_dim
                    visual, visual_seq = self.visual_enc(visual, v_len)  ## bs, a_dim
                    
                if self.hp.use_prefix_p:
                    batch_size = t5_input_id.shape[0]
                    prompt_key_values = self.get_prompt(batch_size=batch_size)
                    prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(DEVICE)
                    attention_mask = torch.cat((prefix_attention_mask, t5_att_mask), dim=1)
                    enc_output = self.T5_encoder(sentences, t5_input_id, attention_mask, t5_labels, prompt_key_values = prompt_key_values, visual=(visual,visual_seq), acoustic=(acoustic,acoustic_seq)) 
                else:
                    enc_output = self.T5_encoder(sentences, t5_input_id, t5_att_mask, t5_labels, visual=(visual,visual_seq), acoustic=(acoustic,acoustic_seq),IS_BAG_list=IS_BAG_list) 
                    
                    
 # (batch_size, seq_len, emb_size)
        else:
            enc_output = self.T5_encoder(sentences, t5_input_id, t5_att_mask, t5_labels)  # (batch_size, seq_len, emb_size)

        logits, loss = enc_output.logits, enc_output.loss
        ### 逻辑上, enc_output则是预测结果
        return logits, loss






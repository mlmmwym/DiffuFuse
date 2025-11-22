
import torch.nn as nn

import torch
import random


class UnimodelAttention(nn.Module):
    def __init__(self,target_len=32,av_dim=64):
        super(UnimodelAttention,self).__init__()
        self.target_len=target_len
        self.av_dim=av_dim
        self.h_fc=nn.Linear(target_len,target_len)
        self.w_fc=nn.Linear(av_dim,av_dim)
        self.sigmoid=nn.Sigmoid()

    def unimodelattention(self,sequence,is_rnn,v_len,output_list,mapping_list):
        attention=torch.stack(output_list,dim=0)

        h_attention=torch.mean(attention,dim=2)
        h_attention=self.sigmoid(self.h_fc(h_attention))

        w_attention=torch.mean(attention,dim=1)
        w_attention=self.sigmoid(self.w_fc(w_attention))

        h_attention=h_attention.unsqueeze(2)
        w_attention=w_attention.unsqueeze(1)

        h_attention=h_attention.repeat(1,1,self.av_dim)
        w_attention=w_attention.repeat(1,self.target_len,1)

        h_attention=list(torch.unbind(h_attention,dim=0))
        w_attention=list(torch.unbind(w_attention,dim=0))

        h_attention_list=self.recover_sequence(h_attention,mapping_list,v_len)
        w_attention_list=self.recover_sequence(w_attention,mapping_list,v_len)

        for i in range(attention.shape[0]):
            av_data=sequence[i]
            av_is_rnn=is_rnn[i]
            where=torch.where(av_is_rnn==1)
            # av_data=av_data[where]
            av_data[where]=(av_data[where]*h_attention_list[i]+av_data[where]*w_attention_list[i])/2.0
            sequence[i]=av_data

        return sequence

    def recover_sequence(self,attention_list,mapping_list,v_len):

        for i in range(len(attention_list)):
            # print(i)
            # print(len(mapping_list))
            attention=attention_list[i]
            mapping=mapping_list[i]
            # print(i)
            # print(attention.shape)
            # print(mapping)
            m_len=v_len[i]
            if m_len==self.target_len:
                attention_list[i]=attention
            elif m_len>self.target_len:
                # print(i)
                # print(attention.shape)
                # print(mapping)
                attention_list[i]=attention[mapping]
            else:
                new_attention=torch.zeros(m_len,self.av_dim,dtype=attention.dtype,device=attention.device)
                B = torch.ones(mapping.shape[0],dtype=attention.dtype,device=attention.device)
                div_mapping=torch.zeros(m_len,dtype=attention.dtype,device=attention.device)
                # print(m_len)
                # print(div_mapping.shape)
                # print(mapping)
                div_mapping.index_add_(0,mapping,B)
                new_attention.index_add_(0,mapping,attention)
                new_attention/=div_mapping.unsqueeze(1)
                attention_list[i]=new_attention

        return attention_list

    def resize_sequence(self,sequence):
        current_len = sequence.size(0)
        output = torch.zeros(self.target_len, sequence.size(1),device=sequence.device)


        if current_len == self.target_len:
            mapping = torch.arange(self.target_len,device=sequence.device)
            return sequence, mapping
        elif current_len < self.target_len:
            # 扩展逻辑与前面相同
            mapping = torch.zeros(self.target_len, dtype=torch.long,device=sequence.device)
            repeats = self.target_len // current_len
            remainder = self.target_len % current_len
            repeats_list = [repeats] * current_len
            for i in range(remainder):
                repeats_list[i] += 1
            random.shuffle(repeats_list)
            indices = []
            counter = 0
            for i in range(current_len):
                for _ in range(repeats_list[i]):
                    indices.append(i)
                    mapping[counter] = i
                    counter += 1
            expanded = sequence[indices]
            return expanded, mapping
        else:
            # 缩减逻辑
            mapping = torch.zeros(current_len, dtype=torch.long,device=sequence.device)
            group_size = current_len // self.target_len
            remainder = current_len % self.target_len
            group_list = [group_size] * self.target_len
            for i in range(remainder):
                group_list[i] += 1
            random.shuffle(group_list)
            start_idx = 0
            counter = 0
            for size in group_list:
                output[counter, :] = sequence[start_idx:start_idx + size, :].mean(dim=0)
                mapping[start_idx:start_idx + size] = counter
                start_idx += size
                counter += 1
            return output, mapping
    def forward(self,sequence,v_len):
        sequence=(sequence[0].permute(1,0,2) , sequence[1].permute(1, 0))
        is_rnn=sequence[1]
        sequence=sequence[0]
        # is_rnn=torch.sum(is_rnn,dim=1)
        output_list=[]
        mapping_list=[]
        for i in range(sequence.shape[0]):
            sample=sequence[i]
            is_sample=is_rnn[i]
            # print(v_len[i],torch.sum(is_sample))
            output, mapping=self.resize_sequence(sample[torch.where(is_sample==1)])
            output_list.append(output)
            # output_list.append(output)
            mapping_list.append(mapping)
        sequence=self.unimodelattention(sequence,is_rnn,torch.sum(is_rnn,dim=1),output_list,mapping_list)
        return (sequence.permute(1,0,2),is_rnn.permute(1, 0))

class BimodalAttention(nn.Module):
    def __init__(self,target_len=32,av_dim=32):
        super(BimodalAttention,self).__init__()
        self.target_len = target_len
        self.av_dim = av_dim
        self.h_fc=nn.Linear(target_len,target_len)
        self.w_fc=nn.Linear(av_dim,av_dim)
        self.conv=nn.Conv2d(in_channels=2,out_channels=1,kernel_size=1,stride=1,padding=0)
        self.sigmoid=nn.Sigmoid()

    def recover_sequence(self,attention_list,mapping_list,v_len):

        for i in range(len(attention_list)):
            # print(i)
            # print(len(mapping_list))
            attention=attention_list[i]
            mapping=mapping_list[i]
            # print(i)
            # print(attention.shape)
            # print(mapping)
            m_len=v_len[i]
            if m_len==self.target_len:
                attention_list[i]=attention
            elif m_len>self.target_len:
                # print(i)
                # print(attention.shape)
                # print(mapping)
                attention_list[i]=attention[mapping]
            else:
                new_attention=torch.zeros(m_len,self.av_dim,dtype=attention.dtype,device=attention.device)
                B = torch.ones(mapping.shape[0],dtype=attention.dtype,device=attention.device)
                div_mapping=torch.zeros(m_len,dtype=attention.dtype,device=attention.device)
                # print(m_len)
                # print(div_mapping.shape)
                # print(mapping)
                div_mapping.index_add_(0,mapping,B)
                new_attention.index_add_(0,mapping,attention)
                new_attention/=div_mapping.unsqueeze(1)
                attention_list[i]=new_attention

        return attention_list
    def bimodelattention(self,acoustic_seq,visual_seq,IS_BAG_list,bag_len,output_list,mapping_list):



        attention=torch.stack(output_list,dim=0)

        c_attention=self.sigmoid(self.conv(attention))

        hw_attention=torch.mean(attention,dim=1)

        h_attention=torch.mean(hw_attention,dim=2)
        h_attention=self.sigmoid(self.h_fc(h_attention))

        w_attention=torch.mean(hw_attention,dim=1)
        w_attention=self.sigmoid(self.w_fc(w_attention))

        h_attention=h_attention.unsqueeze(2)
        w_attention=w_attention.unsqueeze(1)

        h_attention=h_attention.repeat(1,1,self.av_dim)
        w_attention=w_attention.repeat(1,self.target_len,1)
        c_attention=c_attention.squeeze()

        h_attention=list(torch.unbind(h_attention,dim=0))
        w_attention=list(torch.unbind(w_attention,dim=0))
        c_attention=list(torch.unbind(c_attention,dim=0))

        h_attention_list=self.recover_sequence(h_attention,mapping_list,bag_len)
        w_attention_list=self.recover_sequence(w_attention,mapping_list,bag_len)
        c_attention_list=self.recover_sequence(c_attention,mapping_list,bag_len)

        for i in range(acoustic_seq.shape[0]):
            acoustic=acoustic_seq[i]
            visual=visual_seq[i]

            is_bag=IS_BAG_list[i]

            where=torch.where(is_bag==1)
            # av_data=av_data[where]
            acoustic[where]=(acoustic[where]*h_attention_list[i]+acoustic[where]*w_attention_list[i]+acoustic[where]*c_attention_list[i])/3.0
            visual[where]=(visual[where]*h_attention_list[i]+visual[where]*w_attention_list[i]+visual[where]*c_attention_list[i])/3.0

            acoustic_seq[i]=acoustic
            visual_seq[i]=visual


        return acoustic_seq,visual_seq

    def resize_sequence(self,sequence):
        current_len = sequence.size(0)
        output = torch.zeros(self.target_len, sequence.size(1),device=sequence.device)


        if current_len == self.target_len:
            mapping = torch.arange(self.target_len,device=sequence.device)
            return sequence, mapping
        elif current_len < self.target_len:
            # 扩展逻辑与前面相同
            mapping = torch.zeros(self.target_len, dtype=torch.long,device=sequence.device)
            repeats = self.target_len // current_len
            remainder = self.target_len % current_len
            repeats_list = [repeats] * current_len
            for i in range(remainder):
                repeats_list[i] += 1
            random.shuffle(repeats_list)
            indices = []
            counter = 0
            for i in range(current_len):
                for _ in range(repeats_list[i]):
                    indices.append(i)
                    mapping[counter] = i
                    counter += 1
            expanded = sequence[indices]
            return expanded, mapping
        else:
            # 缩减逻辑
            mapping = torch.zeros(current_len, dtype=torch.long,device=sequence.device)
            group_size = current_len // self.target_len
            remainder = current_len % self.target_len
            group_list = [group_size] * self.target_len
            for i in range(remainder):
                group_list[i] += 1
            random.shuffle(group_list)
            start_idx = 0
            counter = 0
            for size in group_list:
                output[counter, :] = sequence[start_idx:start_idx + size, :].mean(dim=0)
                mapping[start_idx:start_idx + size] = counter
                start_idx += size
                counter += 1
            return output, mapping


    def forward(self,acoustic_seq,visual_seq,IS_BAG_list):
        seq_len=acoustic_seq.shape[0]
        bag_len=torch.sum(IS_BAG_list,dim=1)

        output_list=[]
        mapping_list=[]
        for i in range(seq_len):
            aco_sample=acoustic_seq[i]
            vis_sample=visual_seq[i]
            is_sample=IS_BAG_list[i]
            # print(v_len[i],torch.sum(is_sample))

            aco_sample, aco_mapping=self.resize_sequence(aco_sample[torch.where(is_sample==1)])
            vis_sample, vis_mapping=self.resize_sequence(vis_sample[torch.where(is_sample==1)])


            output_list.append(torch.concat([aco_sample.unsqueeze(0),vis_sample.unsqueeze(0)],dim=0))
            # output_list.append(output)
            mapping_list.append(vis_mapping)
        acoustic_seq, visual_seq=self.bimodelattention(acoustic_seq,visual_seq,IS_BAG_list,bag_len,output_list,mapping_list)
        return acoustic_seq,visual_seq
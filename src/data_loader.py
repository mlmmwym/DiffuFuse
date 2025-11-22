import numpy as np


import torch
from torch.utils.data import DataLoader, Dataset
from transformers import T5Tokenizer
from utils.tools import contain_nonum, is_number

from create_dataset import MOSI, MOSEI, MOSEII, MOSELD, MOSELDMP, IEMOCAP, MELD, EmotionLines, Laptops, Restaurants, PAD, UNK
from config import DEVICE
model_path = '../t5-base'
tokenizer = T5Tokenizer.from_pretrained(model_path)


class MSADataset(Dataset):
    def __init__(self, config):
        self.config = config
        ## Fetch dataset
        if "mosi" in str(config.data_dir).lower():
            dataset = MOSI(config)
            self.multi = True
        elif "mosei" in str(config.data_dir).lower():
            dataset = MOSEI(config)
            self.multi = True
        elif 'moselap' in str(config.data_dir).lower():
            dataset = MOSELAP(config)
            self.multi = True
        elif 'moselprs' in str(config.data_dir).lower():
            dataset = MOSELPRS(config)
            self.multi = True
        elif 'moseldmp' in str(config.data_dir).lower():
            dataset = MOSELDMP(config)
            self.multi = True
        elif 'moseld' in str(config.data_dir).lower():
            dataset = MOSELD(config)

            self.multi = True
        elif 'mos' in str(config.data_dir).lower():
            dataset = MOSEII(config)
            self.multi = True
        elif 'iemocap' in str(config.data_dir).lower():
            dataset = IEMOCAP(config)
            self.multi = True
        elif 'meld' in str(config.data_dir).lower():
            dataset = MELD(config)
            self.multi = True
        elif 'emotionlines' in str(config.data_dir).lower():
            dataset = EmotionLines(config)
            self.multi = False
        elif 'laptops' in str(config.data_dir).lower():
            dataset = Laptops(config)
            self.multi = False
        elif 'restaurants' in str(config.data_dir).lower():
            dataset = Restaurants(config)
            self.multi = False
        else:
            print("Dataset not defined correctly")
            exit()


        self.data, self.word2id, self.pretrained_emb = dataset.get_data(config.mode)
        self.len = len(self.data)
        config.word2id = self.word2id
        # config.pretrained_emb = self.pretrained_emb

    @property
    def tva_dim(self):
        t_dim = 512
        if self.multi:
            if self.config.dataset == 'meld':
                print("meld")
                # return t_dim, 0, self.data[0][0][2].shape[0]
                return t_dim, self.data[0][0][1].shape[1], self.data[0][0][2].shape[1]
            else:
                print('t_dim:{},va_dim:{}'.format(t_dim, str((self.data[0][0][1].shape[1], self.data[0][0][2].shape[1]))))
                return t_dim, self.data[0][0][1].shape[1], self.data[0][0][2].shape[1]
        else:
            return t_dim, 0, 0

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


def get_loader(hp, config, prompt_dict = None, shuffle=True):
    """Load DataLoader of given DialogDataset"""

    dataset = MSADataset(config)

    print('mode:{}'.format(config.mode))
    config.data_len = len(dataset)
    config.tva_dim = dataset.tva_dim

    if config.mode == 'train':
        hp.n_train = len(dataset)
    elif config.mode == 'valid':
        hp.n_valid = len(dataset)
    elif config.mode == 'test':
        hp.n_test = len(dataset)
    elif config.mode == 'test_mosi':
        hp.n_mosi_test = len(dataset)
    elif config.mode == 'test_mosei':
        hp.n_mosei_test = len(dataset)
    elif config.mode == 'test_meld':
        hp.n_meld_test = len(dataset)
    elif config.mode == 'test_iemocap':
        hp.n_iemocap_test = len(dataset)
    elif config.mode == 'test_laps':
        hp.n_laps_test = len(dataset)
    elif config.mode == 'test_res':
        hp.n_res_test = len(dataset)

    # task_prefix = "sst2 sentence: "
    task_prefix = ""

    dataset_name = config.dataset

    def collate_fn(batch):
        '''
        Collate functions assume batch = [Dataset[i] for i in index_set]
        '''
        # for later use we sort the batch in descending order of length
        if dataset_name != 'meld0000':
            batch = sorted(batch, key=lambda x: len(x[0][3]), reverse=True)

            v_lens = []
            a_lens = []
            labels = []
            ids = []
            labels_float=[]

            for sample in batch:
                # if sample[0][1].shape[1] != 35:
                #     print(sample[0][1].shape)
                # if sample[0][2].shape[1] != 74:
                #     print(sample[0][2].shape)
                if len(sample[0]) > 4: # unaligned case
                    v_lens.append(torch.IntTensor([sample[0][4]]))
                    a_lens.append(torch.IntTensor([sample[0][5]]))
                else:   # aligned cases
                    v_lens.append(torch.IntTensor([len(sample[0][3])]))
                    a_lens.append(torch.IntTensor([len(sample[0][3])]))
                # labels.append(torch.from_numpy(sample[1]))
                labels.append(sample[1])
                ids.append(sample[2])
                labels_float.append(float(sample[3]))
            vlens = torch.cat(v_lens)
            alens = torch.cat(a_lens)
            # labels = torch.cat(labels, dim=0)

            # MOSEI sentiment labels locate in the first column of sentiment matrix
            # if labels.size(1) == 7:
            #     labels = labels[:,0][:,None]

            # Rewrite this
            def pad_sequence(sequences, target_len=-1, batch_first=False, padding_value=0.0):
                if target_len < 0:
                    max_size = sequences[0].size()
                    trailing_dims = max_size[1:]
                else:
                    max_size = target_len
                    trailing_dims = sequences[0].size()[1:]

                max_len = max([s.size(0) for s in sequences])
                if batch_first:
                    out_dims = (len(sequences), max_len) + trailing_dims
                else:
                    out_dims = (max_len, len(sequences)) + trailing_dims

                out_tensor = sequences[0].new_full(out_dims, padding_value)
                for i, tensor in enumerate(sequences):
                    length = tensor.size(0)
                    # use index notation to prevent duplicate references to the tensor
                    if batch_first:
                        out_tensor[i, :length, ...] = tensor
                    else:
                        out_tensor[:length, i, ...] = tensor
                return out_tensor

            def pad_visual(img_list):
            
                pad_img_list=[img_list[0]]
                for i in range(1,len(img_list)):
                    pad_img_list.append((img_list[i-1]+img_list[i])/2)
                    pad_img_list.append(img_list[i])
                return pad_img_list
            def pad_acoustic(aco_list):
                pad_aco_list=[aco_list[0],aco_list[0]]
                for i in range(1,len(aco_list)):
                    pad_aco_list.append(aco_list[i])
                    pad_aco_list.append(aco_list[i])
                return pad_aco_list
            def divide_list_evenly(lst, p):
                k = len(lst)  # 列表的总元素个数
                if p <= 0:
                    return []  # 如果分组数为0或负数，则返回空列表
                if p > k:
                    return [lst]  # 如果分组数大于元素个数，返回整个列表
                
                # 每组至少的元素个数
                # min_size = k // p
                min_size = torch.div(k, p, rounding_mode='floor')
                # 需要增加一个元素的组数
                extra = k % p

                result = []
                start = 0
                # 步进为p除以额外元素的数量，用于决定均匀放置额外元素
                # step = p // extra if extra != 0 else 0
                step = torch.div(p, extra, rounding_mode='floor') if extra != 0 else 0
                current_extra = 0

                for i in range(p):
                    if current_extra < extra and (i == 0 or i % step == 0):
                        size = min_size + 1
                        current_extra += 1
                    else:
                        size = min_size
                    end = start + size
                    result.append(lst[start:end])
                    start = end

                return result
            def process_av(imgs,acos):                
                pad_imgs,pad_acos,vis_is_rnn_list,aco_is_rnn_list,is_bag_list=imgs,acos,[1 for i in range(len(imgs))],[1 for i in range(len(acos))],[0 for i in range(max(len(imgs),len(acos)))]
                
                # print(len(pad_imgs),len(pad_acos),len(vis_is_rnn_list),len(aco_is_rnn_list),len(is_bag_list))
                is_bag_list[-1]=1
                if len(imgs)>=len(acos):
                    dist=len(imgs)-len(acos)
                    pad_acos=[torch.zeros(acos[0].shape,dtype=acos[0].dtype) for i in range(dist)]+acos
                    aco_is_rnn_list=[0 for i in range(dist)]+aco_is_rnn_list
                else:                    
                    dist=len(acos)-len(imgs)
                    pad_imgs=[torch.zeros(imgs[0].shape,dtype=imgs[0].dtype) for i in range(dist)]+imgs
                    vis_is_rnn_list=[0 for i in range(dist)]+vis_is_rnn_list
                

                pad_imgs=torch.stack(pad_imgs,dim=0)
                pad_acos=torch.stack(pad_acos,dim=0)
                vis_is_rnn_list=torch.tensor(vis_is_rnn_list)
                aco_is_rnn_list=torch.tensor(aco_is_rnn_list)
                is_bag_list=torch.tensor(is_bag_list)
                # print(pad_imgs.shape,pad_acos.shape,vis_is_rnn_list.shape,aco_is_rnn_list.shape,is_bag_list.shape)
                # exit(0)
                return pad_imgs,pad_acos,vis_is_rnn_list,aco_is_rnn_list,is_bag_list

            def process(visual, vlens, acoustic, alens,t5_att_mask):
                # print(inputs_seq)
                # print(t5_input_id,t5_input_id.shape)
                # print(vlens,alens)
                # print(acoustic.shape)

                # for i in range(acoustic.size(1)):
                #     print(i,acoustic[:,i,:])
                # exit(0)
                # print(t5_att_mask)
                tlens=torch.sum(t5_att_mask,dim=1)
                # print(tlens)

                visual_list=[torch.FloatTensor(sample[0][1]) for sample in batch]
                acoustic_list=[torch.FloatTensor(sample[0][2]) for sample in batch]
                # print([v.shape for v in visual_list],[a.shape for a in acoustic_list])

                for k in range(len(batch)):
                    vis=visual_list[k]
                    # print(vis)
                    img_list=[vis[i] for i in range(vis.size(0))]
                    # print([i.shape for i in img_list])
                    # print(len(img_list))
                    while len(img_list)<tlens[k]:
                        img_list=pad_visual(img_list)
                    # print(len(img_list))
                    img_list=divide_list_evenly(img_list, tlens[k])
                    visual_list[k]=img_list
                    
                    # print(tlens[k],len(img_list))
                    # print(len(vis_list))
                    # for v in vis_list:
                    # exit(0)
                for k in range(len(batch)):
                    aco=acoustic_list[k]
                    # print(aco)
                    aco_list=[aco[i] for i in range(aco.size(0))]
                    # print([i.shape for i in aco_list])
                    # print(len(aco_list))
                    while len(aco_list)<tlens[k]:
                        aco_list=pad_acoustic(aco_list)
                    # print(len(aco_list))
                    aco_list=divide_list_evenly(aco_list, tlens[k])
                    acoustic_list[k]=aco_list
                    
                    # print(tlens[k],len(aco_list))
                    # print(len(aco_list))
                    # for v in aco_list:
                    # exit(0)
                VIS_IS_rnn_list,ACO_IS_rnn_list,IS_BAG_list=[],[],[]
                for k in range(len(batch)):
                    VIS_is_rnn_list,ACO_is_rnn_list,IS_bag_list=[],[],[]
                    for tl in range(tlens[k]):
                        imgs=visual_list[k][tl]
                        acos=acoustic_list[k][tl]
                        pad_imgs,pad_acos,vis_is_rnn_list,aco_is_rnn_list,is_bag_list=process_av(imgs,acos)
                        VIS_is_rnn_list.append(vis_is_rnn_list)
                        ACO_is_rnn_list.append(aco_is_rnn_list)
                        IS_bag_list.append(is_bag_list)
                        visual_list[k][tl]=pad_imgs
                        acoustic_list[k][tl]=pad_acos
                        # print(imgs)
                        # print(acos)
                    visual_list[k]=torch.cat(visual_list[k],dim=0)
                    acoustic_list[k]=torch.cat(acoustic_list[k],dim=0)
                    VIS_IS_rnn_list.append(torch.cat(VIS_is_rnn_list,dim=0))
                    ACO_IS_rnn_list.append(torch.cat(ACO_is_rnn_list,dim=0))
                    IS_BAG_list.append(torch.cat(IS_bag_list,dim=0))

                visual_list=pad_sequence(visual_list, target_len=torch.tensor([visual_list[i].shape[0] for i in range(len(batch))],dtype=visual_list[0].dtype).max().item())
                acoustic_list=pad_sequence(acoustic_list, target_len=torch.tensor([acoustic_list[i].shape[0] for i in range(len(batch))],dtype=visual_list[0].dtype).max().item())
                VIS_IS_rnn_list=pad_sequence(VIS_IS_rnn_list, target_len=torch.tensor([VIS_IS_rnn_list[i].shape[0] for i in range(len(batch))],dtype=visual_list[0].dtype).max().item())
                ACO_IS_rnn_list=pad_sequence(ACO_IS_rnn_list, target_len=torch.tensor([ACO_IS_rnn_list[i].shape[0] for i in range(len(batch))],dtype=visual_list[0].dtype).max().item())
                IS_BAG_list=pad_sequence(IS_BAG_list, target_len=torch.tensor([IS_BAG_list[i].shape[0] for i in range(len(batch))],dtype=visual_list[0].dtype).max().item())
                # print([visual_list[i].shape[0] for i in range(len(batch))])
                # print([acoustic_list[i].shape[0] for i in range(len(batch))])
                # print([VIS_IS_rnn_list[i].shape[0] for i in range(len(batch))])
                # print([ACO_IS_rnn_list[i].shape[0] for i in range(len(batch))])
                # print(torch.tensor([IS_BAG_list[i].shape[0] for i in range(len(batch))]))
                # print(IS_BAG_list)
                # print(visual_list.shape)
                # print(acoustic_list.permute(1,0,2)[-1][12:16,:])
                # print(VIS_IS_rnn_list.shape)
                # print(ACO_IS_rnn_list.permute(1,0)[-1])
                # print(IS_BAG_list.shape)
                # exit(0)
                return visual_list,acoustic_list,VIS_IS_rnn_list,ACO_IS_rnn_list,IS_BAG_list
                

            # sentences = pad_sequence([torch.LongTensor(sample[0][0]) for sample in batch],padding_value=PAD)
            visual = pad_sequence([torch.FloatTensor(sample[0][1]) for sample in batch], target_len=vlens.max().item())
            acoustic = pad_sequence([torch.FloatTensor(sample[0][2]) for sample in batch],target_len=alens.max().item())

            ## BERT-based features input prep

            # SENT_LEN = min(sentences.size(0),50)
            SENT_LEN = 50
            # Create bert indices using tokenizer

            max_source_length = 512
            max_target_length = 4


            inputs_seq = []
            outputs_seq = []
            prompt_emb = []
            prompt_id = []
            # A = -0.0001
            # B = 0.0001  # 小数的范围A ~ B
            # C = 6
            for sample in batch:
                # text = " ".join(sample[0][3])
                text = sample[0][3]
                # print('ori sample:{}'.format(text))
                # score = str(sample[1][0][0])
                score = str(sample[1])
                # source = sample[2]
                # print('source:{}'.format(source))

                # if source == 0:
                #     if is_number(score):
                #         text = 'the output of ' + text + 'is a number'
                #     else:
                #         text = 'the output of ' + text + 'is a category'
                # else:
                #     if is_number(score):
                #         text = 'the output of ' + text + 'is a category'
                #     else:
                #         text = 'the output of ' + text + 'is a number'
                #
                # if hp.use_k_prompt:
                #     if source == 0:
                #         if is_number(score): ### 回归任务
                #             # print('regression score:{}'.format(score))
                #             prompt_id.append(list(range(0, hp.k_prompt)))
                #         elif score in ['anger', 'joy', 'surprise', 'disgust', 'fear', 'neutral', 'sadness']: ### 分类任务
                #             # print('classification score:{}'.format(score))
                #             prompt_id.append(list(range(hp.k_prompt, 2 * hp.k_prompt)))
                #         else: ### 提取任务
                #             # print('extraction score:{}'.format(score))
                #             prompt_id.append(list(range(2 * hp.k_prompt, 3 * hp.k_prompt)))
                #     else: ### 提取任务
                #         # print('extraction score:{}'.format(score))
                #         prompt_id.append(list(range(3 * hp.k_prompt, 4 * hp.k_prompt)))


                # if contain_nonum(score):
                #     text = 'prefix: regression ' + text
                # else:
                #     text = 'prefix: classification ' + text
                    # prompt_emb.append(prompt_dict['classification'])
                # a = random.uniform(A, B)
                # deviation = round(a, C)
                # score = str(sample[1][0][0]+deviation)
                # score = '. '.join(str(sample[1][0][0]).split('.'))
                # print('before:{}, after:{}'.format(str(sample[1][0][0]), score))

                inputs_seq.append(text)
                outputs_seq.append(score)
                if prompt_dict != None:
                    if is_number(score):
                        prompt_emb.append(prompt_dict['regression'])
                    else:
                        prompt_emb.append(prompt_dict['classification'])


            encoding = tokenizer(
                [task_prefix + sequence for sequence in inputs_seq],
                return_tensors="pt", padding=True
            )
            # T5 model things are batch_first
            t5_input_id = encoding.input_ids


            # for i in range(acoustic.size(1)):
            #     print(i,acoustic[:,i,:])
            # exit(0)
            t5_att_mask = encoding.attention_mask
  
            visual_list,acoustic_list,VIS_IS_rnn_list,ACO_IS_rnn_list,IS_BAG_list=process(visual, vlens, acoustic, alens,t5_att_mask)

            target_encoding = tokenizer(
            outputs_seq, padding="longest"
            )
            t5_labels = target_encoding.input_ids
            t5_labels = torch.tensor(t5_labels)
            t5_labels[t5_labels == tokenizer.pad_token_id] = -100
            # lengths are useful later in using RNNs
            # lengths = torch.LongTensor([len(sample[0][0]) for sample in batch])
            prompt_embs = torch.FloatTensor(prompt_emb)
            prompt_ids = torch.IntTensor(prompt_id)
            if (vlens <= 0).sum() > 0:
                vlens[np.where(vlens == 0)] = 1

            # return [], visual.to(DEVICE), vlens, acoustic.to(DEVICE), alens, labels,labels_float, t5_input_id.to(DEVICE), t5_att_mask.to(DEVICE), t5_labels.to(DEVICE), prompt_embs.to(DEVICE), prompt_ids.to(DEVICE),ids
            
            return [], (visual_list.to(DEVICE),VIS_IS_rnn_list.to(DEVICE)), vlens, (acoustic_list.to(DEVICE),ACO_IS_rnn_list.to(DEVICE)), alens,\
             labels,labels_float, t5_input_id.to(DEVICE), t5_att_mask.to(DEVICE), t5_labels.to(DEVICE), prompt_embs.to(DEVICE), prompt_ids.to(DEVICE),ids,IS_BAG_list.to(DEVICE)
            
        else:
            ### 没有视频模态
            inputs_seq = []
            outputs_seq = []
            # A = -0.0001
            # B = 0.0001  # 小数的范围A ~ B
            # C = 6
            for sample in batch:
                text = " ".join(sample[0][3])
                label = str(sample[1])

                inputs_seq.append(text)
                outputs_seq.append(label)

            encoding = tokenizer(
                [task_prefix + sequence for sequence in inputs_seq],
                return_tensors="pt", padding=True
            )
            # T5 model things are batch_first
            t5_input_id = encoding.input_ids
            t5_att_mask = encoding.attention_mask
            target_encoding = tokenizer(
                outputs_seq, padding="longest"
            )
            t5_labels = target_encoding.input_ids
            t5_labels = torch.tensor(t5_labels)
            t5_labels[t5_labels == tokenizer.pad_token_id] = -100

            acoustic = torch.FloatTensor([sample[0][2] for sample in batch])
            labels = [sample[1] for sample in batch]
            return None, None, None, acoustic.to(DEVICE), None, labels, t5_input_id.to(
                DEVICE), t5_att_mask.to(DEVICE), t5_labels.to(DEVICE), None, None


    data_loader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn)

    return data_loader

# def get_two_loader(hp, config, shuffle=True):
#     """Load DataLoader of given DialogDataset"""
#     ### 只在混合训练中的测试部分使用
#     dataset = MSADataset(config)
#
#     print(config.mode)
#     config.data_len = len(dataset)
#     config.tva_dim = dataset.tva_dim
#
#
#     hp.n_test = len(dataset)
#
#     task_prefix = "sst2 sentence: "
#
#     def collate_fn(batch):
#         '''
#         Collate functions assume batch = [Dataset[i] for i in index_set]
#         '''
#         # for later use we sort the batch in descending order of length
#         labels = []
#         ids = []
#
#         for sample in batch:
#             ids.append(sample[2].strip())
#             label = sample[1].strip()
#             labels.append(label)
#
#
#         inputs_seq = []
#         outputs_seq = []
#         for sample in batch:
#             text =sample[0].strip()
#             score = str(sample[1])
#             inputs_seq.append(text),
#             outputs_seq.append(score)
#
#         encoding = tokenizer(
#             [task_prefix + sequence for sequence in inputs_seq],
#             return_tensors="pt", padding=True
#         )
#         # T5 model things are batch_first
#         t5_input_id = encoding.input_ids
#         t5_att_mask = encoding.attention_mask
#         target_encoding = tokenizer(
#             outputs_seq, padding="longest"
#         )
#         t5_labels = target_encoding.input_ids
#         t5_labels = torch.tensor(t5_labels)
#         t5_labels[t5_labels == tokenizer.pad_token_id] = -100
#         # lengths are useful later in using RNNs
#
#         return None, None, None, None, None, labels, t5_input_id, t5_att_mask, t5_labels, ids
#
#     data_loader = DataLoader(
#         dataset=dataset,
#         batch_size=config.batch_size,
#         shuffle=shuffle,
#         collate_fn=collate_fn)
#
#     return data_loader

def get_single_modal_loader(hp, config, shuffle=True):
    """Load DataLoader of given DialogDataset"""

    dataset = MSADataset(config)

    print(config.mode)
    config.data_len = len(dataset)
    config.tva_dim = dataset.tva_dim

    if config.mode == 'train':
        hp.n_train = len(dataset)
    elif config.mode == 'valid':
        hp.n_valid = len(dataset)
    elif config.mode == 'test':
        hp.n_test = len(dataset)

    task_prefix = "sst2 sentence: "

    def collate_fn(batch):
        '''
        Collate functions assume batch = [Dataset[i] for i in index_set]
        '''
        # for later use we sort the batch in descending order of length
        labels = []
        ids = []

        for sample in batch:
            ids.append(sample[2].strip())
            label = sample[1].strip()
            labels.append(label)


        inputs_seq = []
        outputs_seq = []
        for sample in batch:
            text =sample[0].strip()
            score = str(sample[1])
            inputs_seq.append(text),
            outputs_seq.append(score)

        encoding = tokenizer(
            [task_prefix + sequence for sequence in inputs_seq],
            return_tensors="pt", padding=True
        )
        # T5 model things are batch_first
        t5_input_id = encoding.input_ids
        t5_att_mask = encoding.attention_mask
        target_encoding = tokenizer(
            outputs_seq, padding="longest"
        )
        t5_labels = target_encoding.input_ids
        t5_labels = torch.tensor(t5_labels)
        t5_labels[t5_labels == tokenizer.pad_token_id] = -100
        # lengths are useful later in using RNNs

        return None, None, None, None, None, labels, t5_input_id, t5_att_mask, t5_labels, ids

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn)

    return data_loader
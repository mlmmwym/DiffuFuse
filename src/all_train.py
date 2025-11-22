import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from pathlib import Path
import pprint
from torch import optim
import time
import argparse

# sys.path.append('/data/lxj/UniMSE')
# sys.path.append('/data/lxj/UniMSE/src')
# sys.path.append('/data/lxj/UniMSE/src/modules')
# sys.path.append('/data/lxj/UniMSE/src/utils')
from utils import *
from solver import Solver
from data_loader import get_loader, get_single_modal_loader

# path to a pretrained word embedding file
word_emb_path = '/home/henry/glove/glove.840B.300d.txt'
assert (word_emb_path is not None)
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

username = Path.home().name
project_dir = Path(__file__).resolve().parent.parent
sdk_dir = project_dir.joinpath('CMU-MultimodalSDK')
data_dir = project_dir.joinpath('datasets')
data_dict = {'mosi': data_dir.joinpath('MOSI'), 'mosei': data_dir.joinpath(
    'MOSEI'), 'mos': data_dir.joinpath('MOS'), 'moseld': data_dir.joinpath('MOSELD'),
             'moseldmp': data_dir.joinpath('MOSELDMP'), 'iemocap': data_dir.joinpath('IEMOCAP'),
             'meld': data_dir.joinpath('MELD'), 'emotionlines': data_dir.joinpath('EmotionLines'),
             'laptops': data_dir.joinpath('laptops'), 'restaurants': data_dir.joinpath(('restaurants'))}
optimizer_dict = {'RMSprop': optim.RMSprop, 'Adam': optim.Adam}
activation_dict = {'elu': nn.ELU, "hardshrink": nn.Hardshrink, "hardtanh": nn.Hardtanh,
                   "leakyrelu": nn.LeakyReLU, "prelu": nn.PReLU, "relu": nn.ReLU, "rrelu": nn.RReLU,
                   "tanh": nn.Tanh}

output_dim_dict = {
    'mosi': 1,
    'mosei_senti': 1,
}

criterion_dict = {
    'mosi': 'L1Loss',
    'iemocap': 'CrossEntropyLoss',
    'ur_funny': 'CrossEntropyLoss'
}


def get_args():
    parser = argparse.ArgumentParser(description='MOSI-and-MOSEI Sentiment Analysis')
    parser.add_argument('-f', default='', type=str)

    # Tasks
    parser.add_argument('--dataset', type=str, default='mos',
                        choices=['mosi', 'mosei', 'mos', 'moseld', 'moseldmp', 'iemocap', 'meld', 'emotionlines',
                                 'laptops', 'restaurants'],
                        help='dataset to use (default: mosei)')
    parser.add_argument('--data_path', type=str, default='datasets',
                        help='path for storing the dataset')

    # Dropouts
    parser.add_argument('--dropout_a', type=float, default=0.2,
                        help='dropout of acoustic LSTM out layer')
    parser.add_argument('--dropout_v', type=float, default=0.5,
                        help='dropout of visual LSTM out layer')
    parser.add_argument('--dropout_prj', type=float, default=0.1,
                        help='dropout of projection layer')

    # Architecture
    parser.add_argument('--multiseed', action='store_true', help='training using multiple seed')
    parser.add_argument('--add_va', action='store_true', help='if add va MMILB module')
    parser.add_argument('--n_layer', type=int, default=1,
                        help='number of layers in LSTM encoders (default: 1)') 
    parser.add_argument('--d_vh', type=int, default=32,
                        help='hidden size in visual rnn')
    parser.add_argument('--d_ah', type=int, default=32,
                        help='hidden size in acoustic rnn')
    parser.add_argument('--d_vout', type=int, default=32,
                        help='output size in visual rnn')
    parser.add_argument('--d_aout', type=int, default=32,
                        help='output size in acoustic rnn')
    parser.add_argument('--bidirectional', action='store_true', help='Whether to use bidirectional rnn')
    parser.add_argument('--d_prjh', type=int, default=128,
                        help='hidden size in projection network')
    parser.add_argument('--pretrain_emb', type=int, default=512,
                        help='dimension of pretrained model output')

    # Activations
    parser.add_argument('--hidden_size', default=768)
    parser.add_argument('--gradient_accumulation_step', default=5)

    # Training Setting
    parser.add_argument('--batch_size', type=int, default=80, metavar='N',
                        help='batch size (default: 32)')
    parser.add_argument('--clip', type=float, default=1.0,
                        help='gradient clip value (default: 0.8)')
    parser.add_argument('--lr_main', type=float, default=0.00001,
                        help='initial learning rate for main model parameters (default: 1e-3)')
    parser.add_argument('--lr_T5', type=float, default=3e-5,
                        help='initial learning rate for bert parameters (default: 5e-5)')
    parser.add_argument('--lr_adapter', type=float, default=0.00001,
                        help='initial learning rate for mmilb parameters (default: 1e-3)')
    parser.add_argument('--lr_info', type=float, default=0.00001,
                        help='initial learning rate for mmilb parameters (default: 0.0001)')

    parser.add_argument('--weight_decay_main', type=float, default=1e-4,
                        help='L2 penalty factor of the main Adam optimizer')
    parser.add_argument('--weight_decay_adapter', type=float, default=1e-4,
                        help='L2 penalty factor of the main Adam optimizer')
    parser.add_argument('--weight_decay_T5', type=float, default=1e-4,
                        help='L2 penalty factor of the main Adam optimizer')
    parser.add_argument('--weight_decay_info', type=float, default=1e-4,
                        help='L2 penalty factor of the main Adam optimizer')

    #### subnetwork parameter
    parser.add_argument('--embed_dropout', type=float, default=1e-4,
                        help='embed_drop')
    parser.add_argument('--attn_dropout', type=float, default=1e-4,
                        help='attn_dropout')
    parser.add_argument('--attn_mask', action='store_false',
                        help='use attention mask for Transformer (default: true)')
    parser.add_argument('--num_heads', type=int, default=2,
                        help='num_heads')
    parser.add_argument('--relu_dropout', type=float, default=1e-4,
                        help='relu_dropout')
    parser.add_argument('--res_dropout', type=float, default=1e-4,
                        help='res_dropout')
    parser.add_argument('--num_layers', type=int, default=2, help='num_layers')
    #### subnetwork parameter

    parser.add_argument('--optim', type=str, default='Adam',
                        help='optimizer to use (default: Adam)')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='number of epochs (default: 40)')
    parser.add_argument('--when', type=int, default=20,
                        help='when to decay learning rate (default: 20)')
    parser.add_argument('--patience', type=int, default=10,
                        help='when to stop training if best never change')
    parser.add_argument('--update_batch', type=int, default=1,
                        help='update batch interval')

    # Logistics
    parser.add_argument('--log_interval', type=int, default=100,
                        help='frequency of result logging (default: 100)')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')

    parser.add_argument('--use_adapter', type=bool, default=True)
    parser.add_argument('--adapter_name', type=str, default='mag', choices=['ffn', 'parallel', 'cross-atten', 'mag'])
    parser.add_argument('--adapter_layer', type=int, default=3, choices=[1, 2, 3, 4])
    parser.add_argument('--fine_T5', type=bool, default=True, help='whether finetune T5')
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--fine_T5_layers', type=list, default=['block.10', 'block.11'])

    parser.add_argument('--save', type=bool, default=True)
    #### 对比学习
    parser.add_argument('--info_nce', type=bool, default=True, help='whether use info_nce_loss')
    parser.add_argument('--use_info_nce_num', type=int, default=3, help='the number of used info_nce',
                        choices=[3, 4, 5])
    parser.add_argument('--use_cl', type=bool, default=True, help='whether use info_nce_loss')
    parser.add_argument('--cl_name', type=str, default='info_nce', help='the number of used info_nce',
                        choices=['info_nce', 'info_mi'])
    ### 对比学习

    ### 可视化
    parser.add_argument('--visualize', type=bool, default=False)

    ### 可视化

    #### MAG
    parser.add_argument("--beta_shift", type=float, default=0.0)
    parser.add_argument("--dropout_prob", type=float, default=0.0)
    #### MAG

    ###p-tune v2###
    parser.add_argument('--use_prefix_p', type=bool, default=False)
    parser.add_argument('--pre_seq_len', type=int, default=8, choices=[1, 10])
    parser.add_argument('--prompt_hidden_size', type=int, default=64)
    parser.add_argument('--prefix_hidden_size', type=int, default=64)
    parser.add_argument('--num_hidden_layers', type=int, default=1)
    parser.add_argument('--prefix_projection', type=bool, default=False)
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.3)
    ###p-tune v2###

    parser.add_argument('--s_dim', type=int, default=30, help='the projection dim of text, video and audio')

    parser.add_argument('--multi', type=bool, default=True, help='modality setting')
    parser.add_argument('--fuse', type=bool, default=True, help='joint training')

    parser.add_argument('--pred_type', type=str, default='classification', help='modality setting',
                        choices=['regression', 'classification', 'generation'])
    # parser.add_argument('--seed_range', action='append',type=int,default=[0,100])
    parser.add_argument('--seed_start', type=int, default=1040, help='Start of seed range')
    parser.add_argument('--seed_end', type=int, default=1050, help='End of seed range')
    parser.add_argument('--log_name', type=str, default='train.txt', help='End of seed range')
    # parser.add_argument('--log_dir', type=str, default='../Log/log.txt', help='Name of the model to use')
    args = parser.parse_args()
    return args


def str2bool(v):
    """string to boolean"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Config(object):
    def __init__(self, data, mode='train'):
        """Configuration Class: set kwargs as class attributes with setattr"""
        self.dataset_dir = data_dict[data.lower()]
        self.sdk_dir = sdk_dir
        self.mode = mode
        # Glove path
        self.word_emb_path = word_emb_path

        # Data Split ex) 'train', 'valid', 'test'
        self.data_dir = self.dataset_dir
        self.hidden_size = 512

    def __str__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        return config_str


def get_config(dataset='mosi', mode='train', batch_size=2):
    config = Config(data=dataset, mode=mode)

    config.dataset = dataset
    config.batch_size = batch_size

    return config


def set_seed(seed):
    torch.set_default_tensor_type('torch.FloatTensor')
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # torch.set_default_tensor_type('torch.cuda.FloatTensor')

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        use_cuda = True


def train(args=None):
    # print(sys.path.append("/data/lxj/UniMSE/src/utils"))
    # args = get_args()
    dataset = str.lower(args.dataset.strip())
    bs = args.batch_size
    print(args.seed)
    set_seed(args.seed)
    print("Start loading the data....")
    print(args)
    train_config = get_config(dataset, mode='train', batch_size=args.batch_size)
    valid_config = get_config(dataset, mode='valid', batch_size=args.batch_size)

    if args.multi:
        train_loader = get_loader(args, train_config, shuffle=True)
    else:
        train_loader = get_single_modal_loader(args, train_config, shuffle=True)
    print('{} training data loaded!'.format(args.n_train))
    if args.multi:
        valid_loader = get_loader(args, valid_config, shuffle=False)
    else:
        valid_loader = get_single_modal_loader(args, valid_config, shuffle=False)
    print('{} validation data loaded!'.format(args.n_valid))

    # addintional appending
    args.word2id = train_config.word2id

    # architecture parameters
    args.d_tin, args.d_vin, args.d_ain = train_config.tva_dim
    args.dataset = args.data = dataset
    args.when = args.when
    args.n_class = output_dim_dict.get(dataset, 1)
    args.init_checkpoint = '../t5-base/pytorch_model.bin'
    # args.init_checkpoint = '../t5-large/pytorch_model.bin'

    ###adapter
    args.adapter_initializer_range = 0.001

    if dataset == 'mos':
        mosi_test_config = get_config(dataset, mode='test_mosi', batch_size=args.batch_size)
        mosei_test_config = get_config(dataset, mode='test_mosei', batch_size=args.batch_size)

        mosi_test_loader = get_loader(args, mosi_test_config, shuffle=False)
        mosei_test_loader = get_loader(args, mosei_test_config, shuffle=False)

        print('{} MOSI Test data loaded!'.format(args.n_mosi_test))
        print('{} MOSEI Test data loaded!'.format(args.n_mosei_test))
        print('Finish loading the data....')

        solver = Solver(args, train_loader=train_loader, dev_loader=valid_loader,
                        test_loader=(mosi_test_loader, mosei_test_loader), is_train=True)

    elif dataset == 'moseld':
        mosi_test_config = get_config(dataset, mode='test_mosi', batch_size=args.batch_size)
        mosei_test_config = get_config(dataset, mode='test_mosei', batch_size=args.batch_size)
        meld_test_config = get_config(dataset, mode='test_meld', batch_size=args.batch_size)

        mosi_test_loader = get_loader(args, mosi_test_config, shuffle=False)
        mosei_test_loader = get_loader(args, mosei_test_config, shuffle=False)
        meld_test_loader = get_loader(args, meld_test_config, shuffle=False)

        print('{} MOSI Test data loaded!'.format(args.n_mosi_test))
        print('{} MOSEI Test data loaded!'.format(args.n_mosei_test))
        print('{} MELD Test data loaded!'.format(args.n_meld_test))
        print('Finish loading the data....')

        solver = Solver(args, train_loader=train_loader, dev_loader=valid_loader,
                        test_loader=(mosi_test_loader, mosei_test_loader, meld_test_loader), is_train=True)

    elif dataset == 'moseldmp':
        mosi_test_config = get_config(dataset, mode='test_mosi', batch_size=args.batch_size)
        mosei_test_config = get_config(dataset, mode='test_mosei', batch_size=args.batch_size)
        meld_test_config = get_config(dataset, mode='test_meld', batch_size=args.batch_size)
        iemocap_test_config = get_config(dataset, mode='test_iemocap', batch_size=args.batch_size)

        mosi_test_loader = get_loader(args, mosi_test_config, shuffle=False)
        mosei_test_loader = get_loader(args, mosei_test_config, shuffle=False)
        meld_test_loader = get_loader(args, meld_test_config, shuffle=False)
        iemocap_test_loader = get_loader(args, iemocap_test_config, shuffle=False)

        print('{} MOSI Test data loaded!'.format(args.n_mosi_test))
        print('{} MOSEI Test data loaded!'.format(args.n_mosei_test))
        print('{} MELD Test data loaded!'.format(args.n_meld_test))
        print('{} IEMOCAP Test data loaded!'.format(args.n_iemocap_test))

        print('Finish loading the data....')

        solver = Solver(args, train_loader=train_loader, dev_loader=valid_loader,
                        test_loader=(mosi_test_loader, mosei_test_loader, meld_test_loader, iemocap_test_loader),
                        is_train=True)

    else:
        test_config = get_config(dataset, mode='test', batch_size=args.batch_size)
        if not args.multi:
            test_loader = get_single_modal_loader(args, test_config, shuffle=False)
        else:
            test_loader = get_loader(args, test_config, shuffle=False)
        print('{} Test data loaded!'.format(args.n_test))
        print('Finish loading the data....')
        solver = Solver(args, train_loader=train_loader, dev_loader=valid_loader,
                        test_loader=test_loader, is_train=True)

    # pretrained_emb saved in train_config here

    train_and_eval_info = solver.train_and_eval()
    # return mae_list, corr_list, mult_a7_list, mult_a5_list, f_score_list, f_score_non0_list, acc_2_list, acc_2_non0_list,train_loss_list,val_loss_list,test_loss_list
    return train_and_eval_info


import logging
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


def plot_values(title, save_path, value_lists, line_labels, y_min=None, y_max=None):
    epochs = len(value_lists[0])
    x = list(range(1, epochs + 1))

    plt.figure()

    # 遍历所有列表并绘制折线图，使用提供的折线名称
    for values, label in zip(value_lists, line_labels):
        plt.plot(x, values, label=label, marker='o')  # 'o'标记每个点

    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)

    # 设置Y轴的固定刻度范围
    if y_min != None:
        plt.ylim([y_min, y_max])
        plt.gca().yaxis.set_major_locator(MultipleLocator(0.1))

    plt.savefig(save_path)
    # plt.show()


if __name__ == '__main__':

    args = get_args()
    # 设置日志
    logging.basicConfig(filename=args.log_name, level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(message)s')
    os.makedirs('../DiffuFuse/MAE', exist_ok=True)
    os.makedirs('../DiffuFuse/Corr', exist_ok=True)
    os.makedirs('../DiffuFuse/Acc7', exist_ok=True)
    os.makedirs('../DiffuFuse/Acc5', exist_ok=True)
    os.makedirs('../DiffuFuse/F1', exist_ok=True)
    os.makedirs('../DiffuFuse/Acc2', exist_ok=True)
    os.makedirs('../DiffuFuse/Loss', exist_ok=True)
    print(args)
    seed_start, seed_end = args.seed_start, args.seed_end

    Max_acc_dict = {
        'mosei': {'mae': 100, 'corr': 0, 'mult_a7': 0, 'mult_a5': 0, 'f_score': 0, 'f_score_non0': 0, 'acc_2': 0,
                  'acc_2_non0': 0},
        'mosi': {'mae': 100, 'corr': 0, 'mult_a7': 0, 'mult_a5': 0, 'f_score': 0, 'f_score_non0': 0, 'acc_2': 0,
                 'acc_2_non0': 0}}
    Best_seed_dict = {
        'mosei': {'mae': 0, 'corr': 0, 'mult_a7': 0, 'mult_a5': 0, 'f_score': 0, 'f_score_non0': 0, 'acc_2': 0,
                  'acc_2_non0': 0},
        'mosi': {'mae': 0, 'corr': 0, 'mult_a7': 0, 'mult_a5': 0, 'f_score': 0, 'f_score_non0': 0, 'acc_2': 0,
                 'acc_2_non0': 0}}

    # for seed in range(0, 1000):
    #     set_seed(seed)
    #     train_and_eval_info = train(args)

    for seed in range(seed_start, seed_end):
        set_seed(seed)
        start = time.time()
        args = get_args()
        args.seed = seed
        print("|" * 50 + "Seed:{}".format(seed) + "|" * 50)
        # mosi_acc_list,acc_list,meld_acc_list,iemocap_acc_list=train(args)
        train_and_eval_info = train(args)
        for dataname in ['mosei', 'mosi']:

            max_acc_dict, best_seed_dict = Max_acc_dict[dataname], Best_seed_dict[dataname]
            mae_list, corr_list, mult_a7_list, mult_a5_list, f_score_list, f_score_non0_list, acc_2_list, acc_2_non0_list, train_loss_list, val_loss_list, test_loss_list = \
                train_and_eval_info[dataname + '_' + 'mae_list'], train_and_eval_info[dataname + '_' + 'corr_list'], \
                train_and_eval_info[dataname + '_' + 'mult_a7_list'], \
                    train_and_eval_info[dataname + '_' + 'mult_a5_list'], train_and_eval_info[
                    dataname + '_' + 'f_score_list'], train_and_eval_info[dataname + '_' + 'f_score_non0_list'], \
                    train_and_eval_info[dataname + '_' + 'acc_2_list'], train_and_eval_info[
                    dataname + '_' + 'acc_2_non0_list'], train_and_eval_info['train_loss_list'], \
                    train_and_eval_info['val_loss_list'], train_and_eval_info[dataname + '_' + 'test_loss_list']

            print('**************************ACC-{}********************************'.format(dataname))
            if max_acc_dict['mae'] > min(mae_list):
                max_acc_dict['mae'] = min(mae_list)
                best_seed_dict['mae'] = seed

            if max_acc_dict['corr'] < max(corr_list):
                max_acc_dict['corr'] = max(corr_list)
                best_seed_dict['corr'] = seed

            if max_acc_dict['mult_a7'] < max(mult_a7_list):
                max_acc_dict['mult_a7'] = max(mult_a7_list)
                best_seed_dict['mult_a7'] = seed

            if max_acc_dict['mult_a5'] < max(mult_a5_list):
                max_acc_dict['mult_a5'] = max(mult_a5_list)
                best_seed_dict['mult_a5'] = seed

            if max_acc_dict['f_score'] < max(f_score_list):
                max_acc_dict['f_score'] = max(f_score_list)
                best_seed_dict['f_score'] = seed

            if max_acc_dict['f_score_non0'] < max(f_score_non0_list):
                max_acc_dict['f_score_non0'] = max(f_score_non0_list)
                best_seed_dict['f_score_non0'] = seed

            if max_acc_dict['acc_2'] < max(acc_2_list):
                max_acc_dict['acc_2'] = max(acc_2_list)
                best_seed_dict['acc_2'] = seed

            if max_acc_dict['acc_2_non0'] < max(acc_2_non0_list):
                max_acc_dict['acc_2_non0'] = max(acc_2_non0_list)
                best_seed_dict['acc_2_non0'] = seed

            print(max_acc_dict)
            print(best_seed_dict)
            # os.makedirs('../DiffuFuse/seed{}'.format(seed), exist_ok=True)
            plot_values('MAE', '../DiffuFuse/MAE/{}seed{}.png'.format(dataname, seed), [mae_list], ['MAE'], 0,
                        1.5)
            plot_values('Corr', '../DiffuFuse/Corr/{}seed{}.png'.format(dataname, seed), [corr_list], ['Corr'],
                        0, 1)
            plot_values('Acc7', '../DiffuFuse/Acc7/{}seed{}.png'.format(dataname, seed), [mult_a7_list],
                        ['Acc7'], 0, 1)
            plot_values('Acc5', '../DiffuFuse/Acc5/{}seed{}.png'.format(dataname, seed), [mult_a5_list],
                        ['Acc5'], 0, 1)
            plot_values('F1', '../DiffuFuse/F1/{}seed{}.png'.format(dataname, seed),
                        [f_score_list, f_score_non0_list], ['f1_all', 'f1_non0'], 0.5, 1)
            plot_values('Acc2', '../DiffuFuse/Acc2/{}seed{}.png'.format(dataname, seed),
                        [acc_2_list, acc_2_non0_list], ['acc2_all', 'acc2_non0'], 0.5, 1)
            plot_values('Loss', '../DiffuFuse/Loss/{}seed{}.png'.format(dataname, seed),
                        [train_loss_list, val_loss_list, test_loss_list],
                        ['train_loss_list', 'val_loss_list', 'test_loss_list'])

            logging.info("|" * 50 + "Seed:{}".format(seed) + "|" * 50)
            logging.info('**************************ACC-{}********************************'.format(dataname))
            logging.info(f"MAE: {mae_list}")
            logging.info(f"Corr: {corr_list}")
            logging.info(f"Acc7: {mult_a7_list}")
            logging.info(f"Acc5: {mult_a5_list}")
            logging.info(f"F1_All: {f_score_list}")
            logging.info(f"F1_Non0: {f_score_non0_list}")
            logging.info(f"Acc2_All: {acc_2_list}")
            logging.info(f"Acc2_Non0: {acc_2_non0_list}")
            logging.info(f"train_loss_list: {train_loss_list}")
            logging.info(f"val_loss_list: {val_loss_list}")
            logging.info(f"test_loss_list: {test_loss_list}")
            logging.info(max_acc_dict)
            logging.info(best_seed_dict)

        print('**************************End********************************')
        end = time.time()
        duration = end - start
        logging.info('**************************End Time {:5.4f} sec********************************'.format(duration))
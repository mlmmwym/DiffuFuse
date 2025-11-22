# %load ~/data_lxj/UniMSE/src/utils/eval_metrics.py
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
# from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from itertools import chain
import sys
# sys.path.append("~/data_lxj/UniMSE/src/utils")
# import tools
from .tools import *
import operator
import functools
import pandas as pd
import os

polarity_to_number={'negative':'-1','neutral':'0','positive':'1'}
iemocap_label_to_number={'1':'anger','2':'disgust','6':'fear','5':'joy','0':'neutral','3':'sadness','4':'surprise'}
meld_label_to_number={'1':'anger','2':'frustrated','5':'joy','0':'neutral','3':'sadness','4':'excited'}

def get_vt():
    # Load the CSV file

    file_path = '../datasets/MOSEI/MOSEI-label.csv'
    print(os.path.abspath(file_path))
    data = pd.read_csv(file_path)
    # print(data['clip_id'].tolist())
    video_ids = data['video_id'].tolist()
    clip_ids = data['clip_id'].tolist()
    video_clip_ids = [video + '_' + str(clip) for video, clip in zip(video_ids, clip_ids)]
    texts = data['text'].tolist()
    # Display the first few rows of the dataframe to the user
    videototext = {video: text for video, text in zip(video_clip_ids, texts)}
    # data.head()
    return videototext

def multiclass_acc(preds, truths):
    """
    Compute the multiclass accuracy w.r.t. groundtruth

    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))


def weighted_accuracy(test_preds_emo, test_truth_emo):
    true_label = (test_truth_emo > 0)
    predicted_label = (test_preds_emo > 0)
    tp = float(np.sum((true_label==1) & (predicted_label==1)))
    tn = float(np.sum((true_label==0) & (predicted_label==0)))
    p = float(np.sum(true_label==1))
    n = float(np.sum(true_label==0))

    return (tp * (n/p) +tn) / (2*n)

def eval_emotionlines(results, truths):
    #主要通过混淆矩阵来计算
    truths = [item for sublist in truths for item in sublist]
    for i, ele in enumerate(results):
        if ele == 'sad' or ele == 'sa':
            results[i] = 'sadness'
        if ele == 'frust':
            results[i] = 'frustrated'
    # cm = confusion_matrix(results, truths)
    report = classification_report(truths, results, zero_division=1)
    print(report)
    report = classification_report(truths, results,output_dict=True, zero_division=1)
    return report['accuracy']
def eval_iemocap(results, truths):
    #主要通过混淆矩阵来计算
    class_set = {'anger','excited','frustrated','neutral','excited','joy'}
    
    for i in range(len(results)):
        ele = results[i]
        if ele in iemocap_label_to_number.keys(): ele=iemocap_label_to_number[ele]
        else : ele='neutral'
        results[i]=ele
    
    for i, ele in enumerate(results):
        if ele not in class_set:
            if ele == 'sad' or ele == 'sa' or 'sad' in ele:
                results[i] = 'sadness'
            elif ele == 'frust' or ele == 'frustness' or 'frust' in ele:
                results[i] = 'frustrated'
            elif ele == 'joyness' or 'joy' in ele:
                results[i] = 'joy'
            elif 'anger' in ele:
                results[i] = 'anger'
            elif 'exicited' in ele:
                results[i] = 'exicited'
            elif 'neu' in ele:
                results[i] = 'neutral'
            else:
                results[i] = 'neutral'
            
    # cm = confusion_matrix(results, truths)
    report = classification_report(truths, results)
    print(report)
    
def eval_meld(results, truths):
    #主要通过混淆矩阵来计算
    class_set = {'anger','disgust','joy','neutral','sadness','surprise'}

    for i in range(len(results)):
        ele = results[i]
        if ele in meld_label_to_number.keys(): ele=meld_label_to_number[ele]
        else : ele='neutral'
        results[i]=ele    
    
    for i, ele in enumerate(results):
        if ele == 'sad' or ele == 'sa':
            results[i] = 'sadness'
        if ele == 'frust':
            results[i] = 'frustrated'
    # cm = confusion_matrix(results, truths)
    report = classification_report(truths, results)
    print(report)

def eval_laptops_restants(results, truths):
    truths = list(chain(*truths))
    true_count = 0
    for pred, truth in zip(results, truths):
        if pred == truth:
            true_count += 1
    print('true_count:{}, total_count:{}, precision:{}'.format(true_count, len(truths), true_count/len(truths)))
    # f1 = f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')


def eval_mosei_senti(results, truths, ids_list, exclude_zero=False):
    # test_truth = truths.view(-1).cpu().detach().numpy()
    # print(truths)
    # print(results)
    test_truth = np.array(functools.reduce(operator.concat, truths))
    # print(test_truth.shape)
    new_test_truth = []
    # for e in test_truth:
    #     if is_number(e):
    #         new_test_truth.append(float(e))
    #     else:
    #         new_test_truth.append(0.0)
    test_truth = np.array([float(e) for e in test_truth])

    non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])
    preds = []
    for ele in results:
        pred=ele.split(',')
        if len(pred) >1: pred=pred[1]
        else : pred=pred[0]
        if is_number(pred):
            preds.append(float(pred))
        else:
            preds.append(0.0)
            
#     print('test_truth:{}'.format(test_truth))
#     print('preds:{}'.format(preds))
#     preds=preds/2.0
    test_preds = np.array(preds)
    test_preds=np.clip(test_preds, a_min=-3., a_max=3.)
    test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
    test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
    test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
    test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)

    mae = np.mean(np.absolute(test_preds - test_truth))   # Average L1 distance between preds and truths
    corr = np.corrcoef(test_preds, test_truth)[0][1]
    mult_a7 = multiclass_acc(test_preds_a7, test_truth_a7)
    mult_a5 = multiclass_acc(test_preds_a5, test_truth_a5)

    # f_score = f1_score((test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average='weighted')
    tmp = test_truth[non_zeros] > 0
    binary_truth_non0 = np.array([int(ele) for ele in tmp])
    tmp = test_preds[non_zeros] > 0
    binary_preds_non0 =  np.array([int(ele) for ele in tmp])
    f_score_non0 = f1_score(binary_truth_non0, binary_preds_non0, average='weighted')

    acc_2_non0 = accuracy_score(binary_truth_non0, binary_preds_non0)
    binary_truth_has0 = test_truth >= 0
    binary_preds_has0 = test_preds >= 0
    acc_2 = accuracy_score(binary_truth_has0, binary_preds_has0)
    f_score = f1_score(binary_truth_has0, binary_preds_has0, average='weighted')

    # videototext=get_vt()
    # count_t,count_f=0,0
    # avg_t,avg_f=0,0
    # for i in range(len(ids_list)):
    #     t,p=binary_truth_has0[i],binary_preds_has0[i]
    #     if t==p:
    #         count_t+=1
    #         avg_t+=len(videototext[ids_list[i]].split())
    #     else:
    #         count_f+=1
    #         avg_f+=len(videototext[ids_list[i]].split())

    print("MAE: ", mae)
    print("Correlation Coefficient: ", corr)
    print("mult_acc_7: ", mult_a7)
    print("mult_acc_5: ", mult_a5)
    print("F1 score all/non0: {}/{} over {}/{}".format(np.round(f_score,4), np.round(f_score_non0,4), binary_truth_has0.shape[0], binary_truth_non0.shape[0]))
    print("Accuracy all/non0: {}/{}".format(np.round(acc_2,4), np.round(acc_2_non0,4)))
    # print("AvgText True/False: {}/{}".format(np.round(avg_t/count_t, 4), np.round(avg_f/count_f, 4)))

    print("-" * 50)
    return np.round(mae,4),np.round(corr,4),np.round(mult_a7,4),np.round(mult_a5,4),np.round(f_score,4)\
        ,np.round(f_score_non0,4),np.round(acc_2,4),np.round(acc_2_non0,4)

def eval_mosi(results, truths, ids_list, exclude_zero=False):
    return eval_mosei_senti(results, truths, ids_list, exclude_zero)

if __name__=='__main__':
    get_vt()

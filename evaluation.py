import json
import numpy as np
import pickle
from sklearn.metrics import accuracy_score
import argparse
import csv
from sklearn.metrics import confusion_matrix, precision_score, recall_score, auc, roc_curve, roc_auc_score, f1_score

file_base_path = './Results/'
parser = argparse.ArgumentParser(description='')
parser.add_argument('--file_name', default='None', type=str)
args = parser.parse_args()

file_name = args.file_name
file_path = file_base_path + file_name

def sortByKey(array, key):
    sorted_obj = sorted(array, key=lambda x : x[key], reverse=True)
    return sorted_obj

data = []
with open(file_path, 'rb') as f:
    data = pickle.load(f)

def calculateKappa(array):
    for i in range(0, len(array)):
        labels = array[i]['labels']
        probs = array[i]['probs'][:,-1]
        threshold_tr = array[i]['best_threshold_tr']
        threshold_te = array[i]['best_threshold_te']
        tmp_probs_tr = probs >= threshold_tr
        tmp_probs_tr = tmp_probs_tr.astype(float)
        precision_tr = precision_score(labels, tmp_probs_tr, average = None)[1]
        recall_tr = recall_score(labels, tmp_probs_tr, average = None)[1]
        fscore_tr = f1_score(labels, tmp_probs_tr, average = None)[1]
        tmp_probs_te = probs >= threshold_te
        tmp_probs_te = tmp_probs_te.astype(float)
        precision_te = precision_score(labels, tmp_probs_te, average = None)[1]
        recall_te = recall_score(labels, tmp_probs_te, average = None)[1]
        fscore_te = f1_score(labels, tmp_probs_te, average = None)[1]
        array[i]['pre_tr'] = precision_tr
        array[i]['rec_tr'] = recall_tr
        array[i]['fscore_tr'] = fscore_tr
        array[i]['pre_te'] = precision_te
        array[i]['rec_te'] = recall_te
        array[i]['fscore_te'] = fscore_te

def writeToCSV(array, file_path):
    tmp = file_path.split('.')
    file_path = tmp[0] + '.csv'
    with open(file_path, 'wt') as file:
        writer = csv.writer(file)
        writer.writerow(('ROC_AUC', 'Pre-tr', 'Rec-tr', 'F-Score-tr', 'Threshold-tr', 'Pre-0.5', 'Rec-0.5', 'F-Score-0.5',
                         'Pre-te', 'Rec-te', 'F-Score-te', 'Threshold-te', 'Window-Size', 'Lead-Time', 'Epoch-No'))
        for i in range(0, len(array)):
            writer.writerow((array[i]['roc_auc'], array[i]['pre_tr'], array[i]['rec_tr'],
                             array[i]['fscore_tr'], array[i]['best_threshold_tr'], array[i]['precision'],
                             array[i]['recall'], array[i]['f_score'], array[i]['pre_te'], array[i]['rec_te'],
                             array[i]['fscore_te'], array[i]['best_threshold_te'],
                             array[i]['w'], array[i]['l'], array[i]['epoch']))
    file.close()

def topKModels(array, top_k_runs):
    for i in range(0, top_k_runs):
        print 'roc_auc:', array[i]['roc_auc'], 'kappa:', array[i]['kappa'], 'tp:', array[i]['tp'], 'fn:', array[i]['fn'], 'fp:', array[i]['fp'], 'tn:', array[i]['tn'], 'pre:', array[i]['precision'], 'recall:', array[i]['recall'], 'f_score:', array[i]['f_score'], 'n:', array[i]['n'], 'w:', array[i]['w'], 'l:', array[i]['l'], 'batch_size:', array[i]['batch_size'], 'epoch', array[i]['epoch']

def findBestCutoffForFScoreTraining(array):
    for i in range(0, len(array)):
        probs = array[i]['train_probs'][:,-1]
        labels = array[i]['train_labels']
        indices = np.argsort(probs)
        probs = probs[indices]
        labels = labels[indices]
        best_fscore = 0.0
        best_precision = 0.0
        best_recall = 0.0
        best_threshold = 0.5
        for j in range(0, labels.shape[0]):
            threshold = probs[j]
            tmp_probs = probs >= threshold
            tmp_probs = tmp_probs.astype(float)
            precision = precision_score(labels, tmp_probs, average = None)[1]
            recall = recall_score(labels, tmp_probs, average = None)[1]
            fscore = f1_score(labels, tmp_probs, average = None)[1]
            #print fscore
            if(fscore > best_fscore):
                best_fscore = fscore
                best_precision = precision
                best_recall = recall
                best_threshold = threshold
        array[i]['best_threshold_tr'] = best_threshold

def findBestCutoffForFScoreTest(array):
    for i in range(0, len(array)):
        probs = array[i]['probs'][:,-1]
        labels = array[i]['labels']
        indices = np.argsort(probs)
        probs = probs[indices]
        labels = labels[indices]
        best_fscore = 0.0
        best_precision = 0.0
        best_recall = 0.0
        best_threshold = 0.5
        for j in range(0, labels.shape[0]):
            threshold = probs[j]
            tmp_probs = probs >= threshold
            tmp_probs = tmp_probs.astype(float)
            precision = precision_score(labels, tmp_probs, average = None)[1]
            recall = recall_score(labels, tmp_probs, average = None)[1]
            fscore = f1_score(labels, tmp_probs, average = None)[1]
            #print fscore
            if(fscore > best_fscore):
                best_fscore = fscore
                best_precision = precision
                best_recall = recall
                best_threshold = threshold
        array[i]['best_threshold_te'] = best_threshold
        

data = sortByKey(data, 'roc_auc')
#data = sortByKey(data, 'kappa')
findBestCutoffForFScoreTraining(data)
findBestCutoffForFScoreTest(data)
calculateKappa(data)
#top_k_runs = 300
#topKModels(data, top_k_runs)
writeToCSV(data, file_path)

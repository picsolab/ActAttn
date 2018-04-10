import random
import json
import csv
import numpy as np
import pandas as pd
from keras.utils.np_utils import to_categorical
from scipy import stats

def readData(dataset_type):
    
    awn_days = None
    if(dataset_type == 'Charlottesville'):
        awn_days = range(11, 32)
    elif(dataset_type == 'FergusonI'):
        awn_days = range(9, 28)
    elif(dataset_type == 'FergusonII'):
        awn_days = range(11, 41)

    states_data_path = './Data/state_abbrv.txt'
    awn_data_path_base = './Data/' + dataset_type + '/'
    awn_dynamic_features_path = awn_data_path_base + 'features.csv'
    awn_labels_path = awn_data_path_base + 'DVs.csv'

    states = []
    with open(states_data_path, 'rb') as file:
        states = [line.rstrip() for line in file]
    
    regions = [u'west', u'south', u'northeast', u'midwest']
    number_of_static_features = 4 + len(regions)
    number_of_dynamic_features = 50
    number_of_classes = 2
    
    static_features = np.zeros(shape=(len(states), number_of_static_features))
    tmp_static_data = pd.read_csv(awn_labels_path)
    tmp_static_data_length = len(states)
    tmp_static_data_keys = tmp_static_data.keys()
    for i in range(0, tmp_static_data_length):
        state = tmp_static_data[u'state'][i]
        state_index = states.index(state)
        region = tmp_static_data[u'region'][i]
        region_index = regions.index(region.lower())
        for j in range(6, len(tmp_static_data_keys)):
            static_features[state_index][j-6] = tmp_static_data[tmp_static_data_keys[j]][i]
        static_features[state_index][4+region_index] = 1
    #print static_features
    
    
    labels_volume = np.zeros(shape=(len(states), len(awn_days), number_of_classes))
    labels_number_of_events = np.zeros(shape=(len(states), len(awn_days), 1))
    tmp_label_data = pd.read_csv(awn_labels_path)
    tmp_label_data_length = len(tmp_label_data[tmp_label_data.keys()[0]])
    for i in range(0, tmp_label_data_length):
        tmp_date = tmp_label_data[u'time'][i]
        tmp_date = tmp_date.split('-')
        day = int(tmp_date[2])
        state = tmp_label_data[u'state'][i]
        
        state_index = states.index(state)
        day_index = day - awn_days[0]
        num = tmp_label_data[u'num'][i]
        tmp_volume = tmp_label_data[u'cat'][i]
        volume = 100
        if(tmp_volume == 'N'):
            volume = 0
        elif(tmp_volume == 'S'):
            volume = 1
        elif(tmp_volume == 'L'):
            volume = 1
        #volume = to_categorical(volume, num_classes=number_of_classes)
        
        labels_volume[state_index][day_index][volume] = 1
        labels_number_of_events[state_index][day_index][0] = num
    
    #print labels_volume
    #print labels_number_of_events
    
    
    dynamic_features = np.zeros(shape=(len(states), len(awn_days), number_of_dynamic_features))
    tmp_dynamic_data = pd.read_csv(awn_dynamic_features_path)
    tmp_dynamic_data_length = len(tmp_dynamic_data[tmp_dynamic_data.keys()[0]])
    tmp_dynamic_data_keys = tmp_dynamic_data.keys()
    for i in range(0, tmp_dynamic_data_length):
        tmp_date = tmp_dynamic_data['time'][i]
        tmp_date = tmp_date.split('-')
        day = int(tmp_date[2])
        state = tmp_dynamic_data['state'][i]
        
        state_index = states.index(state)
        day_index = day - awn_days[0]
        for j in range(2, len(tmp_dynamic_data_keys)):
            if(j == 2):
                dynamic_features[state_index][day_index][j-2] = tmp_dynamic_data[tmp_dynamic_data_keys[j]][i]
            else:
                dynamic_features[state_index][day_index][j-2] = np.divide(float(tmp_dynamic_data[tmp_dynamic_data_keys[j]][i]), tmp_dynamic_data[tmp_dynamic_data_keys[2]][i])
    #print dynamic_features
    
    
    if(dataset_type == 'FII'):
        first_omitted_days = 10
        dynamic_features = dynamic_features[:, first_omitted_days:, :]
        labels_volume = labels_volume[:, first_omitted_days:, :]
        labels_number_of_events = labels_number_of_events[:, first_omitted_days:, :]
    
    return states, static_features, dynamic_features, labels_volume, labels_number_of_events


def createTensorsBySequential(window_size, lead_time, number_of_test_samples_for_each_state, dataset_type):
    
    states, static_features, dynamic_features, labels_volume, labels_number_of_events = readData(dataset_type)
    number_of_time_steps = dynamic_features.shape[1]
    
    #print labels_volume
    #print labels_volume.shape
    
    ##### Set training start index and test start index #####
    train_start_index = 0
    test_start_index = number_of_time_steps - (number_of_test_samples_for_each_state + lead_time + window_size - 1)
    
    ##### Normalize number of tweets in dynamic features #####
    no_tweets = dynamic_features[:, 0:test_start_index, 0].reshape(-1)
    no_tweet_mean = np.mean(no_tweets)
    no_tweet_std = np.std(no_tweets)
    #print no_tweet_mean
    #print no_tweet_std
    #print '+++++++++++++++'
    for i in range(0, len(states)):
        for j in range(0, number_of_time_steps):
            dynamic_features[i][j][0] = float(dynamic_features[i][j][0] - no_tweet_mean) / no_tweet_std
    
    ##### Normalize static features #####
    #print static_features.shape
    no_states = static_features.shape[0]
    no_static_features = static_features.shape[1]
    for i in range(0, no_static_features):
        if(i == 0 or i == 1): #population and pden
            static_features[:, i] = np.log(static_features[:, i])
        elif(i == 2 or i == 3): #vote and div
            features = static_features[:, i].reshape(-1)
            features_mean = np.mean(features)
            features_std = np.std(features)
            #print features_mean
            #print features_std
            #print '----------------------'
            static_features[:, i] = static_features[:, i] - features_mean
            static_features[:, i] /= features_std
    #print static_features
    
    ##### Prepare training and test data #####
    #training_sample_size = no_states * (test_start_index - window_size - lead_time + 2)
    training_sample_size = no_states * test_start_index
    test_sample_size = no_states * number_of_test_samples_for_each_state
    
    train_x = np.zeros(shape=(training_sample_size, window_size, dynamic_features.shape[2]))
    train_y = np.zeros(shape=(training_sample_size, labels_volume.shape[2]))
    test_x = np.zeros(shape=(test_sample_size, window_size, dynamic_features.shape[2]))
    test_y = np.zeros(shape=(test_sample_size, labels_volume.shape[2]))
    train_static = np.zeros(shape=(training_sample_size, static_features.shape[1]))
    test_static = np.zeros(shape=(test_sample_size, static_features.shape[1]))
    
    train_sides = np.zeros(shape=(training_sample_size, no_states, window_size, dynamic_features.shape[2]))
    test_sides = np.zeros(shape=(test_sample_size, no_states, window_size, dynamic_features.shape[2]))
    
    counter = 0
    for i in range(0, len(states)): 
        #for j in range(train_start_index, (test_start_index - window_size + 1)):
        for j in range(train_start_index, test_start_index):
            for k in range(0, window_size):
                #print counter, k
                train_x[counter][k] = dynamic_features[i][j+k]
            for l in range(0, len(states)):
                for m in range(0, window_size):
                    train_sides[counter][l][m] = dynamic_features[l][j+m]
            train_y[counter] = labels_volume[i][j + window_size + lead_time - 1]
            train_static[counter] = static_features[i]
            counter += 1
    
    counter = 0
    for i in range(0, len(states)): 
        for j in range(test_start_index, (number_of_time_steps - lead_time - window_size + 1)):
            for k in range(0, window_size):
                test_x[counter][k] = dynamic_features[i][j+k]
            for l in range(0, len(states)):
                for m in range(0, window_size):
                    test_sides[counter][l][m] = dynamic_features[l][j+m]
            test_y[counter] = labels_volume[i][j + window_size + lead_time - 1]
            test_static[counter] = static_features[i]
            counter += 1
    
    '''
    print 'train_x size: ', train_x.shape
    print 'train_y size: ', train_y.shape
    print 'test_x size: ', test_x.shape
    print 'test_y size: ', test_y.shape
    print 'train_static size: ', train_static.shape
    print 'test_static size: ', test_static.shape
    '''
    return train_x, train_y, train_static, test_x, test_y, test_static, train_sides, test_sides

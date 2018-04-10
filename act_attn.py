import os
import numpy as np
import tensorflow as tf
from keras.layers import Input, Embedding, LSTM, Dense, Flatten, Activation, RepeatVector, Permute, merge, Multiply, Lambda, concatenate, Dropout, multiply
from keras.models import Model
from keras import optimizers
from keras import backend as K
from keras.utils.np_utils import to_categorical
from sklearn.metrics import confusion_matrix, precision_score, recall_score, auc, roc_curve, roc_auc_score, f1_score
import argparse
import sys
import pickle
from keras.callbacks import ModelCheckpoint
from keras.callbacks import Callback
from keras.regularizers import l1_l2, l1, l2
from keras.regularizers import Regularizer
import data
import argparse

parser = argparse.ArgumentParser(description='Non-Event or Event')
parser.add_argument('--hidden_size', default=16, type=int)
parser.add_argument('--learning_rate', default=0.001, type=float)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--lead_time', default=1, type=int)
parser.add_argument('--window_size', default=1, type=int)
parser.add_argument('--no_epochs', default=10, type=int)
parser.add_argument('--number_of_test_samples_for_each_state', default=5, type=int)
parser.add_argument('--output_file', default='None', type=str)
parser.add_argument('--dataset_type', default='None', type=str)
parser.add_argument('--local_reg_factor', default=0.0001, type=float)
parser.add_argument('--global_reg_factor', default=0.0001, type=float)

args = parser.parse_args()
hidden_size = args.hidden_size
learning_rate = args.learning_rate
batch_size = args.batch_size
no_epochs = args.no_epochs
window_size = args.window_size
lead_time = args.lead_time
number_of_test_samples_for_each_state = args.number_of_test_samples_for_each_state
output_file = args.output_file
dataset_type = args.dataset_type
local_reg_factor = args.local_reg_factor
global_reg_factor = args.global_reg_factor
output_base_path = './Results/'

print 'lead_time:', lead_time, 'window_size:',  window_size, 'number_of_test_samples_for_each_state:', number_of_test_samples_for_each_state, 'batch_size:', batch_size, 'no_epochs:', no_epochs, 'local_reg_factor:', local_reg_factor, 'global_reg_factor:', global_reg_factor

train_x, train_y, train_static, test_x, test_y, test_static, train_sides, test_sides = data.createTensorsBySequential(args.window_size, args.lead_time, args.number_of_test_samples_for_each_state, args.dataset_type)

#print train_x.shape, train_static.shape, train_y.shape, train_sides.shape
#print test_x.shape, test_static.shape, test_y.shape, test_sides.shape   

time_steps = window_size
dynamic_feature_size = train_x.shape[2]
volume_output_size = train_y.shape[1]
static_input_size = train_static.shape[1]
no_states = train_sides.shape[1]

train_sides = np.swapaxes(train_sides, 0, 1)
test_sides = np.swapaxes(test_sides, 0, 1)

class L21(Regularizer):
    """Regularizer for L21 regularization.
    # Arguments
        C: Float; L21 regularization factor.
    """

    def __init__(self, C=0.):
        self.C = K.cast_to_floatx(C)

    def __call__(self, x):
        const_coeff = np.sqrt(K.int_shape(x)[1])
        return self.C*const_coeff*K.sum(K.sqrt(K.sum(K.square(x), axis=1)))

    def get_config(self):
        return {'C': float(self.l1)}


def spatial_attention(inputs):
    
    SINGLE_ATTENTION_VECTOR = False
    
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Dense(int(inputs.shape[1]), activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction_spatial')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec_spatial')(a)
    output_attention_mul = multiply([inputs, a_probs])
    output_attention_mul = Lambda(lambda x: K.sum(x, axis=1), name='dim_reduction_2_spatial')(output_attention_mul)
    return output_attention_mul

def spatiotemporal_attention(inputs):
    
    SINGLE_ATTENTION_VECTOR = False
    
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Dense(int(inputs.shape[1]), activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction_spatiotemporal')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec_spatiotemporal')(a)
    output_attention_mul = multiply([inputs, a_probs])
    output_attention_mul = Lambda(lambda x: K.sum(x, axis=1), name='dim_reduction_2_spatiotemporal')(output_attention_mul)
    return output_attention_mul

class Evaluation_Callback(Callback):
    
    def __init__(self, train_data, validation_data, args):
        super(Callback, self).__init__()

        self.current_epoch = 0
        self.args = args
        self.data_to_be_saved = []
        self.x_train_raw, self.train_y = train_data  #tuple of train X and y
        self.train_x = self.x_train_raw[0]
        self.train_sides = self.x_train_raw[1]
        self.train_static = self.x_train_raw[2]
        self.x_test_raw, self.test_y = validation_data  #tuple of validation X and y
        self.test_x = self.x_test_raw[0]
        self.test_sides = self.x_test_raw[1]
        self.test_static = self.x_test_raw[2]
    
    def on_train_begin(self, logs={}):
        return
 
    def on_train_end(self, logs={}):
        return
 
    def on_epoch_begin(self, epoch, logs={}):
        return
 
    def on_epoch_end(self, epoch, logs={}):
        self.current_epoch += 1
        
        if(self.current_epoch % 2 == 1):
            train_pred = self.model.predict([self.train_x, self.train_sides[0], self.train_sides[1], self.train_sides[2], self.train_sides[3], self.train_sides[4], self.train_sides[5], self.train_sides[6], self.train_sides[7], self.train_sides[8], self.train_sides[9], self.train_sides[10], self.train_sides[11], self.train_sides[12], self.train_sides[13], self.train_sides[14], self.train_sides[15], self.train_sides[16], self.train_sides[17], self.train_sides[18], self.train_sides[19], self.train_sides[20], self.train_sides[21], self.train_sides[22], self.train_sides[23], self.train_sides[24], self.train_sides[25], self.train_sides[26], self.train_sides[27], self.train_sides[28], self.train_sides[29], self.train_sides[30], self.train_sides[31], self.train_sides[32], self.train_sides[33], self.train_sides[34], self.train_sides[35], self.train_sides[36], self.train_sides[37], self.train_sides[38], self.train_sides[39], self.train_sides[40], self.train_sides[41], self.train_sides[42], self.train_sides[43], self.train_sides[44], self.train_sides[45], self.train_sides[46], self.train_sides[47], self.train_sides[48], self.train_sides[49], self.train_sides[50], self.train_static], verbose=0)
            test_pred = self.model.predict([self.test_x, self.test_sides[0], self.test_sides[1], self.test_sides[2], self.test_sides[3], self.test_sides[4], self.test_sides[5], self.test_sides[6], self.test_sides[7], self.test_sides[8], self.test_sides[9], self.test_sides[10], self.test_sides[11], self.test_sides[12], self.test_sides[13], self.test_sides[14], self.test_sides[15], self.test_sides[16], self.test_sides[17], self.test_sides[18], self.test_sides[19], self.test_sides[20], self.test_sides[21], self.test_sides[22], self.test_sides[23], self.test_sides[24], self.test_sides[25], self.test_sides[26], self.test_sides[27], self.test_sides[28], self.test_sides[29], self.test_sides[30], self.test_sides[31], self.test_sides[32], self.test_sides[33], self.test_sides[34], self.test_sides[35], self.test_sides[36], self.test_sides[37], self.test_sides[38], self.test_sides[39], self.test_sides[40], self.test_sides[41], self.test_sides[42], self.test_sides[43], self.test_sides[44], self.test_sides[45], self.test_sides[46], self.test_sides[47], self.test_sides[48], self.test_sides[49], self.test_sides[50], self.test_static], verbose=0)
            
            labels = np.argmax(self.test_y, axis=1)
            train_labels = np.argmax(self.train_y, axis=1)
            predictions = np.argmax(test_pred, axis=1)
            
            conf_mat = confusion_matrix(np.argmax(self.test_y, axis=1), np.argmax(test_pred, axis=1))
            
            precisions = precision_score(np.argmax(self.test_y, axis=1), np.argmax(test_pred, axis=1), average = None)
            recalls = recall_score(np.argmax(self.test_y, axis=1), np.argmax(test_pred, axis=1), average = None)
            f_score = f1_score(np.argmax(self.test_y, axis=1), np.argmax(test_pred, axis=1), average = None)[1]
            
            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(volume_output_size):
                fpr[i], tpr[i], _ = roc_curve(self.test_y[:, i], test_pred[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            
            #print 'ROC-AUC:', roc_auc[1]
            
            # Save Data
            a_dict = {'f_score': f_score,
                      'precision': precisions[1],
                      'recall': recalls[1],
                      'roc_auc': roc_auc[1],
                      'tp': conf_mat[1][1], 'tn': conf_mat[0][0], 'fp': conf_mat[0][1], 'fn': conf_mat[1][0],
                      'batch_size': args.batch_size,
                      'epoch': self.current_epoch,
                      'n': args.number_of_test_samples_for_each_state,
                      'w': args.window_size,
                      'l': args.lead_time,
                      'hidden_size': args.hidden_size,
                      'learning_rate': args.learning_rate,
                      'labels': labels,
                      'predictions': predictions,
                      'probs': test_pred,
                      'train_probs': train_pred,
                      'train_labels': train_labels}
            
            self.data_to_be_saved.append(a_dict)
                
        if(self.current_epoch == self.args.no_epochs):
            output = []
            if os.path.exists(args.output_file):
                with open(args.output_file,'rb') as rfp: 
                    output = pickle.load(rfp)
            output = output + self.data_to_be_saved
            with open(args.output_file,'wb') as wfp:
                pickle.dump(output, wfp)
            self.data_to_be_saved = []
            
            model_save_file = output_base_path + args.dataset_type + '.h5'
            self.model.save_weights(model_save_file)
                
        return
 
    def on_batch_begin(self, batch, logs={}):
        return
 
    def on_batch_end(self, batch, logs={}):
        #self.losses.append(logs.get('loss'))
        return

main_input = Input(shape=(time_steps, dynamic_feature_size), name='main_input')
main_out = RepeatVector(1)(LSTM(hidden_size, kernel_regularizer=L21(local_reg_factor), return_sequences=False)(main_input))

side_inputs = []
side_outs = []
for i in range(1, 52):
    name = 'side_input' + str(i)
    side_input = Input(shape=(time_steps, dynamic_feature_size), name=name)
    side_out = RepeatVector(1)(LSTM(hidden_size, kernel_regularizer=L21(global_reg_factor), return_sequences=False)(side_input))
    side_inputs.append(side_input)
    side_outs.append(side_out)

concated_spatial_sides = concatenate(side_outs, axis=1)
spatial_attention_out = RepeatVector(1)(spatial_attention(concated_spatial_sides))

concated_spatiotemporal = concatenate([main_out, spatial_attention_out], axis=1)
main_att_out = spatiotemporal_attention(concated_spatiotemporal)

static_input = Input(shape=(static_input_size,), name='static_input')
concated_all = concatenate([static_input, main_att_out])

main_output = Dense(volume_output_size, activation='softmax', name='main_output')(concated_all)

inputs = [main_input] + side_inputs + [static_input]
model = Model(inputs=inputs, outputs=[main_output])

adam = optimizers.Adam(lr=learning_rate)
model.compile(optimizer=adam, loss='categorical_crossentropy')

ev_callback = Evaluation_Callback(train_data=([train_x, train_sides, train_static], train_y), validation_data=([test_x, test_sides, test_static], test_y), args=args)


a = np.argmax(train_y, axis=1)
ratio_train = float(len(a[a==0])) / len(a[a==1])
b = np.argmax(test_y, axis=1)
ratio_test = float(len(b[b==0])) / len(b[b==1])
#print ratio_train
#print ratio_test

class_weight = {0 : 1., 1: ratio_train}
history = model.fit({'main_input': train_x, 'side_input1': train_sides[0], 'side_input2': train_sides[1], 'side_input3': train_sides[2], 'side_input4': train_sides[3], 'side_input5': train_sides[4], 'side_input6': train_sides[5], 'side_input7': train_sides[6], 'side_input8': train_sides[7], 'side_input9': train_sides[8], 'side_input10': train_sides[9], 'side_input11': train_sides[10], 'side_input12': train_sides[11], 'side_input13': train_sides[12], 'side_input14': train_sides[13], 'side_input15': train_sides[14], 'side_input16': train_sides[15], 'side_input17': train_sides[16], 'side_input18': train_sides[17], 'side_input19': train_sides[18], 'side_input20': train_sides[19], 'side_input21': train_sides[20], 'side_input22': train_sides[21], 'side_input23': train_sides[22], 'side_input24': train_sides[23], 'side_input25': train_sides[24], 'side_input26': train_sides[25], 'side_input27': train_sides[26], 'side_input28': train_sides[27], 'side_input29': train_sides[28], 'side_input30': train_sides[29], 'side_input31': train_sides[30], 'side_input32': train_sides[31], 'side_input33': train_sides[32], 'side_input34': train_sides[33], 'side_input35': train_sides[34], 'side_input36': train_sides[35], 'side_input37': train_sides[36], 'side_input38': train_sides[37], 'side_input39': train_sides[38], 'side_input40': train_sides[39], 'side_input41': train_sides[40], 'side_input42': train_sides[41], 'side_input43': train_sides[42], 'side_input44': train_sides[43], 'side_input45': train_sides[44], 'side_input46': train_sides[45], 'side_input47': train_sides[46], 'side_input48': train_sides[47], 'side_input49': train_sides[48], 'side_input50': train_sides[49], 'side_input51': train_sides[50], 'static_input': train_static},
                    {'main_output': train_y},
                    epochs = no_epochs,
                    batch_size = batch_size,
                    shuffle = True,
                    verbose = False,
                    class_weight = class_weight,
                    callbacks=[ev_callback])
                    #validation_data = ([test_x, test_static], test_y))

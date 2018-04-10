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

## Example ################
dataset_type = 'Charlottesville'
model_file = './Data/Charlottesville.h5'
saved_file = './Data/Charlottesville.pkl'
window_size = 2
lead_time = 1
number_of_test_samples_for_each_state = 5
hidden_size = 16
local_reg_factor = 0.0001
global_reg_factor = 0.0001
###########################

train_x, train_y, train_static, test_x, test_y, test_static, train_sides, test_sides = data.createTensorsBySequential(window_size, lead_time, number_of_test_samples_for_each_state, dataset_type)

time_steps = window_size
dynamic_feature_size = train_x.shape[2]
volume_output_size = train_y.shape[1]
static_input_size = train_static.shape[1]
no_states = train_sides.shape[1]

train_sides = np.swapaxes(train_sides, 0, 1)
test_sides = np.swapaxes(test_sides, 0, 1)

_, _, _, y, _ = data.readData(dataset_type)
y = np.argmax(y, axis=2)
for i in range(0, y.shape[0]):
    print i, y[i]

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

main_input = Input(shape=(time_steps, dynamic_feature_size), name='main_input')
main_out = RepeatVector(1)(LSTM(hidden_size, kernel_regularizer=L21(local_reg_factor), return_sequences=False)(main_input))

side_inputs = []
side_outs = []
for i in range(1, 52):
    name = 'side_input' + str(i)
    side_input = Input(shape=(time_steps, dynamic_feature_size), name=name)
    side_out = RepeatVector(1)(LSTM(hidden_size, kernel_regularizer=L21(global_reg_factor) return_sequences=False)(side_input))
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

model.load_weights(model_file)

test_pred = model.predict([test_x, test_sides[0], test_sides[1], test_sides[2], test_sides[3], test_sides[4], test_sides[5], test_sides[6], test_sides[7], test_sides[8], test_sides[9], test_sides[10], test_sides[11], test_sides[12], test_sides[13], test_sides[14], test_sides[15], test_sides[16], test_sides[17], test_sides[18], test_sides[19], test_sides[20], test_sides[21], test_sides[22], test_sides[23], test_sides[24], test_sides[25], test_sides[26], test_sides[27], test_sides[28], test_sides[29], test_sides[30], test_sides[31], test_sides[32], test_sides[33], test_sides[34], test_sides[35], test_sides[36], test_sides[37], test_sides[38], test_sides[39], test_sides[40], test_sides[41], test_sides[42], test_sides[43], test_sides[44], test_sides[45], test_sides[46], test_sides[47], test_sides[48], test_sides[49], test_sides[50], test_static], verbose=0)

print model.summary()

print model.layers[51].name
print len(model.layers[51].get_weights())

data_saved = []
with open(saved_file, 'rb') as f:
    data_saved = pickle.load(f)
data_saved = data_saved[len(data_saved)-1]
true_labels = data_saved['labels']
roc_auc = data_saved['roc_auc']

predicted_labels = np.argmax(data_saved['probs'], axis=1)
precision = precision_score(true_labels, predicted_labels, average = None)[1]
recall = recall_score(true_labels, predicted_labels, average = None)[1]
f_score = f1_score(true_labels, predicted_labels, average = None)[1]
print 'thd-0.5-pre:', precision
print 'thd-0.5-rec:', recall
print 'thd-0.5-f-score:', f_score
print 'roc_auc:', roc_auc
print '----------------------------'

predicted_labels_thd = data_saved['probs'][:,-1]
predicted_labels_thd = predicted_labels_thd >= 0.5
predicted_labels_thd = predicted_labels_thd.astype(float)
print 'Predicted Labels Thd. shape:', predicted_labels_thd.shape
print 'True Labels shape:', true_labels.shape
true_predicted_labels = []
for i in range(0, predicted_labels_thd.shape[0]):
    if predicted_labels_thd[i] == true_labels[i] and predicted_labels_thd[i] == 1:
        true_predicted_labels.append(i)
print true_predicted_labels
print '----------------------------'

'''
for e in zip(model.layers[51].trainable_weights, model.layers[51].get_weights()):
    print('Param %s:\n%s' % (e[0],e[1]))
'''

print 'Main Weights'
print model.layers[159].name
main_weights = model.layers[159].get_weights()[0]
main_weights = main_weights[:, 0:hidden_size]
#print weights.shape
main_weights = np.absolute(main_weights)
main_weights[main_weights<0.001] = 0
print np.mean(main_weights, axis = 1)
print '----------------------------'

print 'Side State Weights'
for i in range(51, 102):
    weights = model.layers[i].get_weights()[0]
    weights = weights[:, 0:hidden_size]
    #print weights.shape
    weights = np.absolute(weights)
    weights[weights<0.001] = 0
    print np.mean(weights)
print '----------------------------'

### Calculate 0-weights ###
counter = 0
spatial_counter = 0
temporal_counter = 0
for i in range(51, 102):
    inp_weights = model.layers[i].get_weights()[0]
    reg_weights = model.layers[i].get_weights()[1]
    inp_weights = np.absolute(inp_weights)
    reg_weights = np.absolute(reg_weights)
    for k in range(0, inp_weights.shape[0]):
        for j in range(0, inp_weights.shape[1]):
            if inp_weights[k][j] < 0.001:
                counter += 1
                spatial_counter += 1
    '''
    for k in range(0, reg_weights.shape[0]):
        for j in range(0, reg_weights.shape[1]):
            if reg_weights[k][j] < 0.001:
                counter += 1
                spatial_counter += 1
    '''

inp_weights = model.layers[159].get_weights()[0]
reg_weights = model.layers[159].get_weights()[1]
bias_weights = model.layers[159].get_weights()[2]
print inp_weights.shape
print reg_weights.shape
print bias_weights.shape
inp_weights = np.absolute(inp_weights)
for k in range(0, inp_weights.shape[0]):
    for j in range(0, inp_weights.shape[1]):
        if inp_weights[k][j] < 0.001:
            counter += 1
            temporal_counter += 1
print "counter:", counter
print "temporal_counter:", temporal_counter
print "spatial_counter:", spatial_counter
print '----------------------------'

#print 'AK Weights'
#weights = model.layers[52].get_weights()[0]
#print 'MO Weights'
#weights = model.layers[76].get_weights()[0]
#print 'VA Weights'
#weights = model.layers[97].get_weights()[0]
#print 'IA Weights'
#weights = model.layers[66].get_weights()[0]
print 'CA Weights'
weights = model.layers[55].get_weights()[0]
#print 'NY Weights'
#weights = model.layers[83].get_weights()[0]
#print 'TX Weights'
#weights = model.layers[94].get_weights()[0]
weights = weights[:, 0:hidden_size]
#print weights.shape
weights = np.absolute(weights)
weights[weights<0.001] = 0
print np.mean(weights, axis=1)
print '----------------------------'

print 'Static Input Weights'
print model.layers[171].name
weights = model.layers[171].get_weights() 
print weights
print '----------------------------'


a = []
for i in range(0, 51):
    a.append(model.layers[i].get_input_at(0))
a.append(K.learning_phase())

#print model.layers[160].name
#encoder_func = K.function(a, [model.layers[160].output])
print model.layers[156].name
encoder_func = K.function(a, [model.layers[156].output])
out = encoder_func([test_sides[0], test_sides[1], test_sides[2], test_sides[3], test_sides[4], test_sides[5], test_sides[6], test_sides[7], test_sides[8], test_sides[9], test_sides[10], test_sides[11], test_sides[12], test_sides[13], test_sides[14], test_sides[15], test_sides[16], test_sides[17], test_sides[18], test_sides[19], test_sides[20], test_sides[21], test_sides[22], test_sides[23], test_sides[24], test_sides[25], test_sides[26], test_sides[27], test_sides[28], test_sides[29], test_sides[30], test_sides[31], test_sides[32], test_sides[33], test_sides[34], test_sides[35], test_sides[36], test_sides[37], test_sides[38], test_sides[39], test_sides[40], test_sides[41], test_sides[42], test_sides[43], test_sides[44], test_sides[45], test_sides[46], test_sides[47], test_sides[48], test_sides[49], test_sides[50]])[0]

out = np.absolute(out)
for i in range(0, len(true_predicted_labels)):
    print out[true_predicted_labels[i]].shape
    print np.mean(out[true_predicted_labels[i]], axis = 1)

print '----------------------------'


b = []
print model.layers[159].name
b.append(model.layers[159].get_input_at(0))
for i in range(0, 51):
    b.append(model.layers[i].get_input_at(0))
b.append(K.learning_phase())

print model.layers[166].name
encoder_func2 = K.function(b, [model.layers[166].output])
out = encoder_func2([test_x, test_sides[0], test_sides[1], test_sides[2], test_sides[3], test_sides[4], test_sides[5], test_sides[6], test_sides[7], test_sides[8], test_sides[9], test_sides[10], test_sides[11], test_sides[12], test_sides[13], test_sides[14], test_sides[15], test_sides[16], test_sides[17], test_sides[18], test_sides[19], test_sides[20], test_sides[21], test_sides[22], test_sides[23], test_sides[24], test_sides[25], test_sides[26], test_sides[27], test_sides[28], test_sides[29], test_sides[30], test_sides[31], test_sides[32], test_sides[33], test_sides[34], test_sides[35], test_sides[36], test_sides[37], test_sides[38], test_sides[39], test_sides[40], test_sides[41], test_sides[42], test_sides[43], test_sides[44], test_sides[45], test_sides[46], test_sides[47], test_sides[48], test_sides[49], test_sides[50]])[0]

out = np.absolute(out)
for i in range(0, len(true_predicted_labels)):
    print out[true_predicted_labels[i]].shape
    print np.mean(out[true_predicted_labels[i]], axis = 1)

print '----------------------------'

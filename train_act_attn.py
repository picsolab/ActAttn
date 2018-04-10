import subprocess
import sys
import numpy as np
import argparse

output_base_path = './Results/'
parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_type', default='None', type=str)
args = parser.parse_args()

dataset_type = args.dataset_type
hidden_size = 16
learning_rate = 0.001
window_size = 3
lead_time = 1
number_of_test_samples_for_each_state = 5
batch_size = 32
no_epochs = 500
local_reg_factor = 0.0001
global_reg_factor = 0.0001

output_file = output_base_path + dataset_type.lower() + '.pkl'
command = 'python act_attn.py --lead_time=' + str(lead_time) + ' --window_size=' + str(window_size) + ' --number_of_test_samples_for_each_state=' + str(number_of_test_samples_for_each_state) + ' --batch_size=' + str(batch_size) + ' --no_epochs=' + str(no_epochs) + ' --hidden_size=' + str(hidden_size) + ' --learning_rate=' + str(learning_rate) + ' --output_file=' + output_file + ' --dataset_type=' + dataset_type + ' --local_reg_factor=' + str(local_reg_factor) + ' --global_reg_factor=' + str(global_reg_factor)
process = subprocess.call(command, shell=True)

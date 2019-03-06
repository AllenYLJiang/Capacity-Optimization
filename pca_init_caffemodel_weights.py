import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
from IPython import get_ipython
from sklearn.decomposition import PCA
get_ipython().run_line_magic('matplotlib', 'inline')

# Make sure that caffe is on the python path:
caffe_root = '/path/to/your/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

#!diff ../exper/voc12/config/VGG16/train_trainval.prototxt ../exper/voc12/config/VGG16/train_trainval_featuremap_redu.prototxt

# Load the original network and extract the fully connected layers' parameters.
net = caffe.Net('/path/to/your/pca/train_trainval.prototxt', 
                '/path/to/your/pca/ori.caffemodel',
                caffe.TRAIN)
#net = caffe.Net('/path/to/your//train_trainval.prototxt', 
#                '/path/to/your//train2_iter_20000.caffemodel',
#                caffe.TRAIN)

#params = ['conv1', 'conv2_1/dw', 'conv2_1/sep', 'conv2_2/dw', 'conv2_2/sep', 'conv3_1/dw', 'conv3_1/sep', 'conv4_2/dw', 'conv4_2/sep', 'conv4_1/dw', 'conv4_1/sep', 'conv4_2/dw', 'conv4_2/sep', 'conv5_1/dw', 'conv5_1/sep', 'conv5_2/dw', 'conv5_2/sep', 'conv5_3/dw', 'conv5_3/sep', 'conv5_4/dw', 'conv5_4/sep', 'conv5_5/dw', 'conv5_5/sep', 'conv5_6/dw', 'conv5_6/sep', 'conv6/dw', 'conv6/sep']
params = ['conv4_2']

fc_params = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in params}
fc_params_abs = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in params}

## Load the fully convolutional network to transplant the parameters.
net_full_conv = caffe.Net('/path/to/your/pca/train_trainval_conv4_2_pca.prototxt',
                          '/path/to/your/pca/init.caffemodel',
                          caffe.TRAIN)

params_full_conv = ['conv4_2_group1/dw', 'conv4_2_group2/dw', 'conv4_2_group3/dw', 'conv4_2/sep'] #

conv_params = {pr: (net_full_conv.params[pr][0].data, net_full_conv.params[pr][1].data) for pr in params_full_conv}
conv_params_abs = {pr: (net_full_conv.params[pr][0].data, net_full_conv.params[pr][1].data) for pr in params_full_conv}

# clustering on conv5_1_group1:fc_params['conv5_1_group1'][0][0:3,0:110,0:3,0:3]
num_input_large_loop = 512 # original conv4_2
num_output_small_loop = 512 # original conv4_2
num_sep_output = 512
num_component_each_input = 3
#num_component_output = 5
######################################### Similarity between output 110*27##########################################################################
# This is the number_of_input_channels * (3*3) * 3 (the number of principal components) 
each_input_3x9_output = [None]*num_input_large_loop  
for i in range(len(each_input_3x9_output)):  
    each_input_3x9_output[i] = [0]*(9*num_component_each_input)  
####################################################################################################################################################
#Dimension of conv4_2/dw:  15,110,3,3 && 15 
#Dimension of conv4_2/sep: 110,15,1,1 && 110
################ PCA coefficients of the 3 components inside each input corresponding to num_output_small_loop outputs #############################
################ Dimension 110(input) x 110(output) x num_component_each_input(coefficients of the 3 components in PCA) ############################
# number_of_input_channels * number_of_output_channels * number_of_principal_components
pca_coefficients_all_inputs = [None]*num_input_large_loop
for i in range(len(pca_coefficients_all_inputs)):
    pca_coefficients_all_inputs[i] = [0]*num_output_small_loop
for i in range(num_input_large_loop):
    for j in range(num_output_small_loop):
        pca_coefficients_all_inputs[i][j] = [0]*num_component_each_input
####################################################################################################################################################
for idx_input in np.arange(num_input_large_loop):
    pca = PCA(n_components=num_component_each_input)   
    
    num_rows = num_output_small_loop  
    num_cols = 9
    conv4_2_temp_vector = [None]*num_rows  
    for i in range(len(conv4_2_temp_vector)):  
        conv4_2_temp_vector[i] = [0]*num_cols  
    for idx_inter_kernel in np.arange(num_output_small_loop):
        for idx_within_kernel in np.arange(9):
            conv4_2_temp_vector[idx_inter_kernel][idx_within_kernel] = fc_params['conv4_2'][0][idx_inter_kernel,idx_input,int(idx_within_kernel/3),(idx_within_kernel%3)]

    pca.fit(conv4_2_temp_vector)
    
    for idx_inter_kernel in np.arange(num_output_small_loop):
        vector1 = [0]*9
        vector2 = [0]*9
        for idx_within_kernel in np.arange(9):        
            vector1[idx_within_kernel] = pca.components_[0][idx_within_kernel]
            vector2[idx_within_kernel] = conv4_2_temp_vector[idx_inter_kernel][idx_within_kernel]
        pca_coefficients_all_inputs[idx_input][idx_inter_kernel][0] = (np.dot(vector1,vector2))#/(np.sqrt(np.dot(vector1,vector1))*np.sqrt(np.dot(vector2,vector2)))
        for idx_within_kernel in np.arange(9):        
            vector1[idx_within_kernel] = pca.components_[1][idx_within_kernel]
            vector2[idx_within_kernel] = conv4_2_temp_vector[idx_inter_kernel][idx_within_kernel]
        pca_coefficients_all_inputs[idx_input][idx_inter_kernel][1] = (np.dot(vector1,vector2))#/(np.sqrt(np.dot(vector1,vector1))*np.sqrt(np.dot(vector2,vector2)))
        for idx_within_kernel in np.arange(9):        
            vector1[idx_within_kernel] = pca.components_[2][idx_within_kernel]
            vector2[idx_within_kernel] = conv4_2_temp_vector[idx_inter_kernel][idx_within_kernel]
        pca_coefficients_all_inputs[idx_input][idx_inter_kernel][2] = (np.dot(vector1,vector2))#/(np.sqrt(np.dot(vector1,vector1))*np.sqrt(np.dot(vector2,vector2)))
            
####################################################################################################################################################
    for idx_within_kernel in np.arange(9):
        conv_params['conv4_2_group1/dw'][0][idx_input,0,int(idx_within_kernel/3),(idx_within_kernel%3)] = pca.components_[0][idx_within_kernel]
        conv_params['conv4_2_group2/dw'][0][idx_input,0,int(idx_within_kernel/3),(idx_within_kernel%3)] = pca.components_[1][idx_within_kernel]
        conv_params['conv4_2_group3/dw'][0][idx_input,0,int(idx_within_kernel/3),(idx_within_kernel%3)] = pca.components_[2][idx_within_kernel]
        conv_params['conv4_2_group1/dw'][1][idx_input] = 0
        conv_params['conv4_2_group2/dw'][1][idx_input] = 0
        conv_params['conv4_2_group3/dw'][1][idx_input] = 0
####################################################################################################################################################

for idx_input in np.arange(num_input_large_loop):
    for idx_output in np.arange(num_sep_output):
        conv_params['conv4_2/sep'][0][idx_output,idx_input,0,0] = pca_coefficients_all_inputs[idx_input][idx_output][0]
        conv_params['conv4_2/sep'][0][idx_output,idx_input + num_input_large_loop,0,0] = pca_coefficients_all_inputs[idx_input][idx_output][1]
        conv_params['conv4_2/sep'][0][idx_output,idx_input + num_input_large_loop * 2,0,0] = pca_coefficients_all_inputs[idx_input][idx_output][2]
        conv_params['conv4_2/sep'][1][idx_output] = fc_params['conv4_2'][1][idx_output]
        
net_full_conv.save('/path/to/your/pca/init.caffemodel')

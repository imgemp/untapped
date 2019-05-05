import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from collections import OrderedDict

import numpy as np
import pandas
import lasagne
from untapped.parmesan.layers import SampleLayer
from untapped.parmesan.layers import NormalizingSimplexFlowLayer, LogisticFlowLayer, SoftplusFlowLayer

from untapped.M12 import L1, L2, KL
from untapped.S2S_DGM import SSDGM
from untapped.utilities import timeStamp

from datasets.load_libs import load_process_data as load_process_libs
from datasets.load_raman import load_process_data as load_process_raman
from datasets.load_raman import get_endmembers
from plotting.plotting_libs import make_plots as make_plots_libs
from plotting.plotting_raman2 import make_plots as make_plots_raman
from IPython import embed
#####################################################################
# REQUIRES optimizer='fast_compile' in .theanorc file under [global]
#####################################################################
# THEANO_FLAGS=device=gpu0 python examples/DEMO_fusion.py
##############################################################################################################
##############################################################################################################
# train libs model
##############################################################################################################
##############################################################################################################
embed()
print('#'*50)
print('training libs model')
print('#'*50)

# # load data
xy, ux, waves, names, colors = load_process_libs(DropLastDim=False)
sup_train_x, sup_train_y, sup_valid_x, sup_valid_y, train_x, train_y = [xyi.astype('float32') for xyi in xy]

# rectify data
names = names[0:4] + names[5:] + [names[4]]
colors = list(colors)
colors = colors[0:4] + colors[5:] + [colors[4]]
sup_train_y = sup_train_y[:,[0,1,2,3,5,6,7,8]]
sup_valid_y = sup_valid_y[:,[0,1,2,3,5,6,7,8]]
train_y = train_y[:,[0,1,2,3,5,6,7,8]]
# sup_train_y = np.hstack((sup_train_y[:,[0,1,2,3,5,6,7,8]],sup_train_y[:,4][:,None]))
# sup_valid_y = np.hstack((sup_valid_y[:,[0,1,2,3,5,6,7,8]],sup_valid_y[:,4][:,None]))
# train_y = np.hstack((train_y[:,[0,1,2,3,5,6,7,8]],train_y[:,4][:,None]))

# define variable sizes
num_features = sup_train_x.shape[1]
num_output = sup_train_y.shape[1]
num_latent_z2 = 2

# construct data dictionary
data = {'X':sup_train_x,'y':sup_train_y,
        'X_':train_x,
        'X_valid':sup_valid_x,'y_valid':sup_valid_y,
        'X__valid':train_x}

# include "untapped" label source
data['_y'] = train_y
data['_y_valid'] = train_y
data['z2'] = np.random.uniform(low=-1.5, high=1.5, size=(train_y.shape[0], num_latent_z2)).astype('float32')
data['z2_valid'] = np.random.uniform(low=-1.5, high=1.5, size=(train_y.shape[0], num_latent_z2)).astype('float32')

# define priors
prior_x = None  # uniform distribution over positive intensities
prior_y = None  # uniform distribution over simplex
prior_z2 = None  # uniform distribution over interval

# define the model architecture (x=z1)
model_dict = OrderedDict()

arch = OrderedDict()
arch['hidden'] = [25,10]
arch['gain'] = np.sqrt(2)
arch['nonlin'] = lasagne.nonlinearities.tanh
arch['num_output'] = num_output
arch['sample_layer'] = SampleLayer
arch['flows'] = [NormalizingSimplexFlowLayer]
arch['batch_norm'] = False
model_dict['z1->y'] = OrderedDict([('arch',arch)])

arch = OrderedDict()
arch['hidden'] = [5,5]
arch['gain'] = np.sqrt(2)
arch['nonlin'] = lasagne.nonlinearities.tanh
arch['num_output'] = num_latent_z2
arch['sample_layer'] = SampleLayer
arch['flows'] = [LogisticFlowLayer]
arch['batch_norm'] = False
model_dict['z1y->z2'] = OrderedDict([('arch',arch)])

arch = OrderedDict()
arch['hidden'] = [50]
arch['gain'] = np.sqrt(2)
arch['nonlin'] = lasagne.nonlinearities.tanh
arch['num_output'] = num_features
arch['sample_layer'] = SampleLayer
arch['flows'] = [SoftplusFlowLayer]
arch['batch_norm'] = False
model_dict['yz2->_z1'] = OrderedDict([('arch',arch)])

# set result directory path
res_out='examples/results/fusion/libs/'+timeStamp().format("")

# construct the semi^2-supervised deep generative model
m = SSDGM(num_features,num_output,model_dict=model_dict,
          prior_x=prior_x,prior_y=prior_y,prior_z2=prior_z2,loss_x=L2,loss_y=KL,
          coeff_x=1e-1,coeff_y=1e-1,coeff_x_dis=1,coeff_y_dis=1e-4,coeff_x_prob=0,coeff_y_prob=0,
          batch_size_X__train=1000,batch_size_X__eval=1000,num_epochs=1500,eval_freq=500,lr=1e-2,eq_samples=1,iw_samples=1,
          res_out=res_out)

# fit the model
m.fit(verbose=True,debug=False,**data)

# auto-set title and plot results (saved to res_out)
title = 'M2'
if m.coeff_y_dis > 0:
    if m.coeff_y > 0:
        title = 'Labels Untapped'
    else:
        title = r'Supervised ($\mathbf{y} \rightarrow \mathbf{x}$) M2'

make_plots_libs(m,data,colors,names,waves=waves,sample_size=4,res_out=res_out)

##############################################################################################################
##############################################################################################################
# synthesize libs data
##############################################################################################################
##############################################################################################################

# load mineral data
minerals = pandas.read_csv('examples/datasets/fusion/data_4_Ian.csv',index_col=0)
mineral_names = np.array(minerals.index)
element_names = minerals.columns.values
mineral_compositions = minerals.as_matrix()
composition_sums = mineral_compositions.sum(axis=1)
mineral_compositions = mineral_compositions/composition_sums[:,None]

# load raman data
xy_r, ux_r, waves_r, names_r, colors_r = load_process_raman(DropLastDim=False)
sup_train_x_r, sup_train_y_r, sup_valid_x_r, sup_valid_y_r, train_x_r, train_y_r = [xyi.astype('float32') for xyi in xy_r]
sup_train_x_r *= 1e-4
sup_valid_x_r *= 1e-4
train_x_r *= 1e-4

# rectify data differences (5=Chabazite, 7=Clinochlore, 8=Diamond)
not_in_minerals = [5,7,8]
selected_minerals = list(range(len(names_r)))
for i in not_in_minerals:
  selected_minerals.remove(i)

other_sup_train_y_r = np.sum(sup_train_y_r[:,not_in_minerals],axis=1,keepdims=True)
other_sup_valid_y_r = np.sum(sup_valid_y_r[:,not_in_minerals],axis=1,keepdims=True)
other_train_y_r = np.sum(train_y_r[:,not_in_minerals],axis=1,keepdims=True)

sup_train_y_r = sup_train_y_r[:,selected_minerals]
sup_valid_y_r = sup_valid_y_r[:,selected_minerals]
train_y_r = train_y_r[:,selected_minerals]
# sup_train_y_r = np.hstack((sup_train_y_r[:,selected_minerals],other_sup_train_y_r))
# sup_valid_y_r = np.hstack((sup_valid_y_r[:,selected_minerals],other_sup_valid_y_r))
# train_y_r = np.hstack((train_y_r[:,selected_minerals],other_train_y_r))

min_comp_reduced = np.hstack((mineral_compositions[:,:8],np.sum(mineral_compositions[:,8:],axis=1,keepdims=True)))

# synthesize mineral mixtures
sup_train_y_l = np.dot(sup_train_y_r,min_comp_reduced).astype('float32')
sup_valid_y_l = np.dot(sup_valid_y_r,min_comp_reduced).astype('float32')
train_y_l = np.dot(train_y_r,min_comp_reduced).astype('float32')

# sup_train_y_l = sup_train_y_l/sup_train_y_l.sum(axis=1,keepdims=True)
# sup_valid_y_l = sup_valid_y_l/sup_valid_y_l.sum(axis=1,keepdims=True)
# train_y_l = train_y_l/train_y_l.sum(axis=1,keepdims=True)

print('check:',train_y_l[0].sum())

sup_train_y_l = sup_train_y_l[:,:-1]
sup_valid_y_l = sup_valid_y_l[:,:-1]
train_y_l = train_y_l[:,:-1]

# embed()

# # generate LIBS spectra from mineral mixtures
z2_sup_train = m.getZ2(x=sup_train_x,y=sup_train_y,deterministic=True)
z2_sup_valid = m.getZ2(x=sup_valid_x,y=sup_valid_y,deterministic=True)
z2_train = m.getZ2(x=train_x,deterministic=True)
z2_sup_train_mean = z2_sup_train.mean(axis=0)
z2_sup_valid_mean = z2_sup_valid.mean(axis=0)
z2_train_mean = z2_train.mean(axis=0)
z2_gen_sup_train = z2_sup_train_mean*np.ones((sup_train_y_l.shape[0],z2_sup_train.shape[1])).astype('float32')
z2_gen_sup_valid = z2_sup_valid_mean*np.ones((sup_valid_y_l.shape[0],z2_sup_valid.shape[1])).astype('float32')
z2_gen_train = z2_train_mean*np.ones((train_y_l.shape[0],z2_train.shape[1])).astype('float32')
sup_train_x_l = m.generate(y=sup_train_y_l,z2=z2_gen_sup_train,deterministic=True)
sup_valid_x_l = m.generate(y=sup_valid_y_l,z2=z2_gen_sup_valid,deterministic=True)
train_x_l = m.generate(y=train_y_l,z2=z2_gen_train,deterministic=True)

# construct fused dataset
sup_train_x_lr = np.hstack((sup_train_x_l,sup_train_x_r))
sup_valid_x_lr = np.hstack((sup_valid_x_l,sup_valid_x_r))
train_x_lr = np.hstack((train_x_l,train_x_r))
sup_train_y_lr = sup_train_y_r[:,:-1]
sup_valid_y_lr = sup_valid_y_r[:,:-1]
train_y_lr = train_y_r[:,:-1]
ux_lr = np.append(ux,ux_r)
# waves_lr = np.append(waves,waves_r)
waves_lr = np.arange(sup_train_x_lr.shape[1])
names_lr = mineral_names
colors_lr = colors_r[selected_minerals]

##############################################################################################################
##############################################################################################################
# train data fusion model
##############################################################################################################
##############################################################################################################

print('#'*50)
print('training data fusion model')
print('#'*50)

# define variable sizes
num_features = sup_train_x_lr.shape[1]
num_output = sup_train_y_lr.shape[1]
num_latent_z2 = 2

# construct data dictionary
data = {'X':sup_train_x_lr,'y':sup_train_y_lr,
        'X_':train_x_lr,
        'X_valid':sup_valid_x_lr,'y_valid':sup_valid_y_lr,
        'X__valid':train_x_lr}

# include "untapped" label source
data['_y'] = train_y_lr
data['_y_valid'] = train_y_lr
data['z2'] = np.random.uniform(low=-1.5, high=1.5, size=(train_y_lr.shape[0], num_latent_z2)).astype('float32')
data['z2_valid'] = np.random.uniform(low=-1.5, high=1.5, size=(train_y_lr.shape[0], num_latent_z2)).astype('float32')

# define priors
prior_x = None  # uniform distribution over positive intensities
prior_y = None  # uniform distribution over simplex
prior_z2 = None  # uniform distribution over interval

# define the model architecture (x=z1)
model_dict = OrderedDict()

arch = OrderedDict()
arch['hidden'] = [25,25]
arch['gain'] = np.sqrt(2)
arch['nonlin'] = lasagne.nonlinearities.tanh
arch['num_output'] = num_output
arch['sample_layer'] = SampleLayer
arch['flows'] = [NormalizingSimplexFlowLayer]
arch['batch_norm'] = False
model_dict['z1->y'] = OrderedDict([('arch',arch)])

arch = OrderedDict()
arch['hidden'] = [5,5]
arch['gain'] = np.sqrt(2)
arch['nonlin'] = lasagne.nonlinearities.tanh
arch['num_output'] = num_latent_z2
arch['sample_layer'] = SampleLayer
arch['flows'] = [LogisticFlowLayer]
arch['batch_norm'] = False
model_dict['z1y->z2'] = OrderedDict([('arch',arch)])

arch = OrderedDict()
arch['hidden'] = [50]
arch['gain'] = np.sqrt(2)
arch['nonlin'] = lasagne.nonlinearities.tanh
arch['num_output'] = num_features
arch['sample_layer'] = SampleLayer
arch['flows'] = [SoftplusFlowLayer]
arch['batch_norm'] = False
model_dict['yz2->_z1'] = OrderedDict([('arch',arch)])

# set result directory path
res_out='examples/results/fusion/fusion/'+timeStamp().format("")

# construct the semi^2-supervised deep generative model
m1 = SSDGM(num_features,num_output,variational=False,model_dict=model_dict,
          prior_x=prior_x,prior_y=prior_y,prior_z2=prior_z2,loss_x=L2,loss_y=KL,
          coeff_x=1e-1,coeff_y=1e-2,coeff_x_dis=1e-2,coeff_y_dis=1e-2,coeff_x_prob=1,coeff_y_prob=1,
          num_epochs=1000,eval_freq=100,lr=1e-2,eq_samples=1,iw_samples=1,
          res_out=res_out)
# m1 = SSDGM(num_features,num_output,variational=True,model_dict=model_dict,
#           prior_x=prior_x,prior_y=prior_y,prior_z2=prior_z2,loss_x=L2,loss_y=L1,
#           coeff_x=1e-2,coeff_y=1e-3,coeff_x_dis=10,coeff_y_dis=1e-4,coeff_x_prob=0,coeff_y_prob=0,
#           num_epochs=500,eval_freq=100,lr=1e-2,eq_samples=1,iw_samples=1,
#           res_out=res_out)

# fit the model
m1.fit(verbose=True,debug=False,**data)

data1 = data

# data['X_'] = train_x_lr
# data['X__valid'] = train_x_lr
# include "untapped" label source
# data['_y'] = train_y_lr
# data['_y_valid'] = train_y_lr
# data['z2'] = np.random.uniform(low=-1.5, high=1.5, size=(train_y_lr.shape[0], num_latent_z2)).astype('float32')
# data['z2_valid'] = np.random.uniform(low=-1.5, high=1.5, size=(train_y_lr.shape[0], num_latent_z2)).astype('float32')

# # auto-set title and plot results (saved to res_out)
# title = 'M2'
# if m.coeff_y_dis > 0:
#     if m.coeff_y > 0:
#         title = 'Labels Untapped'
#     else:
#         title = r'Supervised ($\mathbf{y} \rightarrow \mathbf{x}$) M2'

# make_plots_raman(m,data,colors_lr,names_lr,groundtruth=None,
#                  waves=waves_lr,sample_size=4,res_out=res_out,title=title)

##############################################################################################################
##############################################################################################################
# train Raman model
##############################################################################################################
##############################################################################################################

print('#'*50)
print('training raman model')
print('#'*50)

# define variable sizes
num_features = sup_train_x_r.shape[1]
num_output = sup_train_y_lr.shape[1]
num_latent_z2 = 2

# construct data dictionary
data = {'X':sup_train_x_r,'y':sup_train_y_lr,
        'X_':train_x_r,
        'X_valid':sup_valid_x_r,'y_valid':sup_valid_y_lr,
        'X__valid':train_x_r}

# include "untapped" label source
data['_y'] = train_y_lr
data['_y_valid'] = train_y_lr
data['z2'] = np.random.uniform(low=-1.5, high=1.5, size=(train_y_lr.shape[0], num_latent_z2)).astype('float32')
data['z2_valid'] = np.random.uniform(low=-1.5, high=1.5, size=(train_y_lr.shape[0], num_latent_z2)).astype('float32')

# define priors
prior_x = None  # uniform distribution over positive intensities
prior_y = None  # uniform distribution over simplex
prior_z2 = None  # uniform distribution over interval

# define the model architecture (x=z1)
model_dict = OrderedDict()

arch = OrderedDict()
arch['hidden'] = [25,25]
arch['gain'] = np.sqrt(2)
arch['nonlin'] = lasagne.nonlinearities.tanh
arch['num_output'] = num_output
arch['sample_layer'] = SampleLayer
arch['flows'] = [NormalizingSimplexFlowLayer]
arch['batch_norm'] = False
model_dict['z1->y'] = OrderedDict([('arch',arch)])

arch = OrderedDict()
arch['hidden'] = [5,5]
arch['gain'] = np.sqrt(2)
arch['nonlin'] = lasagne.nonlinearities.tanh
arch['num_output'] = num_latent_z2
arch['sample_layer'] = SampleLayer
arch['flows'] = [LogisticFlowLayer]
arch['batch_norm'] = False
model_dict['z1y->z2'] = OrderedDict([('arch',arch)])

arch = OrderedDict()
arch['hidden'] = [50]
arch['gain'] = np.sqrt(2)
arch['nonlin'] = lasagne.nonlinearities.tanh
arch['num_output'] = num_features
arch['sample_layer'] = SampleLayer
arch['flows'] = [SoftplusFlowLayer]
arch['batch_norm'] = False
model_dict['yz2->_z1'] = OrderedDict([('arch',arch)])

# set result directory path
res_out='examples/results/fusion/raman/'+timeStamp().format("")

# construct the semi^2-supervised deep generative model
m2 = SSDGM(num_features,num_output,variational=False,model_dict=model_dict,
          prior_x=prior_x,prior_y=prior_y,prior_z2=prior_z2,loss_x=L2,loss_y=KL,
          coeff_x=1e-1,coeff_y=1e-2,coeff_x_dis=1e-2,coeff_y_dis=1e-2,coeff_x_prob=0,coeff_y_prob=0,
          num_epochs=500,eval_freq=100,lr=1e-2,eq_samples=1,iw_samples=1,
          res_out=res_out)
# m2 = SSDGM(num_features,num_output,variational=True,model_dict=model_dict,
#           prior_x=prior_x,prior_y=prior_y,prior_z2=prior_z2,loss_x=L2,loss_y=L1,
#           coeff_x=1e-2,coeff_y=1e-3,coeff_x_dis=10,coeff_y_dis=1e-4,coeff_x_prob=0,coeff_y_prob=0,
#           num_epochs=500,eval_freq=100,lr=1e-2,eq_samples=1,iw_samples=1,
#           res_out=res_out)

# fit the model
m2.fit(verbose=True,debug=False,**data)

data2 = data

# data['X_'] = train_x_r
# data['X__valid'] = train_x_r
# include "untapped" label source
# data['_y'] = train_y_lr
# data['_y_valid'] = train_y_lr
# data['z2'] = np.random.uniform(low=-1.5, high=1.5, size=(train_y_lr.shape[0], num_latent_z2)).astype('float32')
# data['z2_valid'] = np.random.uniform(low=-1.5, high=1.5, size=(train_y_lr.shape[0], num_latent_z2)).astype('float32')

# # auto-set title and plot results (saved to res_out)
# title = 'M2'
# if m.coeff_y_dis > 0:
#     if m.coeff_y > 0:
#         title = 'Labels Untapped'
#     else:
#         title = r'Supervised ($\mathbf{y} \rightarrow \mathbf{x}$) M2'

# make_plots_raman(m,data,colors_lr,names_lr,groundtruth=None,
#                  waves=waves_r,sample_size=4,res_out=res_out,title=title)

make_plots_raman(m1,m2,data1,data2,names_lr,sample_size=4,res_out=res_out,title=None)



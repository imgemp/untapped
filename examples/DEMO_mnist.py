import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from collections import OrderedDict

import numpy as np
import lasagne
from untapped.parmesan.layers import SampleLayer, BernoulliSampleLayer, ConcreteSampleLayer
from untapped.parmesan.layers import NormalizingSimplexFlowLayer, LogisticFlowLayer

from untapped.M12 import binary_cross_entropy as bince
from untapped.M12 import categorical_cross_entropy as catce
from untapped.S2S_DGM import SSDGM
from untapped.utilities import timeStamp

from datasets.load_mnist import load_data
from plotting.plotting_mnist import make_plots

#####################################################################
# REQUIRES optimizer='fast_compile' in .theanorc file under [global]
#####################################################################

# load data
xy, ux, names, colors = load_data(binary=False)
sup_train_x, sup_train_y, sup_valid_x, sup_valid_y, train_x, train_y = [xyi.astype('float32') for xyi in xy]

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
low = -5.
high = 5.
data['_y'] = train_y
data['_y_valid'] = train_y
data['z2'] = np.random.uniform(low=low, high=high, size=(train_y.shape[0], num_latent_z2)).astype('float32')
data['z2_valid'] = np.random.uniform(low=low, high=high, size=(train_y.shape[0], num_latent_z2)).astype('float32')

# define priors
prior_x = None  # uniform distribution over positive intensities
prior_y = None  # uniform distribution over simplex
prior_z2 = None  # uniform distribution over interval

# define the model architecture (x=z1)
model_dict = OrderedDict()

arch = OrderedDict()
arch['hidden'] = [500]
arch['gain'] = np.sqrt(2)
arch['nonlin'] = lasagne.nonlinearities.softplus
arch['num_output'] = num_output
arch['sample_layer'] = ConcreteSampleLayer
arch['flows'] = []
arch['batch_norm'] = False
model_dict['z1->y'] = OrderedDict([('arch',arch)])

arch = OrderedDict()
arch['hidden'] = [500]
arch['gain'] = np.sqrt(2)
arch['nonlin'] = lasagne.nonlinearities.softplus
arch['num_output'] = num_latent_z2
arch['sample_layer'] = SampleLayer
arch['flows'] = [LogisticFlowLayer]
arch['batch_norm'] = False
model_dict['z1y->z2'] = OrderedDict([('arch',arch)])

arch = OrderedDict()
arch['hidden'] = [500]
arch['gain'] = np.sqrt(2)
arch['nonlin'] = lasagne.nonlinearities.softplus
arch['num_output'] = num_features
arch['sample_layer'] = BernoulliSampleLayer
arch['flows'] = []
arch['batch_norm'] = False
model_dict['yz2->_z1'] = OrderedDict([('arch',arch)])

# set result directory path
res_out='examples/results/mnist/'+timeStamp().format("")

# construct the semi^2-supervised deep generative model
m = SSDGM(num_features,num_output,variational=True,model_dict=model_dict,eq_samples=1,iw_samples=1,
          prior_x=prior_x,prior_y=prior_y,prior_z2=prior_z2,loss_x=bince,loss_y=catce,
          coeff_x=1e-1,coeff_y=0,coeff_x_dis=1,coeff_y_dis=1e-1,coeff_x_prob=1e-1,coeff_y_prob=0,
          num_epochs=1000,eval_freq=100,lr=1e-3,
          batch_size_Xy_train=10000,batch_size_X__train=10000,batch_size__y_train=10000,
          batch_size_Xy_eval=10000,batch_size_X__eval=10000,batch_size__y_eval=10000,
          res_out=res_out)

# fit the model
m.fit(verbose=True,debug=True,**data)

# auto-set title and plot results (saved to res_out)
title = 'M2'
if m.coeff_y_dis > 0:
    if m.coeff_y > 0:
        title = 'Labels Untapped'
    else:
        title = r'Supervised ($\mathbf{y} \rightarrow \mathbf{x}$) M2'

make_plots(m,data,colors,names,sample_size=4,res_out=res_out,title=title)

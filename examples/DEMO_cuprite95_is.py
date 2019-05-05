import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from collections import OrderedDict
from functools import partial
import time

import numpy as np
import lasagne
from untapped.parmesan.layers import SampleLayer
from untapped.parmesan.layers import NormalizingSimplexFlowLayer, LogisticFlowLayer, ConcreteSampleLayer

from untapped.M12 import L2, KL, dirichlet
from untapped.S2S_DGM import SSDGM
from untapped.utilities import timeStamp

from datasets.load_cuprite import load_process_cuprite95
from plotting.plotting_cuprite2 import make_plots

import gzip
from six.moves import cPickle
from functools import reduce

from IPython import embed
#####################################################################
# REQUIRES optimizer='fast_compile' in .theanorc file under [global]
#####################################################################

start = time.time()

# load data
print('Loading data...')
remove_mean = True
xy, ux, waves, names, colors, img_shape, endmembers, simplex = load_process_cuprite95(remove_mean=remove_mean,train=0.8,
                                                                                      dists=True,remove_bad=True,region=3)
sup_train_x, sup_train_y, sup_valid_x, sup_valid_y, train_x, train_y = [xyi.astype('float32') for xyi in xy]

# define variable sizes
num_features = train_x.shape[1]
num_output = train_y.shape[1]
num_latent_z2 = 1

# construct data dictionary
data = {'X':sup_train_x,'y':sup_train_y,
        'X_':train_x,
        'X_valid':sup_valid_x,'y_valid':sup_valid_y,
        'X__valid':train_x}

# include "untapped" label source
data['_y'] = train_y
data['_y_valid'] = train_y
low = -5.
high = 5.
eps = 1.
sample_z2s = lambda: np.random.uniform(low=low, high=high, size=(train_y.shape[0], num_latent_z2)).astype('float32')
data['z2'] = sample_z2s()
data['z2_valid'] = sample_z2s()

# define priors
prior_x = None  # uniform distribution over positive intensities
# prior_y = None  # uniform distribution over simplex
alpha = 1e-1*np.ones(num_output+1)
prior_y = lambda y: dirichlet(y,alpha=alpha,eps=1e-3,axis=-1)  # mixtures of a few minerals are more likely
prior_z2 = None  # uniform distribution over interval

# define the model architecture (x=z1)
model_dict = OrderedDict()

arch = OrderedDict()
arch['hidden'] = [7]
arch['gain'] = np.sqrt(2)
arch['nonlin'] = lasagne.nonlinearities.softplus
arch['num_output'] = num_output
arch['sample_layer'] = SampleLayer
arch['flows'] = [NormalizingSimplexFlowLayer]
# arch['sample_layer'] = ConcreteSampleLayer
# arch['flows'] = []
arch['batch_norm'] = False
model_dict['z1->y'] = OrderedDict([('arch',arch)])

arch = OrderedDict()
arch['hidden'] = []
arch['gain'] = np.sqrt(2)
arch['nonlin'] = lasagne.nonlinearities.softplus
arch['num_output'] = num_latent_z2
arch['sample_layer'] = SampleLayer
arch['flows'] = [partial(LogisticFlowLayer,lo=low-eps,hi=high+eps)]
arch['batch_norm'] = False
model_dict['z1y->z2'] = OrderedDict([('arch',arch)])

arch = OrderedDict()
arch['hidden'] = [7]
arch['gain'] = np.sqrt(2)
arch['nonlin'] = lasagne.nonlinearities.softplus
arch['num_output'] = num_features
arch['sample_layer'] = SampleLayer
arch['flows'] = []
arch['batch_norm'] = False
model_dict['yz2->_z1'] = OrderedDict([('arch',arch)])

# set result directory path
res_out='examples/results/cuprite95/'+timeStamp().format("")

# makeplots = partial(make_plots,data=data,colors=colors,names=names,groundtruth=endmembers,ylim=None,waves=waves,
#                                sample_size=10,ux=ux,remove_mean=remove_mean,res_out=res_out,simplex=simplex)

makeplots = lambda m: make_plots(m,data=data,colors=colors,names=names,groundtruth=endmembers,ylim=None,waves=waves,
                               sample_size=10,ux=ux,remove_mean=remove_mean,res_out=res_out,simplex=simplex)

# construct the semi^2-supervised deep generative model
print('Building model...')
m = SSDGM(num_features,num_output,variational=False,model_dict=model_dict,
          prior_x=prior_x,prior_y=prior_y,prior_z2=prior_z2,loss_x=L2,loss_y=KL,
          coeff_x=1e-1,coeff_y=0.,coeff_x_dis=0.,coeff_y_dis=0.,coeff_x_prob=0.,coeff_y_prob=0.,
          num_epochs=10000,eval_freq=100,lr=1e-3,eq_samples=1,iw_samples=1,
          batch_size_Xy_train=100,batch_size_X__train=100,batch_size__y_train=100,
          batch_size_Xy_eval=100,batch_size_X__eval=100,batch_size__y_eval=100,
          res_out=res_out,make_plots=makeplots,sample_z2s=sample_z2s)

o_net_weights = [w.get_value() for w in m.model_dict['params']['all']]

f = gzip.open('examples/results/cuprite95/2018-02-17-21-47-26/modelepoch1000')
net_weights = cPickle.load(f)
for var, val in zip(m.model_dict['params']['all'],net_weights):
  var.set_value(val.get_value())

print('Obtaining importance weights')

try:
    isw = np.load('examples/results/cuprite95/2018-03-02-19-43-43/isw.npy')
except Exception as e:
    N_samples = 1000
    ycorners = np.random.dirichlet(alpha=0.1*np.ones(train_y.shape[1]+1),size=N_samples)[:,:-1].astype('float32')
    zcorners = np.random.uniform(low=low, high=high, size=(N_samples, num_latent_z2)).astype('float32')
    ydata = np.random.dirichlet(alpha=2.*np.ones(train_y.shape[1]+1),size=N_samples)[:,:-1].astype('float32')
    zdata = np.random.uniform(low=low, high=high, size=(N_samples, num_latent_z2)).astype('float32')
    mu_corners, std_corners = m.get_Xdp(y=ycorners,z2=zcorners)
    mu_data, std_data = m.get_Xdp(y=ydata,z2=zdata)
    # px = lambda x,mu,std: np.exp(- 0.5 * np.log(2*np.pi) - np.log(np.abs(std)) - (x - mu)**2 / (2 * std**2)).mean()
    px = lambda x,mu,std: np.exp(- 0.5 * np.log(2*np.pi) - np.log(np.abs(std)) - (x - mu)**2 / (2 * std**2)).sum(axis=-1)
    pxtrain = lambda mustd: px(train_x,mustd[0],mustd[1])
    pxcorners = reduce((lambda x,y: x+y/N_samples), map(pxtrain,zip(mu_corners,std_corners)))
    pxdata = reduce((lambda x,y: x+y/N_samples), map(pxtrain,zip(mu_data,std_data)))

    # isw = np.array([px(x,mu_corners,std_corners)/px(x,mu_data,std_data) for x in train_x])
    # pxcorners = np.array([px(train_x,mu_corner,std_corner) for mu_corner,std_corner in zip(mu_corners,std_corners)]).mean(axis=0)
    # pxdata = np.array([px(train_x,mu_datum,std_datum) for mu_datum,std_datum in zip(mu_data,std_data)]).mean(axis=0)
    isw = pxcorners/pxdata
    np.save(res_out+'/isw.npy',isw)

m.isw = isw

for var, val in zip(m.model_dict['params']['all'],o_net_weights):
  var.set_value(val)

# fit the model
print('Training...')
# m.fit(verbose=True,debug=False,**data)
m.fit(variational=True,verbose=True,debug=False,**data)

# # auto-set title and plot results (saved to res_out)
# title = 'M2'
# if m.coeff_y_dis > 0:
#     if m.coeff_y > 0:
#         title = 'Labels Untapped'
#     else:
#         title = r'Supervised ($\mathbf{y} \rightarrow \mathbf{x}$) M2'

# print('Plotting...')
# make_plots(m,data,colors,names,groundtruth=endmembers,ylim=None,waves=waves,
#            sample_size=10,ux=ux,remove_mean=remove_mean,res_out=res_out,title=title,simplex=train_y)
# makeplots(m)

end = time.time()
duration = end - start
print('duration: '+str(duration)+' sec')

# embed()

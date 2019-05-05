import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from collections import OrderedDict

import numpy as np
import lasagne
from untapped.parmesan.layers import SampleLayer, BernoulliSampleLayer, ConcreteSampleLayer
from untapped.parmesan.layers import NormalizingSimplexFlowLayer, LogisticFlowLayer

from untapped.M12 import L2, KL, binary_cross_entropy, log_beta
from untapped.S2S_DGM import SSDGM
from untapped.utilities import timeStamp

from sklearn.linear_model import LogisticRegression as LR
from sklearn.grid_search import GridSearchCV

from datasets.load_mnist import load_data
from plotting.plotting_mnist import make_plots

from collections import Counter
import copy
from functools import partial
import time

from IPython import embed

#####################################################################
# REQUIRES optimizer='fast_compile' in .theanorc file under [global]
#####################################################################


def assign_permutation(p):
  '''
  Given a square n x n matrix of probabilities or scores where each row is assumed
  to represent a single category out of the n-categories, this method returns a labeling
  of the n rows.
  If a category is "requested" by a single row, the row is labeled with that category
  If multiple rows "request" the same category, the row with the highest score gets the category
  '''
  # p = m_lr.predict_proba(means)
  assert p.shape[0] == p.shape[1]
  if p.shape[0] == 1:
    return np.array([0])
  labels = np.argmax(p,axis=1)
  not_dups = np.array([item for item, count in Counter(labels).items() if count == 1])
  dups = np.array([item for item, count in Counter(labels).items() if count > 1])
  if len(dups) > 0:
    rows = [idx for idx in range(p.shape[0]) if labels[idx] in dups]
    cols = [idx for idx in range(p.shape[1]) if idx not in not_dups]
    if len(rows) == p.shape[0]:
      winner = np.argmax(p[:,dups[0]])
      rows = [idx for idx in rows if idx != winner]
      cols = [idx for idx in cols if idx != dups[0]]
    p_new = p[rows,:]
    p_new = p_new[:,cols]
    sub_labels = assign_permutation(p_new)
    labels[rows] = np.array(cols)[sub_labels]
  return labels


# load data
xy, ux, names, colors = load_data(binary=False)
sup_train_x, sup_train_y, sup_valid_x, sup_valid_y, train_x, train_y = [xyi.astype('float32') for xyi in xy]

print('Training logistic regression...')
# set up data for logistic regression
x_lr = np.vstack((sup_train_x,sup_valid_x))
y_lr = np.argmax(np.vstack((sup_train_y,sup_valid_y)),axis=1)

# define hyperparameter grid (+ any arguments that should be passed to fit)
# model_parameters = {'C':np.logspace(-2,2,num=5)}

# perform gridsearch on logistic regression with cross-validation
# model = GridSearchCV(LR(), model_parameters, cv=5, verbose=2)
# model.fit(X=x_lr,y=y_lr)

# # retrieve best model
# m_lr = model.best_estimator_
m_lr = LR().fit(x_lr,y_lr)
print('Done.')

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
low = -1.5
high = 1.5
data['_y'] = train_y
data['_y_valid'] = train_y
data['z2'] = np.random.uniform(low=low, high=high, size=(train_y.shape[0], num_latent_z2)).astype('float32')
data['z2_valid'] = np.random.uniform(low=low, high=high, size=(train_y.shape[0], num_latent_z2)).astype('float32')

data_ref = dict(data.items())

# define priors
maxs = train_x.max(axis=0)
sums = train_x.sum(axis=1)
alpha_maxs = np.ones(maxs.shape)/(1+1e-3)
beta_maxs = np.ones(maxs.shape)/(maxs+1e-3)
sums_mean = np.clip(np.mean(sums),1e-3,1-1e-3)
sums_var = np.var(sums) + 1e-3
alpha_sums = ((1-sums_mean)/sums_var - 1/sums_mean)*sums_mean**2.
beta_sums = alpha_sums*(1/sums_mean-1)
alpha = np.concatenate((alpha_maxs,[alpha_sums]))
beta = np.concatenate((beta_maxs,[beta_sums]))
prior_x = lambda x: log_beta(x,alpha,beta)
# prior_x = None  # uniform distribution over positive intensities
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
arch['flows'] = [LogisticFlowLayer]  # [partial(LogisticFlowLayer,lo=-5.5,hi=5.5)]
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
root_res_out='examples/results/mnist/'+timeStamp().format("")

num_trials = 1

for trial in range(num_trials):
  start = time.time()

  res_out = root_res_out + str(trial) + '/'
  # construct the semi^2-supervised deep generative model
  m = SSDGM(num_features,num_output,variational=True,model_dict=model_dict,eq_samples=1,iw_samples=1,
            prior_x=prior_x,prior_y=prior_y,prior_z2=prior_z2,loss_x=L2,loss_y=KL,
            coeff_x=1e-1,coeff_y=1e-1,coeff_x_dis=1,coeff_y_dis=0,coeff_x_prob=1e-1,coeff_y_prob=0,
            num_epochs=1000,eval_freq=100,lr=1e-3,
            batch_size_Xy_train=10000,batch_size_X__train=10000,batch_size__y_train=10000,
            batch_size_Xy_eval=10000,batch_size_X__eval=10000,batch_size__y_eval=10000,
            seed=None,res_out=res_out)
  # embed()
  # fit the model
  m.fit(verbose=True,debug=True,**data)

  finish = time.time()
  print('duration: '+str(finish-start))

  # auto-set title and plot results (saved to res_out)
  title = 'M2'
  if m.coeff_y > 0:
    title = 'Labels Untapped'
  elif m.coeff_y_dis > 0:
    title = r'Supervised ($\mathbf{y} \rightarrow \mathbf{x}$) M2'

  # data_ref['X'] = m._sample_bernoulli_image(data['X'])
  # data_ref['X_'] = m._sample_bernoulli_image(data['X_'])
  # data_ref['X_valid'] = m._sample_bernoulli_image(data['X_valid'])
  # data_ref['X__valid'] = m._sample_bernoulli_image(data['X__valid'])
  
  endmembers = make_plots(m,data_ref,colors,names,sample_size=4,res_out=res_out+'preperm',title=title)

  p = m_lr.predict_proba(endmembers)
  num_digits = len(np.unique(m_lr.predict(endmembers)))
  p_sub = p[5:,:][:,5:]
  perm_sub = assign_permutation(p_sub)
  perm = np.array([0,1,2,3,4] + list(5+perm_sub))
  print(perm)

  data_perm = copy.copy(data)
  data_perm['y'] = data['y'][:,perm]
  data_perm['y_valid'] = data['y_valid'][:,perm]

  make_plots(m,data_perm,colors,names,sample_size=4,res_out=res_out+'perm',title=title,num_digits=num_digits)

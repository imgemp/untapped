"""
Semi^2-Supervised Deep Generative Model

Kingma, Diederik P., et al. "Semi-supervised learning with deep generative models." 2014.
Kingma, Diederik P., and Max Welling. "Auto-encoding variational bayes." 2013.
Rezende, Danilo Jimenez, and Shakir Mohamed. "Variational inference with normalizing flows." 2015.
Burda, Yuri, Roger Grosse, and Ruslan Salakhutdinov. "Importance weighted autoencoders." 2015.
Extension of https://github.com/casperkaae/parmesan

SSDGM is a class to build and train a generative model using deep nets.
The model construction consists of recognition networks, p(y,z|x;phi), that
can be used to infer latent variables (Z) and possibly response
variables (Y) as well as generative networks, p(x|y,z;theta), that can be used
to approximate the true data distribution.  The model is flexible enough
to be trained with a variational upper bound or as a deep autoencoder in the
limiting case of zero variance for the posteriors and uniform prior.  The
model can be trained with x-y pairs, solo x's, or solo y's.  In some domains,
y's can be obtained on their own and so SSDGM accomodates those scenarios.

     x
     |
     v
_x<--z1-->y
       \  | \
        \,v  \,
         z2->_z1  
"""

import os
import shutil
import gzip
from six.moves import cPickle
import copy
import time
import itertools
import warnings
warnings.filterwarnings("ignore")

from collections import OrderedDict

from untapped.utilities import timeStamp, toLog, toScreen

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator

import theano
import theano.tensor as T
from theano.ifelse import ifelse
theano.config.floatX = 'float32'
theano.config.exception_verbosity = 'high'
theano.config.traceback.limit = 20

import lasagne

from untapped.parmesan.layers import (NICE,
                             NormalizeLayer,
                             ScaleAndShiftLayer,
                             NormalizingSimplexFlowLayer,
                             SampleLayer)
from untapped.parmesan.distributions import log_normal2

from untapped.M12 import M1, M2, L2

from IPython import embed


class SSDGM(BaseEstimator):
    """Semi-Supervised Deep Generative Model"""
    def __init__(self,num_features,num_output,model_dict=None,variational=False,
        model_type=0,iw_samples=1,eq_samples=1,
        lr=0.001,anneal_lr_factor=0.9995,anneal_lr_epoch=500,
        prior_x=None,prior_y=None,prior_z1=None,prior_z2=None,
        nonlin_enc=None,nonlin_dec=None,
        gain_enc=np.sqrt(2),gain_dec=np.sqrt(2),
        hidden=[],num_latent_z1=100,num_latent_z2=2,nflows=0,simplexflow=True,batch_norm=False,
        default_data={},loss_x=L2,loss_y=L2,
        coeff_x=1,coeff_y=0,coeff_x_dis=0,coeff_y_dis=0,coeff_x_prob=0,coeff_y_prob=0,
        batch_size_Xy_train=100,batch_size_X__train=100,batch_size__y_train=100,
        batch_size_Xy_eval=100,batch_size_X__eval=100,batch_size__y_eval=100,
        num_epochs=10,eval_freq=5,
        res_out='results/'+timeStamp().format(""),seed=1234,verbose=True,debug=True):
        """Initialize the model.
        Parameters
        ----------
        num_features : int, required
            number of features in the data, X.shape[1]
        num_output : int, required
            number of features in the response variables, y.shape[1]
        model_dict : OrderedDict, optional
            OrderedDict containing net architectures for each model component
            *this overrides all other architecture related settings passed to SSDGM
        variational : boolean, optional
            False (default) = reconstruction error, True = variational objective
        model_type : int in [0,2], optional
            specifies M1, M2, or M12 model from Kingma's 2014 paper
        iw_samples : int, optional
            number of importance weighted samples
        eq_samples : int, optional
            number of samples to estimate E(z|x)
        lr : float, optional
            learning rate
        anneal_lr_factor : float, optional
            factor to reduce learning rate by at each anneal epoch
            (see below)
        anneal_lr_epoch : int, optional
            anneal learning rate at these intervals
        prior_x : symbolic theano probability density function
            prior distribution for x
        prior_y : symbolic theano probability density function
            prior distribution for y
        prior_z1 : symbolic theano probability density function
            prior distribution for z1
        prior_z2 : symbolic theano probability density function
            prior distribution for z2
        nonlin_enc : theano.tensor nonlinearity, optional
            nonlinearity to apply at each encoding (recognition) layer
        nonlin_dec : theano.tensor nonlinearity, optional
            nonlinearity to apply at each decoding (generative) layer
        gain_enc : float, optional
            encoding weights initialized with glorot_uniform(gain)
        gain_dec : float, optional
            decoding weights initialized with glorot_uniform(gain)
        hidden : list, optional
            list of number of units in each layer
        num_latent_z1 : int, optional
            number of units in first latent layer
        num_latent_z2 : int, optional
            number of units in second latent layer
        nflows : int, optional
            number of planar normalizing flows following each
            normal distribution
        simplexflow : bool, optional
            whether to include a normalizing flow to the simplex
            after the planar flows
        batch_norm : bool, optional
            whether to include batch normalization
        default_data : dictionary of numpy arrays, optional
            default training data arrays (to be split into train/validation by GridSearchCV)
        loss_x : loss function for x variable, optional
            supervised loss objective for x
        loss_y : loss function for y variable, optional
            supervised loss objective for y
        coeff_x : float, optional
            coefficient to scale the objective for x data
        coeff_y : float, optional
            coefficient to scale the objective for y data
        coeff_x_dis : float, optional
            coefficient to scale the discriminative objective for labeled x data
        coeff_y_dis : float, optional
            coefficient to scale the discriminative objective for labeled y data
        coeff_x_prob : float, optional
            coefficient to scale the probabilistic discriminative objective for labeled x data
        coeff_y_prob : float, optional
            coefficient to scale the probabilistic discriminative objective for labeled y data
        batch_size_Xy_train : int, optional
            batch size for training / passing Xy data through the model
        batch_size_X__train : int, optional
            batch size for training / passing X_ data through the model
        batch_size__y_train : int, optional
            batch size for training / passing _y data through the model
        batch_size_Xy_eval : int, optional
            batch size for evaluation / passing Xy data through the model
        batch_size_X__eval : int, optional
            batch size for evaluation / passing X_ data through the model
        batch_size__y_eval : int, optional
            batch size for evaluation / passing _y data through the model
        num_epochs : int, optional
            number of epochs to train model for
        eval_freq : int, optional
            evaluate model every `eval_freq` epochs
        res_out : str, optional
            path to save results to
        seed : int or None, optional
            seed for random number generator, seed not set if None
        verbose : bool, optional
            whether to print out objective values during training/fit
        debug : bool, optional
            whether to enter interactive IPython mode when NaN encountered in objective value
        """

        self._process_inputs(num_features,num_output,model_dict,variational,model_type,iw_samples,eq_samples,
            lr,anneal_lr_factor,anneal_lr_epoch,prior_x,prior_y,prior_z1,prior_z2,nonlin_enc,nonlin_dec,gain_enc,gain_dec,
            hidden,num_latent_z1,num_latent_z2,nflows,simplexflow,batch_norm,default_data,
            loss_x,loss_y,coeff_x,coeff_y,coeff_x_dis,coeff_y_dis,coeff_x_prob,coeff_y_prob,
            batch_size_Xy_train,batch_size_X__train,batch_size__y_train,
            batch_size_Xy_eval,batch_size_X__eval,batch_size__y_eval,
            num_epochs,eval_freq,res_out,seed,verbose,debug)
        self._setup_log_output()
        self._build_net()
        self._add_functionality()

    def _process_inputs(self,num_features,num_output,model_dict,variational,model_type,iw_samples,eq_samples,
            lr,anneal_lr_factor,anneal_lr_epoch,prior_x,prior_y,prior_z1,prior_z2,nonlin_enc,nonlin_dec,gain_enc,gain_dec,
            hidden,num_latent_z1,num_latent_z2,nflows,simplexflow,batch_norm,default_data,
            loss_x,loss_y,coeff_x,coeff_y,coeff_x_dis,coeff_y_dis,coeff_x_prob,coeff_y_prob,
            batch_size_Xy_train,batch_size_X__train,batch_size__y_train,
            batch_size_Xy_eval,batch_size_X__eval,batch_size__y_eval,
            num_epochs,eval_freq,res_out,seed,verbose,debug):

        self.num_features = num_features
        self.num_output = num_output
        if model_dict is None:
            model_dict = OrderedDict()
        self.model_dict = model_dict
        self.variational = variational
        self.model_type = model_type
        self.iw_samples = iw_samples
        self.eq_samples = eq_samples
        self.lr = lr
        self.anneal_lr_factor = anneal_lr_factor
        self.anneal_lr_epoch = anneal_lr_epoch
        self.prior_x = prior_x
        self.prior_y = prior_y
        self.prior_z1 = prior_z1
        self.prior_z2 = prior_z2
        self.nonlin_enc = nonlin_enc
        self.nonlin_dec = nonlin_dec
        self.gain_enc = gain_enc
        self.gain_dec = gain_dec
        self.hidden = hidden
        self.num_latent_z1 = num_latent_z1
        self.num_latent_z2 = num_latent_z2
        self.nflows = nflows
        self.simplexflow = simplexflow
        self.batch_norm = batch_norm
        self.default_data = default_data
        self.loss_x = loss_x
        self.loss_y = loss_y
        self.coeff_x = coeff_x
        self.coeff_y = coeff_y
        self.coeff_x_dis = coeff_x_dis
        self.coeff_y_dis = coeff_y_dis
        self.coeff_x_prob = coeff_x_prob
        self.coeff_y_prob = coeff_y_prob
        self.batch_size_Xy_train = batch_size_Xy_train
        self.batch_size_X__train = batch_size_X__train
        self.batch_size__y_train = batch_size__y_train
        self.batch_size_Xy_eval = batch_size_Xy_eval
        self.batch_size_X__eval = batch_size_X__eval
        self.batch_size__y_eval = batch_size__y_eval
        self.num_epochs = num_epochs
        self.eval_freq = eval_freq
        self.res_out = res_out
        self.seed = seed
        self.verbose = verbose
        self.debug = debug

        code = self._process_inputs.__func__.__code__
        args = code.co_varnames[1:code.co_argcount]
        self.args_dict = OrderedDict([(input,self.__getattribute__(input))
                                      for input in args])

    def _setup_log_output(self):
        if not os.path.exists(self.res_out):
            os.makedirs(self.res_out)

        # save SSDGM file to res_out
        filename_script = os.path.basename(os.path.realpath(__file__))
        shutil.copy(os.path.realpath(__file__), os.path.join(self.res_out, filename_script))
        self.filename_script = filename_script

        logfile = os.path.join(self.res_out, 'logfile.log')
        model_out = os.path.join(self.res_out, 'model')
        self.logfile = logfile
        self.model_out = model_out

    def _log_model_setup(self):
        # write model parameters to header of logfile
        description = []
        description.append('######################################################')
        description.append('# --Model Params--')
        for name in self.args_dict.keys():
            val = self.__getattribute__(name)
            self.args_dict[name] = val
            if name == 'model_dict':
                newval = OrderedDict()
                for k,v in val.items():
                    if 'arch' in v:
                        newval[k] = v['arch']
                val = newval
            description.append("# " + name + ":\t" + str(val))
        description.append('######################################################')
        self.description = description
        with open(self.logfile,'w') as f:
            for l in description:
                f.write(l + '\n')

    def _build_net(self):
        if self.seed is not None:
            np.random.seed(self.seed)
        self._build_blueprint()
        self._construct_from_blueprint(self.model_dict)

    def _build_blueprint(self):
        model_dict = self.model_dict
        self._init_symbols(model_dict)
        self._init_net_archs(model_dict)
        model_dict['objs'] = OrderedDict()
        model_dict['params'] = OrderedDict()
        self.model_dict = model_dict

    def _init_symbols(self,model_dict):
        sym_iw_samples = T.iscalar('iw_samples')
        sym_eq_samples = T.iscalar('eq_samples')
        sym_lr = T.scalar('lr')
        sym_x = T.matrix('x')
        sym_y = T.matrix('y')
        sym_z1 = T.matrix('z1')
        sym_z2 = T.matrix('z2')
        sym_x_sup = T.matrix('x_sup')
        sym_y_sup = T.matrix('y_sup')
        temp = T.scalar('temp')
        sym_names = ['x','y','z1','z2','x_sup','y_sup','eq_samples','iw_samples','temp','lr']
        syms = [sym_x,sym_y,sym_z1,sym_z2,sym_x_sup,sym_y_sup,sym_eq_samples,sym_iw_samples,temp,sym_lr]
        model_dict['sym'] = OrderedDict(zip(sym_names,syms))

    def _init_net_archs(self,model_dict):
        if 'z1->y' in model_dict:
            assert ('yz2->_z1' in model_dict and 'z1y->z2' in model_dict), 'S2S_DGM msg: z1->y in model_dict implies yz2->_z1 and z1y->z2 must exist also.'
            self.num_latent_z1 = model_dict['yz2->_z1']['arch']['num_output']
            self.num_latent_z2 = model_dict['z1y->z2']['arch']['num_output']
            if 'x->z1' in model_dict:
                assert 'z1->_x' in model_dict, 'S2S_DGM msg: x->z1 in model_dict implies z1->_x must exist also.'
                self.model_type = 2
            else:
                assert self.num_features == self.num_latent_z1, 'S2S_DGM msg: # x must equal # z1 for M2 model (x=z1).'
                self.model_type = 1
        elif 'x->z1' in model_dict:
            assert 'z1->_x' in model_dict, 'S2S_DGM msg: x->z1 in model_dict implies z1->_x must exist also.'
            self.model_type = 0

        model_type = self.model_type
        num_features = self.num_features
        num_output = self.num_output
        num_latent_z1 = self.num_latent_z1
        num_latent_z2 = self.num_latent_z2

        flows = [NICE]*self.nflows + \
                [NormalizingSimplexFlowLayer]*self.simplexflow

        if model_type in [0,2]:
            if not 'x->z1' in model_dict:
                arch = OrderedDict()
                arch['hidden'] = self.hidden
                arch['gain'] = self.gain_enc
                arch['nonlin'] = self.nonlin_enc
                arch['sample_layer'] = SampleLayer
                if model_type == 0:
                    arch['num_output'] = num_output
                    if num_output > 1:
                        arch['flows'] = flows
                    else:
                        arch['flows'] = []
                else:
                    arch['num_output'] = num_latent_z1
                    if num_latent_z1 > 1:
                        arch['flows'] = flows[:-1]
                    else:
                        arch['flows'] = []
                arch['batch_norm'] = self.batch_norm
                model_dict['x->z1'] = OrderedDict([('arch',arch)])

            if not 'z1->_x' in model_dict:
                arch = OrderedDict()
                arch['hidden'] = self.hidden
                arch['gain'] = self.gain_enc
                arch['nonlin'] = self.nonlin_enc
                arch['num_output'] = num_features
                arch['sample_layer'] = SampleLayer
                arch['flows'] = []
                arch['batch_norm'] = self.batch_norm
                model_dict['z1->_x'] = OrderedDict([('arch',arch)])

        if model_type in [1,2]:
            if not 'z1->y' in model_dict:
                arch = OrderedDict()
                arch['hidden'] = self.hidden
                arch['gain'] = self.gain_enc
                arch['nonlin'] = self.nonlin_enc
                arch['num_output'] = num_output
                arch['sample_layer'] = SampleLayer
                if num_output > 1:
                    arch['flows'] = flows
                else:
                    arch['flows'] = []
                arch['batch_norm'] = self.batch_norm
                model_dict['z1->y'] = OrderedDict([('arch',arch)])

            if not 'z1y->z2' in model_dict:
                arch = OrderedDict()
                arch['hidden'] = self.hidden
                arch['gain'] = self.gain_enc
                arch['nonlin'] = self.nonlin_enc
                arch['num_output'] = num_latent_z2
                arch['sample_layer'] = SampleLayer
                if num_latent_z2 > 1:
                    arch['flows'] = flows[:-1]
                else:
                    arch['flows'] = []
                arch['batch_norm'] = self.batch_norm
                model_dict['z1y->z2'] = OrderedDict([('arch',arch)])

            if not 'yz2->_z1' in model_dict:
                arch = OrderedDict()
                arch['hidden'] = self.hidden
                arch['gain'] = self.gain_dec
                arch['nonlin'] = self.nonlin_dec
                arch['sample_layer'] = SampleLayer
                if model_type == 1:
                    arch['num_output'] = num_features
                else:
                    arch['num_output'] = num_latent_z1
                arch['flows'] = []
                arch['batch_norm'] = self.batch_norm
                model_dict['yz2->_z1'] = OrderedDict([('arch',arch)])

        batch_norms = []
        if model_type in [0,2]:
            batch_norms += [model_dict['x->z1']['arch']['batch_norm']]
            batch_norms += [model_dict['z1->_x']['arch']['batch_norm']]
        if model_type in [1,2]:
            batch_norms += [model_dict['z1->y']['arch']['batch_norm']]
            batch_norms += [model_dict['z1y->z2']['arch']['batch_norm']]
            batch_norms += [model_dict['yz2->_z1']['arch']['batch_norm']]
        self.batch_norm = any(batch_norms)

    def _construct_from_blueprint(self,model_dict):
        # Input layer
        l_input = lasagne.layers.InputLayer((None, self.num_features))

        # Track important layers
        self.x = l_input
        self.z1 = None
        self._x = None
        self.y = None
        self.z2 = None
        self._z1 = None

        # # Track trainable params
        # params = OrderedDict()

        # Track construction
        M1_built = False
        M2_built = False

        # Track input layer
        M_input = l_input

        # M1 Construction
        if self.model_type in [0,2]:
            M1_built = M1(M_input,model_dict,
                          prior_x=self.prior_x,prior_z1=self.prior_z1,
                          loss_x=self.loss_x,loss_y=self.loss_y)
        if M1_built:
            n_flows_x_z1 = len(model_dict['x->z1']['arch']['flows'])
            z1_name = 'l_z' + (n_flows_x_z1 > 0)*str(n_flows_x_z1)
            z1 = model_dict['x->z1']['key_layers'][z1_name]
            self.z1 = z1
            # params['x->z1'] = lasagne.layers.get_all_params([z1],trainable=True)

            M_input = z1

            n_flows_z1__x = len(model_dict['z1->_x']['arch']['flows'])
            _x_name = 'l_z' + (n_flows_z1__x > 0)*str(n_flows_z1__x)
            _x = model_dict['z1->_x']['key_layers'][_x_name]

            self._x = _x
            # params['M1'] = lasagne.layers.get_all_params([_x],trainable=True)
            # params['z1->_x'] = list(set(params['M1'])-set(params['x->z1']))

        # M2 Construction
        if self.model_type in [1,2]:
            if self.model_type == 1:
                self.prior_xz1 = self.prior_x
            else:
                self.prior_xz1 = self.prior_z1
            M2_built = M2(M_input,model_dict,
                          prior_xz1=self.prior_xz1,prior_y=self.prior_y,prior_z2=self.prior_z2,
                          model_type=self.model_type,
                          loss_x=self.loss_x,loss_y=self.loss_y)
        if M2_built:
            n_flows_z1_y = len(model_dict['z1->y']['arch']['flows'])
            y_name = 'l_z' + (n_flows_z1_y > 0)*str(n_flows_z1_y)
            y = model_dict['z1->y']['key_layers'][y_name]

            self.y = y
            # params['z1->y'] = lasagne.layers.get_all_params([y],trainable=True)
            # if M1_built:
            #     params['z1->y'] = list(set(params['z1->y'])-set(params['M1']))

            n_flows_z1y_z2 = len(model_dict['z1y->z2']['arch']['flows'])
            z2_name = 'l_z' + (n_flows_z1y_z2 > 0)*str(n_flows_z1y_z2)
            z2 = model_dict['z1y->z2']['key_layers'][z2_name]

            self.z2 = z2
            # params['z1y->z2'] = lasagne.layers.get_all_params([z2],trainable=True)
            # params['z1y->z2'] = list(set(params['z1y->z2'])-set(params['z1->y']))
            # if M1_built:
            #     params['z1y->z2'] = list(set(params['z1y->z2'])-set(params['M1']))

            n_flows_yz2__z1 = len(model_dict['yz2->_z1']['arch']['flows'])
            _z1_name = 'l_z' + (n_flows_yz2__z1 > 0)*str(n_flows_yz2__z1)
            _z1 = model_dict['yz2->_z1']['key_layers'][_z1_name]

            self._z1 = _z1
            # params['yz2->_z1'] = lasagne.layers.get_all_params([_z1],trainable=True)
            # params['yz2->_z1'] = list(set(params['yz2->_z1'])-set(params['z1y->z2'])-set(params['z1->y']))
            # if M1_built:
            #     params['yz2->_z1'] = list(set(params['yz2->_z1'])-set(params['M1']))
            # params['M2'] = list(params['z1->y'])+list(params['z1y->z2'])+list(params['yz2->_z1'])

            if M1_built:
                print('removing M1 built components for M1M2 model.')
                # remove M1 supervised and y-unsupervised components
                # key layer outputs
                for key in list(model_dict['x->z1']['key_outputs'].keys()):
                    if 'y' in key or 'sup' in key:
                        model_dict['x->z1']['key_outputs'].pop(key,None)
                for key in list(model_dict['z1->_x']['key_outputs'].keys()):
                    if 'y' in key or 'sup' in key:
                        model_dict['z1->_x']['key_outputs'].pop(key,None)
                # key layer objectives
                for typ in ['var','nonvar']:
                    for key in list(model_dict['x->z1']['key_objs'][typ].keys()):
                        if 'y' in key or 'sup' in key:
                            model_dict['x->z1']['key_objs'][typ].pop(key,None)
                    for key in list(model_dict['z1->_x']['key_objs'][typ].keys()):
                        if 'y' in key or 'sup' in key:
                            model_dict['z1->_x']['key_objs'][typ].pop(key,None)
                # key objectives
                for typ in ['var','nonvar']:
                    for key in list(model_dict['objs'][typ].keys()):
                        if 'M1' in key and ('y' in key or 'sup' in key):
                            model_dict['objs'][typ].pop(key,None)

                # params['all'] = params['M1']+params['M2']
                model_dict['params']['all'] = model_dict['params']['M1']+model_dict['params']['M2']
                model_dict['params']['nonvar'] = model_dict['params']['M1_nonvar'] + \
                                                      model_dict['params']['M2_nonvar']
                model_dict['params']['var'] = model_dict['params']['M1_var'] + \
                                                    model_dict['params']['M2_var']
            else:
                self.z1 = self.x
                self._x = self._z1

                # params['all'] = params['M2']
                model_dict['params']['all'] = model_dict['params']['M2']
                model_dict['params']['nonvar'] = model_dict['params']['M2_nonvar']
                model_dict['params']['var'] = model_dict['params']['M2_var']
        else:
            assert M1_built, 'S2S_DGM msg: Either M1 or M2 must be built'

            # params['all'] = params['M1']
            model_dict['params']['all'] = model_dict['params']['M1']
            model_dict['params']['nonvar'] = model_dict['params']['M1_nonvar']
            model_dict['params']['var'] = model_dict['params']['M1_var']
        
        # self.params = params

        self.M1_built = M1_built
        self.M2_built = M2_built

        self._switch_variational(self.variational)

    def _switch_variational(self,variational=False):
        if variational:
            typ = 'var'
        else:
            typ = 'nonvar'

        model_dict = self.model_dict

        obj_x = []
        obj_y = []
        obj_x_sup = []
        obj_y_sup = []
        obj_x_sup_dis = []
        obj_y_sup_dis = []

        if self.M1_built:
            obj_x += [model_dict['objs'][typ]['M1_x']]
        if self.M2_built:
            obj_x += [model_dict['objs'][typ]['M2_x']]
            obj_y += [model_dict['objs'][typ]['M2_y']]
            obj_x_sup += [model_dict['objs'][typ]['M2_x_sup']]
            obj_y_sup += [model_dict['objs'][typ]['M2_y_sup']]
            obj_x_sup_dis += [model_dict['objs'][typ]['M2_x_sup_dis']]
            obj_y_sup_dis += [model_dict['objs'][typ]['M2_y_sup_dis']]
        else:
            obj_y += [model_dict['objs'][typ]['M1_y']]
            obj_x_sup += [model_dict['objs'][typ]['M1_x_sup']]
            obj_y_sup += [model_dict['objs'][typ]['M1_y_sup']]
            obj_x_sup_dis += [model_dict['objs'][typ]['M1_x_sup_dis']]
            obj_y_sup_dis += [model_dict['objs'][typ]['M1_y_sup_dis']]

        self.obj_x = obj_x
        self.obj_y = obj_y
        self.obj_x_sup = obj_x_sup
        self.obj_y_sup = obj_y_sup
        self.obj_x_sup_dis = obj_x_sup_dis
        self.obj_y_sup_dis = obj_y_sup_dis

        return typ

    def _add_functionality(self):
        model_type = self.model_type
        syms = self.model_dict['sym']

        # get x
        self.getX = lambda x: x

        # get z1
        self.getZ1 = self._make_getZ1(model_type,syms)

        # get x reconstruction
        self.get_X = self._make_get_X(model_type,syms)
        self.generate = self.get_X

        # get y
        self.getY = self._make_getY(model_type,syms)
        self.predict = self.getY

        # get z2
        self.getZ2 = self._make_getZ2(model_type,syms)

        # get z1 reconstruction
        self.get_Z1 = self._make_get_Z1(model_type,syms)

        # get y & z2 - uses functions getZ1, getY, and getZ2
        self.getYZ2 = self._make_get_YZ2(model_type)

        # prob want a get y and get z2 reconstruction

        # get X unsupervised error
        self.getX_obj = self._make_getX_obj(model_type,syms)

        # get Y unsupervised error
        self.getY_obj = self._make_getY_obj(model_type,syms)

        # get X supervised error
        self.getXsup_obj = self._make_getXsup_obj(model_type,syms)

        # get Y supervised error
        self.getYsup_obj = self._make_getYsup_obj(model_type,syms)

        # get X->Y supervised discriminative error (prediction error)
        self.getXY_obj = self._make_getXY_obj(model_type,syms)

        # get Y->X supervised discriminative error (inverse prediction error)
        self.getYX_obj = self._make_getYX_obj(model_type,syms)

        self.useful_functions = ['fit', 'predict', 'generate', 'loss', 'score',
                                 'getX_obj', 'getY_obj',
                                 'getXsup_obj', 'getYsup_obj',
                                 'getXY_obj', 'getYX_obj',
                                 'getX', 'getY', 'getYZ2', 'getZ1', 'getZ2',
                                 'get_X', 'get_Z1']


    def _make_getX_obj(self,model_type,syms):
        # note eq and iw samples only used for non-deterministic
        one = np.cast['int32'](1)
        kwargs = {'givens':{syms['eq_samples']:one,syms['iw_samples']:one},'on_unused_input':'ignore',
                  'name':'getX_obj'}

        return theano.function([syms['x']],T.sum(self.obj_x),**kwargs)

    def _make_getY_obj(self,model_type,syms):
        # note eq and iw samples only used for non-deterministic
        one = np.cast['int32'](1)
        kwargs = {'givens':{syms['eq_samples']:one,syms['iw_samples']:one},'on_unused_input':'ignore',
                  'name':'getY_obj'}

        return theano.function([syms['y'],syms['z2']],T.sum(self.obj_y),**kwargs)

    def _make_getXsup_obj(self,model_type,syms):
        # note eq and iw samples only used for non-deterministic
        one = np.cast['int32'](1)
        kwargs = {'givens':{syms['eq_samples']:one,syms['iw_samples']:one},'on_unused_input':'ignore',
                  'name':'getXsup_obj'}

        return theano.function([syms['x_sup'],syms['y_sup']],T.sum(self.obj_x_sup),**kwargs)

    def _make_getYsup_obj(self,model_type,syms):
        # note eq and iw samples only used for non-deterministic
        one = np.cast['int32'](1)
        kwargs = {'givens':{syms['eq_samples']:one,syms['iw_samples']:one},'on_unused_input':'ignore',
                  'name':'getYsup_obj'}

        return theano.function([syms['x_sup'],syms['y_sup']],T.sum(self.obj_y_sup),**kwargs)

    def _make_getXY_obj(self,model_type,syms):
        # note eq and iw samples only used for non-deterministic
        one = np.cast['int32'](1)
        kwargs = {'givens':{syms['eq_samples']:one,syms['iw_samples']:one},'on_unused_input':'ignore',
                  'name':'getXY_obj'}

        return theano.function([syms['x_sup'],syms['y_sup']],T.sum(self.obj_x_sup_dis),**kwargs)

    def _make_getYX_obj(self,model_type,syms):
        # note eq and iw samples only used for non-deterministic
        one = np.cast['int32'](1)
        kwargs = {'givens':{syms['eq_samples']:one,syms['iw_samples']:one},'on_unused_input':'ignore',
                  'name':'getYX_obj'}

        return theano.function([syms['x_sup'],syms['y_sup']],T.sum(self.obj_y_sup_dis),**kwargs)

    def loss(self,X,y):
        # loss is weighted supervised error for composition prediction and spectra generation
        return self.coeff_x_dis*self.getXY_obj(X,y) + self.coeff_y_dis*self.getYX_obj(X,y)

    def score(self,X,y):
        # score is negative loss
        return -self.loss(X,y)

    def _make_getZ1(self,model_type,syms):
        # note eq and iw samples only used for non-deterministic
        one = np.cast['int32'](1)
        kwargs = {'givens':{syms['eq_samples']:one,syms['iw_samples']:one},'on_unused_input':'ignore'}

        from_x_det = lasagne.layers.get_output(self.z1,inputs={self.x:syms['x']},deterministic=True)
        from_x_nondet = lasagne.layers.get_output(self.z1,inputs={self.x:syms['x']},deterministic=False)
        from_x_det = theano.function([syms['x']],from_x_det,**kwargs)
        from_x_nondet = theano.function([syms['x']],from_x_nondet,**kwargs)

        if model_type in [0,2]:
            def getZ1(x=None,deterministic=not self.variational):
                # require input
                assert x is not None, 'S2S_DGM msg: Must provide an input for x.'

                if deterministic:
                    return from_x_det(x)
                else:
                    return from_x_nondet(x)
        else:
            def getZ1(x=None,deterministic=not self.variational):
                raise NotImplementedError('S2S_DGM msg: z1 is treated as input in M2 model. A getZ1 function is unnecessary.')

        return getZ1

    def _make_get_X(self,model_type,syms):
        # note eq and iw samples only used for non-deterministic
        one = np.cast['int32'](1)
        kwargs = {'givens':{syms['eq_samples']:one,syms['iw_samples']:one},'on_unused_input':'ignore'}
        
        from_x_det = lasagne.layers.get_output(self._x,inputs={self.x:syms['x']},deterministic=True)
        from_x_nondet = lasagne.layers.get_output(self._x,inputs={self.x:syms['x']},deterministic=False)
        from_x_det = theano.function([syms['x']],from_x_det,**kwargs)
        from_x_nondet = theano.function([syms['x']],from_x_nondet,**kwargs)

        from_z1_det = lasagne.layers.get_output(self._x,inputs={self.z1:syms['z1']},deterministic=True)
        from_z1_nondet = lasagne.layers.get_output(self._x,inputs={self.z1:syms['z1']},deterministic=False)
        from_z1_det = theano.function([syms['z1']],from_z1_det,**kwargs)
        from_z1_nondet = theano.function([syms['z1']],from_z1_nondet,**kwargs)

        if model_type == 0:
            def get_X(x=None,y=None,z1=None,deterministic=not self.variational):
                # require input
                assert not (x is None and y is None and z1 is None), 'S2S_DGM msg: Must provide an input for x or y/z1.'

                # check z1/y relationship
                if z1 is None:
                    z1 = y
                elif y is not None:
                    assert np.allclose(z1,y), 'S2S_DGM msg: z1 and y are the same variable - getX expecting the same value.'

                # priority given to y/z1 - assuming better reconstruction possible
                if z1 is not None:
                    if deterministic:
                        return from_z1_det(z1)
                    else:
                        return from_z1_nondet(z1)
                else:
                    if deterministic:
                        return from_x_det(x)
                    else:
                        return from_x_nondet(x)

        elif model_type in [1,2]:
            # to get from y & z2
            # get _z1 from y&z2
            _z1_from_yz2_det = lasagne.layers.get_output(self._z1,inputs={self.y:syms['y'],self.z2:syms['z2']},deterministic=True)
            _z1_from_yz2_nondet = lasagne.layers.get_output(self._z1,inputs={self.y:syms['y'],self.z2:syms['z2']},deterministic=False)
            # get _x from _z1
            _x_from__z1_det = lasagne.layers.get_output(self._x,inputs={self.z1:_z1_from_yz2_det},deterministic=True)
            _x_from__z1_nondet = lasagne.layers.get_output(self._x,inputs={self.z1:_z1_from_yz2_nondet},deterministic=False)
            # rename
            from_yz2_det = _x_from__z1_det
            from_yz2_nondet = _x_from__z1_nondet
            from_yz2_det = theano.function([syms['y'],syms['z2']],from_yz2_det,**kwargs)
            from_yz2_nondet = theano.function([syms['y'],syms['z2']],from_yz2_nondet,**kwargs)
            
            if model_type == 1:
                # to get from x & y
                # get z1 from x,
                z1_from_x_det = lasagne.layers.get_output(self.z1,inputs={self.x:syms['x']},deterministic=True)
                z1_from_x_nondet = lasagne.layers.get_output(self.z1,inputs={self.x:syms['x']},deterministic=False)
                # then get z2 from z1&y,
                z2_from_z1y_det = lasagne.layers.get_output(self.z2,inputs={self.z1:z1_from_x_det,self.y:syms['y']},deterministic=True)
                z2_from_z1y_nondet = lasagne.layers.get_output(self.z2,inputs={self.z1:z1_from_x_nondet,self.y:syms['y']},deterministic=False)
                # then get _x/_z1 from y&z2
                _x_from_yz2_det = lasagne.layers.get_output(self._x,inputs={self.y:syms['y'],self.z2:z2_from_z1y_det},deterministic=True)
                _x_from_yz2_nondet = lasagne.layers.get_output(self._x,inputs={self.y:syms['y'],self.z2:z2_from_z1y_nondet},deterministic=False)
                # rename
                from_xy_det = _x_from_yz2_det
                from_xy_nondet = _x_from_yz2_nondet
                from_xy_det = theano.function([syms['x'],syms['y']],from_xy_det,**kwargs)
                from_xy_nondet = theano.function([syms['x'],syms['y']],from_xy_nondet,**kwargs)

                # to get from z1 & y
                # get z2 from z1&y,
                z2_from_z1y_det = lasagne.layers.get_output(self.z2,inputs={self.z1:syms['z1'],self.y:syms['y']},deterministic=True)
                z2_from_z1y_nondet = lasagne.layers.get_output(self.z2,inputs={self.z1:syms['z1'],self.y:syms['y']},deterministic=False)
                # then get _x/_z1 from y&z2
                _x_from_yz2_det = lasagne.layers.get_output(self._x,inputs={self.y:syms['y'],self.z2:z2_from_z1y_det},deterministic=True)
                _x_from_yz2_nondet = lasagne.layers.get_output(self._x,inputs={self.y:syms['y'],self.z2:z2_from_z1y_nondet},deterministic=False)
                # rename
                from_z1y_det = _x_from_yz2_det
                from_z1y_nondet = _x_from_yz2_nondet
                from_z1y_det = theano.function([syms['z1'],syms['y']],from_z1y_det,**kwargs)
                from_z1y_nondet = theano.function([syms['z1'],syms['y']],from_z1y_nondet,**kwargs)

                # to get from x & z2
                # get z1 from x,
                z1_from_x_det = lasagne.layers.get_output(self.z1,inputs={self.x:syms['x']},deterministic=True)
                z1_from_x_nondet = lasagne.layers.get_output(self.z1,inputs={self.x:syms['x']},deterministic=False)
                # then get y from z1,
                y_from_z1_det = lasagne.layers.get_output(self.y,inputs={self.z1:z1_from_x_det},deterministic=True)
                y_from_z1_nondet = lasagne.layers.get_output(self.y,inputs={self.z1:z1_from_x_nondet},deterministic=False)
                # then get _x/_z1 from y&z2
                _x_from_yz2_det = lasagne.layers.get_output(self._x,inputs={self.y:y_from_z1_det,self.z2:syms['z2']},deterministic=True)
                _x_from_yz2_nondet = lasagne.layers.get_output(self._x,inputs={self.y:y_from_z1_nondet,self.z2:syms['z2']},deterministic=False)
                # rename
                from_xz2_det = _x_from_yz2_det
                from_xz2_nondet = _x_from_yz2_nondet
                from_xz2_det = theano.function([syms['x'],syms['z2']],from_xz2_det,**kwargs)
                from_xz2_nondet = theano.function([syms['x'],syms['z2']],from_xz2_nondet,**kwargs)

                # to get from z1 & z2
                # get y from z1,
                y_from_z1_det = lasagne.layers.get_output(self.y,inputs={self.z1:syms['z1']},deterministic=True)
                y_from_z1_nondet = lasagne.layers.get_output(self.y,inputs={self.z1:syms['z1']},deterministic=False)
                # then get _x from _z1
                _x_from__z1_det = lasagne.layers.get_output(self._x,inputs={self.z1:_z1_from_yz2_det},deterministic=True)
                _x_from__z1_nondet = lasagne.layers.get_output(self._x,inputs={self.z1:_z1_from_yz2_nondet},deterministic=False)
                # then get _x from y&z2
                _x_from_yz2_det = lasagne.layers.get_output(self._x,inputs={self.y:y_from_z1_det,self.z2:syms['z2']},deterministic=True)
                _x_from_yz2_nondet = lasagne.layers.get_output(self._x,inputs={self.y:y_from_z1_nondet,self.z2:syms['z2']},deterministic=False)
                # rename
                from_z1z2_det = _x_from_yz2_det
                from_z1z2_nondet = _x_from_yz2_nondet
                from_z1z2_det = theano.function([syms['z1'],syms['z2']],from_z1z2_det,**kwargs)
                from_z1z2_nondet = theano.function([syms['z1'],syms['z2']],from_z1z2_nondet,**kwargs)

                def get_X(x=None,y=None,z1=None,z2=None,deterministic=not self.variational):
                    given = (x is not None, y is not None, z1 is not None, z2 is not None)
                    given = tuple([int(g) for g in given])
                    allowed = set([(1,0,0,0),(0,0,1,0),(1,1,0,0),(0,1,1,0),(1,0,0,1),(0,0,1,1),(0,1,0,1)])
                    assert given in allowed, 'S2S_DGM msg: Input ' + repr(given) + ' is not an allowed pairing. [x,y,z1,z2] in ' + repr(allowed)

                    if given == (1,0,0,0):
                        if deterministic:
                            return from_x_det(x)
                        else:
                            return from_x_nondet(x)
                    elif given == (0,0,1,0):
                        if deterministic:
                            return from_z1_det(z1)
                        else:
                            return from_z1_nondet(z1)
                    elif given == (1,1,0,0):
                        if deterministic:
                            return from_xy_det(x,y)
                        else:
                            return from_xy_nondet(x,y)
                    elif given == (0,1,1,0):
                        if deterministic:
                            return from_z1y_det(z1,y)
                        else:
                            return from_z1y_nondet(z1,y)
                    elif given == (1,0,0,1):
                        if deterministic:
                            return from_xz2_det(x,z2)
                        else:
                            return from_xz2_nondet(x,z2)
                    elif given == (0,0,1,1):
                        if deterministic:
                            return from_z1z2_det(z1,z2)
                        else:
                            return from_z1z2_nondet(z1,z2)
                    elif given == (0,1,0,1):
                        if deterministic:
                            return from_yz2_det(y,z2)
                        else:
                            return from_yz2_nondet(y,z2)
            else:
                def get_X(x=None,y=None,z1=None,z2=None,deterministic=not self.variational):
                    given = (x is not None, y is not None, z1 is not None, z2 is not None)
                    given = tuple([int(g) for g in given])
                    allowed = set([(1,0,0,0),(0,0,1,0),(1,1,0,0),(0,1,1,0),(1,0,0,1),(0,0,1,1),(0,1,0,1)])
                    assert given in allowed, 'S2S_DGM msg: Input ' + repr(given) + ' is not an allowed pairing. [x,y,z1,z2] in ' + repr(allowed)

                    if given == (1,0,0,0):
                        if deterministic:
                            return from_x_det(x)
                        else:
                            return from_x_nondet(x)
                    elif given == (0,0,1,0):
                        if deterministic:
                            return from_z1_det(z1)
                        else:
                            return from_z1_nondet(z1)
                    elif given == (1,1,0,0):
                        if deterministic:
                            return from_x_det(x)
                        else:
                            return from_x_nondet(x)
                    elif given == (0,1,1,0):
                        if deterministic:
                            return from_z1_det(z1)
                        else:
                            return from_z1_nondet(z1)
                    elif given == (1,0,0,1):
                        if deterministic:
                            return from_x_det(x)
                        else:
                            return from_x_nondet(x)
                    elif given == (0,0,1,1):
                        if deterministic:
                            return from_z1_det(z1)
                        else:
                            return from_z1_nondet(z1)
                    elif given == (0,1,0,1):
                        if deterministic:
                            return from_yz2_det(y,z2)
                        else:
                            return from_yz2_nondet(y,z2)

        return get_X

    def _make_getY(self,model_type,syms):
        # note eq and iw samples only used for non-deterministic
        one = np.cast['int32'](1)
        kwargs = {'givens':{syms['eq_samples']:one,syms['iw_samples']:one},'on_unused_input':'ignore'}
        
        if model_type == 0:
            from_x_det = lasagne.layers.get_output(self.z1,inputs={self.x:syms['x']},deterministic=True)
            from_x_nondet = lasagne.layers.get_output(self.z1,inputs={self.x:syms['x']},deterministic=False)
            from_x_det = theano.function([syms['x']],from_x_det,**kwargs)
            from_x_nondet = theano.function([syms['x']],from_x_nondet,**kwargs)

            def getY(x=None,deterministic=not self.variational):
                '''
                Treats z1 as y in M1 model.
                '''
                # require input
                assert x is not None, 'S2S_DGM msg: Must provide an input for x.'

                if deterministic:
                    return from_x_det(x)
                else:
                    return from_x_nondet(x)
        else:
            # from x
            from_x_det = lasagne.layers.get_output(self.y,inputs={self.x:syms['x']},deterministic=True)
            from_x_nondet = lasagne.layers.get_output(self.y,inputs={self.x:syms['x']},deterministic=False)
            from_x_det = theano.function([syms['x']],from_x_det,name='get_y_from_x_det',**kwargs)
            from_x_nondet = theano.function([syms['x']],from_x_nondet,**kwargs)

            # from z1
            from_z1_det = lasagne.layers.get_output(self.y,inputs={self.z1:syms['z1']},deterministic=True)
            from_z1_nondet = lasagne.layers.get_output(self.y,inputs={self.z1:syms['z1']},deterministic=False)
            from_z1_det = theano.function([syms['z1']],from_z1_det,**kwargs)
            from_z1_nondet = theano.function([syms['z1']],from_z1_nondet,name='getY_from_z1_nondet',**kwargs)

            def getY(x=None,z1=None,deterministic=not self.variational):
                assert not (x is None and z1 is None), 'S2S_DGM msg: Must provide an input for x or z1.'

                if z1 is not None:
                    if deterministic:
                        return from_z1_det(z1)
                    else:
                        return from_z1_nondet(z1)
                else:
                    if deterministic:
                        return from_x_det(x)
                    else:
                        return from_x_nondet(x)

        return getY

    def _make_getZ2(self,model_type,syms):
        # note eq and iw samples only used for non-deterministic
        one = np.cast['int32'](1)
        kwargs = {'givens':{syms['eq_samples']:one,syms['iw_samples']:one},'on_unused_input':'ignore'}
        
        if model_type == 0:
            def getZ2(x=None,deterministic=not self.variational):
                raise NotImplementedError('S2S_DGM msg: z2 is not a part of the M1 model.')
        else:
            # from x
            from_x_det = lasagne.layers.get_output(self.z2,inputs={self.x:syms['x']},deterministic=True)
            from_x_nondet = lasagne.layers.get_output(self.z2,inputs={self.x:syms['x']},deterministic=False)
            from_x_det = theano.function([syms['x']],from_x_det,**kwargs)
            from_x_nondet = theano.function([syms['x']],from_x_nondet,**kwargs)

            # from x&y
            from_xy_det = lasagne.layers.get_output(self.z2,inputs={self.x:syms['x'],self.y:syms['y']},deterministic=True)
            from_xy_nondet = lasagne.layers.get_output(self.z2,inputs={self.x:syms['x'],self.y:syms['y']},deterministic=False)
            from_xy_det = theano.function([syms['x'],syms['y']],from_xy_det,**kwargs)
            from_xy_nondet = theano.function([syms['x'],syms['y']],from_xy_nondet,**kwargs)

            # from z1
            from_z1_det = lasagne.layers.get_output(self.z2,inputs={self.z1:syms['z1']},deterministic=True)
            from_z1_nondet = lasagne.layers.get_output(self.z2,inputs={self.z1:syms['z1']},deterministic=False)
            from_z1_det = theano.function([syms['z1']],from_z1_det,**kwargs)
            from_z1_nondet = theano.function([syms['z1']],from_z1_nondet,**kwargs)

            # from z1&y
            from_z1y_det = lasagne.layers.get_output(self.z2,inputs={self.z1:syms['z1'],self.y:syms['y']},deterministic=True)
            from_z1y_nondet = lasagne.layers.get_output(self.z2,inputs={self.z1:syms['z1'],self.y:syms['y']},deterministic=False)
            from_z1y_det = theano.function([syms['z1'],syms['y']],from_z1y_det,**kwargs)
            from_z1y_nondet = theano.function([syms['z1'],syms['y']],from_z1y_nondet,**kwargs)

            def getZ2(x=None,y=None,z1=None,deterministic=not self.variational):
                given = (x is not None, y is not None, z1 is not None)
                given = tuple([int(g) for g in given])
                allowed = set([(1,0,0),(1,1,0),(0,0,1),(0,1,1)])
                assert given in allowed, 'S2S_DGM msg: Input ' + repr(given) + ' is not an allowed pairing. [x,y,z1] in ' + repr(allowed)

                if given == (1,0,0):
                    if deterministic:
                        return from_x_det(x)
                    else:
                        return from_x_nondet(x)
                if given == (1,1,0):
                    if deterministic:
                        return from_xy_det(x,y)
                    else:
                        return from_xy_nondet(x,y)
                elif given == (0,0,1):
                    if deterministic:
                        return from_z1_det(z1)
                    else:
                        return from_z1_nondet(z1)
                elif given == (0,1,1):
                    if deterministic:
                        return from_z1y_det(z1,y)
                    else:
                        return from_z1y_nondet(z1,y)

        return getZ2

    def _make_get_Z1(self,model_type,syms):
        # note eq and iw samples only used for non-deterministic
        one = np.cast['int32'](1)
        kwargs = {'givens':{syms['eq_samples']:one,syms['iw_samples']:one},'on_unused_input':'ignore'}

        if model_type == 0:
            def get_Z1(x=None,y=None,z1=None,z2=None,deterministic=not self.variational):
                raise NotImplementedError('S2S_DGM msg: z1 reconstruction is not a part of the M1 model.')
        else:
            # from x
            from_x_det = lasagne.layers.get_output(self._z1,inputs={self.x:syms['x']},deterministic=True)
            from_x_nondet = lasagne.layers.get_output(self._z1,inputs={self.x:syms['x']},deterministic=False)
            from_x_det = theano.function([syms['x']],from_x_det,**kwargs)
            from_x_nondet = theano.function([syms['x']],from_x_nondet,**kwargs)

            # from z1
            from_z1_det = lasagne.layers.get_output(self._z1,inputs={self.z1:syms['z1']},deterministic=True)
            from_z1_nondet = lasagne.layers.get_output(self._z1,inputs={self.z1:syms['z1']},deterministic=False)
            from_z1_det = theano.function([syms['z1']],from_z1_det,**kwargs)
            from_z1_nondet = theano.function([syms['z1']],from_z1_nondet,**kwargs)

            # to get from x & y
            # get z1 from x,
            z1_from_x_det = lasagne.layers.get_output(self.z1,inputs={self.x:syms['x']},deterministic=True)
            z1_from_x_nondet = lasagne.layers.get_output(self.z1,inputs={self.x:syms['x']},deterministic=False)
            # then get z2 from z1&y,
            z2_from_z1y_det = lasagne.layers.get_output(self.z2,inputs={self.z1:z1_from_x_det,self.y:syms['y']},deterministic=True)
            z2_from_z1y_nondet = lasagne.layers.get_output(self.z2,inputs={self.z1:z1_from_x_nondet,self.y:syms['y']},deterministic=False)
            # then get _z1 from y&z2
            _z1_from_yz2_det = lasagne.layers.get_output(self._z1,inputs={self.y:syms['y'],self.z2:z2_from_z1y_det},deterministic=True)
            _z1_from_yz2_nondet = lasagne.layers.get_output(self._z1,inputs={self.y:syms['y'],self.z2:z2_from_z1y_nondet},deterministic=False)
            # rename
            from_xy_det = _z1_from_yz2_det
            from_xy_nondet = _z1_from_yz2_nondet
            from_xy_det = theano.function([syms['x'],syms['y']],from_xy_det,**kwargs)
            from_xy_nondet = theano.function([syms['x'],syms['y']],from_xy_nondet,**kwargs)

            # to get from z1 & y
            # get z2 from z1&y,
            z2_from_z1y_det = lasagne.layers.get_output(self.z2,inputs={self.z1:syms['z1'],self.y:syms['y']},deterministic=True)
            z2_from_z1y_nondet = lasagne.layers.get_output(self.z2,inputs={self.z1:syms['z1'],self.y:syms['y']},deterministic=False)
            # then get _x from y&z2
            _z1_from_yz2_det = lasagne.layers.get_output(self._z1,inputs={self.y:syms['y'],self.z2:z2_from_z1y_det},deterministic=True)
            _z1_from_yz2_nondet = lasagne.layers.get_output(self._z1,inputs={self.y:syms['y'],self.z2:z2_from_z1y_nondet},deterministic=False)
            # rename
            from_z1y_det = _z1_from_yz2_det
            from_z1y_nondet = _z1_from_yz2_nondet
            from_z1y_det = theano.function([syms['z1'],syms['y']],from_z1y_det,**kwargs)
            from_z1y_nondet = theano.function([syms['z1'],syms['y']],from_z1y_nondet,**kwargs)

            # to get from x & z2
            # get z1 from x,
            z1_from_x_det = lasagne.layers.get_output(self.z1,inputs={self.x:syms['x']},deterministic=True)
            z1_from_x_nondet = lasagne.layers.get_output(self.z1,inputs={self.x:syms['x']},deterministic=False)
            # then get y from z1,
            y_from_z1_det = lasagne.layers.get_output(self.y,inputs={self.z1:z1_from_x_det},deterministic=True)
            y_from_z1_nondet = lasagne.layers.get_output(self.y,inputs={self.z1:z1_from_x_nondet},deterministic=False)
            # then get _x from y&z2
            _z1_from_yz2_det = lasagne.layers.get_output(self._z1,inputs={self.y:y_from_z1_det,self.z2:syms['z2']},deterministic=True)
            _z1_from_yz2_nondet = lasagne.layers.get_output(self._z1,inputs={self.y:y_from_z1_nondet,self.z2:syms['z2']},deterministic=False)
            # rename
            from_xz2_det = _z1_from_yz2_det
            from_xz2_nondet = _z1_from_yz2_nondet
            from_xz2_det = theano.function([syms['x'],syms['z2']],from_xz2_det,**kwargs)
            from_xz2_nondet = theano.function([syms['x'],syms['z2']],from_xz2_nondet,**kwargs)

            # to get from z1&z2
            # get y from z1,
            y_from_z1_det = lasagne.layers.get_output(self.y,inputs={self.z1:syms['z1']},deterministic=True)
            y_from_z1_nondet = lasagne.layers.get_output(self.y,inputs={self.z1:syms['z1']},deterministic=False)
            # then get _z1 from y&z2
            _z1_from_yz2_det = lasagne.layers.get_output(self._z1,inputs={self.y:y_from_z1_det,self.z2:syms['z2']},deterministic=True)
            _z1_from_yz2_nondet = lasagne.layers.get_output(self._z1,inputs={self.y:y_from_z1_nondet,self.z2:syms['z2']},deterministic=False)
            # rename
            from_z1z2_det = _z1_from_yz2_det
            from_z1z2_nondet = _z1_from_yz2_nondet
            from_z1z2_det = theano.function([syms['z1'],syms['z2']],from_z1z2_det,**kwargs)
            from_z1z2_nondet = theano.function([syms['z1'],syms['z2']],from_z1z2_nondet,**kwargs)

            # from y&z2
            from_yz2_det = lasagne.layers.get_output(self._z1,inputs={self.y:syms['y'],self.z2:syms['z2']},deterministic=True)
            from_yz2_nondet = lasagne.layers.get_output(self._z1,inputs={self.y:syms['y'],self.z2:syms['z2']},deterministic=False)
            from_yz2_det = theano.function([syms['y'],syms['z2']],from_yz2_det,**kwargs)
            from_yz2_nondet = theano.function([syms['y'],syms['z2']],from_yz2_nondet,**kwargs)

            def get_Z1(x=None,y=None,z1=None,z2=None,deterministic=not self.variational):
                given = (x is not None, y is not None, z1 is not None, z2 is not None)
                given = tuple([int(g) for g in given])
                allowed = set([(1,0,0,0),(0,0,1,0),(1,1,0,0),(0,1,1,0),(1,0,0,1),(0,0,1,1),(0,1,0,1)])
                assert given in allowed, 'S2S_DGM msg: Input ' + repr(given) + ' is not an allowed pairing. [x,y,z1,z2] in ' + repr(allowed)

                if given == (1,0,0,0):
                    if deterministic:
                        return from_x_det(x)
                    else:
                        return from_x_nondet(x)
                elif given == (0,0,1,0):
                    if deterministic:
                        return from_z1_det(z1)
                    else:
                        return from_z1_nondet(z1)
                elif given == (1,1,0,0):
                    if deterministic:
                        return from_xy_det(x,y)
                    else:
                        return from_xy_nondet(x,y)
                elif given == (0,1,1,0):
                    if deterministic:
                        return from_z1y_det(z1,y)
                    else:
                        return from_z1y_nondet(z1,y)
                elif given == (1,0,0,1):
                    if deterministic:
                        return from_xz2_det(x,z2)
                    else:
                        return from_xz2_nondet(x,z2)
                elif given == (0,0,1,1):
                    if deterministic:
                        return from_z1z2_det(z1,z2)
                    else:
                        return from_z1z2_nondet(z1,z2)
                elif given == (0,1,0,1):
                    if deterministic:
                        return from_yz2_det(y,z2)
                    else:
                        return from_yz2_nondet(y,z2)

        return get_Z1

    def _make_get_YZ2(self,model_type):
        if model_type == 0:
            def get_YZ2(*args,**kwargs):
                raise NotImplementedError('S2S_DGM msg: z2 is not a part of the M1 model.')
        elif model_type == 1:
            def get_YZ2(x=None,z1=None,deterministic=not self.variational):
                assert (x is not None) or (z1 is not None), 'S2S_DGM msg: Must provide an input for x/z1'
                if x is None:
                    x = z1

                y = self.getY(x=x,deterministic=deterministic)
                z2 = self.getZ2(x=x,y=y,deterministic=deterministic)
                return y, z2
        else:
            def get_YZ2(x=None,z1=None,deterministic=not self.variational):
                if z1 is None:
                    z1 = self.getZ1(x=x,deterministic=deterministic)
                y = self.getY(z1=z1,deterministic=deterministic)
                z2 = self.getZ2(z1=z1,y=y,deterministic=deterministic)
                return y, z2

        return get_YZ2

    def fit(self,X=None,y=None,X_=None,_y=None,z2=None,
            X_valid=None,y_valid=None,X__valid=None,_y_valid=None,z2_valid=None,
            params=None,variational=None,lr=None,verbose=True,debug=True):

        if self.verbose != verbose:
            self.verbose = not self.verbose
        if self.debug != debug:
            self.debug = not self.debug

        if X_ is None and 'X_' in self.default_data:
            X_ = self.default_data['X_']
        if (_y is None or z2 is None) and ('_y' in self.default_data and 'z2' in self.default_data):
            _y = self.default_data['_y']
            z2 = self.default_data['z2']

        if params is None:
            params = self.model_dict['params']['all']

        if variational is None:
            variational = self.variational
        else:
            self.variational = variational
        typ = self._switch_variational(variational)

        if lr is not None:
            assert isinstance(lr,float) and lr >= 0., 'S2S_DGM msg: learning rate must be positive float.'
            self.lr = lr

        self._build_update(params=params,typ=typ,X=X,y=y,X_=X_,_y=_y,z2=z2,
                           X_valid=X_valid,y_valid=y_valid,X__valid=X__valid,_y_valid=_y_valid,z2_valid=z2_valid)

        data_train, shared_train = self._prep_data(X,y,X_,_y,z2)
        data_valid, shared_valid = self._prep_data(X_valid,y_valid,X__valid,_y_valid,z2_valid,
                                                   validation=True)

        batch_symbols_train = self._batch_symbols('_train')
        batch_symbols_valid = self._batch_symbols('_valid')
        self.model_dict['sym'].update(batch_symbols_train)
        self.model_dict['sym'].update(batch_symbols_valid)
        model_dict = self.model_dict

        inputs_train = self._init_inputs(model_dict,batch_symbols_train)
        inputs_valid = self._init_inputs(model_dict,batch_symbols_valid,validation=True)

        reports_train, outputs_train = self._init_outputs(model_dict,typ=typ,obj_type='train',
                                                          X=X,y=y,X_=X_,_y=_y,z2=z2)
        reports_eval, outputs_eval = self._init_outputs(model_dict,typ=typ,obj_type='eval',
                                                        X=X_valid,y=y_valid,
                                                        X_=X__valid,_y=_y_valid,
                                                        z2=z2_valid)

        givens_train = self._init_givens(model_dict,data_train,shared_train,sym_suffix='_train')
        givens_valid = self._init_givens(model_dict,data_valid,shared_valid,sym_suffix='_valid')

        self.train_model = self._get_outputs_(inputs_train,outputs_train,
                                              givens=givens_train,
                                              updates=self._updates)

        # self.eval_model = self._get_outputs_(inputs_valid+[model_dict['sym']['lr']],outputs_eval,
        #                                      givens=givens_valid,
        #                                      updates=self._updates)

        # Due to some very strange theano behavior, need to include updates as
        # part of eval_model and use zero learning rate
        # Try defining train_model with updates=None to reproduce strange behavior
        # Original eval_model below gave floating point exception on gpu
        self.eval_model = self._get_outputs_(inputs_valid,outputs_eval,
                                             givens=givens_valid,
                                             updates=None)

        # if self.batch_norm:
        #     self.collect_out = lasagne.layers.get_output(self._x,
        #                                                  model_dict['sym']['x'],
        #                                                  deterministic=True,collect=True)
        #     if not (X_ is None and X is None):
        #         if X is None:
        #             batch_X = shared_train[2][:int(min(1000,data_train[2].shape[0]))]
        #         else:
        #             batch_X = shared_train[0][:int(min(1000,data_train[0].shape[0]))]
        #         self.f_collect = theano.function([model_dict['sym']['eq_samples'],model_dict['sym']['iw_samples']],
        #                                           self.collect_out,
        #                                           givens={model_dict['sym']['x']: batch_X})
        #     else:
        #         print('X not available to compute batch statistics. Ignoring batch normalization.')
        #         self.batch_norm = False

        self.f_collect = self._prep_batchnorm(X_,X,shared_train,data_train,model_dict)

        self.train_epoch = self._build_process_epoch(self.train_model,'train')
        self.eval_epoch = self._build_process_epoch(self.eval_model,'eval')

        # build_run_epochs
        self.run_epochs = self._build_run_epochs(self.train_epoch,self.eval_epoch)

        # save model parameters to logfile
        self._log_model_setup()

        # run epoch
        self.run_epochs(self.lr,self.eq_samples,self.iw_samples,
                        data_train,data_valid,
                        shared_train,shared_valid,
                        reports_train,reports_eval,
                        self.batch_size_Xy_train,self.batch_size_X__train,self.batch_size__y_train,
                        self.batch_size_Xy_eval,self.batch_size_X__eval,self.batch_size__y_eval)

    def _build_update(self,params=None,typ='nonvar',X=None,y=None,X_=None,_y=None,z2=None,
                      X_valid=None,y_valid=None,X__valid=None,_y_valid=None,z2_valid=None):
        if params is None:
            params = self.model_dict['params']['all']
        self._build_objectives(self.model_dict,typ=typ,X=X,y=y,X_=X_,_y=_y,z2=z2,
                           X_valid=X_valid,y_valid=y_valid,X__valid=X__valid,_y_valid=_y_valid,z2_valid=z2_valid)
        grads = self._derive_gradient(params,typ=typ)
        self._grads = grads
        self._updates = self._build_updates(params,grads)

    def _build_objectives(self,model_dict,typ='nonvar',X=None,y=None,X_=None,_y=None,z2=None,
                          X_valid=None,y_valid=None,X__valid=None,_y_valid=None,z2_valid=None):
        model_dict['objs'][typ]['train'] = 0
        model_dict['objs'][typ]['eval'] = 0

        if not (X is None or y is None):
            if self.coeff_x_prob > 0 and typ != 'nonvar':
                model_dict['objs'][typ]['train'] += self.coeff_x*T.sum(self.obj_x_sup)
            if self.coeff_y_prob > 0 and typ != 'nonvar':
                model_dict['objs'][typ]['train'] += self.coeff_y*T.sum(self.obj_y_sup)
            if self.coeff_x_dis > 0:
                model_dict['objs'][typ]['train'] += self.coeff_x_dis*T.sum(self.obj_x_sup_dis)
            if self.coeff_y_dis > 0:
                model_dict['objs'][typ]['train'] += self.coeff_y_dis*T.sum(self.obj_y_sup_dis)
        if not (X_valid is None or y_valid is None):
            if self.coeff_x_prob > 0 and typ != 'nonvar':
                model_dict['objs'][typ]['eval'] += self.coeff_x*T.sum(self.obj_x_sup)
            if self.coeff_y_prob > 0 and typ != 'nonvar':
                model_dict['objs'][typ]['eval'] += self.coeff_y*T.sum(self.obj_y_sup)
            if self.coeff_x_dis > 0:
                model_dict['objs'][typ]['eval'] += self.coeff_x_dis*T.sum(self.obj_x_sup_dis)
            if self.coeff_y_dis > 0:
                model_dict['objs'][typ]['eval'] += self.coeff_y_dis*T.sum(self.obj_y_sup_dis)
        if X_ is not None:
            if self.coeff_x > 0:
                model_dict['objs'][typ]['train'] += self.coeff_x*T.sum(self.obj_x)
        if X__valid is not None:
            if self.coeff_x > 0:
                model_dict['objs'][typ]['eval'] += self.coeff_x*T.sum(self.obj_x)
        if _y is not None and (self.model_type == 0 or z2 is not None):
            if self.coeff_y > 0:
                model_dict['objs'][typ]['train'] += self.coeff_y*T.sum(self.obj_y)
        if _y_valid is not None and (self.model_type == 0 or z2_valid is not None):
            if self.coeff_y > 0:
                model_dict['objs'][typ]['eval'] += self.coeff_y*T.sum(self.obj_y)

    def _derive_gradient(self,params=None,typ='nonvar'):
        if params is None:
            params = self.model_dict['params']['all']
        grads = T.grad(self.model_dict['objs'][typ]['train'],params,disconnected_inputs='ignore')
        return grads

    def _build_updates(self,params,grads,clip_grad=1.,max_norm=5.):
        mgrads = lasagne.updates.total_norm_constraint(grads,max_norm=max_norm)
        cgrads = [T.clip(g, -clip_grad, clip_grad) for g in mgrads]
        sym_lr = self.model_dict['sym']['lr']
        updates = lasagne.updates.adam(cgrads,params,beta1=0.9,beta2=0.999,
                                       epsilon=1e-4,learning_rate=sym_lr)
        return updates

    def _prep_data(self,X=None,y=None,X_=None,_y=None,z2=None,validation=False):
        data = []
        shared = []
        for datum in [X,y,X_,_y,z2]:
            if datum is None:
                data += [datum]
                shared += [datum]
            else:
                datum_casted = datum.astype(theano.config.floatX)
                data += [datum_casted]
                shared += [theano.shared(datum_casted,borrow=True)]
        if not validation:
            sh_temp = theano.shared(np.cast[theano.config.floatX](0.),borrow=True)
            shared += [sh_temp]

        return data, shared

    def _batch_symbols(self,suffix='_train'):
        sym_index = T.iscalar('index'+suffix)
        sym_batch_size_Xy = T.iscalar('batch_size_Xy'+suffix)
        sym_batch_size_X_ = T.iscalar('batch_size_X_'+suffix)
        sym_batch_size__y = T.iscalar('batch_size__y'+suffix)
        sym_names = ['index'+suffix,'batch_size_Xy'+suffix,
                     'batch_size_X_'+suffix,'batch_size__y'+suffix]
        syms = [sym_index,sym_batch_size_Xy,sym_batch_size_X_,sym_batch_size__y]
        return OrderedDict(zip(sym_names,syms))

    def _init_inputs(self,model_dict,batch_symbols,validation=False):
        # index, bs_Xy, bs_X_, bs__y, lr, eq_samp, iw_samp
        syms = model_dict['sym']
        others = [syms[key] for key in ('eq_samples','iw_samples')]
        if not validation:
            others += [syms['lr']]
        inputs = list(batch_symbols.values()) + others
        return inputs

    def _init_outputs(self,model_dict,typ='nonvar',obj_type='train',X=None,y=None,X_=None,_y=None,z2=None):
        objs = model_dict['objs'][typ]
        reports = []

        if not (X is None or y is None) or X_ is not None or _y is not None:
            reports += [obj_type]

        if not (X is None or y is None):
            if self.coeff_x_dis > 0:
                reports += [k for k in objs.keys() if 'x' in k and 'sup_dis' in k]
            if self.coeff_y_dis > 0:
                reports += [k for k in objs.keys() if 'y' in k and 'sup_dis' in k]
            if typ != 'nonvar':
                if self.coeff_x_prob > 0:
                    reports += [k for k in objs.keys() if 'x' in k and 'sup' in k and 'dis' not in k]
                if self.coeff_y_prob > 0:
                    reports += [k for k in objs.keys() if 'y' in k and 'sup' in k and 'dis' not in k]
        if X_ is not None:
            if self.coeff_x > 0:
                reports += [k for k in objs.keys() if ('M1_x' in k or 'M2_x' in k) and 'sup' not in k]
        if _y is not None and (self.model_type == 0 or z2 is not None):
            if self.coeff_y > 0:
                reports += [k for k in objs.keys() if ('M1_y' in k or 'M2_y' in k) and 'sup' not in k]

        outputs = [objs[k] for k in reports]
        return reports, outputs

    def _init_givens(self,model_dict,data,shared,sym_suffix='_train'):
        keys = ['x_sup','y_sup','x','y','z2']
        syms = model_dict['sym']

        givens = {}
        for key,d,sh in zip(keys,data,shared[:5]):
            if sh is not None:
                givens.update({syms[key]:self._get_slice(d,sh,key,syms,sym_suffix)})

        if len(shared) > 5:
            sh_temp = shared[5]
            givens.update({syms['temp']:sh_temp})

        return givens

    def _get_slice(self,d,sh,sym,syms,sym_suffix='_train'):
        N_samples = d.shape[0]
        sym_index = syms['index'+sym_suffix]

        # retrieve corresponding batch size
        if sym in ('x_sup','y_sup'):
            sym_batch_size = syms['batch_size_Xy'+sym_suffix]
        elif sym == 'x':
            sym_batch_size = syms['batch_size_X_'+sym_suffix]
        elif sym in ('y','z2'):
            sym_batch_size = syms['batch_size__y'+sym_suffix]
        else:
            raise ValueError('S2S_DGM msg: symbol '+str(sym)+' is not a valid option')

        # handle wraparound indexing
        start = sym_index * sym_batch_size % N_samples
        end = T.clip(start + sym_batch_size, 0, N_samples)
        q,r = divmod(sym_batch_size-(end-start), N_samples)
        # theano doesn't support tiling `0` times, so need ifelse
        batch = ifelse(T.gt(q,0),
                       T.concatenate([sh[slice(start,end)]]+[T.tile(sh,(q,1))]+[sh[slice(0,r)]]),
                       T.concatenate([sh[slice(start,end)]]+[sh[slice(0,r)]]),
                       name=sym+sym_suffix)

        return batch

    def _get_outputs_(self,inputs,outputs,givens,updates=None,i=None):
        if i is not None and i < len(outputs):
            outputs = outputs[i]
        if givens is not None:
            out = theano.function(inputs,outputs,
                                  givens=givens,
                                  updates=updates,
                                  on_unused_input='ignore')
            return out
        else:
            return None

    def _prep_batchnorm(self,X_,X,shared_train,data_train,model_dict):
        # collects statistics using up to 10 x (# of samples in training batch)
        if self.batch_norm:
            self.collect_out = lasagne.layers.get_output(self._x,
                                                         model_dict['sym']['x'],
                                                         deterministic=True,collect=True)
            if not (X_ is None and X is None):
                if X is None:
                    N_samples = int(min(10*self.batch_size_X__train,data_train[2].shape[0]))
                    batch_X = shared_train[2][:N_samples]
                else:
                    N_samples = int(min(10*self.batch_size_Xy_train,data_train[0].shape[0]))
                    batch_X = shared_train[0][:N_samples]
                return theano.function([model_dict['sym']['eq_samples'],model_dict['sym']['iw_samples']],
                                       self.collect_out,
                                       givens={model_dict['sym']['x']: batch_X})
            else:
                print('X not available to compute batch statistics. Ignoring batch normalization.')
                self.batch_norm = False
                return lambda eq,iw: None

    def _build_process_epoch(self,process_model,process_type='train'):
        def _process_epoch(reports,
                           lr,eq_samples,iw_samples,
                           N_Xy=None,N_X_=None,N__y=None,
                           batch_size_Xy=None,batch_size_X_=None,batch_size__y=None):
            if process_type in reports:
                Ns = [N_Xy,N_X_,N__y]
                batch_sizes = [batch_size_Xy,batch_size_X_,batch_size__y]
                n_batch_opts = [float(a)/b for a,b in zip(Ns,batch_sizes) if not (a is None or b is None)]
                n_batches = int(np.max([np.ceil(n_batch_opts)]))
                empty_lists = [[] for i in range(len(reports))]
                record = OrderedDict(zip(reports,empty_lists))

                if process_type == 'eval' and self.batch_norm:
                    if self.verbose:
                        print('Computing batch normalization statistics...')
                    _ = self.f_collect(1,1)  # collect BN stats on train
                    # _ is required, otherwise function will be removed from theano graph
                    # because theano sees output is unused

                NaN = False

                for b in range(n_batches):
                    # index, bs_Xy, bs_X_, bs__y, eq_samp, iw_samp, lr
                    if process_type == 'train':
                        outs = process_model(b, batch_size_Xy, batch_size_X_, batch_size__y,
                                             eq_samples, iw_samples, lr)
                    else:
                        # outs = process_model(b, batch_size_Xy, batch_size_X_, batch_size__y,
                        #                      eq_samples, iw_samples, 0.)
                        outs = process_model(b, batch_size_Xy, batch_size_X_, batch_size__y,
                                             eq_samples, iw_samples)
                    for idx,out in enumerate(outs):
                        record[reports[idx]] += [out]

                if np.any(np.isnan(record[process_type])):
                    NaN = True
                    if self.verbose:
                        print('NaN encountered in ('+process_type+') cost!')
                    if self.debug:
                        embed()

                results = OrderedDict()
                for key in reports:
                    result = record[key][0]
                    if isinstance(result,np.ndarray):
                        if result.shape == ():
                            results[key] = np.mean(record[key])
                        else:
                            results[key] = np.concatenate(record[key])
                    elif isinstance(result,float):
                        results[key] = np.mean(record[key])
                    else:
                        raise NotImplementedError('S2S_DGM msg: Not sure what to do with output type')
                return results, NaN
            else:
                return OrderedDict(), False
        return _process_epoch

    def _build_run_epochs(self,train_epoch,eval_epoch):
        def _run_epochs(lr,eq_samples,iw_samples,
                        data_train,data_valid,
                        shared_train,shared_valid,
                        reports_train,reports_eval,
                        batch_size_Xy_train,batch_size_X__train,batch_size__y_train,
                        batch_size_Xy_eval,batch_size_X__eval,batch_size__y_eval):

            sh_temp = shared_train[-1]

            N_Xy_train, N_X__train, N__y_train = [None if data_train[i] is None else data_train[i].shape[0] for i in [0,2,3]]
            N_Xy_eval, N_X__eval, N__y_eval = [None if data_valid[i] is None else data_valid[i].shape[0] for i in [0,2,3]]

            # init records
            train_record = self._init_record(reports_train)
            eval1_record = self._init_record(reports_eval)
            evalN_record = self._init_record(reports_eval)

            xepochs = []
            for epoch in range(1, 1+self.num_epochs):
                t0 = time.time()

                self._shuffle_data(data_train,shared_train)
                self._shuffle_data(data_valid,shared_valid)

                # set temperature
                # temperature = min(1.,.01+(epoch-1)/10000.)
                temperature = 1.
                sh_temp.set_value(temperature)

                train_out, NaN = train_epoch(reports_train,
                                             lr,eq_samples,iw_samples,
                                             N_Xy_train,N_X__train,N__y_train,
                                             batch_size_Xy_train,batch_size_X__train,batch_size__y_train)
                if NaN:
                    break

                if epoch >= self.anneal_lr_epoch:
                    # annealing learning rate
                    lr = lr*self.anneal_lr_factor

                if epoch % self.eval_freq == 0:
                    t = time.time() - t0

                    for key,val in train_out.items():
                        train_record[key] += [val]

                    if self.verbose:
                        print('\ncalculating LL eq=1, iw=1')
                    eval1_out, NaN = eval_epoch(reports_eval,
                                                lr,1,1,
                                                N_Xy_eval,N_X__eval,N__y_eval,
                                                batch_size_Xy_eval,batch_size_X__eval,batch_size__y_eval)
                    for key,val in eval1_out.items():
                        eval1_record[key] += [val]

                    if self.verbose:
                        print('calculating LL eq=1, iw=5')
                    # smaller batch size to reduce memory requirements
                    evalN_out, NaN = eval_epoch(reports_eval,
                                                lr,1,5,
                                                N_Xy_eval,N_X__eval,N__y_eval,
                                                batch_size_Xy_eval//5,batch_size_X__eval//5,batch_size__y_eval//5)
                    for key,val in evalN_out.items():
                        evalN_record[key] += [val]

                    xepochs += [epoch]

                    self._log(epoch,t,lr,eq_samples,iw_samples,train_out,eval1_out,evalN_out)

                    # save model every 1000'th epochs
                    if epoch % 1000 == 0:

                        if os.path.isfile(self.model_out + 'epoch%i' % (epoch-1)):
                            os.remove(self.model_out + 'epoch%i' % (epoch-1))
                        f = gzip.open(self.model_out + 'epoch%i' % (epoch), 'wb')
                        cPickle.dump(self.model_dict['params']['all'], f, protocol=cPickle.HIGHEST_PROTOCOL)
                        f.close()

                    self._save_plots(xepochs,train_record,eval1_record,evalN_record)

            self.train_record = train_record
            self.eval1_record = eval1_record
            self.evalN_record = evalN_record
        return _run_epochs

    def _init_record(self,reports):
        empty_lists = [[] for i in range(len(reports))]
        record = OrderedDict(zip(reports,copy.deepcopy(empty_lists)))
        return record

    def _shuffle_data(self,data,shared):
        # data: [X,y,X_,_y,z2]
        if data[0] is not None:
            indices_Xy = np.arange(data[0].shape[0])
            np.random.shuffle(indices_Xy)
        elif data[1] is not None:
            indices_Xy = np.arange(data[1].shape[0])
            np.random.shuffle(indices_Xy)
        if data[3] is not None:
            indices__yz2 = np.arange(data[3].shape[0])
            np.random.shuffle(indices__yz2)
        elif data[4] is not None:
            indices__yz2 = np.arange(data[4].shape[0])
            np.random.shuffle(indices__yz2)
        for idx,datum in enumerate(data):
            if datum is not None:
                if idx in [0,1]:
                    shared[idx].set_value(datum[indices_Xy])
                elif idx in [3,4]:
                    shared[idx].set_value(datum[indices__yz2])
                else:
                    np.random.shuffle(datum)
                    shared[idx].set_value(datum)

    def _log(self,epoch,t,lr,eq_samples,iw_samples,train_out,eval1_out,evalN_out):
        epoch_format = "\n*Epoch=%d\tTime=%.2f\tLR=%.5f\teq_samples=%d\t" + \
                        "iw_samples=%d\n"
        epoch_vals = (epoch, t, lr, eq_samples, iw_samples)
        with open(self.logfile,'a') as f:
            f.write(epoch_format % epoch_vals)

        sc_train = dict((k,v) for k,v in train_out.items() if 'l_' not in k)
        sc_eval1 = dict((k,v) for k,v in eval1_out.items() if 'l_' not in k)
        sc_evalN = dict((k,v) for k,v in evalN_out.items() if 'l_' not in k)

        if self.verbose:
            print(epoch_format % epoch_vals)
            toScreen(prefix='Train_',mydict=sc_train)
            toScreen(prefix='Eval1_',mydict=sc_eval1)
            toScreen(prefix='EvalN_',mydict=sc_evalN)

        logout = toLog(prefix='Train_',mydict=sc_train)
        logout += toLog(prefix='Eval1_',mydict=sc_eval1)
        logout += toLog(prefix='EvalN_',mydict=sc_evalN)
        with open(self.logfile, 'a') as f:
            f.write(logout)

    def _save_plots(self,xepochs,train_record,eval1_record,evalN_record):
        plt.figure(figsize=[12,12])
        for key,val in train_record.items():
            if not isinstance(val[0],np.ndarray):
                plt.plot(xepochs,val,'o-',label=key)
        plt.xlabel('Epochs')
        plt.ylabel('log()')
        plt.grid('on')
        plt.title('Train')
        lgd = plt.legend(loc='center left',bbox_to_anchor=(1.05, 1))
        plt.savefig(self.res_out+'/train.png',additional_artists=[lgd],bbox_inches='tight')
        plt.close()

        plt.figure(figsize=[12,12])
        for key,val in eval1_record.items():
            if not isinstance(val[0],np.ndarray):
                plt.plot(xepochs,val,'o-',label=key)
        plt.xlabel('Epochs')
        plt.ylabel('log()')
        plt.grid('on')
        plt.title('Eval L1')
        lgd = plt.legend(loc='center left',bbox_to_anchor=(1.05, 1))
        plt.savefig(self.res_out+'/eval_L1.png',additional_artists=[lgd],bbox_inches='tight')
        plt.close()

        plt.figure(figsize=[12,12])
        for key,val in evalN_record.items():
            if not isinstance(val[0],np.ndarray):
                plt.plot(xepochs,val,'o-',label=key)
        plt.xlabel('Epochs')
        plt.ylabel('log()'), plt.grid('on')
        plt.title('Eval LN')
        lgd = plt.legend(loc='center left',bbox_to_anchor=(1.05, 1))
        plt.savefig(self.res_out+'/eval_LN.png',additional_artists=[lgd],bbox_inches='tight')
        plt.close()












import numpy as np
import theano
import theano.tensor as T
theano.config.floatX = 'float32'
theano.config.exception_verbosity = 'high'
theano.config.traceback.limit = 20
import lasagne
from lasagne.layers.helper import get_all_layers
from untapped.parmesan.layers import SampleLayer, BernoulliSampleLayer, ConcreteSampleLayer, ListIndexLayer
from untapped.parmesan.distributions import log_normal as log_norm  # affects SampleLayer nonlinearity argument
from untapped.parmesan.distributions import log_bernoulli as log_bern
from untapped.parmesan.distributions import log_gumbel_softmax as log_gumsoft
from collections import OrderedDict
from scipy.special import gammaln
import copy


class M12(object):
    """
    Create M1 & M2 Models from "Semi-supervised Learning with Deep Generative
    Models"
    """
    def __init__(self,l_input,nets):
        assert isinstance(nets,dict)
        M1 = set(nets.keys()) == set(['x->z','z->_x'])
        M2 = set(nets.keys()) == set(['x->y','xy->z','yz->_x'])
        assert M1 or M2
        net_layers = {key:get_all_layers(layer)
                      for key,layer in nets.items()
                      if layer is not None}
        if M1:
            x = l_input
            net_layers['x->z'][0].input_layer = x
            z = net_layers['x->z'][-1]
            net_layers['z->_x'][0].input_layer = z
        else:
            x = l_input
            net_layers['x->y'][0].input_layer = x
            xy = [l_input,net_layers['x->y'][-1]]
            xy = lasagne.layers.ConcatLayer(xy,axis=-1)
            net_layers['xy->z'][0].input_layer = xy
            yz = [net_layers['x->y'][-1],net_layers['xy->z'][-1]]
            yz = lasagne.layers.ConcatLayer(yz,axis=-1)
            net_layers['yz->_x'][0].input_layer = yz

        self.net_layers = net_layers

        # make layer names unique and build name map
        name_map = {net:None for net in net_layers.keys()}
        for net, layers in net_layers.items():
            names = [layer.name for layer in layers]
            # ensure uniqueness
            if len(names) != len(np.unique(names)):
                counts = [1]*len(np.unique(names))
                seen = set()
                names = []
                for layer in layers:
                    if layer.name not in seen:
                        seen.add(layer.name)
                        names += [layer.name]
                    else:
                        layer.name = layer.name + str(counts[layer.name])
                        names += layer.name
                        counts[layer.name] += 1
            # build name map
            name_map[net] = {name:layers[idx] for idx,name in enumerate(names)}
        self.name_map = name_map

    def getByName(self,net,names):
        if isinstance(names,list):
            return [self.name_map[net][name] for name in names]
        else:
            assert isinstance(names,str)
            return self.name_map[net][names]


def FlowNet(input_shape,sym_eq_samples,sym_iw_samples,hidden,gain,nonlin,
            num_output,sample_layer,flows,batch_norm):
    """
    Creates deep net --> sample layer --> flows
    """
    params = {}

    l_new = input_shape
    glorot_uni = lasagne.init.GlorotUniform
    for n,h in enumerate(hidden):
        l_new = lasagne.layers.DenseLayer(l_new,
                                          num_units=h,
                                          W=glorot_uni(gain),
                                          nonlinearity=nonlin,
                                          name='l_'+str(n))
        if batch_norm:
            l_new = lasagne.layers.batch_norm(l_new)
    if len(hidden) == 0:
        l_new = IdentityLayer(l_new,name='identity')

    l_z, l_mu = sample(l_new,num_output,sym_eq_samples,sym_iw_samples,sample_layer,gain)

    params['nonvar'] = lasagne.layers.get_all_params([l_mu],trainable=True)

    # Normalizing Flow
    l_invs = []
    l_zk = l_z
    for idx,flow in enumerate(flows):
        l_nf = flow(l_zk,name='l_nf_'+str(idx))
        l_zk = ListIndexLayer(l_nf,index=0,name='l_z'+str(idx+1))
        if l_nf.inverse is not None:
            l_invs += [l_nf.inverse]

    l_z0 = lasagne.layers.InputLayer((None, num_output))
    if len(l_invs) > 0:
        assert len(l_invs) == len(flows)
        for _idx,l_inv in enumerate(l_invs[::-1]):
            idx = len(l_invs) - _idx - 1
            inv, arg = l_inv
            l_z0 = inv(l_z0,arg,name='l_if_'+str(idx))
    else:
        l_z0 = IdentityLayer(l_z0,name='l_if_0')

    params['all'] = lasagne.layers.get_all_params([l_zk])
    params['var'] = list(set(params['all'])-set(params['nonvar']))

    return l_zk, l_z0, params


def sample(l_new,num_output,sym_eq_samples,sym_iw_samples,sample_layer,gain):
    glorot_uni = lasagne.init.GlorotUniform
    if sample_layer is SampleLayer:
        l_mu = lasagne.layers.DenseLayer(l_new,num_units=num_output,W=glorot_uni(gain),
                                     nonlinearity=None,name='l_mu')
        l_sigma = lasagne.layers.DenseLayer(l_new,num_units=num_output,W=lasagne.init.Constant(0),
                                            nonlinearity=theano.tensor.nnet.softplus,
                                            name='l_log_var',
                                            b=lasagne.init.Constant(0))
        # Sample layer
        l_z = SampleLayer(mean=l_mu,log_var=l_sigma,
                          eq_samples=sym_eq_samples,iw_samples=sym_iw_samples,
                          nonlinearity=lambda x: x,
                          name='l_z')
    elif sample_layer is BernoulliSampleLayer:
        l_mu = lasagne.layers.DenseLayer(l_new,num_units=num_output,W=glorot_uni(gain),
                                        nonlinearity=theano.tensor.nnet.sigmoid,
                                        name='l_mu',
                                        b=lasagne.init.Constant(0))
        # Sample layer
        l_z = BernoulliSampleLayer(mean=l_mu,
                                   eq_samples=sym_eq_samples,iw_samples=sym_iw_samples,
                                   name='l_z')
    elif sample_layer is ConcreteSampleLayer:
        l_logits = lasagne.layers.DenseLayer(l_new,num_units=num_output,W=glorot_uni(gain),
                                             nonlinearity=None,
                                             name='l_logits',
                                             b=lasagne.init.Constant(0))
        # Sample layer
        l_z = ConcreteSampleLayer(logits=l_logits,
                                  eq_samples=sym_eq_samples,iw_samples=sym_iw_samples,
                                  name='l_z')
        l_mu = l_logits
    else:
        raise NotImplementedError(sample_layer+' is not an implemented option.')

    return l_z, l_mu


class IdentityLayer(lasagne.layers.Layer):
    """
    Dummy layer that returns input
    """
    def __init__(self, incoming, **kwargs):
        super(IdentityLayer, self).__init__(incoming, **kwargs)

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, **kwargs):
        return input


def M1(M_input,model_dict,variational=True,
       prior_x=None,prior_z1=None,coeff_sup=1.,
       loss_x=None,loss_y=None):
    ''' Build Model '''
    syms = model_dict['sym']
    sym_x = syms['x']
    sym_y = syms['y']
    sym_x_sup = syms['x_sup']
    sym_y_sup = syms['y_sup']
    sym_eq_samples = syms['eq_samples']
    sym_iw_samples = syms['iw_samples']
    det = not variational
    try:
        # Latent recognition model q(z|x)
        x_shape = M_input.output_shape
        q_z_x, z0, params_x_z1 = FlowNet(x_shape,sym_eq_samples,sym_iw_samples,
                                         **model_dict['x->z1']['arch'])
        model_dict['params']['x->z1'] = params_x_z1['all']
        model_dict['params']['x->z1__nonvar'] = params_x_z1['nonvar']
        model_dict['params']['x->z1__var'] = params_x_z1['var']

        # Generative model p(x|z)
        z_shape = q_z_x.output_shape
        q_x_z, x0, params_z1__x = FlowNet(z_shape,sym_eq_samples,sym_iw_samples,
                                          **model_dict['z1->_x']['arch'])
        model_dict['params']['z1->_x'] = params_x_z1['all']
        model_dict['params']['z1->_x__nonvar'] = params_z1__x['nonvar']
        model_dict['params']['z1->_x__var'] = params_z1__x['var']

        model_dict['params']['M1'] = model_dict['params']['x->z1'] + model_dict['params']['z1->_x']
        model_dict['params']['M1_nonvar'] = model_dict['params']['x->z1__nonvar'] + \
                                            model_dict['params']['z1->_x__nonvar']
        model_dict['params']['M1_var'] = model_dict['params']['x->z1__var'] + \
                                         model_dict['params']['z1->_x__var']

        # Connect model components
        nets_M1 = OrderedDict([('x->z',q_z_x),('z->_x',q_x_z)])
        model = M12(M_input,nets_M1)
    except Exception as e:
        print('Exception: '+repr(e))
        print('M1 build failed!')
        return False

    ''' Get Key Layers '''
    layers_x_z1, z1_name, names_x_z1 = get_key_layers_names(model,model_dict['x->z1']['arch'],'x->z')
    layers_z1__x, _x_name, names_z1__x = get_key_layers_names(model,model_dict['z1->_x']['arch'],'z->_x')

    # Record name lists
    zipped = zip(names_x_z1,layers_x_z1)
    model_dict['x->z1']['key_layers'] = OrderedDict(zipped)
    zipped = zip(names_z1__x,layers_z1__x)
    model_dict['z1->_x']['key_layers'] = OrderedDict(zipped)

    ''' Get Key Layer Outputs '''
    model_dict['x->z1']['key_outputs'] = OrderedDict()
    model_dict['z1->_x']['key_outputs'] = OrderedDict()

    # x-unsupervised
    # encode
    model_outputs(model_dict,'x->z1','nondet_x',inputs=sym_x,deterministic=False)
    model_outputs(model_dict,'x->z1','det_x',inputs=sym_x,deterministic=True)
    # decode
    model_outputs(model_dict,'z1->_x','nondet_x',inputs=sym_x,deterministic=False)
    model_outputs(model_dict,'z1->_x','det_x',inputs=sym_x,deterministic=True)
    
    # y-unsupervised
    z1 = model_dict['x->z1']['key_layers'][z1_name]
    # encode
    model_outputs(model_dict,'z1->_x','nondet_y',inputs={z1:sym_y},deterministic=False)
    model_outputs(model_dict,'z1->_x','det_y',inputs={z1:sym_y},deterministic=True)
    # decode
    _x_out = model_dict['z1->_x']['key_outputs']['nondet_y'][_x_name]
    model_outputs(model_dict,'x->z1','nondet_y',inputs=_x_out,deterministic=False)
    _x_out = model_dict['z1->_x']['key_outputs']['det_y'][_x_name]
    model_outputs(model_dict,'x->z1','det_y',inputs=_x_out,deterministic=True)

    # supervised
    # x -> y
    model_outputs(model_dict,'x->z1','nondet_x_sup',inputs=sym_x_sup,deterministic=False)
    model_outputs(model_dict,'x->z1','det_x_sup',inputs=sym_x_sup,deterministic=True)
    # y -> x
    model_outputs(model_dict,'z1->_x','nondet_y_sup',
                  inputs={z1:sym_y_sup},deterministic=False)
    model_outputs(model_dict,'z1->_x','det_y_sup',
                  inputs={z1:sym_y_sup},deterministic=True)

    ''' Create Objective Components '''
    x_num_units = model_dict['z1->_x']['arch']['num_output']
    y_num_units = model_dict['x->z1']['arch']['num_output']

    # x -> z1 layer
    model_dict['x->z1']['key_objs'] = OrderedDict()

    # variational (recognition loss, x->y supervised loss,
    #              generative loss for inverse net,
    #              supervised generative loss for inverse net)
    model_dict['x->z1']['key_objs']['var'] = OrderedDict()

    # recognition loss
    layer_outs = model_dict['x->z1']['key_outputs']['nondet_x']
    nondet_rec_z1 = LL_rec(1,layer_outs,y_num_units,sym_eq_samples,sym_iw_samples,power=1)
    model_dict['x->z1']['key_objs']['var']['nondet_x'] = nondet_rec_z1

    layer_outs = model_dict['x->z1']['key_outputs']['det_x']
    det_rec_z1 = LL_rec(1,layer_outs,y_num_units,sym_eq_samples,sym_iw_samples,power=1)
    model_dict['x->z1']['key_objs']['var']['det_x'] = det_rec_z1

    # x->y supervised loss
    layer_outs = model_dict['x->z1']['key_outputs']['nondet_x_sup']
    nondet_z1_out = layer_outs[z1_name]
    nondet_sup = sup_error(nondet_z1_out,sym_y_sup,y_num_units,
                           sym_eq_samples,sym_iw_samples,loss=loss_y)
    model_dict['x->z1']['key_objs']['var']['nondet_sup'] = nondet_sup

    layer_outs = model_dict['x->z1']['key_outputs']['det_x_sup']
    det_z1_out = layer_outs[z1_name]
    det_sup = sup_error(det_z1_out,sym_y_sup,y_num_units,
                        sym_eq_samples,sym_iw_samples,loss=loss_y)
    model_dict['x->z1']['key_objs']['var']['det_sup'] = det_sup

    # generative loss for inverse net
    sym_y0 = lasagne.layers.get_output(z0,inputs=sym_y,deterministic=True)

    x_layer_outs = model_dict['z1->_x']['key_outputs']['nondet_y']
    _y_layer_outs = model_dict['x->z1']['key_outputs']['nondet_y']
    nondet_gen_y = LL_gen(1,x_layer_outs,_y_layer_outs,sym_y0,
                          x_num_units,y_num_units,
                          sym_eq_samples,sym_iw_samples,power=2,priors=prior_x)
    model_dict['x->z1']['key_objs']['var']['nondet_y'] = nondet_gen_y

    x_layer_outs = model_dict['z1->_x']['key_outputs']['det_y']
    _y_layer_outs = model_dict['x->z1']['key_outputs']['det_y']
    det_gen_y = LL_gen(1,x_layer_outs,_y_layer_outs,sym_y0,
                       x_num_units,y_num_units,
                       sym_eq_samples,sym_iw_samples,power=2,priors=prior_x)
    model_dict['x->z1']['key_objs']['var']['det_y'] = det_gen_y

    # supervised generative loss for inverse net
    sym_y0 = lasagne.layers.get_output(z0,inputs=sym_y_sup,deterministic=True)

    # x_layer_outs = model_dict['z1->_x']['key_outputs']['nondet_y_sup']
    # x_layer_outs = {'l_sample':sym_x_sup*T.ones_like(x_layer_outs['l_sample'])}
    # x_layer_outs = {'l_sample':sym_x_sup*T.ones_like(sym_x_sup)}
    _y_layer_outs = model_dict['x->z1']['key_outputs']['nondet_x_sup']
    # nondet_gen_y = LL_gen(1,x_layer_outs,_y_layer_outs,sym_y0,
    #                       x_num_units,y_num_units,
    #                       sym_eq_samples,sym_iw_samples,priors=prior_x)
    nondet_gen_y = LL_gen(1,[],_y_layer_outs,sym_y0,
                          None,y_num_units,
                          sym_eq_samples,sym_iw_samples,power=1,priors=None)
    model_dict['x->z1']['key_objs']['var']['nondet_y_sup'] = nondet_gen_y

    # x_layer_outs = model_dict['z1->_x']['key_outputs']['det_y_sup']
    # x_layer_outs = {'l_sample':sym_x_sup*T.ones_like(x_layer_outs['l_sample'])}
    _y_layer_outs = model_dict['x->z1']['key_outputs']['det_x_sup']
    # det_gen_y = LL_gen(1,x_layer_outs,_y_layer_outs,sym_y0,
    #                    x_num_units,y_num_units,
    #                    sym_eq_samples,sym_iw_samples,priors=prior_x)
    det_gen_y = LL_gen(1,[],_y_layer_outs,sym_y0,
                       None,y_num_units,
                       sym_eq_samples,sym_iw_samples,power=1,priors=None)
    model_dict['x->z1']['key_objs']['var']['det_y_sup'] = det_gen_y

    # non-variational (x->y supervised loss,reconstruction error for inverse net)
    model_dict['x->z1']['key_objs']['nonvar'] = OrderedDict()

    # x->y supervised error
    model_dict['x->z1']['key_objs']['nonvar']['det_sup'] = model_dict['x->z1']['key_objs']['var']['det_sup']

    # reconstruction error for inverse net
    _y_layer_outs = model_dict['x->z1']['key_outputs']['det_y']
    det_z1_out = _y_layer_outs[z1_name]
    det_gen_z1 = sup_error(det_z1_out,sym_y,y_num_units,
                           sym_eq_samples**2,sym_iw_samples**2,
                           loss=loss_y)
    model_dict['x->z1']['key_objs']['nonvar']['det_y'] = det_gen_z1

    # z1 -> _x layer
    model_dict['z1->_x']['key_objs'] = OrderedDict()

    # variational (generative loss, supervised generative loss,
    #              y->x supervised loss,
    #              recognition loss for inverse net)
    model_dict['z1->_x']['key_objs']['var'] = OrderedDict()

    # generative loss
    sym_x0 = lasagne.layers.get_output(x0,inputs=sym_x,deterministic=True)

    y_layer_outs = model_dict['x->z1']['key_outputs']['nondet_x']
    _x_layer_outs = model_dict['z1->_x']['key_outputs']['nondet_x']
    nondet_gen_x = LL_gen(1,y_layer_outs,_x_layer_outs,sym_x0,
                          y_num_units,x_num_units,
                          sym_eq_samples,sym_iw_samples,power=2,priors=prior_z1)
    model_dict['z1->_x']['key_objs']['var']['nondet_x'] = nondet_gen_x

    y_layer_outs = model_dict['x->z1']['key_outputs']['det_x']
    _x_layer_outs = model_dict['z1->_x']['key_outputs']['det_x']
    det_gen_x = LL_gen(1,y_layer_outs,_x_layer_outs,sym_x0,
                       y_num_units,x_num_units,
                       sym_eq_samples,sym_iw_samples,power=2,priors=prior_z1)
    model_dict['z1->_x']['key_objs']['var']['det_x'] = det_gen_x

    # supervised generative loss
    sym_x0 = lasagne.layers.get_output(x0,inputs=sym_x_sup,deterministic=True)

    # y_layer_outs = model_dict['x->z1']['key_outputs']['nondet_x_sup']
    # y_layer_outs = {'l_sample':sym_y_sup*T.ones_like(y_layer_outs['l_sample'])}
    _x_layer_outs = model_dict['z1->_x']['key_outputs']['nondet_y_sup']
    # nondet_gen_x = LL_gen(1,y_layer_outs,_x_layer_outs,sym_x0,
    #                       y_num_units,x_num_units,
    #                       sym_eq_samples,sym_iw_samples,priors=prior_z1)
    nondet_gen_x = LL_gen(1,[],_x_layer_outs,sym_x0,
                          None,x_num_units,
                          sym_eq_samples,sym_iw_samples,power=1,priors=None)
    model_dict['z1->_x']['key_objs']['var']['nondet_x_sup'] = nondet_gen_x

    # y_layer_outs = model_dict['x->z1']['key_outputs']['det_x_sup']
    # y_layer_outs = {'l_sample':sym_y_sup*T.ones_like(y_layer_outs['l_sample'])}
    _x_layer_outs = model_dict['z1->_x']['key_outputs']['det_y_sup']
    # det_gen_x = LL_gen(1,y_layer_outs,_x_layer_outs,sym_x0,
    #                    y_num_units,x_num_units,
    #                    sym_eq_samples,sym_iw_samples,priors=prior_z1)
    det_gen_x = LL_gen(1,[],_x_layer_outs,sym_x0,
                       None,x_num_units,
                       sym_eq_samples,sym_iw_samples,power=1,priors=None)
    model_dict['z1->_x']['key_objs']['var']['det_x_sup'] = det_gen_x

    # y->x supervised loss
    layer_outs = model_dict['z1->_x']['key_outputs']['nondet_y_sup']
    nondet_x_out = layer_outs[_x_name]
    nondet_sup = sup_error(nondet_x_out,sym_x_sup,x_num_units,
                           sym_eq_samples,sym_iw_samples,loss=loss_x)
    model_dict['z1->_x']['key_objs']['var']['nondet_sup'] = nondet_sup

    layer_outs = model_dict['z1->_x']['key_outputs']['det_y_sup']
    det_x_out = layer_outs[_x_name]
    det_sup = sup_error(det_x_out,sym_x_sup,x_num_units,
                        sym_eq_samples,sym_iw_samples,loss=loss_x)
    model_dict['z1->_x']['key_objs']['var']['det_sup'] = det_sup

    # recognition loss for inverse net
    layer_outs = model_dict['z1->_x']['key_outputs']['nondet_y']
    nondet_rec_x = LL_rec(1,layer_outs,x_num_units,sym_eq_samples,sym_iw_samples,power=1)
    model_dict['z1->_x']['key_objs']['var']['nondet_y'] = nondet_rec_x

    layer_outs = model_dict['z1->_x']['key_outputs']['det_y']
    det_rec_x = LL_rec(1,layer_outs,x_num_units,sym_eq_samples,sym_iw_samples,power=1)
    model_dict['z1->_x']['key_objs']['var']['det_y'] = det_rec_x

    # non-variational (reconstruction error, y->x supervised loss)
    model_dict['z1->_x']['key_objs']['nonvar'] = OrderedDict()

    # reconstruction error
    _x_layer_outs = model_dict['z1->_x']['key_outputs']['det_x']
    det_x_out = _x_layer_outs[_x_name]
    det_gen_x = sup_error(det_x_out,sym_x,x_num_units,
                          sym_eq_samples**2,sym_iw_samples**2,
                          loss=loss_x)
    model_dict['z1->_x']['key_objs']['nonvar']['det_x'] = det_gen_x

    # y->x supervised error
    model_dict['z1->_x']['key_objs']['nonvar']['det_sup'] = model_dict['z1->_x']['key_objs']['var']['det_sup']

    # Objectives
    model_dict['objs']['var'] = OrderedDict()
    model_dict['objs']['nonvar'] = OrderedDict()

    # M1 objective (x-unsupervised)
    # variational
    model_dict['objs']['var']['M1_x_rec_y'] = model_dict['x->z1']['key_objs']['var']['nondet_x']['sum']
    model_dict['objs']['var']['M1_x_gen_x'] = model_dict['z1->_x']['key_objs']['var']['nondet_x']['sum']
    model_dict['objs']['var']['M1_x'] = model_dict['objs']['var']['M1_x_rec_y'] - model_dict['objs']['var']['M1_x_gen_x']

    # non-variational
    model_dict['objs']['nonvar']['M1_x'] = model_dict['z1->_x']['key_objs']['nonvar']['det_x']

    # M1 objective (y-unsupervised)
    model_dict['objs']['var']['M1_y_rec_x'] = model_dict['z1->_x']['key_objs']['var']['nondet_y']['sum']
    model_dict['objs']['var']['M1_y_gen_y'] = model_dict['x->z1']['key_objs']['var']['nondet_y']['sum']
    model_dict['objs']['var']['M1_y'] = model_dict['objs']['var']['M1_y_rec_x'] - model_dict['objs']['var']['M1_y_gen_y']

    # non-variational
    model_dict['objs']['nonvar']['M1_y'] = model_dict['x->z1']['key_objs']['nonvar']['det_y']

    # M1 objective (supervised)
    # variational
    model_dict['objs']['var']['M1_x_sup_gen_x_sup'] = model_dict['z1->_x']['key_objs']['var']['nondet_x_sup']['sum']
    model_dict['objs']['var']['M1_x_sup'] = -model_dict['objs']['var']['M1_x_sup_gen_x_sup']
    model_dict['objs']['var']['M1_y_sup_gen_y_sup'] = model_dict['x->z1']['key_objs']['var']['nondet_y_sup']['sum']
    model_dict['objs']['var']['M1_y_sup'] = -model_dict['objs']['var']['M1_y_sup_gen_y_sup']

    model_dict['objs']['var']['M1_x_sup_dis'] = model_dict['x->z1']['key_objs']['var']['nondet_sup']
    model_dict['objs']['var']['M1_y_sup_dis'] = model_dict['z1->_x']['key_objs']['var']['nondet_sup']

    # non-variational
    model_dict['objs']['nonvar']['M1_x_sup'] = 0*model_dict['z1->_x']['key_objs']['nonvar']['det_sup']  # zero'd to avoid double counting objective
    model_dict['objs']['nonvar']['M1_y_sup'] = 0*model_dict['x->z1']['key_objs']['nonvar']['det_sup']  # zero'd to avoid double counting objective

    model_dict['objs']['nonvar']['M1_x_sup_dis'] = model_dict['x->z1']['key_objs']['nonvar']['det_sup']
    model_dict['objs']['nonvar']['M1_y_sup_dis'] = model_dict['z1->_x']['key_objs']['nonvar']['det_sup']

    return True


def M2(M_input,model_dict,variational=True,
       prior_xz1=None,prior_y=None,prior_z2=None,coeff_sup=1.,model_type=1,
       loss_x=None,loss_y=None):
    ''' Build Model '''
    assert model_type in [1,2]
    if model_type == 1:
        M1M2 = False
    else:
        M1M2 = True
    syms = model_dict['sym']
    sym_x = syms['x']
    sym_y = syms['y']
    sym_z1 = syms['z1']
    sym_z2 = syms['z2']
    sym_x_sup = syms['x_sup']
    sym_y_sup = syms['y_sup']
    sym_eq_samples = syms['eq_samples']
    sym_iw_samples = syms['iw_samples']
    det = not variational
    try:
        # Supervised recognition model q(y|z1)
        z1_shape = M_input.output_shape
        q_y_z1, y0, params_z1_y = FlowNet(z1_shape,sym_eq_samples,sym_iw_samples,
                                          **model_dict['z1->y']['arch'])
        model_dict['params']['z1->y'] = params_z1_y['all']
        model_dict['params']['z1->y__nonvar'] = params_z1_y['nonvar']
        model_dict['params']['z1->y__var'] = params_z1_y['var']

        # Latent recognition model q(z2|z1,y)
        z1_shape = M_input.output_shape
        y_shape = q_y_z1.output_shape
        z1y_shape = add_shape(z1_shape,y_shape)
        q_z2_z1y, z20, params_z1y_z2 = FlowNet(z1y_shape,sym_eq_samples,sym_iw_samples,
                                               **model_dict['z1y->z2']['arch'])
        model_dict['params']['z1y->z2'] = params_z1y_z2['all']
        model_dict['params']['z1y->z2__nonvar'] = params_z1y_z2['nonvar']
        model_dict['params']['z1y->z2__var'] = params_z1y_z2['var']

        ###############################################################
        # Inverse latent recognition model q(z2|y) * z1 integrated out
        ###############################################################

        # Generative model p(z1|y,z2)
        z2_shape = q_z2_z1y.output_shape
        yz2_shape = add_shape(y_shape,z2_shape)
        q_z1_yz2, z10, params_yz2__z1 = FlowNet(yz2_shape,sym_eq_samples,sym_iw_samples,
                                                **model_dict['yz2->_z1']['arch'])
        model_dict['params']['yz2->_z1'] = params_yz2__z1['all']
        model_dict['params']['yz2->_z1__nonvar'] = params_yz2__z1['nonvar']
        model_dict['params']['yz2->_z1__var'] = params_yz2__z1['var']

        model_dict['params']['M2'] = model_dict['params']['z1->y'] + \
                                     model_dict['params']['z1y->z2'] + \
                                     model_dict['params']['yz2->_z1']
        model_dict['params']['M2_nonvar'] = model_dict['params']['z1->y__nonvar'] + \
                                            model_dict['params']['z1y->z2__nonvar'] + \
                                            model_dict['params']['yz2->_z1__nonvar']
        model_dict['params']['M2_var'] = model_dict['params']['z1->y__var'] + \
                                         model_dict['params']['z1y->z2__var'] + \
                                         model_dict['params']['yz2->_z1__var']

        nets_M2 = OrderedDict([('x->y',q_y_z1),
                               ('xy->z',q_z2_z1y),
                               ('yz->_x',q_z1_yz2)])
        model = M12(M_input,nets_M2)
    except Exception as e:
        print('Exception: '+repr(e))
        print('M2 build failed!')
        return False

    ''' Get Key Layers '''
    layers_z1_y, y_name, names_z1_y = get_key_layers_names(model,model_dict['z1->y']['arch'],'x->y')
    layers_z1y_z2, z2_name, names_z1y_z2 = get_key_layers_names(model,model_dict['z1y->z2']['arch'],'xy->z')
    layers_yz2__z1, _z1_name, names_yz2__z1 = get_key_layers_names(model,model_dict['yz2->_z1']['arch'],'yz->_x')

    # Record name lists
    zipped = zip(names_z1_y,layers_z1_y)
    model_dict['z1->y']['key_layers'] = OrderedDict(zipped)
    zipped = zip(names_z1y_z2,layers_z1y_z2)
    model_dict['z1y->z2']['key_layers'] = OrderedDict(zipped)
    zipped = zip(names_yz2__z1,layers_yz2__z1)
    model_dict['yz2->_z1']['key_layers'] = OrderedDict(zipped)

    ''' Get Key Layer Outputs '''
    model_dict['z1->y']['key_outputs'] = OrderedDict()
    model_dict['z1y->z2']['key_outputs'] = OrderedDict()
    model_dict['yz2->_z1']['key_outputs'] = OrderedDict()

    # auxiliary layers
    l_input = get_all_layers(M_input)[0]
    z1 = M_input
    y = model_dict['z1->y']['key_layers'][y_name]
    z2 = model_dict['z1y->z2']['key_layers'][z2_name]
    # _z1 = model_dict['yz2->_z1']['key_layers'][_z1_name]

    # x-unsupervised
    # encode
    model_outputs(model_dict,'z1->y','nondet_x',inputs=sym_x,deterministic=False)
    model_outputs(model_dict,'z1->y','det_x',inputs=sym_x,deterministic=True)

    z1_nondet_x = lasagne.layers.get_output(z1,inputs=sym_x,deterministic=False)
    z1_nondet_x_tiled = T.tile(z1_nondet_x,(sym_eq_samples*sym_iw_samples,1))
    z1_det_x = lasagne.layers.get_output(z1,inputs=sym_x,deterministic=True)
    z1_det_x_tiled = T.tile(z1_det_x,(sym_eq_samples*sym_iw_samples,1))
    y_nondet_x = model_dict['z1->y']['key_outputs']['nondet_x']['l_sample']
    y_det_x = model_dict['z1->y']['key_outputs']['det_x']['l_sample']

    model_outputs(model_dict,'z1y->z2','nondet_x',inputs={z1:z1_nondet_x_tiled,y:y_nondet_x},deterministic=False)
    model_outputs(model_dict,'z1y->z2','det_x',inputs={z1:z1_det_x_tiled,y:y_det_x},deterministic=True)

    # decode
    z2_nondet_x = model_dict['z1y->z2']['key_outputs']['nondet_x']['l_sample']
    z2_det_x = model_dict['z1y->z2']['key_outputs']['det_x']['l_sample']
    y_nondet_x_tiled = T.tile(y_nondet_x,(sym_eq_samples*sym_iw_samples,1))
    y_det_x_tiled = T.tile(y_det_x,(sym_eq_samples*sym_iw_samples,1))

    model_outputs(model_dict,'yz2->_z1','nondet_x',inputs={y:y_nondet_x_tiled,z2:z2_nondet_x},deterministic=False)
    model_outputs(model_dict,'yz2->_z1','det_x',inputs={y:y_det_x_tiled,z2:z2_det_x},deterministic=True)
    
    # y-unsupervised
    # encode
    model_outputs(model_dict,'yz2->_z1','nondet_y',inputs={y:sym_y,z2:sym_z2},deterministic=False)
    model_outputs(model_dict,'yz2->_z1','det_y',inputs={y:sym_y,z2:sym_z2},deterministic=True)
    # decode
    nondet_z1_out = model_dict['yz2->_z1']['key_outputs']['nondet_y'][_z1_name]
    model_outputs(model_dict,'z1->y','nondet_y',inputs={M_input:nondet_z1_out},deterministic=False)
    det_z1_out = model_dict['yz2->_z1']['key_outputs']['det_y'][_z1_name]
    model_outputs(model_dict,'z1->y','det_y',inputs={M_input:det_z1_out},deterministic=True)

    # supervised
    # z1 -> y
    model_outputs(model_dict,'z1->y','nondet_x_sup',inputs=sym_x_sup,deterministic=False)
    model_outputs(model_dict,'z1->y','det_x_sup',inputs=sym_x_sup,deterministic=True)

    # z1y -> z2
    z1_nondet_x_sup = lasagne.layers.get_output(z1,inputs=sym_x_sup,deterministic=False)
    z1_det_x_sup = lasagne.layers.get_output(z1,inputs=sym_x_sup,deterministic=True)
    if M1M2:
        sym_y_sup_tiled = T.tile(sym_y_sup,(sym_eq_samples*sym_iw_samples,1))
    else:
        sym_y_sup_tiled = sym_y_sup
    model_outputs(model_dict,'z1y->z2','nondet_xy_sup',inputs={z1:z1_nondet_x_sup,y:sym_y_sup_tiled},deterministic=False)
    model_outputs(model_dict,'z1y->z2','det_xy_sup',inputs={z1:z1_det_x_sup,y:sym_y_sup_tiled},deterministic=True)

    # yz2 -> _z1
    sym_y_sup_tiled = T.tile(sym_y_sup_tiled,(sym_eq_samples*sym_iw_samples,1))
    z2_nondet_xy = model_dict['z1y->z2']['key_outputs']['nondet_xy_sup']['l_sample']
    z2_det_xy = model_dict['z1y->z2']['key_outputs']['det_xy_sup']['l_sample']
    model_outputs(model_dict,'yz2->_z1','nondet_xy_sup',inputs={z2:z2_nondet_xy,y:sym_y_sup_tiled},deterministic=False)
    model_outputs(model_dict,'yz2->_z1','det_xy_sup',inputs={z2:z2_det_xy,y:sym_y_sup_tiled},deterministic=True)

    ''' Create Objective Components '''
    z1_num_units = model_dict['yz2->_z1']['arch']['num_output']
    y_num_units = model_dict['z1->y']['arch']['num_output']
    z2_num_units = model_dict['z1y->z2']['arch']['num_output']

    # z1 -> y layer
    model_dict['z1->y']['key_objs'] = OrderedDict()

    # variational (recognition loss, z1->y supervised loss,
    #              generative loss for inverse net,
    #              supervised generative loss for inverse net)
    model_dict['z1->y']['key_objs']['var'] = OrderedDict()

    # recognition loss
    layer_outs = model_dict['z1->y']['key_outputs']['nondet_x']
    nondet_rec_y = LL_rec(1,layer_outs,y_num_units,sym_eq_samples,sym_iw_samples,power=1)
    model_dict['z1->y']['key_objs']['var']['nondet_x'] = nondet_rec_y

    layer_outs = model_dict['z1->y']['key_outputs']['det_x']
    det_rec_y = LL_rec(1,layer_outs,y_num_units,sym_eq_samples,sym_iw_samples,power=1)
    model_dict['z1->y']['key_objs']['var']['det_x'] = det_rec_y

    # z1->y supervised loss
    layer_outs = model_dict['z1->y']['key_outputs']['nondet_x_sup']
    nondet_y_out = layer_outs[y_name]
    nondet_sup = sup_error(nondet_y_out,sym_y_sup,y_num_units,
                           sym_eq_samples**(1+M1M2),sym_iw_samples**(1+M1M2),loss=loss_y)
    model_dict['z1->y']['key_objs']['var']['nondet_sup'] = nondet_sup

    layer_outs = model_dict['z1->y']['key_outputs']['det_x_sup']
    det_y_out = layer_outs[y_name]
    det_sup = sup_error(det_y_out,sym_y_sup,y_num_units,
                        sym_eq_samples**(1+M1M2),sym_iw_samples**(1+M1M2),loss=loss_y)
    model_dict['z1->y']['key_objs']['var']['det_sup'] = det_sup

    # generative loss for inverse net
    sym_y0 = lasagne.layers.get_output(y0,inputs=sym_y,deterministic=True)

    z1_layer_outs = model_dict['yz2->_z1']['key_outputs']['nondet_y']
    _y_layer_outs = model_dict['z1->y']['key_outputs']['nondet_y']
    nondet_gen_y = LL_gen(1,z1_layer_outs,_y_layer_outs,sym_y0,
                          z1_num_units,y_num_units,
                          sym_eq_samples,sym_iw_samples,power=2,priors=prior_xz1)
    model_dict['z1->y']['key_objs']['var']['nondet_y'] = nondet_gen_y

    z1_layer_outs = model_dict['yz2->_z1']['key_outputs']['det_y']
    _y_layer_outs = model_dict['z1->y']['key_outputs']['det_y']
    det_gen_y = LL_gen(1,z1_layer_outs,_y_layer_outs,sym_y0,
                       z1_num_units,y_num_units,
                       sym_eq_samples,sym_iw_samples,power=2,priors=prior_xz1)
    model_dict['z1->y']['key_objs']['var']['det_y'] = det_gen_y

    # supervised generative loss for inverse net
    sym_y0_sup = lasagne.layers.get_output(y0,inputs=sym_y_sup,deterministic=True)
    z1_layer_outs = model_dict['yz2->_z1']['key_outputs']['nondet_xy_sup']
    z1_sup = lasagne.layers.get_output(M_input,inputs={l_input:sym_x_sup},deterministic=False)
    # z1_layer_outs = {'l_sample':z1_sup*T.ones_like(z1_layer_outs['l_sample'])}
    z1_layer_outs = {'l_sample':z1_sup*T.ones_like(z1_sup)}
    _y_layer_outs = model_dict['z1->y']['key_outputs']['nondet_x_sup']
    # nondet_gen_y = LL_gen(1,z1_layer_outs,_y_layer_outs,sym_y0_sup,
    #                       z1_num_units,y_num_units,
    #                       sym_eq_samples,sym_iw_samples,power=1,priors=prior_xz1)
    nondet_gen_y = LL_gen(1,z1_layer_outs,_y_layer_outs,sym_y0_sup,
                          z1_num_units,y_num_units,
                          sym_eq_samples,sym_iw_samples,power=1+M1M2,priors=prior_xz1)
    model_dict['z1->y']['key_objs']['var']['nondet_y_sup'] = nondet_gen_y

    z1_layer_outs = model_dict['yz2->_z1']['key_outputs']['det_xy_sup']
    z1 = lasagne.layers.get_output(M_input,inputs={l_input:sym_x},deterministic=True)
    # z1_layer_outs = {'l_sample':z1*T.ones_like(z1_layer_outs['l_sample'])}
    z1_layer_outs = {'l_sample':z1*T.ones_like(z1)}
    _y_layer_outs = model_dict['z1->y']['key_outputs']['det_x_sup']
    # det_gen_y = LL_gen(1,z1_layer_outs,_y_layer_outs,sym_y0,
    #                    z1_num_units,y_num_units,
    #                    sym_eq_samples,sym_iw_samples,power=1,priors=prior_xz1)
    det_gen_y = LL_gen(1,z1_layer_outs,_y_layer_outs,sym_y0,
                       z1_num_units,y_num_units,
                       sym_eq_samples,sym_iw_samples,power=1+M1M2,priors=prior_xz1)
    model_dict['z1->y']['key_objs']['var']['det_y_sup'] = det_gen_y

    # non-variational (z1->y supervised loss,reconstruction error for inverse net)
    model_dict['z1->y']['key_objs']['nonvar'] = OrderedDict()

    # z1->y supervised error
    model_dict['z1->y']['key_objs']['nonvar']['det_sup'] = model_dict['z1->y']['key_objs']['var']['det_sup']

    # reconstruction error for inverse net
    # this power should prob be 3 if sampling z's as well
    # i think z's are set to zeros for now in SSDGM somewhere maybe?
    _y_layer_outs = model_dict['z1->y']['key_outputs']['det_y']
    det_y_out = _y_layer_outs[y_name]
    det_gen_y = sup_error(det_y_out,sym_y,y_num_units,
                          sym_eq_samples**2,sym_iw_samples**2,
                          loss=loss_y)
    model_dict['z1->y']['key_objs']['nonvar']['det_y'] = det_gen_y

    # z1y -> z2 layer
    model_dict['z1y->z2']['key_objs'] = OrderedDict()

    # variational (recognition loss, supervised recognition loss,
    #              z1y->z2 supervised loss IGNORED, generative loss for inverse net IGNORED)
    model_dict['z1y->z2']['key_objs']['var'] = OrderedDict()

    # recognition loss
    layer_outs = model_dict['z1y->z2']['key_outputs']['nondet_x']
    # nondet_rec_z2 = LL_rec(1,layer_outs,z2_num_units,sym_eq_samples,sym_iw_samples,power=2)
    nondet_rec_z2 = LL_rec(1,layer_outs,z2_num_units,sym_eq_samples,sym_iw_samples,power=2+M1M2)
    model_dict['z1y->z2']['key_objs']['var']['nondet_x'] = nondet_rec_z2

    layer_outs = model_dict['z1y->z2']['key_outputs']['det_x']
    # det_rec_z2 = LL_rec(1,layer_outs,z2_num_units,sym_eq_samples,sym_iw_samples,power=2)
    det_rec_z2 = LL_rec(1,layer_outs,z2_num_units,sym_eq_samples,sym_iw_samples,power=2+M1M2)
    model_dict['z1y->z2']['key_objs']['var']['det_x'] = det_rec_z2

    # supervised recognition loss
    layer_outs = model_dict['z1y->z2']['key_outputs']['nondet_xy_sup']
    # nondet_rec_z2 = LL_rec(1,layer_outs,z2_num_units,sym_eq_samples,sym_iw_samples,power=1)
    nondet_rec_z2 = LL_rec(1,layer_outs,z2_num_units,sym_eq_samples,sym_iw_samples,power=1+M1M2)
    model_dict['z1y->z2']['key_objs']['var']['nondet_x_sup'] = nondet_rec_z2

    layer_outs = model_dict['z1y->z2']['key_outputs']['det_xy_sup']
    # det_rec_z2 = LL_rec(1,layer_outs,z2_num_units,sym_eq_samples,sym_iw_samples,power=1)
    det_rec_z2 = LL_rec(1,layer_outs,z2_num_units,sym_eq_samples,sym_iw_samples,power=1+M1M2)
    model_dict['z1y->z2']['key_objs']['var']['det_x_sup'] = det_rec_z2

    # yz2 -> _z1 layer
    model_dict['yz2->_z1']['key_objs'] = OrderedDict()

    # variational (generative loss, supervised generative loss,
    #              yz2->z1 supervised loss,
    #              recognition loss for inverse net)
    model_dict['yz2->_z1']['key_objs']['var'] = OrderedDict()

    # generative loss
    z1 = lasagne.layers.get_output(M_input,inputs={l_input:sym_x},deterministic=True)
    _z10 = lasagne.layers.get_output(z10,inputs=z1,deterministic=True)
    y_layer_outs = model_dict['z1->y']['key_outputs']['nondet_x']
    y_layer_outs['l_sample'] = T.tile(y_layer_outs['l_sample'],(sym_eq_samples*sym_iw_samples,1))
    z2_layer_outs = model_dict['z1y->z2']['key_outputs']['nondet_x']
    _z1_layer_outs = model_dict['yz2->_z1']['key_outputs']['nondet_x']
    nondet_gen_z1 = LL_gen(1,[y_layer_outs,z2_layer_outs],_z1_layer_outs,_z10,
                           [y_num_units,z2_num_units],z1_num_units,
                           sym_eq_samples,sym_iw_samples,power=3,priors=[prior_y,prior_z2])
    model_dict['yz2->_z1']['key_objs']['var']['nondet_x'] = nondet_gen_z1

    y_layer_outs = model_dict['z1->y']['key_outputs']['det_x']
    y_layer_outs['l_sample'] = T.tile(y_layer_outs['l_sample'],(sym_eq_samples*sym_iw_samples,1))
    z2_layer_outs = model_dict['z1y->z2']['key_outputs']['det_x']
    _z1_layer_outs = model_dict['yz2->_z1']['key_outputs']['det_x']
    det_gen_z1 = LL_gen(1,[y_layer_outs,z2_layer_outs],_z1_layer_outs,_z10,
                        [y_num_units,z2_num_units],z1_num_units,
                        sym_eq_samples,sym_iw_samples,power=3,priors=[prior_y,prior_z2])
    model_dict['yz2->_z1']['key_objs']['var']['det_x'] = det_gen_z1

    # supervised generative loss
    z1_sup = lasagne.layers.get_output(M_input,inputs={l_input:sym_x_sup},deterministic=True)
    _z10_sup = lasagne.layers.get_output(z10,inputs=z1_sup,deterministic=True)
    # y_layer_outs = model_dict['z1->y']['key_outputs']['nondet_x_sup']
    # y_layer_outs = {'l_sample':sym_y_sup*T.ones_like(y_layer_outs['l_sample'])}
    # y_layer_outs = {'l_sample':sym_y_sup*T.ones_like(sym_y_sup)}
    z2_layer_outs = model_dict['z1y->z2']['key_outputs']['nondet_xy_sup']
    _z1_layer_outs = model_dict['yz2->_z1']['key_outputs']['nondet_xy_sup']
    # nondet_gen_z1 = LL_gen(1,[y_layer_outs,z2_layer_outs],_z1_layer_outs,_z10_sup,
    #                        [y_num_units,z2_num_units],z1_num_units,
    #                        sym_eq_samples,sym_iw_samples,power=1+M1M2,priors=[prior_y,prior_z2])
    nondet_gen_z1 = LL_gen(1,z2_layer_outs,_z1_layer_outs,_z10_sup,
                           z2_num_units,z1_num_units,
                           sym_eq_samples,sym_iw_samples,power=2,priors=prior_z2)
    model_dict['yz2->_z1']['key_objs']['var']['nondet_x_sup'] = nondet_gen_z1

    # y_layer_outs = model_dict['z1->y']['key_outputs']['det_x_sup']
    # y_layer_outs = {'l_sample':sym_y_sup*T.ones_like(y_layer_outs['l_sample'])}
    # y_layer_outs = {'l_sample':sym_y_sup*T.ones_like(sym_y_sup)}
    z2_layer_outs = model_dict['z1y->z2']['key_outputs']['det_xy_sup']
    _z1_layer_outs = model_dict['yz2->_z1']['key_outputs']['det_xy_sup']
    # det_gen_z1 = LL_gen(1,[y_layer_outs,z2_layer_outs],_z1_layer_outs,_z10_sup,
    #                     [y_num_units,z2_num_units],z1_num_units,
    #                     sym_eq_samples,sym_iw_samples,power=1+M1M2,priors=[prior_y,prior_z2])
    det_gen_z1 = LL_gen(1,z2_layer_outs,_z1_layer_outs,_z10_sup,
                        z2_num_units,z1_num_units,
                        sym_eq_samples,sym_iw_samples,power=2,priors=prior_z2)
    model_dict['yz2->_z1']['key_objs']['var']['det_x_sup'] = det_gen_z1

    # yz2->_z1 supervised loss
    layer_outs = model_dict['yz2->_z1']['key_outputs']['nondet_xy_sup']
    nondet_z1_out = layer_outs[_z1_name]
    nondet_sup = sup_error(nondet_z1_out,z1_sup,z1_num_units,
                           sym_eq_samples**2,sym_iw_samples**2,loss=loss_x)
    model_dict['yz2->_z1']['key_objs']['var']['nondet_sup'] = nondet_sup

    layer_outs = model_dict['yz2->_z1']['key_outputs']['det_xy_sup']
    det_z1_out = layer_outs[_z1_name]
    det_sup = sup_error(det_z1_out,z1_sup,z1_num_units,
                        sym_eq_samples**2,sym_iw_samples**2,loss=loss_x)
    model_dict['yz2->_z1']['key_objs']['var']['det_sup'] = det_sup

    # recognition loss for inverse net
    layer_outs = model_dict['yz2->_z1']['key_outputs']['nondet_y']
    nondet_rec_z1 = LL_rec(1,layer_outs,z1_num_units,sym_eq_samples,sym_iw_samples,power=1)
    model_dict['yz2->_z1']['key_objs']['var']['nondet_y'] = nondet_rec_z1

    layer_outs = model_dict['yz2->_z1']['key_outputs']['det_y']
    det_rec_z1 = LL_rec(1,layer_outs,z1_num_units,sym_eq_samples,sym_iw_samples,power=1)
    model_dict['yz2->_z1']['key_objs']['var']['det_y'] = det_rec_z1

    # non-variational (reconstruction error, yz2->z1 supervised loss)
    model_dict['yz2->_z1']['key_objs']['nonvar'] = OrderedDict()

    # reconstruction error
    _z1_layer_outs = model_dict['yz2->_z1']['key_outputs']['det_x']
    det_z1_out = _z1_layer_outs[_z1_name]
    det_gen_z1 = sup_error(det_z1_out,z1,z1_num_units,
                           sym_eq_samples**3,sym_iw_samples**3,
                           loss=loss_x)
    model_dict['yz2->_z1']['key_objs']['nonvar']['det_x'] = det_gen_z1

    # yz2->z1 supervised error
    model_dict['yz2->_z1']['key_objs']['nonvar']['det_sup'] = model_dict['yz2->_z1']['key_objs']['var']['det_sup']

    # Objectives
    if 'var' not in model_dict['objs']:
        model_dict['objs']['var'] = OrderedDict()
    if 'nonvar' not in model_dict['objs']:
        model_dict['objs']['nonvar'] = OrderedDict()

    # M2 objective (x-unsupervised)
    # variational
    model_dict['objs']['var']['M2_x_rec_y'] = model_dict['z1->y']['key_objs']['var']['nondet_x']['sum']
    model_dict['objs']['var']['M2_x_rec_z2'] = model_dict['z1y->z2']['key_objs']['var']['nondet_x']['sum']
    model_dict['objs']['var']['M2_x_gen_z1'] = model_dict['yz2->_z1']['key_objs']['var']['nondet_x']['sum']
    model_dict['objs']['var']['M2_x'] = model_dict['objs']['var']['M2_x_rec_y'] + model_dict['objs']['var']['M2_x_rec_z2'] - model_dict['objs']['var']['M2_x_gen_z1']

    # non-variational
    model_dict['objs']['nonvar']['M2_x'] = model_dict['yz2->_z1']['key_objs']['nonvar']['det_x']

    # M2 objective (y-unsupervised)
    model_dict['objs']['var']['M2_y_rec_z1'] = model_dict['yz2->_z1']['key_objs']['var']['nondet_y']['sum']
    model_dict['objs']['var']['M2_y_gen_y'] = model_dict['z1->y']['key_objs']['var']['nondet_y']['sum']
    model_dict['objs']['var']['M2_y'] = model_dict['objs']['var']['M2_y_rec_z1'] - model_dict['objs']['var']['M2_y_gen_y']

    # non-variational
    model_dict['objs']['nonvar']['M2_y'] = model_dict['z1->y']['key_objs']['nonvar']['det_y']

    # M2 objective (supervised)
    # variational
    model_dict['objs']['var']['M2_x_sup_rec_z2_sup'] = model_dict['z1y->z2']['key_objs']['var']['nondet_x_sup']['sum']
    model_dict['objs']['var']['M2_x_sup_gen_z1_sup'] = model_dict['yz2->_z1']['key_objs']['var']['nondet_x_sup']['sum']
    model_dict['objs']['var']['M2_x_sup'] = model_dict['objs']['var']['M2_x_sup_rec_z2_sup'] - model_dict['objs']['var']['M2_x_sup_gen_z1_sup']

    model_dict['objs']['var']['M2_y_sup_gen_y_sup'] = model_dict['z1->y']['key_objs']['var']['nondet_y_sup']['sum']
    model_dict['objs']['var']['M2_y_sup'] = -model_dict['objs']['var']['M2_y_sup_gen_y_sup']

    model_dict['objs']['var']['M2_x_sup_dis'] = model_dict['z1->y']['key_objs']['var']['nondet_sup']
    model_dict['objs']['var']['M2_y_sup_dis'] = model_dict['yz2->_z1']['key_objs']['var']['nondet_sup']

    # non-variational
    model_dict['objs']['nonvar']['M2_x_sup'] = 0*model_dict['yz2->_z1']['key_objs']['nonvar']['det_sup']  # zero'd to avoid double counting objective
    model_dict['objs']['nonvar']['M2_y_sup'] = 0*model_dict['z1->y']['key_objs']['nonvar']['det_sup']  # zero'd to avoid double counting objective
    
    model_dict['objs']['nonvar']['M2_x_sup_dis'] = model_dict['z1->y']['key_objs']['nonvar']['det_sup']
    model_dict['objs']['nonvar']['M2_y_sup_dis'] = model_dict['yz2->_z1']['key_objs']['nonvar']['det_sup']

    return True


def get_key_layers_names(model,arch,net):
    if arch['sample_layer'] is SampleLayer:
        names = ['l_mu','l_log_var','l_z']
    elif arch['sample_layer'] is BernoulliSampleLayer:
        names = ['l_mu','l_z']
    elif arch['sample_layer'] is ConcreteSampleLayer:
        names = ['l_logits','l_z']
    else:
        raise NotImplementedError(arch['sample_layer']+' is not implemented.')

    # Retrieve key model layers
    n_flows = len(arch['flows'])
    names = names + (n_flows > 0)*['l_z'+str(n_flows)]
    sample_name = names[-1]
    layers = model.getByName(net,names)

    # Retrieve log determinant layers
    names_nf = ['l_nf_'+str(idx) for idx in range(n_flows)]
    l_nfs = model.getByName(net,names_nf)
    l_logdet_J = [ListIndexLayer(l_nf,index=1) for l_nf in l_nfs]
    layers += l_logdet_J
    names = names + names_nf

    return layers, sample_name, names


def reshape(x, shape=None):
    if isinstance(x,np.ndarray):
        print('reshaping numpy array')
        if shape is not None:
            return x.reshape(shape)
        else:
            raise ValueError('Expected argument ''shape''')
    elif isinstance(x,theano.gof.type.PureType.Variable):
        print('dimshuffling theano variable')
        return x.dimshuffle(0,'x','x',1)
    else:
        raise TypeError('Unexpected output type')


def sup_error(predicted,actual,num_output,eq_samples,iw_samples,loss=None):
    predicted = predicted.reshape((-1, eq_samples, iw_samples, num_output))
    actual = actual.dimshuffle(0, 'x', 'x', 1)
    if loss is None:
        sup_train = T.mean(0.*actual,axis=3)
    else:
        sup_train = loss(predicted,actual)
    return T.mean(sup_train)


def L2(predicted,actual):
    return T.sqrt(T.sum(T.sqr(predicted-actual),axis=3))


def REP(predicted,actual):
    return T.sum(T.abs_((predicted-actual)/actual),axis=3)


def KL(predicted,actual):
    eps = 1e-6
    _actual = actual.clip(eps,1.)
    _predicted = predicted.clip(eps,1.)
    KL_most = T.sum(_actual*(T.log(_actual)-T.log(_predicted)),axis=3)
    actual_last = (1.-T.sum(actual,axis=3)).clip(eps,1.)
    predicted_last = (1.-T.sum(predicted,axis=3)).clip(eps,1.)
    KL_last = actual_last*(T.log(actual_last)-T.log(predicted_last))
    return KL_most + KL_last


def LL_rec(t, layer_outs, num_units, eq_samples, iw_samples, power=1):
    assert isinstance(power,int) and power >= 1
    pre_shape = (-1,eq_samples**(power-1),iw_samples**(power-1),num_units)
    post_shape = (-1,eq_samples**power,iw_samples**power,num_units)

    partial_LL = OrderedDict([('sum',0)])
    layer_outs_keys = layer_outs.keys()

    # Recognition likelihood: q(zk|x) = q(z0|x) - logdet
    layer_names_A = ['l_z','l_mu']
    layer_names_B = ['l_z','l_logits']
    in_A = all(key in layer_outs_keys for key in layer_names_A)
    in_B = all(key in layer_outs_keys for key in layer_names_B)
    if in_A:
        name = 'mu'
    else:
        name = 'logits'

    # q(z0|x)
    if in_A or in_B:
        z0 = layer_outs['l_z']
        partial_LL['x'] = z0
        z0 = z0.reshape(post_shape)
        
        z0_mu = layer_outs['l_'+name]
        partial_LL['x_'+name] = z0_mu
        if power == 1:
            z0_mu = z0_mu.dimshuffle(0,'x','x',1)
        else:
            z0_mu = T.tile(z0_mu.reshape(pre_shape),(1,eq_samples,iw_samples,1))

        if 'l_log_var' in layer_outs_keys:
            z0_log_var = layer_outs['l_log_var']
            partial_LL['x_log_var'] = z0_log_var
            if power == 1:
                z0_log_var = z0_log_var.dimshuffle(0,'x','x',1)
            else:
                z0_log_var = T.tile(z0_log_var.reshape(pre_shape),(1,eq_samples,iw_samples,1))
            log_q0z0_given_x = log_norm(z0, z0_mu, z0_log_var).sum(axis=3)
        else:
            if in_A:
                log_q0z0_given_x = log_bern(z0, z0_mu).sum(axis=3)
            else:
                log_q0z0_given_x = log_gumsoft(z0, z0_mu)

        partial_LL['log_q0z0_given_x'] = log_q0z0_given_x.mean()
        partial_LL['sum'] += log_q0z0_given_x

    # logdet
    if any('l_nf' in key for key in layer_outs_keys):
        logdet_Js = [v for k,v in layer_outs.items() if 'l_nf_' in k]
        new_shape = post_shape[:-1]
        logdet_Js = [ldj.reshape(new_shape) for ldj in logdet_Js]
        sum_logdet_Js = sum(logdet_Js)
        partial_LL['sum_logdet_Js'] = sum_logdet_Js.mean()
        partial_LL['sum'] -= sum_logdet_Js

    partial_LL['sum'] = logsumexp(partial_LL['sum'])

    return partial_LL


def LL_gen(t, zs_layer_outs, x_layer_outs, x, zs_num_units, x_num_units,
           eq_samples, iw_samples, power=1, priors=None):
    assert isinstance(power,int) and power >= 1
    partial_LL = OrderedDict([('sum',0)])

    # Generative likelihood: p(x,zk) = p(zk) + p(x|zk)
    # with flows: p(xk,zk) = p(zk) + p(xk|zk) = p(zk) + p(x0|zk) - logdet

    if isinstance(zs_layer_outs,dict):
        zs_layer_outs = [zs_layer_outs]
        assert not isinstance(zs_num_units,list)
        zs_num_units = [zs_num_units]
        assert not isinstance(priors,list)
        priors = [priors]

    for i, z_layer_outs in enumerate(zs_layer_outs):
        z_layer_outs_keys = z_layer_outs.keys()
        pre_shape = (-1,eq_samples**(power-1),iw_samples**(power-1),zs_num_units[i])

        # p(zk)
        if 'l_sample' in z_layer_outs_keys:
            zk = z_layer_outs['l_sample']
            if power == 1:
                zk = zk.dimshuffle(0,'x','x',1)
            else:
                zk = zk.reshape(pre_shape)
            if priors[i] is not None:
                log_pzk = priors[i](zk)
                partial_LL['log_pzk'+str(i)] = log_pzk.mean()
                partial_LL['sum'] += T.tile(log_pzk,(1,eq_samples,iw_samples))

    x_layer_outs_keys = x_layer_outs.keys()
    pre_shape = (-1,eq_samples**(power-1),iw_samples**(power-1),x_num_units)
    post_shape = (-1,eq_samples**power,iw_samples**power,x_num_units)

    in_A = 'l_mu' in x_layer_outs_keys
    in_B = 'l_logits' in x_layer_outs_keys
    if in_A:
        name = 'mu'
    else:
        name = 'logits'

    # p(x|zk)
    if in_A or in_B:
        x = x.dimshuffle(0,'x','x',1)

        x_mu = x_layer_outs['l_'+name]
        if power == 1:
            x_mu = x_mu.dimshuffle(0,'x','x',1)
        else:
            x_mu = x_mu.reshape(pre_shape)

        if 'l_log_var' in x_layer_outs_keys:
            x_log_var = x_layer_outs['l_log_var']
            if power == 1:
                x_log_var = x_log_var.dimshuffle(0,'x','x',1)
            else:
                x_log_var = x_log_var.reshape(pre_shape)        
            log_px_given_zk = log_norm(x, x_mu, x_log_var).sum(axis=3)
        else:
            if in_A:
                log_px_given_zk = log_bern(x, x_mu).sum(axis=3)
            else:
                log_px_given_zk = log_gumsoft(x, x_mu)

        partial_LL['log_px_given_zk'] = log_px_given_zk.mean()
        partial_LL['sum'] += T.tile(log_px_given_zk,(1,eq_samples,iw_samples))

    # logdet
    if any('l_nf' in key for key in x_layer_outs_keys):
        logdet_Js = [v for k,v in x_layer_outs.items() if 'l_nf_' in k]
        new_shape = post_shape[:-1]
        logdet_Js = [ldj.reshape(new_shape) for ldj in logdet_Js]
        sum_logdet_Js = sum(logdet_Js)
        partial_LL['sum_logdet_Js'] = sum_logdet_Js.mean()
        partial_LL['sum'] -= sum_logdet_Js

    partial_LL['sum'] = logsumexp(partial_LL['sum'])

    return partial_LL


# def logsumexp(a,sf=np.cast['float32'](1e-5)):
def logsumexp(a,sf=1e-5):
    # all log_*** should have dimension (batch_size, eq_samples, iw_samples)
    # Calculate the LL using log-sum-exp to avoid underflow
    # input size: (batch_size, eq_samples, iw_samples)
    a_max = T.max(a, axis=2, keepdims=True)
    # size: (batch_size, eq_samples, 1)

    # LL is calculated using Eq (8) in Burda et al.
    # Working from inside out of the calculation below:
    # T.exp(a-a_max): (batch_size, eq_samples, iw_samples)
    # -> subtract a_max to avoid overflow. a_max is specific for  each set of
    # importance samples and is broadcasted over the last dimension.
    #
    # T.log( T.mean(T.exp(a-a_max), axis=2) ): (batch_size, eq_samples)
    # -> This is the log of the sum over the importance weighted samples
    #
    # The outer T.mean() computes the mean over eq_samples and batch_size
    #
    # Lastly we add T.mean(a_max) to correct for the log-sum-exp trick
    LL = T.mean(a_max) + T.mean(T.log(sf + T.mean(T.exp(a-a_max), axis=2)))

    return LL


# def dirichlet(x,alpha,eps=np.cast['float32'](1e-3),axis=-1):
def dirichlet(x,alpha,eps=1e-3,axis=-1):
    x_last = 1.-T.sum(x,axis=axis)
    logBeta = gammaln(alpha).sum() - gammaln(alpha.sum())
    ll = T.sum(T.log(x.clip(eps,1.))*(alpha[:-1]-1.),axis=axis)
    ll_last = T.log(x_last.clip(eps,1.))*(alpha[-1]-1.)
    log_pzk = -logBeta + ll + ll_last
    return log_pzk


def categorical(x,p,axis=-1):
    return T.sum(p**x,axis=-1)


def add_shape(sh_a,sh_b):
    sh = []
    for a,b in zip(sh_a,sh_b):
        if (a is None) or (b is None):
            sh += [None]
        else:
            sh += [a+b]
    return tuple(sh)


def last_zk(layer_outs):
    ks = [k for k,v in layer_outs.items() if 'l_z' in k]
    k = sorted(ks,reverse=True)[0]
    zk = layer_outs[k]
    return zk


def model_outputs(model_dict,net,new_key,inputs,deterministic):
    if 'key_layers' in model_dict[net]:
        layers = model_dict[net]['key_layers'].values()
        outputs = lasagne.layers.get_output(layers,inputs=inputs,
                                            deterministic=deterministic)
        names = model_dict[net]['key_layers'].keys()
        new_dict = OrderedDict(zip(names,outputs))
        new_dict['l_sample'] = last_zk(new_dict)
        model_dict[net]['key_outputs'][new_key] = new_dict

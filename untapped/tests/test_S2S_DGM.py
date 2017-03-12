import unittest

import os 
os.environ['THEANO_FLAGS'] = "device=cpu" 
import itertools

import numpy as np

import theano
theano.config.exception_verbosity = 'high'
theano.config.traceback.limit = 20

import lasagne

from untapped.parmesan.distributions import log_normal2
from untapped.S2S_DGM import SSDGM


def log_unexpected(f,prefix,e,fail):
    print(prefix,e)
    if not ('S2S_DGM msg: ' in str(e) or 'got an unexpected keyword argument' in str(e)):
        f.write(prefix+' '+str(e)+'\n')
        fail = True
    return fail


class TestS2SDGM(unittest.TestCase):
    def test_arg_sweep(self):
        fail = False

        num_features = 10
        num_output = 2
        num_latent_z2 = 1
        N_Xy = 100
        N_X_ = 100
        N__y = 100
        N_z1 = 100

        X = np.random.rand(N_Xy,num_features).astype('float32')
        y = np.random.rand(N_Xy,num_output).astype('float32')
        X_ = np.random.rand(N_X_,num_features).astype('float32')
        _y = np.random.rand(N__y,num_output).astype('float32')
        z2 = np.random.uniform(low=-1.5, high=1.5, size=(N__y,num_latent_z2)).astype('float32')

        X_valid = np.random.rand(N_Xy,num_features).astype('float32')
        y_valid = np.random.rand(N_Xy,num_output).astype('float32')
        X__valid = np.random.rand(N_X_,num_features).astype('float32')
        _y_valid = np.random.rand(N__y,num_output).astype('float32')
        z2_valid = np.random.uniform(low=-1.5, high=1.5, size=(N__y,num_latent_z2)).astype('float32')

        data_groups_0 = [{'X':X,'y':y,'X_valid':X,'y_valid':y},
                         {'X_':X_,'X__valid':X_},
                         {'_y':_y,'_y_valid':_y},
                         {'X':X,'y':y,'X_valid':X,'y_valid':y,'X_':X_,'X__valid':X_},
                         {'X':X,'y':y,'X_valid':X,'y_valid':y,'_y':_y,'_y_valid':_y},
                         {'X':X,'y':y,'X_valid':X,'y_valid':y,'X_':X_,'X__valid':X_,'_y':_y,'_y_valid':_y}]
        data_groups_12 = [{'X':X,'y':y,'X_valid':X,'y_valid':y},
                          {'X_':X_,'X__valid':X_},
                          {'_y':_y,'_y_valid':_y,'z2':z2,'z2_valid':z2},
                          {'X':X,'y':y,'X_valid':X,'y_valid':y,'X_':X_,'X__valid':X_},
                          {'X':X,'y':y,'X_valid':X,'y_valid':y,'_y':_y,'_y_valid':_y,'z2':z2,'z2_valid':z2},
                          {'X':X,'y':y,'X_valid':X,'y_valid':y,'X_':X_,'X__valid':X_,'_y':_y,'_y_valid':_y,'z2':z2,'z2_valid':z2}]
        data_groups = [data_groups_0,data_groups_12,data_groups_12]

        prior_x = lambda x: log_normal2(x,mean=0.,log_var=0.).sum(axis=3)
        prior_y = lambda y: log_normal2(y,mean=0.,log_var=0.).sum(axis=3)
        prior_z1 = lambda z1: log_normal2(z1,mean=0.,log_var=0.).sum(axis=3)
        prior_z2 = lambda z2: log_normal2(z2,mean=0.,log_var=0.).sum(axis=3)

        res_out = 'untapped/tests/results'
        testlog = os.path.join(res_out, 'testlog.log')

        with open(testlog, 'a') as testfile:

            for model_type in [0,1,2]:
                print('model_type',model_type)

                for variational in [False,True]:
                    print('variational',variational)

                    # handle z1 ambiguity
                    if model_type == 0:
                        num_latent_z1 = num_output
                    elif model_type == 1:
                        num_latent_z1 = num_features
                    else:
                        num_latent_z1 = 5
                    z1 = np.random.rand(N_z1,num_latent_z1).astype('float32')

                    m = SSDGM(num_features,num_output,num_latent_z1=num_latent_z1,num_latent_z2=num_latent_z2,
                              prior_x=prior_x,prior_y=prior_y,prior_z1=prior_z1,prior_z2=prior_z2,
                              coeff_x=1,coeff_y=1e-4,coeff_x_dis=1,coeff_y_dis=1e-4,coeff_x_prob=1e-4,coeff_y_prob=1e-4,
                              num_epochs=10,eval_freq=5,lr=3e-3,eq_samples=2,iw_samples=3,
                              batch_size_Xy_train=50,batch_size_X__train=25,batch_size__y_train=75,
                              batch_size_Xy_eval=125,batch_size_X__eval=40,batch_size__y_eval=57,
                              nonlin_enc=lasagne.nonlinearities.tanh,nonlin_dec=lasagne.nonlinearities.tanh,
                              variational=variational,model_type=model_type,res_out=res_out)

                    fs = [m.getZ1,
                          m.get_X,
                          m.getY,
                          m.getZ2,
                          m.get_Z1,
                          m.getYZ2]

                    input_dict = {'x':X,'y':y,'z1':z1,'z2':z2}
                    kwarg_list = []
                    for n in range(1,len(input_dict.keys())):
                        for keys in itertools.combinations(input_dict,n):
                            kwarg_list += [dict([(key,input_dict[key]) for key in keys])]

                    for kwargs in kwarg_list:
                        for f in fs:
                            try:
                                res = f(deterministic=True,**kwargs)
                                if res is None:
                                    print(f.__name__,'no output')
                            except Exception as e:
                                fail = log_unexpected(testfile,f.__name__,e,fail)
                    
                    try:
                        res = m.getX_obj(X)
                        if res is None:
                            raise ValueError('S2S_DGM msg: output is None')
                        else:
                            print('getX_obj=',res)
                    except Exception as e:
                        fail = log_unexpected(testfile,m.getX_obj.name,e,fail)

                    try:
                        res = m.getY_obj(y,z2)
                        if res is None:
                            raise ValueError('S2S_DGM msg: output is None')
                        else:
                            print('getY_obj=',res)
                    except Exception as e:
                        fail = log_unexpected(testfile,m.getY_obj.name,e,fail)

                    try:
                        res = m.getXsup_obj(X,y)
                        if res is None:
                            raise ValueError('S2S_DGM msg: output is None')
                        else:
                            print('getXsup_obj=',res)
                    except Exception as e:
                        fail = log_unexpected(testfile,m.getXsup_obj.name,e,fail)

                    try:
                        res = m.getYsup_obj(X,y)
                        if res is None:
                            raise ValueError('S2S_DGM msg: output is None')
                        else:
                            print('getYsup_obj=',res)
                    except Exception as e:
                        fail = log_unexpected(testfile,m.getYsup_obj.name,e,fail)


                    try:
                        res = m.getXY_obj(X,y)
                        if res is None:
                            raise ValueError('S2S_DGM msg: output is None')
                        else:
                            print('getXY_obj=',res)
                    except Exception as e:
                        fail = log_unexpected(testfile,m.getXY_obj.name,e,fail)

                    try:
                        res = m.getYX_obj(X,y)
                        if res is None:
                            raise ValueError('S2S_DGM msg: output is None')
                        else:
                            print('getYX_obj=',res)
                    except Exception as e:
                        fail = log_unexpected(testfile,getYX_obj.name,e,fail)

                    for data in data_groups[model_type]:
                        print(data.keys())
                        m.fit(verbose=False,**data)

        assert not fail, 'See untapped/tests/results/testlog.log for additional details.'


if __name__ == '__main__':
    test = tests.test_S2S_DGM()


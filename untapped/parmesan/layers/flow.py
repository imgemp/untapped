import lasagne
import theano.tensor as T
import numpy as np

sf = 1e-5

class NormalizingPlanarFlowLayer(lasagne.layers.Layer):
    """
    Normalizing Planar Flow Layer as described in Rezende et
    al. [REZENDE]_ (Equation numbers and appendixes refers to this paper)
    Eq. (8) is used for calculating the forward transformation f(z).
    The last term of eq. (13) is also calculated within this layer and
    returned as an output for computational reasons. Furthermore, the
    transformation is ensured to be invertible using the constrains
    described in appendix A.1

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape
    u,w : Theano shared variable, numpy array or callable
        An initializer for the weights of the layer. If a shared variable or a
        numpy array is provided the shape should be (num_inputs, num_units).
        See :meth:`Layer.create_param` for more information.
    b : Theano shared variable, numpy array, callable or None
        An initializer for the biases of the layer. If a shared variable or a
        numpy array is provided the shape should be (num_units,).
        If None is provided the layer will have no biases.
        See :meth:`Layer.create_param` for more information.

    References
    ----------
        ..  [REZENDE] Rezende, Danilo Jimenez, and Shakir Mohamed.
            "Variational Inference with Normalizing Flows."
            arXiv preprint arXiv:1505.05770 (2015).
    """
    def __init__(self, incoming, u=lasagne.init.Normal(),
                 w=lasagne.init.Normal(),
                 b=lasagne.init.Constant(0.0), **kwargs):
        super(NormalizingPlanarFlowLayer, self).__init__(incoming, **kwargs)
        
        num_latent = int(np.prod(self.input_shape[1:]))
        
        self.u = self.add_param(u, (num_latent,), name="u")
        self.w = self.add_param(w, (num_latent,), name="w")
        self.b = self.add_param(b, tuple(), name="b") # scalar
    
    def get_output_shape_for(self, input_shape):
        return input_shape
    
    
    def get_output_for(self, input, **kwargs):
        # 1) calculate u_hat to ensure invertibility (appendix A.1 to)
        # 2) calculate the forward transformation of the input f(z) (Eq. 8)
        # 3) calculate u_hat^T psi(z) 
        # 4) calculate logdet-jacobian log|1 + u_hat^T psi(z)| to be used in the LL function
        
        z = input
        # z is (batch_size, num_latent_units)
        uw = T.dot(self.u,self.w)
        muw = -1 + T.nnet.softplus(uw) # = -1 + T.log(1 + T.exp(uw))
        u_hat = self.u + (muw - uw) * T.transpose(self.w) / T.sum(self.w**2)
        zwb = T.dot(z,self.w) + self.b
        f_z = z + u_hat.dimshuffle('x',0) * lasagne.nonlinearities.tanh(zwb).dimshuffle(0,'x')
        
        psi = T.dot( (1-lasagne.nonlinearities.tanh(zwb)**2).dimshuffle(0,'x'),  self.w.dimshuffle('x',0)) # tanh(x)dx = 1 - tanh(x)**2
        psi_u = T.dot(psi, u_hat)

        logdet_jacobian = T.log(T.abs_(1 + psi_u))
        
        return [f_z, logdet_jacobian]

class NormalizingSimplexFlowLayer(lasagne.layers.Layer):
    """
    A flow from Rn to the simplex

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape
    u,w : Theano shared variable, numpy array or callable
        An initializer for the weights of the layer. If a shared variable or a
        numpy array is provided the shape should be (num_inputs, num_units).
        See :meth:`Layer.create_param` for more information.
    b : Theano shared variable, numpy array, callable or None
        An initializer for the biases of the layer. If a shared variable or a
        numpy array is provided the shape should be (num_units,).
        If None is provided the layer will have no biases.
        See :meth:`Layer.create_param` for more information.

    References
    ----------
        ..  [REZENDE] Rezende, Danilo Jimenez, and Shakir Mohamed.
            "Variational Inference with Normalizing Flows."
            arXiv preprint arXiv:1505.05770 (2015).
    """
    def __init__(self, incoming, **kwargs):
        super(NormalizingSimplexFlowLayer, self).__init__(incoming, **kwargs)
        # NEW ##################
        self.inverse = [NormalizingSimplexFlowLayer_Inverse,None]

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, **kwargs):
        # 1) calculate u_hat to ensure invertibility (appendix A.1 to)
        # 2) calculate the forward transformation of the input f(z) (Eq. 8)
        # 3) calculate u_hat^T psi(z)
        # 4) calculate logdet-jacobian log|1 + u_hat^T psi(z)| to be used in the
        #    LL function

        z = input
        # # z is (batch_size, num_latent_units)
        # hz = T.exp(z)  # batch_size x num_latent_units
        # K = (T.sum(hz,axis=1) + 1).dimshuffle(0,'x')  # batch_size x 1
        # f_z = 1./K * hz  # batch_size x num_latent_units

        # f_z = T.exp(z)/(T.exp(z).sum(1,keepdims=True)+1.)
        z_max = T.max(z,axis=1,keepdims=True)
        z_max = T.clip(z_max,0,z_max.max())

        z_adj = z-z_max
        sumexp = T.sum(T.exp(z_adj),axis=1,keepdims=True)
        K = T.log(sf+sumexp+T.exp(-z_max))
        f_z = T.exp(z_adj - K)

        # uw = T.dot(self.u,self.w)
        # muw = -1 + T.nnet.softplus(uw) # = -1 + T.log(1 + T.exp(uw))
        # u_hat = self.u + (muw - uw) * T.transpose(self.w) / T.sum(self.w**2)
        # zwb = T.dot(z,self.w) + self.b
        # f_z = z + u_hat.dimshuffle('x',0) * \
            # lasagne.nonlinearities.tanh(zwb).dimshuffle(0,'x')

        # dhz/dz = hz for exponential
        u = -f_z  # -h_z/K
        v = f_z  # dh_z/dz/K
        A = f_z  # matrix where each row represents matrix diagonal dh_z/dz/K

        # psi = T.sum(v*A*u,axis=1)
        psi = T.sum(u,axis=1)

        # detA = T.prod(A,axis=1)
        
        # temp = T.log(T.abs_(1 + sf + psi))
        logdet_jacobian = T.log(sf+ T.abs_(1 + psi)) + T.sum(T.log(sf + A),axis=1)

        # psi = T.dot( (1-lasagne.nonlinearities.tanh(zwb)**2).dimshuffle(0,'x'),  self.w.dimshuffle('x',0)) # tanh(x)dx = 1 - tanh(x)**2
        # psi_u = T.dot(psi, u_hat)

        # logdet_jacobian = T.log(T.abs_(1 + psi_u))

        return [f_z, logdet_jacobian]

class NormalizingSimplexFlowLayer_Inverse(lasagne.layers.Layer):
    """
    A flow from the simplex to Rn

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape
    u,w : Theano shared variable, numpy array or callable
        An initializer for the weights of the layer. If a shared variable or a
        numpy array is provided the shape should be (num_inputs, num_units).
        See :meth:`Layer.create_param` for more information.
    b : Theano shared variable, numpy array, callable or None
        An initializer for the biases of the layer. If a shared variable or a
        numpy array is provided the shape should be (num_units,).
        If None is provided the layer will have no biases.
        See :meth:`Layer.create_param` for more information.

    References
    ----------
        ..  [REZENDE] Rezende, Danilo Jimenez, and Shakir Mohamed.
            "Variational Inference with Normalizing Flows."
            arXiv preprint arXiv:1505.05770 (2015).
    """
    def __init__(self, incoming, arg=None, **kwargs):
        super(NormalizingSimplexFlowLayer_Inverse, self).__init__(incoming, **kwargs)
        # self.inverse = NormalizingSimplexFlowLayer

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, **kwargs):
        z = input
        # z is (batch_size, num_latent_units)
        zn = 1. - T.sum(z,axis=1,keepdims=True)

        z = T.clip(z,sf,1.)
        zn = T.clip(zn,sf,1.)

        f_z = T.log(z) - T.log(zn)

        return f_z

class LogisticFlowLayer(lasagne.layers.Layer):
    """
    An elementwise flow from Rn to the [-1,1] hypercube

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape
    u,w : Theano shared variable, numpy array or callable
        An initializer for the weights of the layer. If a shared variable or a
        numpy array is provided the shape should be (num_inputs, num_units).
        See :meth:`Layer.create_param` for more information.
    b : Theano shared variable, numpy array, callable or None
        An initializer for the biases of the layer. If a shared variable or a
        numpy array is provided the shape should be (num_units,).
        If None is provided the layer will have no biases.
        See :meth:`Layer.create_param` for more information.

    References
    ----------
        ..  [REZENDE] Rezende, Danilo Jimenez, and Shakir Mohamed.
            "Variational Inference with Normalizing Flows."
            arXiv preprint arXiv:1505.05770 (2015).
    """
    def __init__(self, incoming, lo=-np.pi/2, hi=np.pi/2, **kwargs):
        super(LogisticFlowLayer, self).__init__(incoming, **kwargs)
        
        self.scale = hi-lo
        self.offset = lo

        arg = (self.scale,self.offset)
        self.inverse = [LogisticFlowLayer_Inverse,arg]

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, **kwargs):
        # 1) calculate u_hat to ensure invertibility (appendix A.1 to)
        # 2) calculate the forward transformation of the input f(z) (Eq. 8)
        # 3) calculate u_hat^T psi(z)
        # 4) calculate logdet-jacobian log|1 + u_hat^T psi(z)| to be used in the
        #    LL function

        z = input
        # # z is (batch_size, num_latent_units)

        # f_standard = 1. / (1. + T.exp(-z))
        f_standard = T.nnet.sigmoid(z)
        
        f_z = self.scale * f_standard + self.offset

        logdet_jacobian = T.log(sf + T.prod(self.scale*f_standard*(1.-f_standard),axis=1))
        # logdet_jacobian = T.sum(T.log(sf + self.scale*f_standard*(1.-f_standard)),axis=1)

        # f_z = 1 / (1 + T.exp(-z))

        # logdet_jacobian = T.log(T.abs_(T.prod(f_z*(1-f_z),axis=1)))

        return [f_z, logdet_jacobian]

class LogisticFlowLayer_Inverse(lasagne.layers.Layer):
    """
    An elementwise flow from the [-1,1] hypercube to Rn

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape
    u,w : Theano shared variable, numpy array or callable
        An initializer for the weights of the layer. If a shared variable or a
        numpy array is provided the shape should be (num_inputs, num_units).
        See :meth:`Layer.create_param` for more information.
    b : Theano shared variable, numpy array, callable or None
        An initializer for the biases of the layer. If a shared variable or a
        numpy array is provided the shape should be (num_units,).
        If None is provided the layer will have no biases.
        See :meth:`Layer.create_param` for more information.

    References
    ----------
        ..  [REZENDE] Rezende, Danilo Jimenez, and Shakir Mohamed.
            "Variational Inference with Normalizing Flows."
            arXiv preprint arXiv:1505.05770 (2015).
    """
    def __init__(self, incoming, arg=(1,0), **kwargs):
        super(LogisticFlowLayer_Inverse, self).__init__(incoming, **kwargs)

        self.scale = arg[0]
        self.offset = arg[1]

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, **kwargs):
        z = input
        # z is (batch_size, num_latent_units)

        z = (z-self.offset)/self.scale

        z = T.clip(z,sf,1.-sf)
        
        f_z = T.log( z / (1-z) )

        return f_z

class SoftplusFlowLayer(lasagne.layers.Layer):
    """
    An elementwise flow from Rn to R+ elementwise

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape
    u,w : Theano shared variable, numpy array or callable
        An initializer for the weights of the layer. If a shared variable or a
        numpy array is provided the shape should be (num_inputs, num_units).
        See :meth:`Layer.create_param` for more information.
    b : Theano shared variable, numpy array, callable or None
        An initializer for the biases of the layer. If a shared variable or a
        numpy array is provided the shape should be (num_units,).
        If None is provided the layer will have no biases.
        See :meth:`Layer.create_param` for more information.

    References
    ----------
        ..  [REZENDE] Rezende, Danilo Jimenez, and Shakir Mohamed.
            "Variational Inference with Normalizing Flows."
            arXiv preprint arXiv:1505.05770 (2015).
    """
    def __init__(self, incoming, **kwargs):
        super(SoftplusFlowLayer, self).__init__(incoming, **kwargs)

        self.inverse = [SoftplusFlowLayer_Inverse,None]

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, **kwargs):
        # 1) calculate u_hat to ensure invertibility (appendix A.1 to)
        # 2) calculate the forward transformation of the input f(z) (Eq. 8)
        # 3) calculate u_hat^T psi(z)
        # 4) calculate logdet-jacobian log|1 + u_hat^T psi(z)| to be used in the
        #    LL function

        z = input
        # # z is (batch_size, num_latent_units)

        f_z = T.nnet.softplus(z)


        ez = T.exp(z)
        df_z = ez/(1+ez)
        # logdet_jacobian = T.sum(T.log(sf+df_z),axis=1)
        logdet_jacobian = T.log(sf + T.prod(df_z,axis=1))

        return [f_z, logdet_jacobian]

class SoftplusFlowLayer_Inverse(lasagne.layers.Layer):
    """
    An elementwise flow from R+ to Rn elementwise

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape
    u,w : Theano shared variable, numpy array or callable
        An initializer for the weights of the layer. If a shared variable or a
        numpy array is provided the shape should be (num_inputs, num_units).
        See :meth:`Layer.create_param` for more information.
    b : Theano shared variable, numpy array, callable or None
        An initializer for the biases of the layer. If a shared variable or a
        numpy array is provided the shape should be (num_units,).
        If None is provided the layer will have no biases.
        See :meth:`Layer.create_param` for more information.

    References
    ----------
        ..  [REZENDE] Rezende, Danilo Jimenez, and Shakir Mohamed.
            "Variational Inference with Normalizing Flows."
            arXiv preprint arXiv:1505.05770 (2015).
    """
    def __init__(self, incoming, arg=None, **kwargs):
        super(SoftplusFlowLayer_Inverse, self).__init__(incoming, **kwargs)

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, **kwargs):
        z = input
        # z is (batch_size, num_latent_units)

        ez = T.exp(z)
        f_z = T.log(sf+T.clip(ez-1,0,ez.max()-1))

        return f_z

class IdentityFlowLayer(lasagne.layers.Layer):
    """
    Normalizing Planar Flow Layer as described in Rezende et
    al. [REZENDE]_ (Equation numbers and appendixes refers to this paper)
    Eq. (8) is used for calculating the forward transformation f(z).
    The last term of eq. (13) is also calculated within this layer and
    returned as an output for computational reasons. Furthermore, the
    transformation is ensured to be invertible using the constrains
    described in appendix A.1

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape
    u,w : Theano shared variable, numpy array or callable
        An initializer for the weights of the layer. If a shared variable or a
        numpy array is provided the shape should be (num_inputs, num_units).
        See :meth:`Layer.create_param` for more information.
    b : Theano shared variable, numpy array, callable or None
        An initializer for the biases of the layer. If a shared variable or a
        numpy array is provided the shape should be (num_units,).
        If None is provided the layer will have no biases.
        See :meth:`Layer.create_param` for more information.

    References
    ----------
        ..  [REZENDE] Rezende, Danilo Jimenez, and Shakir Mohamed.
            "Variational Inference with Normalizing Flows."
            arXiv preprint arXiv:1505.05770 (2015).
    """
    def __init__(self, incoming, **kwargs):
        super(IdentityFlowLayer, self).__init__(incoming, **kwargs)
        # NEW ##################
        self.inverse = None  # what's the point of having one

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, **kwargs):
        return input

class NICE(lasagne.layers.Layer):
    """
    A flow from Rn to the simplex

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape
    u,w : Theano shared variable, numpy array or callable
        An initializer for the weights of the layer. If a shared variable or a
        numpy array is provided the shape should be (num_inputs, num_units).
        See :meth:`Layer.create_param` for more information.
    b : Theano shared variable, numpy array, callable or None
        An initializer for the biases of the layer. If a shared variable or a
        numpy array is provided the shape should be (num_units,).
        If None is provided the layer will have no biases.
        See :meth:`Layer.create_param` for more information.

    References
    ----------
        ..  [REZENDE] Rezende, Danilo Jimenez, and Shakir Mohamed.
            "Variational Inference with Normalizing Flows."
            arXiv preprint arXiv:1505.05770 (2015).
    """
    def __init__(self, incoming, nonlin=lasagne.nonlinearities.tanh, nlayers=3, **kwargs):
        super(NICE, self).__init__(incoming, **kwargs)

        num_latent = int(np.prod(self.input_shape[1:]))
        assert num_latent > 1, 'Input dimension must be greater than 1 for NICE to work'

        d = num_latent//2
        D = num_latent
        
        Ws = []
        bs = []
        for n in range(nlayers):
            # W_init = lasagne.init.Constant(0.0)  # lasagne.init.GlorotUniform()
            W_init = lasagne.init.GlorotUniform()
            b_init = lasagne.init.Constant(0.0)
            if n % 2:
                in_shape = d
                out_shape = D-d
            else:
                in_shape = D-d
                out_shape = d
            W = self.add_param(W_init, (in_shape,out_shape), name="W_"+str(n))
            b = self.add_param(b_init, (out_shape,), name="b_"+str(n))
            Ws += [W]
            bs += [b]
        self.Ws = Ws
        self.bs = bs

        arg = (nonlin,list(zip(Ws,bs)))

        self.inverse = [NICE_Inverse,arg]
        self.nonlin = nonlin
        self.nlayers = nlayers

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, **kwargs):
        # 1) calculate u_hat to ensure invertibility (appendix A.1 to)
        # 2) calculate the forward transformation of the input f(z) (Eq. 8)
        # 3) calculate u_hat^T psi(z)
        # 4) calculate logdet-jacobian log|1 + u_hat^T psi(z)| to be used in the
        #    LL function

        z = input
        # # z is (batch_size, num_latent_units)

        d = z.shape[1]//2
        D = z.shape[1]

        for n in range(self.nlayers):
            x1, x2 = T.split(z, [d, D-d], 2, axis=1)
            if n % 2:
                m_x1 = self.nonlin(T.dot(x1,self.Ws[n]) + self.bs[n].dimshuffle('x', 0))
                m_x2 = 0
            else:
                m_x1 = 0
                m_x2 = self.nonlin(T.dot(x2,self.Ws[n]) + self.bs[n].dimshuffle('x', 0))

            y1 = x1 + m_x2
            y2 = x2 + m_x1

            z = T.concatenate([y1,y2],axis=1)

        f_z = z
        logdet_jacobian = 0.*T.sum(input,axis=1)

        return [f_z, logdet_jacobian]

class NICE_Inverse(lasagne.layers.Layer):
    """
    A flow from the simplex to Rn

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape
    u,w : Theano shared variable, numpy array or callable
        An initializer for the weights of the layer. If a shared variable or a
        numpy array is provided the shape should be (num_inputs, num_units).
        See :meth:`Layer.create_param` for more information.
    b : Theano shared variable, numpy array, callable or None
        An initializer for the biases of the layer. If a shared variable or a
        numpy array is provided the shape should be (num_units,).
        If None is provided the layer will have no biases.
        See :meth:`Layer.create_param` for more information.

    References
    ----------
        ..  [REZENDE] Rezende, Danilo Jimenez, and Shakir Mohamed.
            "Variational Inference with Normalizing Flows."
            arXiv preprint arXiv:1505.05770 (2015).
    """
    def __init__(self, incoming, arg, **kwargs):
        super(NICE_Inverse, self).__init__(incoming, **kwargs)
        self.nonlin = arg[0]
        self.Wbs = arg[1]

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, **kwargs):
        z = input
        # z is (batch_size, num_latent_units)

        d = z.shape[1]//2
        D = z.shape[1]

        for n in range(len(self.Wbs)-1,-1,-1):
            y1, y2 = T.split(z, [d, D-d], 2, axis=1)
            W,b = self.Wbs[n]
            if n % 2:
                m_y1 = self.nonlin(T.dot(y1,W) + b.dimshuffle('x', 0))
                m_y2 = 0
            else:
                m_y1 = 0
                m_y2 = self.nonlin(T.dot(y2,W) + b.dimshuffle('x', 0))

            x1 = y1 - m_y2
            x2 = y2 - m_y1

            z = T.concatenate([x1,x2],axis=1)

        f_z = z

        return f_z

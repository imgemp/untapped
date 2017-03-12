"""
Tools to train semi^2-supervised deep generative models in Theano
"""

try:
    import theano
except ImportError:  # pragma: no cover
    raise ImportError("""Could not import Theano.
Please make sure you install a recent enough version of Theano.
""")
else:
    del theano


from . import utilities
from . import M12
from . import S2S_DGM

import pkg_resources
__version__ = pkg_resources.get_distribution("untapped").version
del pkg_resources
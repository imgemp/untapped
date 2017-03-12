Untapped
=======
Untapped provides a sklearn-compatible class for constructing a semi^2-supervised deep generative model built with neural networks and trained with variational inference.

Installation
------------
Untapped depends heavily on the `Lasagne <http://github.com/Lasagne/Lasagne>`_ and
`Theano <http://deeplearning.net/software/theano>`_ libraries.
Please make sure you have these installed before installing Untapped.
Untapped also depends heavily on a modifed version of `Parmesan <https://github.com/casperkaae/parmesan>`_.
We recommend installing untapped in a virtual environment (i.e., `download Anaconda <https://www.continuum.io/downloads>`_).

**Install Untapped**

.. code-block:: bash

  git clone https://github.com/imgemp/untapped.git
  cd untapped
  conda create --name untapped python
  source activate untapped

  pip install -r requirements.txt
  python setup.py install

  OR

  pip install -r requirements-dev.txt
  python setup.py develop
  python setup.py test


Documentation
-------------
At the moment untapped primarily includes

* A semi^2-supervised deep generative model (^2 for unlabeled x and unlabeled y) equipped with training procedures for both variational inference and vanilla reconstruction error. Models M1, M2, and M12 from Kingma et. al are implemented.

Please see the source code and code examples for further details.

Examples
-------------
* **examples/DEMO_crism.py**: M2 model trained on hyperspectral data obtained under CRISM (Mars satellite instrument) like conditions
* **examples/DEMO_libs.py**: M2 model trained on LIBS spectral data obtained from Mars Curiosity rover
* **examples/DEMO_raman.py**: M2 model trained on Raman spectral data of mineral mixtures
* **examples/DEMO_mnist.py**: M2 model trained on MNIST data with half labels missing (never see 5-9)

**Usage example**:
python examples/DEMO_crism.py

Development
-----------
Untapped is a work in progress, inputs, contributions and bug reports are very welcome.

The library is developed by
    * Ian Gemp

References
-----------

* Kingma, D. P., & Welling, M. (2013). Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114.
* Burda, Y., Grosse, R., & Salakhutdinov, R. (2015). Importance Weighted Autoencoders. arXiv preprint arXiv:1509.00519.
* Rezende, D. J., & Mohamed, S. (2015). Variational Inference with Normalizing Flows. arXiv preprint arXiv:1505.05770.


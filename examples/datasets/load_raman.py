import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import KFold

from matplotlib.pyplot import cm

from untapped.utilities import load_url

# NOTE: the Ishikawa paper had a mistake!
# They put Annite in the Plagioclase group,
# but it should really be in the Mica group.
ISHIKAWA_MINERALS = set([
    # Plagioclase
    'Albite','Andesine','Annite','Anorthite',
    'Bytownite','Labradorite','Oligoclase',
    # Pyroxene
    'Augite','Clinoenstatite','Diopside','Enstatite',
    'Ferrosilite','Hedenbergite','Jadeite','Spodumene',
    # K-Spar
    'Anorthoclase','Microcline','Orthoclase',
    # Mica
    'Lepidolite','Muscovite','Phlogopite','Trilithionite','Zinnwaldite',
    # Olivine
    'Fayalite','Forsterite',
    # Quartz
    'Quartz'
])


def resample(spectrum, target_bands):
    waves = spectrum[:,0]
    intensities = spectrum[:,1]
    m = intensities.min()
    return np.interp(target_bands, waves, intensities, left=m, right=m)


def zero_one(x):
    x -= np.min(x,axis=1)[:,None]
    return (x.T/np.ptp(x,axis=1)).T


def get_endmembers(dataset='examples/datasets/raman/raman.pkl.gz'):
    x, y, waves, majors = load_url('https://people.cs.umass.edu/~imgemp/datasets/raman.pkl.gz',dataset)
    x = zero_one(x)
    endmems = np.zeros((y.shape[1],x.shape[1]))
    for i in range(y.shape[1]):
        samples = (y[:,i] == 1.)
        endmems[i] = np.mean(x[samples],axis=0)
    return endmems


def load_process_data(dataset='examples/datasets/raman/raman.pkl.gz',trial=0,n_folds=2,remove_mean=False,log_x=False):
    x, y, waves, majors = load_url('https://people.cs.umass.edu/~imgemp/datasets/raman.pkl.gz',dataset)

    # normalize
    # preprocessing.normalize(x,norm='l1',copy=False)

    ux = x.mean(axis=0)
    if remove_mean:
        # mean zero
        x -= ux

    # remove last column
    y = y[:,:-1]

    # select train-validation split
    cv = KFold(n_splits=n_folds,shuffle=True,random_state=0).split(X=x,y=y)
    if trial is not None:
        for num,split in enumerate(cv):
            if num >= trial:
                break
    # print('Only '+str(num)+' splits available. Using last split.')
    
    train_idx, valid_idx = split

    x_train = x[train_idx]
    y_train = y[train_idx]

    x_valid = x[valid_idx]
    y_valid = y[valid_idx]

    x_unsup = x
    y_unsup = y

    xy = (x_train, y_train, x_valid, y_valid, x_unsup, y_unsup)

    names = [major.decode('UTF-8') for major in majors]

    colors = cm.rainbow(np.linspace(0,1,len(majors)))

    return xy, ux, waves, names, colors

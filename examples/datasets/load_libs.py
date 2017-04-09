import numpy as np
from sklearn.preprocessing import normalize
from sklearn import preprocessing
from sklearn.model_selection import LeavePGroupsOut
from sklearn.model_selection import StratifiedKFold

from matplotlib.pyplot import cm

from untapped.utilities import load_url

# Mask for LIBS channels
ALAMOS_MASK = np.zeros(6144, dtype=bool)
ALAMOS_MASK[110:1994] = True
ALAMOS_MASK[2169:4096] = True
ALAMOS_MASK[4182:5856] = True


def load_process_data(dataset='examples/datasets/libs/libs.pkl.gz',big=True,trial=0,hardCV=True,
                      cv_param=3,remove_mean=False,log_x=False,eps=0.,DropLastDim=True):

    small_means, big_means, cal_data = load_url('http://www-anw.cs.umass.edu/public_data/untapped/libs.pkl.gz',dataset)

    # unlabeled dataset
    if big and big_means is not None:
        unsup_x = big_means.clip(eps,np.inf)
    else:
        unsup_x = small_means.clip(eps,np.inf)

    # labeled dataset
    sup_data, meta, split = load_cal_targets(cal_data,majors_only=True,big=big,
                                             trial=trial,hardCV=hardCV,
                                             cv_param=cv_param)
    sup_x, sup_y = sup_data
    sup_x = sup_x.clip(eps,np.inf)

    # unpack meta data
    waves_orig, elements, types, le = meta
    elements = [e.decode('UTF-8') for e in elements]

    # unpack train-validation indices
    train_idx, valid_idx = split

    # report train-validation sample types
    num_train = len(train_idx)
    num_valid = len(valid_idx)
    total = sup_x.shape[0]
    train_types = set(le.inverse_transform(types[train_idx]))
    valid_types = set(le.inverse_transform(types[valid_idx]))
    print('Training on {:d}/{:d}: {!r}'.format(num_train,total,train_types))
    print('Validating on {:d}/{:d}: {!r}'.format(num_valid,total,valid_types))

    # mask-normalize-meanzero all data with libsnorm3
    mask = True
    all_x = np.vstack((sup_x,unsup_x))
    all_x = mask_norm(all_x,mask=mask,norm=True)

    if log_x:
        all_x = np.log(all_x)

    ux = all_x.mean(axis=0)
    if remove_mean:
        all_x -= ux
    sup_x = all_x[:sup_x.shape[0]]
    unsup_x = all_x[sup_x.shape[0]:]

    # mask waves meta data to match preprocessing
    waves = mask_norm(waves_orig[np.newaxis,:],mask=mask,norm=False).squeeze()

    # labels should be in [0,1] with sum <= 1
    sup_y = sup_y/100.
    gt1 = np.sum(sup_y,axis=1) > 1.
    gt1_sum = np.sum(sup_y[gt1,:],axis=1)[:,np.newaxis]
    sup_y[gt1,:] = sup_y[gt1,:]/gt1_sum

    N = 100
    num_y = sup_y.shape[1]
    mx = 0.85
    rest = (1-mx)/(num_y-1)
    I = np.ones((num_y,num_y))*rest + np.diag(np.ones(num_y))*(mx-rest)
    unsup_y = np.tile(I,(N,1))

    # unsup_y = np.tile(np.eye(sup_y.shape[1]),(N,1))
    # unsup_y += np.random.rand(*unsup_y.shape) * 0.1/unsup_y.shape[1]
    # unsup_y = np.exp(5*unsup_y)/np.exp(5*unsup_y).sum(axis=1)[:,np.newaxis]

    if DropLastDim:
        # remove last column
        sup_y = sup_y[:,:-1]
        unsup_y = unsup_y[:,:-1]

    # trim down unsupervised samples
    Nx = Ny = 500
    x_unsup = unsup_x[np.random.choice(unsup_x.shape[0],size=Nx,replace=False)]
    y_unsup = unsup_y[np.random.choice(unsup_y.shape[0],size=Ny,replace=False)]

    # split supervised data into train and valid
    x_train = sup_x[train_idx]
    y_train = sup_y[train_idx]

    x_valid = sup_x[valid_idx]
    y_valid = sup_y[valid_idx]

    # package meta data
    meta = waves, (types, le), elements, ux
    meta_split = (train_idx,valid_idx), (train_types, valid_types)

    xy = (x_train, y_train, x_valid, y_valid, x_unsup, y_unsup)
    xy_names = ('x_train', 'y_train', 'x_valid', 'y_valid', 'x_unsup', 'y_unsup')
    print('Data Shapes:')
    for name, d in zip(xy_names,xy):
        print(name,d.shape)

    colors = cm.rainbow(np.linspace(0,1,len(elements)))

    return xy, ux, waves, elements, colors


### depricated: big_means from url is already preprocessed
def load_mars_means(big=True):
    print('WARNING: big_means downloaded from ALL-website is already preprocessed.')
    if big:
        d = np.load('examples/datasets/libs/big_means.npy')

        # naive outlier removal
        thresh = 0.5
        scaled = (d.T/d.max(axis=1)).T
        counts = np.sum(scaled > thresh,axis=1)
        counts_sorted = np.argsort(counts)
        d = d[counts_sorted[:-20]]
        np.random.shuffle(d)

        return d
    else:
        return np.load('examples/datasets/libs/small_means.npy')


def load_cal_targets(data,majors_only=True,big=True,trial=0,hardCV=True,cv_param=3):
    x, y, majors, waves, types = data
    if majors_only:
        y = y[:,:-5]
        majors = majors[:-5]

    print('Unique Rock Types:')
    print(np.unique(types))

    # remove nan data
    bad = np.any(np.isnan(y.squeeze()),axis=1)
    x = x[~bad]
    y = y[~bad]
    types = types[~bad]

    # remove Macusanite & KGA-MED, + Graphite (4) and Titanium (2) because not enough samples
    isMacusanite = (types == 'MACUSANITE')
    isKGA_MED = (types == 'KGA-MED')
    isGoodRock = ~(isMacusanite + isKGA_MED)
    x = x[isGoodRock]
    y = y[isGoodRock]
    types = types[isGoodRock]

    # set seed for reproducibility of label encoding & cv splits
    np.random.seed(0)

    # encode sample types
    le = preprocessing.LabelEncoder()

    types = le.fit_transform(types)

    if trial == -3:
        inds = list(range(x.shape[0]))
        np.random.shuffle(inds)
        train = inds[:x.shape[0]//2]
        valid = inds[x.shape[0]//2:]
        return (x,y),(waves,majors,types,le),(train,valid)

    if trial == -2:
        train = range(x.shape[0])
        valid = range(x.shape[0])
        return (x,y),(waves,majors,types,le),(train,valid)

    # select train-validation split
    if hardCV:
        if trial == -1:
            # Original split that showed promising results for VAE
            raise ValueError('Need to add back in Macusanite for this to work')
            train_ids = [1, 5, 6, 7]
            valid_ids = [0, 3, 4, 2]
            train = [idx for idx,val in enumerate(types) if val in train_ids]
            valid = [idx for idx,val in enumerate(types) if val in valid_ids]
            return (x,y),(waves,majors,types,le),(train,valid)
        cv = LeavePGroupsOut(n_groups=cv_param).split(X=x,y=y,groups=types)
    else:
        cv = StratifiedKFold(n_splits=cv_param).split(X=x,y=y,groups=types)
    if trial is not None:
        for num,split in enumerate(cv):
            if num >= trial:
                return (x,y),(waves,majors,types,le),split
    print('Only '+str(num)+' splits available. Returning last split.')
    return (x,y),(waves,majors,types,le),split


def mask_norm(x,mask=True,norm=True):
    if norm:
        x = libs_norm3(x)
    if mask:
        x = x[:,ALAMOS_MASK]
    return x


def libs_norm3(shots, copy=True):
    shots = np.array(shots, copy=copy, ndmin=2)
    num_chan = shots.shape[1]
    assert num_chan in (6143, 6144, 5485)
    if num_chan == 6143:
        a, b = 2047, 4097
    elif num_chan == 6144:
        a, b = 2048, 4098
    elif num_chan == 5485:
        a, b = 1884, 3811
    normalize(shots[:, :a], norm='l1', copy=False)
    normalize(shots[:,a:b], norm='l1', copy=False)
    normalize(shots[:, b:], norm='l1', copy=False)
    return shots

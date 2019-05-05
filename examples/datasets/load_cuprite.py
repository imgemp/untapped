import numpy as np
# from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

from itertools import product

import spectral.io.envi as envi
import pandas as pd

from scipy.io import loadmat, savemat
from os.path import join

from sklearn.decomposition import PCA

from untapped.utilities import load_url

from IPython import embed
# def get_endmembers(dataset='examples/datasets/cuprite/cuprite.pkl.gz'):
#     x, y, endmembers = load_url('http://www-anw.cs.umass.edu/public_data/untapped/cuprite.pkl.gz',dataset)
#     return endmembers


# def get_data(dataset='examples/datasets/cuprite/cuprite.pkl.gz'):
#     x, y, _ = load_url('http://www-anw.cs.umass.edu/public_data/untapped/cuprite.pkl.gz',dataset)

#     names = ['olivine','diopside','bytownite']
#     colors = ['r','g','b']

#     return x, y, names, colors


# def get_data(dataset='examples/datasets/cuprite/cup95eff_dbl.img'):
#     img = envi.open(dataset+'.hdr', dataset)
#     img_np = np.asarray(img.asarray())
#     x = img_np.reshape((-1,img_np.shape[-1]))

#     return x
def get_endmembers(dataset='examples/datasets/cuprite/labeled_data.npz'):
    return np.load(dataset)['endmembers']


def save_labels(dataset='examples/datasets/cuprite/cup95_av_dbl_all_roi.txt',train=0.6):
    data = pd.read_csv(dataset,header=31,delim_whitespace=True)
    header = list(data)
    mat = data.as_matrix()
    pt_ids = mat[:,0]
    coords = mat[:,1:3]
    X = mat[:,3:]
    row_markers = [r for r,pt_id in enumerate(pt_ids) if pt_id==1] + [len(pt_ids)]
    roi_names = ['Playa', 'Varnished Tuff', 'Silica', 'Alunite', 'Kaolinite', 'Buddingtonite', 'Calcite']
    roi_coords = [coords[l:r] for l,r in zip(row_markers[:-1],row_markers[1:])]
    roi_center_idxs = [np.argmin(np.linalg.norm(rc-np.mean(rc,axis=0),axis=1)) for rc in roi_coords]
    roi_X = [X[l:r].astype('float32') for l,r in zip(row_markers[:-1],row_markers[1:])]
    endmembers = [x[rc] for x,rc in zip(roi_X,roi_center_idxs)]
    y_corners = [y for y in np.vstack((np.eye(len(roi_X)-1),np.zeros(len(roi_X)-1))).astype('float32')]
    roi_y = [np.tile(y[None], (x.shape[0],1)) for x,y in zip(roi_X,y_corners)]
    x_all = np.vstack(roi_X)
    y_all = np.vstack(roi_y)
    x_sup = np.vstack([x[:int(train*x.shape[0])] for x in roi_X])
    y_sup = np.vstack([y[:int(train*y.shape[0])] for y in roi_y])
    x_valid = np.vstack([x[int(train*x.shape[0]):] for x in roi_X])
    y_valid = np.vstack([y[int(train*y.shape[0]):] for y in roi_y])
    # colors = ['r','b','g','k','c','m','y']
    colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255),(255,0,255),(176,48,96)]
    colors = [tuple(np.array(c)/255.) for c in colors]
    save_image(y_all,coords,colors)
    np.savez_compressed('examples/datasets/cuprite/labeled_data.npz',endmembers=endmembers,
                        x_sup=x_sup,y_sup=y_sup,x_valid=x_valid,y_valid=y_valid,colors=colors,
                        x_all=x_all,y_all=y_all,roi_names=roi_names,markers=row_markers,coords=coords)

def save_image(y,coords,colors,file='examples/datasets/cuprite/cuprite_img.png'):
    rc = coords - np.min(coords,axis=0)
    xymn = coords.min(axis=0).astype(int)
    xymx = coords.max(axis=0).astype(int)
    rng = [xymn[0],xymx[0],xymn[1],xymx[1]]
    rc[:,1] = np.max(rc[:,1]) - rc[:,1]
    rc = rc.astype(int)
    rc[:,[0,1]] = rc[:,[1,0]]

    rows, cols = np.max(rc,axis=0) + 1
    img = np.zeros((rows,cols,3))
    for idx, r, c in zip(np.argmax(y,axis=1),rc[:,0],rc[:,1]):
        img[r,c,:] = np.array(colors[idx])

    plt.imshow(img, extent=rng)
    plt.savefig(file)


def save_labels_big():
    data = loadmat('examples/datasets/cuprite/cuprite95_map/spcs_av95.mat')['spcs_av95']
    waves = np.array([data[0][xi][3] for xi in range(data.shape[1])]).squeeze()[0]
    spectra = np.array([data[0][xi][4] for xi in range(data.shape[1])]).squeeze()
    x_all = spectra
    colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255),(255,0,255),(176,48,96)]
    colors = [tuple(np.array(c)/255.) for c in colors]
    np.savez_compressed('examples/datasets/cuprite/labeled_data_big.npz',x_all=x_all,waves=waves,colors=colors)


def cupritebig():
    colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255),(255,0,255),(176,48,96)]
    colors = np.array([tuple(np.array(c)/255.) for c in colors])
    data = loadmat('examples/datasets/cuprite/cuprite95_map/tetscores_cuprite95.mat')
    imgid = np.argmax(data['scores'],axis=2)
    img = np.zeros(imgid.shape+(3,))
    img = colors[imgid]
    plt.imshow(img)
    plt.savefig('examples/datasets/cuprite/cupritebig_img.png')


def get_labels(dataset='examples/datasets/cuprite/labeled_data.npz'):
    return np.load(dataset)


def get_data(dataset='examples/datasets/cuprite/data.npy'):
    return np.load(dataset)


def load_process_data(remove_mean=True):
    x = get_data()
    xy = get_labels()
    x_sup = xy['x_sup']
    y_sup = xy['y_sup']
    x_valid = xy['x_valid']
    y_valid = xy['y_valid']
    names = xy['roi_names']
    colors = xy['colors']
    x_all = np.vstack((x,x_sup))
    
    # mean zero
    ux = x_all.mean(axis=0)
    if remove_mean:
        x -= ux
        x_sup -= ux

    x_unsup = x

    print('\tbuilding simplex for _y...')
    simplex = []
    for point in product(*([np.linspace(0,1,10)]*(y_sup.shape[1]+1))):
        if np.sum(point) == 1:
            simplex += [point]
    simplex = np.asarray(simplex).astype('float32')
    y_unsup = simplex[:,:-1]
    # y_unsup1 = np.random.rand(167, 2) * 0.05
    # y_unsup2 = np.hstack((y_unsup1[:, 0, None], 1. - y_unsup1.sum(axis=1)[None].T))
    # y_unsup3 = np.hstack((1. - y_unsup1.sum(axis=1)[None].T, y_unsup1[:, 1, None]))
    # y_unsup = np.vstack((y_unsup1, y_unsup2, y_unsup3))

    xy = (x_sup, y_sup, x_valid, y_valid, x_unsup, y_unsup)
    xy_names = ('x_train', 'y_train', 'x_valid', 'y_valid', 'x_unsup', 'y_unsup')
    print('Data Shapes:')
    for name, d in zip(xy_names,xy):
        print(name,d.shape)
    # file = 'examples/datasets/crism/wavelength_SWIR.mat'
    # waves = loadmat(file)
    # waves = waves['wavelength'].squeeze()
    waves = np.arange(x.shape[1])

    return xy, ux, waves, names, colors


def load_process_data_big(remove_mean=True):
    data = np.load('examples/datasets/cuprite/labeled_data_big.npz')
    x_all = data['x_all']
    waves = data['waves']
    colors = data['colors']
    
    # mean zero
    ux = x_all.mean(axis=0)
    if remove_mean:
        x_all -= ux

    x_unsup = x_all

    simplex = []
    for point in product(*([np.linspace(0,1,10)]*len(colors))):
        if np.sum(point) == 1:
            simplex += [point]
    simplex = np.asarray(simplex).astype('float32')
    y_unsup = simplex[:,:-1]

    xy = (x_unsup, y_unsup)
    xy_names = ('x_unsup', 'y_unsup')
    print('Data Shapes:')
    for name, d in zip(xy_names,xy):
        print(name,d.shape)

    names = [str(i) for i in range(len(colors))]

    return xy, ux, waves, names, colors


def softmax(x,nnz_only=True,axis=-1):
    xexp = np.exp(x)
    if nnz_only:
        zeros = x==0.
        xexp[zeros] = 1e-5
    return xexp/xexp.sum(axis,keepdims=True)


def load_process_cuprite95(remove_mean=True,train=0.6,dists=False,seed=1234,remove_bad=False,region=None):
    waves = np.load('examples/datasets/cuprite/labeled_data_big.npz')['waves']
    root = 'examples/datasets/cuprite/cuprite95/'
    x = envi.open(root+'cuprite95.hdr',root+'cuprite95').load()
    img_shape = x.shape
    N_channels = x.shape[2]
    x = x.reshape((-1,N_channels))

    region_1 = range(0,105)
    region_2 = range(114,151)
    region_3 = range(169,N_channels)

    if region is None and remove_bad:
        x = np.hstack([x[:,region_1],x[:,region_2],x[:,region_3]])
        waves = np.concatenate([waves[region_1],waves[region_2],waves[region_3]])
    elif region==1:
        x = x[:,region_1]
        waves = waves[region_1]
    elif region==2:
        x = x[:,region_2]
        waves = waves[region_2]
    elif region==3:
        x = x[:,region_3]
        waves = waves[region_3]

    live_samples = x.min(axis=1)>0.
    x = x[live_samples]

    colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255),(255,0,255),(176,48,96)]
    colors = np.array([tuple(np.array(c)/255.) for c in colors])
    tetscores = loadmat('examples/datasets/cuprite/cuprite95_map/tetscores_cuprite95.mat')
    scores = tetscores['scores']
    N_classes = scores.shape[2]
    scores = scores.reshape(-1,N_classes)[live_samples]
    
    nonzeros = scores.sum(axis=1)>0
    # x = x[nonzeros]
    # scores = scores[nonzeros]

    N_samples = x.shape[0]

    topk = 1
    endmembers = []
    pure_scores = tetscores['scores_pure'].reshape(-1,N_classes)[live_samples]  # [nonzeros]
    for c in range(N_classes):
        inds = pure_scores[:,c].argsort()[::-1][:topk]
        endmembers += [np.mean(x[inds],axis=0)]
    endmembers = np.array(endmembers)

    if dists:
        ydist = softmax(scores/18.,axis=1)
        y = ydist[:,:-1]
    else:
        y = np.argmax(scores,axis=1)
        y = np.eye(N_classes)[y][:,:-1]

    if dists:
        subset = np.random.choice(x.shape[0],50000,replace=False)
        # subset = ydist.max(axis=1)>=0.8
        # subset = []
        # for c in range(N_classes):
        #     subset += list(ydist[:,c].argsort()[-10:])
        x = x[subset]
        y = y[subset]
        nonzeros = nonzeros[subset]
        N_samples = x.shape[0]
        # print(N_samples)
        # print(np.bincount(ydist[subset].argmax(axis=1)))
    
    assert x.shape[0] == y.shape[0]
    N = x.shape[0]
    N_train = int(train*N)

    np.random.seed(seed)
    inds = np.arange(N)
    np.random.shuffle(inds)
    x = x[inds]
    y = y[inds]
    nonzeros = nonzeros[inds]

    x_all = np.array(x).astype('float')
    y_all = np.array(y).astype('float')

    # group indices by type
    y_all_ext = np.hstack([y_all,1-y_all.sum(axis=1,keepdims=True)])
    classes = np.argmax(y_all_ext,axis=1)
    classes[nonzeros] = N_classes
    ordered = np.argsort(classes)
    x_all = x_all[ordered]
    y_all = y_all[ordered]
    # array([ 4586, 18114, 19503, 19553, 26715, 34109])
    
    # normalize by area
    # x_all /= x_all.sum(axis=1,keepdims=True)
    
    # mean zero
    ux = x_all.mean(axis=0)
    if remove_mean:
        x_all -= ux
        endmembers -= ux
    xrng = x_all.max() - x_all.min()
    x_all /= xrng
    endmembers /= xrng
    ux /= xrng

    # x_sup = x_all[:N_train]
    # y_sup = y_all[:N_train]
    x_sup = x_all
    y_sup = y_all
    x_valid = x_all[N_train:]
    y_valid = y_all[N_train:]
    x_unsup = x_all

    print('\tbuilding simplex for _y...')
    try:
        simplex = np.load(root+'simplex.npy')
        y_unsup = np.load(root+'y_unsup.npy')
    except Exception as e:
        simplex = []
        for point in product(*([np.linspace(0,1,10)]*len(colors))):
            if np.sum(point) == 1:
                simplex += [point]
        simplex = np.asarray(simplex).astype('float32')
        # y_unsup = simplex[:,:-1]
        y_unsup = np.vstack((np.eye(N_classes-1),np.zeros(N_classes-1))).astype('float32')
        np.save(root+'simplex.npy',simplex)
        np.save(root+'y_unsup.npy',y_unsup)

    xy = (x_sup, y_sup, x_valid, y_valid, x_unsup, y_unsup)
    xy_names = ('x_train', 'y_train', 'x_valid', 'y_valid', 'x_unsup', 'y_unsup')
    print('Data Shapes:')
    for name, d in zip(xy_names,xy):
        print(name,d.shape)

    # names = [str(i) for i in range(len(colors))]
    # names = ['Playa', 'Varnished Tuff', 'Silica', 'Alunite', 'Kaolinite', 'Buddingtonite', 'Calcite']
    names = ['Alunite','Calcite','Chalcedony','Dickite','Kaolinite','Montmorillonite','Muscovite']

    return xy, ux, waves, names, colors, img_shape, endmembers, simplex[:,:-1]


if __name__ == '__main__':
    x, y, w, n, c = load_process_data()

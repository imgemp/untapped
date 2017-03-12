import numpy as np
# from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

from scipy.io import loadmat, savemat
from os.path import join

from sklearn.decomposition import PCA


def get_endmembers(path='examples/datasets/crism/endmembers.txt'):
    return np.loadtxt(path)


def get_data(path='examples/datasets/crism/multiAngle_measurements'):
    names = [['ternaryMix1_position1.mat', 'mix1Samp_pos1'],
             ['ternaryMix1_position2.mat', 'mix1Samp_pos2'],
             ['ternaryMix1_samplesAngled.mat', 'mix1Samp_angle']]
    suffix = '_Abun'
    x_arr = []
    y_arr = []
    for group in names:
        data = loadmat(join(path, group[0]))
        x_arr.append(data[group[1]])
        y_arr.append(data[group[1] + suffix])

    x = x_arr[0]
    y = y_arr[0]
    for i in range(1, len(x_arr)):
        x = np.append(x, x_arr[i], axis=0)
        y = np.append(y, y_arr[i], axis=0)

    names = ['olivine','diopside','bytownite']
    colors = ['r','g','b']

    return x, y, names, colors


def load_process_data(remove_mean=True,plot=False):
    x, y, names, colors = get_data()
    assert np.allclose(y.sum(axis=1), 1.)
    n = y.shape[1]
    y_all = y[:]
    y = y[:,:-1]  # remove last column

    # mean zero
    ux = x.mean(axis=0)
    if remove_mean:
        x -= ux

    # grab center of simplex
    center = 1./n
    pm = .7*center
    in_center = np.logical_and(np.all(y_all>(center-pm), axis=1), np.all(y_all<(center + pm), axis=1))

    x_sup = x[in_center]
    y_sup = y[in_center]

    num_train = min(x_sup.shape[0], 500)
    inds = np.random.permutation(x_sup.shape[0])  # range
    np.random.shuffle(inds)

    x_train = x_sup[inds[:num_train]]
    y_train = y_sup[inds[:num_train]]

    x_valid = x
    y_valid = y

    x_unsup = x_sup

    y_unsup1 = np.random.rand(167, 2) * 0.05
    y_unsup2 = np.hstack((y_unsup1[:, 0, None], 1. - y_unsup1.sum(axis=1)[None].T))
    y_unsup3 = np.hstack((1. - y_unsup1.sum(axis=1)[None].T, y_unsup1[:, 1, None]))
    y_unsup = np.vstack((y_unsup1, y_unsup2, y_unsup3))

    if plot:
        fig = plt.figure()
        ax = Axes3D(fig)

        ax.set_axis_bgcolor('white')
        ax.w_xaxis.set_pane_color((0.0, 1.0, 0.0, 0.2))
        ax.w_yaxis.set_pane_color((1.0, 0.0, 0.0, 0.2))
        ax.w_zaxis.set_pane_color((0.0, 0.0, 1.0, 0.2))
        ax.grid(False)

        third = 1-y_unsup.sum(axis=1)
        skip = 20
        y_not = y[np.logical_not(in_center)]
        ax.scatter(y_unsup[::skip,0],y_unsup[::skip,1],third[::skip], c='w', marker='o', s=200, facecolor='w', alpha=1.0, zorder=2)
        ax.scatter(y_not[:, 0], y_not[:, 1], 1 - y_not.sum(axis=1), c='k', marker='o', s=20, facecolor='k', alpha=0.2, zorder=1)
        ax.scatter(y_sup[:,0],y_sup[:,1],1-y_sup.sum(axis=1), c='w', marker='^', s=200, zorder=2)
        ax.plot([1,0],[0,1],[0,0],'k-',lw=2,alpha=1.0, zorder=0)
        ax.plot([0,0],[1,0],[0,1],'k-',lw=2,alpha=1.0, zorder=0)
        ax.plot([0,1],[0,0],[1,0],'k-',lw=2,alpha=1.0, zorder=0)
        
        ax.elev = 10
        ax.azim = 75
        
        ax.set_xlabel('olivine',fontsize=32)
        ax.set_ylabel('dyopside',fontsize=32)
        ax.set_zlabel('bytownite',fontsize=32)

        ticks = np.linspace(0,1,4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        
        plt.savefig('crism_simplex.png',bbox_inches='tight')

    xy = (x_train, y_train, x_valid, y_valid, x_unsup, y_unsup)
    print([arr.shape for arr in xy])
    # file = 'examples/datasets/crism/wavelength_SWIR.mat'
    # waves = loadmat(file)
    # waves = waves['wavelength'].squeeze()
    waves = np.arange(x.shape[1])

    return xy, ux, waves, names, colors

def compute_radii(allX,ally):
    pca_main = PCA(n_components=3)
    Xnew = pca_main.fit_transform(allX)

    grouped = {}
    y_unique = []
    for idy, y in enumerate(ally):
        if repr(y) not in grouped:
            grouped[repr(y)] = [idy]
            y_unique += [y]
        else:
            grouped[repr(y)] += [idy]

    pca = PCA(n_components=3)
    for y in grouped:
        inds = grouped[y]
        pca.fit(Xnew[inds])
        cov = pca.get_covariance()
        # r = np.linalg.eigvals(cov)
        cov_main = pca_main.inverse_transform(cov)
        r = np.max(np.linalg.svd(cov_main)[1])  # same as eigvals(cov)
        grouped[y] = (r,inds)

    rs = []
    for y in y_unique:
        rs += [grouped[repr(y)][0]]
    rs = np.asarray(rs)
    y_unique = np.asarray(y_unique)

    data = {'radii':rs,'matching_y':y_unique}

    savemat('radii.mat',data)


if __name__ == '__main__':
    x, y, w = load_process_data()

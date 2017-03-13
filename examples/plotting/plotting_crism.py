import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from itertools import product

from sklearn.cross_decomposition import PLSRegression

import numpy as np


def make_plots(m,data,colors,names,groundtruth=None,waves=None,sample_size=10,ux=0,
               remove_mean=False,log_x=False,ylim=(0.3,1),res_out='',title=None):
    inds_sup_train = np.random.choice(data['X'].shape[0],size=sample_size)
    inds_sup_valid = np.random.choice(data['X_valid'].shape[0],size=sample_size)
    inds_train_x = np.random.choice(data['X_'].shape[0],size=sample_size)
    inds_train_y = np.random.choice(data['_y'].shape[0],size=sample_size)

    y = np.hstack([data['y'],1-data['y'].sum(axis=1,keepdims=True)])
    y_valid = np.hstack([data['y_valid'],1-data['y_valid'].sum(axis=1,keepdims=True)])
    y_corners = np.vstack((np.eye(data['y'].shape[1]),np.zeros(data['y'].shape[1]))).astype('float32')

    converter = lambda x: x.decode('UTF-8')
    converters = {0:converter,1:converter,2:converter}
    true_samples = np.loadtxt('datasets/triangle/calculatedWeights.csv',delimiter=',',converters=converters)

    simplex = []
    for point in product(*([np.linspace(0,1,50)]*y.shape[1])):
        if np.sum(point) == 1:
            simplex += [point]
    simplex = np.asarray(simplex).astype('float32')
    simplex = simplex[:,:-1]

    if waves is None:
        waves = np.arange(data['X'].shape[1])

    if remove_mean:
        _ux = ux
    else:
        _ux = 0

    if log_x:
        f = lambda x: np.exp(x)
    else:
        f = lambda x: x

    if ylim is not None:
        force_ylim = True

    pls_XY = PLSRegression(n_components=8,scale=False)
    pls_XY.fit(data['X'],y)
    pred_train_pls = pls_XY.predict(data['X'])
    pred_train_pls = (pred_train_pls.T/np.sum(pred_train_pls,axis=1)).T
    pred_valid_pls = pls_XY.predict(data['X_valid'])
    pred_valid_pls = (pred_valid_pls.T/np.sum(pred_valid_pls,axis=1)).T
    score_pred_train_pls = KL(pred_train_pls,y)
    score_pred_valid_pls = KL(pred_valid_pls,y_valid)

    pls_YX = PLSRegression(n_components=min(8,y.shape[1]),scale=False)
    pls_YX.fit(y,data['X'])
    gen_train_pls = pls_YX.predict(y)
    gen_valid_pls = pls_YX.predict(y_valid)
    score_gen_train_pls = L2(gen_train_pls,data['X'])
    score_gen_valid_pls = L2(gen_valid_pls,data['X_valid'])

    pred_train = m.predict(x=data['X'],deterministic=True)
    pred_train = np.hstack([pred_train,1-pred_train.sum(axis=1,keepdims=True)])
    score_pred_train = KL(pred_train,y)
    pred_valid = m.predict(x=data['X_valid'],deterministic=True)
    pred_valid = np.hstack([pred_valid,1-pred_valid.sum(axis=1,keepdims=True)])
    score_pred_valid = KL(pred_valid,y_valid)

    KLstats = []
    KLstats += [minmaxKL(pred_train_pls,y,true_samples)]
    KLstats += [minmaxKL(pred_valid_pls,y_valid,true_samples)]
    KLstats += [minmaxKL(pred_train,y,true_samples)]
    KLstats += [minmaxKL(pred_valid,y_valid,true_samples)]
    KLstats = np.asarray(KLstats)
    np.save(res_out+'/KLstats.npy',KLstats)
    print(KLstats)

    if m.model_type in [1,2]:
        z2_train = m.getZ2(x=data['X'],y=data['y'],deterministic=True)
        z2_valid = m.getZ2(x=data['X_valid'],y=data['y_valid'],deterministic=True)
        z2_train_mean = z2_train.mean(axis=0)
        z2_valid_mean = z2_valid.mean(axis=0)
        z2_gen_train = z2_train_mean*np.ones_like(z2_train).astype('float32')
        z2_gen_valid = z2_valid_mean*np.ones_like(z2_valid).astype('float32')
        z2_gen_manifold = z2_valid_mean*np.ones((simplex.shape[0],z2_valid.shape[1])).astype('float32')
        z2_gen_endmembers = z2_train_mean*np.ones((y_corners.shape[0],z2_train.shape[1])).astype('float32')
        gen_train = f(_ux + m.generate(y=data['y'][inds_sup_train],z2=z2_gen_train[inds_sup_train],deterministic=True))  # true by default for non-variational, variational default is False
        gen_valid = f(_ux + m.generate(y=data['y_valid'][inds_sup_valid],z2=z2_gen_valid[inds_sup_valid],deterministic=True))
        manifold = f(_ux + m.generate(y=simplex,z2=z2_gen_manifold,deterministic=True))
        endmembers = f(_ux + m.generate(y=y_corners,z2=z2_gen_endmembers,deterministic=True))
        if m.variational:
            endmembers_dists = []
            for idx_c, c in enumerate(y_corners):
                endmembers_dist = [f(_ux + m.generate(y=np.atleast_2d(c),z2=z2_gen_endmembers[idx_c:idx_c+1],deterministic=False)).squeeze() for i in range(sample_size)]
                endmembers_dists += [np.asarray(endmembers_dist)]
            endmembers_dists = endmembers_dists
    else:
        gen_train = f(_ux + m.generate(y=data['y'][inds_sup_train],deterministic=True))  # true by default for non-variational, variational default is False
        gen_valid = f(_ux + m.generate(y=data['y_valid'][inds_sup_valid],deterministic=True))
        manifold = f(_ux + m.generate(y=simplex,deterministic=True))
        endmembers = f(_ux + m.generate(y=y_corners,deterministic=True))
        if m.variational:
            endmembers_dists = []
            for idx_c, c in enumerate(y_corners):
                endmembers_dist = [f(_ux + m.generate(y=np.atleast_2d(c),deterministic=False)).squeeze() for i in range(sample_size)]
                endmembers_dists += [np.asarray(endmembers_dist)]
            endmembers_dists = endmembers_dists
    recon_train = f(_ux + m.generate(x=data['X_'][inds_train_x],deterministic=True))
    recon_sup_valid = f(_ux + m.generate(x=data['X_valid'][inds_sup_valid],deterministic=True))

    fs = 24
    fs_tick = 18

    # change xticks to be names
    p = 100
    plt.plot(p*y[inds_sup_train][0],'k',lw=2,label='Ground Truth')
    ssdgm_label = 'SSDGM ({:.3f})'.format(score_pred_train)
    plt.plot(p*pred_train[inds_sup_train][0],'r-.',lw=2,label=ssdgm_label)
    pls_label = 'PLS ({:.3f})'.format(score_pred_train_pls)
    plt.plot(p*pred_train_pls[inds_sup_train][0],'b-.',lw=2,label=pls_label)
    plt.plot(p*y[inds_sup_train].T,'k',lw=2)
    plt.plot(p*pred_train[inds_sup_train].T,'r-.',lw=2)
    plt.plot(p*pred_train_pls[inds_sup_train].T,'b-.',lw=2)
    plt.title('Predicting Composition - Training Error', fontsize=fs)
    plt.ylabel('Composition (%)', fontsize=fs)
    ax = plt.gca()
    ax.set_ylim((0,1*p))
    ax.set_xticks(np.arange(y.shape[1]))
    ax.set_xticklabels(names, fontsize=fs)
    ax.tick_params(axis='x',direction='out',top='off',length=10,labelsize=fs_tick)
    lgd = plt.legend(loc='center left',bbox_to_anchor=(1, 0.5))
    ax = plt.gca()
    plt.savefig(res_out+'/comp_train.png',additional_artists=[lgd],bbox_inches='tight')
    plt.close()

    plt.plot(p*y_valid[inds_sup_valid][0],'k',lw=2,label='Ground Truth')
    ssdgm_label = 'SSDGM ({:.3f})'.format(score_pred_valid)
    plt.plot(p*pred_valid[inds_sup_valid][0],'r-.',lw=2,label=ssdgm_label)
    pls_label = 'PLS ({:.3f})'.format(score_pred_valid_pls)
    plt.plot(p*pred_valid_pls[inds_sup_valid][0],'b-.',lw=2,label=pls_label)
    plt.plot(p*y_valid[inds_sup_valid].T,'k',lw=2)
    plt.plot(p*pred_valid[inds_sup_valid].T,'r-.',lw=2)
    plt.plot(p*pred_valid_pls[inds_sup_valid].T,'b-.',lw=2)
    plt.title('Predicting Composition - Validation Error', fontsize=fs)
    plt.ylabel('Composition (%)', fontsize=fs)
    ax = plt.gca()
    ax.set_ylim((0,1*p))
    ax.set_xticks(np.arange(y.shape[1]))
    ax.set_xticklabels(names, fontsize=fs)
    ax.tick_params(axis='x',direction='out',top='off',length=10,labelsize=fs_tick)
    lgd = plt.legend(loc='center left',bbox_to_anchor=(1, 0.5))
    ax = plt.gca()
    plt.savefig(res_out+'/comp_valid.png',additional_artists=[lgd],bbox_inches='tight')
    plt.close()

    plt.plot(waves,f(_ux+data['X'][inds_sup_train]).T,'k')
    plt.plot(waves,gen_train.T,'r-.')
    plt.title('Generating Spectra - Training Error', fontsize=fs)
    plt.xlabel('Channels', fontsize=fs)
    plt.ylabel('Intensities', fontsize=fs)
    plt.tick_params(axis='both', which='major', labelsize=fs_tick)
    if force_ylim:
        plt.gca().set_ylim(ylim)
    plt.savefig(res_out+'/genspectra_train.png')
    plt.close()

    plt.plot(waves,f(_ux+data['X_valid'][inds_sup_valid]).T,'k')
    plt.plot(waves,gen_valid.T,'r-.')
    plt.title('Generating Spectra - Validation Error', fontsize=fs)
    plt.xlabel('Channels', fontsize=fs)
    plt.ylabel('Intensities', fontsize=fs)
    plt.tick_params(axis='both', which='major', labelsize=fs_tick)
    if force_ylim:
        plt.gca().set_ylim(ylim)
    plt.savefig(res_out+'/genspectra_valid.png')
    plt.close()

    if m.variational:
        for endmember, color, name in zip(endmembers,colors,names):
            plt.plot(waves,endmember,color=color,lw=2,label=name)
        for endmember_dist, color in zip(endmembers_dists,colors):
            plt.plot(waves,endmember_dist.T,'-.',color=color,lw=1)
        plt.title('Generating Endmembers with Distributions', fontsize=fs)
        plt.xlabel('Channels', fontsize=fs)
        plt.ylabel('Intensities', fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs_tick)
        lgd = plt.legend(loc='center left',bbox_to_anchor=(1, 0.5))
        ax = plt.gca()
        if force_ylim:
            ax.set_ylim(ylim)
        plt.savefig(res_out+'/endmembers_dist.png',additional_artists=[lgd],bbox_inches='tight')
        plt.close()

    for endmember, color, name in zip(endmembers,colors,names):
        plt.plot(waves,endmember,color=color,lw=2,label=name)
    plt.title('Generating Endmembers', fontsize=fs)
    plt.xlabel('Channels', fontsize=fs)
    plt.ylabel('Intensities', fontsize=fs)
    plt.tick_params(axis='both', which='major', labelsize=fs_tick)
    lgd = plt.legend(loc='center left',bbox_to_anchor=(1, 0.5))
    if m.variational:
        plt.gca().set_ylim(ax.get_ylim())
    if force_ylim:
        plt.gca().set_ylim(ylim)
    plt.savefig(res_out+'/endmembers_means.png',additional_artists=[lgd],bbox_inches='tight')
    plt.close()

    for endmember, color, name in zip(endmembers,colors,names):
        plt.plot(waves,endmember,color=color,lw=2,label=name)
    for endmember, color, name in zip(groundtruth,colors,names):
        plt.plot(waves,endmember[:len(waves)],color=color,lw=6,alpha=0.4)
    score_gen_endmembers = L2(endmembers,groundtruth[:,:len(waves)])
    if title is None:
        plt.title('Generating Endmembers with Ground Truth ({:.3f})'.format(score_gen_endmembers), fontsize=fs)
    else:
        plt.title(title+' ({:.3f})'.format(score_gen_endmembers), fontsize=fs)
    plt.xlabel('Channels', fontsize=fs)
    plt.ylabel('Intensities', fontsize=fs)
    plt.tick_params(axis='both', which='major', labelsize=fs_tick)
    lgd = plt.legend(loc='lower right', fontsize=fs)
    # lgd = plt.legend(loc='center left',bbox_to_anchor=(1, 0.5))
    if m.variational:
        plt.gca().set_ylim(ax.get_ylim())
    if force_ylim:
        plt.gca().set_ylim(ylim)
    plt.savefig(res_out+'/endmembers_means_with_groundtruth.png',additional_artists=[lgd],bbox_inches='tight')
    plt.close()

    plt.plot(waves,manifold.T,color='lightgray',lw=1,alpha=0.1)
    for endmember, color, name in zip(groundtruth,colors,names):
        plt.plot(waves,endmember[:len(waves)],color=color,lw=6,alpha=1.0)
    plt.title('Spectral Manifold', fontsize=fs)
    plt.xlabel('Channels', fontsize=fs)
    plt.ylabel('Intensities', fontsize=fs)
    plt.tick_params(axis='both', which='major', labelsize=fs_tick)
    lgd = plt.legend(loc='center left',bbox_to_anchor=(1, 0.5))
    if m.variational:
        plt.gca().set_ylim(ax.get_ylim())
    if force_ylim:
        plt.gca().set_ylim(ylim)
    plt.savefig(res_out+'/manifold.png',bbox_inches='tight')
    plt.close()

    plt.plot(waves,f(_ux+data['X_'][inds_train_x]).T,'k')
    plt.plot(waves,recon_train.T,'r-.')
    plt.title('Reconstructing Spectra - Training Error', fontsize=fs)
    plt.xlabel('Channels', fontsize=fs)
    plt.ylabel('Intensities', fontsize=fs)
    plt.tick_params(axis='both', which='major', labelsize=fs_tick)
    if force_ylim:
        plt.gca().set_ylim(ylim)
    plt.savefig(res_out+'/recon_train.png')
    plt.close()

    plt.plot(waves,f(_ux+data['X_valid'][inds_sup_valid]).T,'k')
    plt.plot(waves,recon_sup_valid.T,'r-.')
    plt.title('Reconstructing Spectra - Validation Error', fontsize=fs)
    plt.xlabel('Channels', fontsize=fs)
    plt.ylabel('Intensities', fontsize=fs)
    plt.tick_params(axis='both', which='major', labelsize=fs_tick)
    if force_ylim:
        plt.gca().set_ylim(ylim)
    plt.savefig(res_out+'/recon_valid.png')
    plt.close()

    if m.model_type in [1,2]:
        # need to use vertical lines to denote edges of datasets
        # write dataset i in middle of range on xlabel
        for i in range(z2_train.shape[1]):
            plt.plot(z2_train[:,i],'r-.')
            plt.title('Nuisance Variable '+str(i)+' - Training', fontsize=fs)
            plt.tick_params(axis='both', which='major', labelsize=fs_tick)
            plt.savefig(res_out+'/nuisance_train_'+str(i)+'.png')
            plt.close()

            plt.plot(z2_valid[:,i],'r-.')
            ax = plt.gca()
            ylim = ax.get_ylim()
            # should make this general if possible
            plt.plot([1866,1866],[-5,5],'k--')
            plt.plot([1866+1742,1866+1742],[-5,5],'k--')
            # plt.plot([1866+1742+1746,1866+1742+1746],[-5,5],'k--')
            ax.set_ylim(ylim)
            plt.title('Nuisance Variable '+str(i)+' - Validation', fontsize=fs)
            plt.tick_params(axis='both', which='major', labelsize=fs_tick)
            plt.savefig(res_out+'/nuisance_valid_'+str(i)+'.png')
            plt.close()

def KL(pred,true):
    eps = 1e-6
    _true = np.clip(true,eps,1.)
    _pred = np.clip(pred,eps,1.)
    KL = np.sum(_true*(np.log(_true)-np.log(_pred)),axis=-1).mean()
    return KL

def minmaxKL(pred,true,true_samples):
    KLs = []
    for _pred,_true in zip(pred,true):
        if np.allclose(_true,[0.3,0.3,0.4]):
            KLs += [KL(_pred,_true_sample) for _true_sample in true_samples]
    return [np.min(KLs), np.max(KLs), np.mean(KLs)]


def L2(pred,true):
    return np.linalg.norm(pred-true,axis=1).mean(axis=0)

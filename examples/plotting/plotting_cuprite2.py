import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

from itertools import product
from collections import Counter

from sklearn.cross_decomposition import PLSRegression

import numpy as np

from IPython import embed
def make_plots(m,data,colors,names,groundtruth=None,waves=None,sample_size=10,ux=0,permute=True,
               remove_mean=False,log_x=False,ylim=(0.3,1),res_out='',title=None, simplex=None):
    inds_sup_train = np.random.choice(data['X'].shape[0],size=sample_size)
    inds_sup_valid = np.random.choice(data['X_valid'].shape[0],size=sample_size)
    inds_train_x = np.random.choice(data['X_'].shape[0],size=sample_size)
    inds_train_y = np.random.choice(data['_y'].shape[0],size=sample_size)

    y = np.hstack([data['y'],1-data['y'].sum(axis=1,keepdims=True)])
    y_valid = np.hstack([data['y_valid'],1-data['y_valid'].sum(axis=1,keepdims=True)])
    y_corners = np.vstack((np.eye(data['y'].shape[1]),np.zeros(data['y'].shape[1]))).astype('float32')

    # perm = None
    # num_unique = None
    # if permute and groundtruth is not None:
    #     p = m.predict(x=groundtruth,deterministic=True)
    #     p = np.hstack([p,1-p.sum(axis=1,keepdims=True)])
    #     print(p)
    #     num_unique = len(np.unique(p.argmax(axis=1)))
    #     perm = assign_permutation(p.T)
    #     print(perm)
    #     y = y[:,perm]
    #     y_valid = y_valid[:,perm]
    #     y_corners = y_corners[perm,:]
    #     print(num_unique)

    # auto-set title and plot results (saved to res_out)
    if m.model_type == 0:
        title = 'M1'
    elif m.model_type == 1:
        title = 'M2'
    else:
        title = 'M1+2'
    if m.coeff_y_dis > 0:
        if m.coeff_y > 0:
            title += 'with Labels Untapped'
        else:
            title += r'with Supervised ($\mathbf{y} \rightarrow \mathbf{x}$)'

    if simplex is None:
        print('\tbuilding simplex for _y...')
        simplex = []
        for point in product(*([np.linspace(0,1,5)]*y.shape[1])):
            if np.sum(point) == 1:
                simplex += [point]
        simplex = np.asarray(simplex).astype('float32')
        simplex = simplex[:,:-1]

    if waves is None:
        waves = np.arange(data['X'].shape[1])

    if remove_mean:
        _ux = ux
        groundtruth = groundtruth+ux
    else:
        _ux = 0

    if log_x:
        f = lambda x: np.exp(x)
    else:
        f = lambda x: x

    force_ylim = False
    if ylim is not None:
        force_ylim = True

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
        endmembers_dists = []
        if m.variational:
            for idx_c, c in enumerate(y_corners):
                endmembers_dist = [f(_ux + m.generate(y=np.atleast_2d(c),z2=z2_gen_endmembers[idx_c:idx_c+1],deterministic=False)).squeeze() for i in range(sample_size)]
                endmembers_dists += [np.asarray(endmembers_dist)]
            endmembers_dists = endmembers_dists
    else:
        gen_train = f(_ux + m.generate(y=data['y'][inds_sup_train],deterministic=True))  # true by default for non-variational, variational default is False
        gen_valid = f(_ux + m.generate(y=data['y_valid'][inds_sup_valid],deterministic=True))
        manifold = f(_ux + m.generate(y=simplex,deterministic=True))
        endmembers = f(_ux + m.generate(y=y_corners,deterministic=True))
        endmembers_dists = []
        if m.variational:
            for idx_c, c in enumerate(y_corners):
                endmembers_dist = [f(_ux + m.generate(y=np.atleast_2d(c),deterministic=False)).squeeze() for i in range(sample_size)]
                endmembers_dists += [np.asarray(endmembers_dist)]
            endmembers_dists = endmembers_dists
    recon_train = f(_ux + m.generate(x=data['X_'][inds_train_x],deterministic=True))
    recon_sup_valid = f(_ux + m.generate(x=data['X_valid'][inds_sup_valid],deterministic=True))

    perm = None
    num_unique = None
    # if permute and groundtruth is not None:

    scores = []
    for endmember in endmembers:
        score = []
        for gt in groundtruth[:,:len(waves)]:
            offset = np.mean(gt - endmember)
            score += [-np.linalg.norm(endmember+offset-gt)]
        scores += [score]
    scores = np.array(scores)
    # scores = np.array([-np.linalg.norm(endmember-groundtruth[:,:len(waves)],axis=1) for endmember in endmembers])
    # embed()
    num_unique = len(np.unique(scores.argmax(axis=1)))
    # print(scores)
    perm = np.argsort(assign_permutation(scores))
    print('endmember permutation:\n',perm)
    endmembers_perm = endmembers[perm,:]
    if m.variational:
        endmembers_dists_perm = np.asarray(endmembers_dists)[perm]
    # print(num_unique)

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
    pred_valid = m.predict(x=data['X_valid'],deterministic=True)
    pred_valid = np.hstack([pred_valid,1-pred_valid.sum(axis=1,keepdims=True)])
    if permute:
        pred_train = pred_train[:,perm]
        pred_valid = pred_valid[:,perm]
    score_pred_train = KL(pred_train,y)
    score_pred_valid = KL(pred_valid,y_valid)

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
    plt.ylabel('Composition (%)', fontsize=fs_tick)
    ax = plt.gca()
    ax.set_ylim((0,1*p))
    ax.set_xticks(np.arange(y.shape[1]))
    ax.set_xticklabels(names, fontsize=fs)
    ax.tick_params(axis='x',direction='out',top='off',length=10,labelsize=fs_tick)
    plt.xticks(rotation=90)
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
    plt.ylabel('Composition (%)', fontsize=fs_tick)
    ax = plt.gca()
    ax.set_ylim((0,1*p))
    ax.set_xticks(np.arange(y.shape[1]))
    ax.set_xticklabels(names, fontsize=fs)
    ax.tick_params(axis='x',direction='out',top='off',length=10,labelsize=fs_tick)
    plt.xticks(rotation=90)
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
        if m.variational:
            plt.gca().set_ylim(ax.get_ylim())
        if force_ylim:
            plt.gca().set_ylim(ylim)
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
        plt.plot(waves,endmember,color=color,lw=2,label=name,zorder=1)
    for endmember, color, name in zip(groundtruth,colors,names):
        plt.plot(waves,endmember[:len(waves)],color=color,lw=6,alpha=0.4,zorder=0)
    score_gen_endmembers = L2(endmembers,groundtruth[:,:len(waves)])
    if title is None:
        plt.title('Generating Endmembers with Ground Truth ({:.3f})'.format(score_gen_endmembers), fontsize=fs)
    else:
        plt.title(title+' ({:.3f})'.format(score_gen_endmembers), fontsize=fs)
    plt.xlabel('Channels', fontsize=fs)
    plt.ylabel('Intensities', fontsize=fs)
    plt.tick_params(axis='both', which='major', labelsize=fs_tick)
    # lgd = plt.legend(loc='lower right',bbox_to_anchor=(1, 0.5))
    lgd = plt.legend(loc='center left',bbox_to_anchor=(1, 0.5))
    if m.variational:
        plt.gca().set_ylim(ax.get_ylim())
    if force_ylim:
        plt.gca().set_ylim(ylim)
    plt.savefig(res_out+'/endmembers_means_with_groundtruth.png',additional_artists=[lgd],bbox_inches='tight')
    plt.close()

    for endmember, gt, color, name in zip(endmembers_perm,groundtruth[:,:len(waves)],colors,names):
        offset = np.mean(gt - endmember)
        plt.plot(waves,endmember+offset,color=color,lw=2,label=name,zorder=1)
        plt.plot(waves,gt,color=color,lw=6,alpha=0.4,zorder=0)
    offsets = np.mean(groundtruth[:,:len(waves)]-endmembers_perm,axis=1,keepdims=True)
    score_gen_endmembers = L2(endmembers_perm+offsets,groundtruth[:,:len(waves)])
    if title is None:
        plt.title('Generating Endmembers Offset with Ground Truth ({:.3f})'.format(score_gen_endmembers), fontsize=fs)
    else:
        plt.title(title+' ({:.3f})'.format(score_gen_endmembers), fontsize=fs)
    plt.xlabel('Channels', fontsize=fs)
    plt.ylabel('Intensities', fontsize=fs)
    plt.tick_params(axis='both', which='major', labelsize=fs_tick)
    # lgd = plt.legend(loc='lower right',bbox_to_anchor=(1, 0.5))
    lgd = plt.legend(loc='center left',bbox_to_anchor=(1, 0.5))
    if m.variational:
        plt.gca().set_ylim(ax.get_ylim())
    if force_ylim:
        plt.gca().set_ylim(ylim)
    plt.savefig(res_out+'/endmembers_means_offset_with_groundtruth.png',additional_artists=[lgd],bbox_inches='tight')
    plt.close()

    N_endmembers = len(endmembers_perm)
    fig, axarr = plt.subplots(N_endmembers,sharex=True,sharey=True,figsize=(5,15))
    if title is None:
        fig.suptitle('Generating Endmembers with Groundtruth', y=1.02, fontsize=fs)
    else:
        fig.suptitle(title, y=1.02, fontsize=fs)
    lgds = []
    for idx, endmember, color, name in zip(range(N_endmembers),endmembers_perm,colors,names):
        axarr[idx].plot(waves,endmember,color=color,lw=2,label=name,zorder=1)
        lgds += [axarr[idx].legend(loc='center left',bbox_to_anchor=(1, 0.5))]
        axarr[idx].tick_params(axis='both', which='major', labelsize=fs_tick)
        if idx == N_endmembers//2:
            axarr[idx].set_ylabel('Intensities', fontsize=fs)
    plt.xlabel('Channels', fontsize=fs)
    # fig.subplots_adjust(hspace=0)
    plt.savefig(res_out+'/endmembers_means_separated.png',additional_artists=lgds,bbox_inches='tight')
    plt.close()

    fig, axarr = plt.subplots(N_endmembers,sharex=True,sharey=True,figsize=(5,15))
    if title is None:
        fig.suptitle('Generating Endmembers', y=1.02, fontsize=fs)
    else:
        fig.suptitle(title, y=1.02, fontsize=fs)
    lgds = []
    for idx, endmember, gt, color, name in zip(range(N_endmembers),endmembers_perm,groundtruth[:,:len(waves)],colors,names):
        offset = np.mean(gt - endmember)
        axarr[idx].plot(waves,endmember + offset,color=color,lw=2,label=name,zorder=1)
        axarr[idx].plot(waves,gt,color=color,lw=6,alpha=0.4,zorder=0)
        lgds += [axarr[idx].legend(loc='center left',bbox_to_anchor=(1, 0.5))]
        axarr[idx].tick_params(axis='both', which='major', labelsize=fs_tick)
        if idx == N_endmembers//2:
            axarr[idx].set_ylabel('Intensities', fontsize=fs)
    plt.xlabel('Channels', fontsize=fs)
    # fig.subplots_adjust(hspace=0)
    plt.savefig(res_out+'/endmembers_means_separated_with_groundtruth.png',additional_artists=lgds,bbox_inches='tight')
    plt.close()

    if m.variational:
        fig, axarr = plt.subplots(N_endmembers,sharex=True,sharey=True,figsize=(5,15))
        if title is None:
            fig.suptitle('Generating Endmembers with Distributions', y=1.02, fontsize=fs)
        else:
            fig.suptitle(title, y=1.02, fontsize=fs)
        lgds = []
        for idx, endmember, endmember_dist, gt, color, name in zip(range(N_endmembers),endmembers_perm,endmembers_dists_perm,groundtruth[:,:len(waves)],colors,names):
            offset = np.mean(gt - endmember)
            axarr[idx].plot(waves,endmember + offset,color=color,lw=2,label=name,zorder=1)
            axarr[idx].plot(waves,(endmember_dist+offset).T,'-.',color=color,lw=1)
            axarr[idx].plot(waves,gt,color=color,lw=6,alpha=0.4,zorder=0)
            lgds += [axarr[idx].legend(loc='center left',bbox_to_anchor=(1, 0.5))]
            axarr[idx].tick_params(axis='both', which='major', labelsize=fs_tick)
            if idx == N_endmembers//2:
                axarr[idx].set_ylabel('Intensities', fontsize=fs)
        plt.xlabel('Channels', fontsize=fs)
        # fig.subplots_adjust(hspace=0)
        plt.savefig(res_out+'/endmembers_dists_separated_with_groundtruth.png',additional_artists=lgds,bbox_inches='tight')
        plt.close()

    plt.plot(waves,manifold.T,color='lightgray',lw=6,alpha=0.1)
    for endmember, color, name in zip(groundtruth,colors,names):
        plt.plot(waves,endmember[:len(waves)],color=color,lw=2,alpha=1.0)
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
        labels = y.argmax(axis=1)
        zero_score = np.linalg.norm(y-1./y.shape[1],axis=1)<1e-3
        labels[zero_score] = y.shape[1]
        cutoffs = np.cumsum(np.bincount(labels))[:-1]
        cutoffs_valid = np.cumsum(np.bincount(y_valid.argmax(axis=1)))[:-1]
        cutoff_x = np.hstack((cutoffs[:,None],cutoffs[:,None])).T
        cutoff_x_valid = np.hstack((cutoffs_valid[:,None],cutoffs_valid[:,None])).T
        for i in range(z2_train.shape[1]):
            plt.plot(z2_train[:,i],'r-.')
            ax = plt.gca()
            ylim = ax.get_ylim()
            cutoff_y = np.ones_like(cutoff_x).astype('float')
            cutoff_y[0] *= ylim[0]
            cutoff_y[1] *= ylim[1]
            ax.set_ylim(ylim)
            plt.plot(cutoff_x,cutoff_y,'k--')
            plt.title('Nuisance Variable '+str(i)+' - Training', fontsize=fs)
            plt.tick_params(axis='both', which='major', labelsize=fs_tick)
            plt.savefig(res_out+'/nuisance_train_'+str(i)+'.png')
            plt.close()

            plt.plot(z2_valid[:,i],'r-.')
            ax = plt.gca()
            ylim = ax.get_ylim()
            # should make this general if possible
            # plt.plot([1866,1866],[-5,5],'k--')
            # plt.plot([1866+1742,1866+1742],[-5,5],'k--')
            # plt.plot([1866+1742+1746,1866+1742+1746],[-5,5],'k--')
            cutoff_y_valid = np.ones_like(cutoff_x_valid).astype('float')
            cutoff_y_valid[0] *= ylim[0]
            cutoff_y_valid[1] *= ylim[1]
            ax.set_ylim(ylim)
            plt.plot(cutoff_x_valid,cutoff_y_valid,'k--')
            plt.title('Nuisance Variable '+str(i)+' - Validation', fontsize=fs)
            plt.tick_params(axis='both', which='major', labelsize=fs_tick)
            plt.savefig(res_out+'/nuisance_valid_'+str(i)+'.png')
            plt.close()

    np.savez_compressed(res_out+'/plotfiles.npz',
                        inds_sup_train=inds_sup_train,inds_sup_valid=inds_sup_valid,
                        inds_train_x=inds_train_x,inds_train_y=inds_train_y,
                        y=y,y_valid=y_valid,y_corners=y_corners,simplex=simplex,
                        waves=waves,remove_mean=remove_mean,ux=ux,log_x=log_x,force_ylim=force_ylim,
                        data=data,gen_train=gen_train,gen_valid=gen_valid,manifold=manifold,
                        endmembers=endmembers,endmembers_dists=endmembers_dists,recon_train=recon_train,
                        recon_sup_valid=recon_sup_valid,p=p,colors=colors,names=names,
                        groundtruth=groundtruth,title=title,ylim=ylim,permute=permute,perm=perm,
                        num_unique=num_unique,offsets=offsets)

    # embed()

def KL(pred,true):
    eps = 1e-6
    _true = np.clip(true,eps,1.)
    _pred = np.clip(pred,eps,1.)
    KL = np.sum(_true*(np.log(_true)-np.log(_pred)),axis=-1).mean()
    return KL


def L2(pred,true):
    return np.linalg.norm(pred-true,axis=1).mean(axis=0)

def assign_permutation(p):
  '''
  Given a square n x n matrix of probabilities or scores where each row is assumed
  to represent a single category out of the n-categories, this method returns a labeling
  of the n rows.
  If a category is "requested" by a single row, the row is labeled with that category
  If multiple rows "request" the same category, the row with the highest score gets the category
  '''
  assert p.shape[0] == p.shape[1]
  if p.shape[0] == 1:
    return np.array([0])
  labels = np.argmax(p,axis=1)
  not_dups = np.array([item for item, count in Counter(labels).items() if count == 1])
  dups = np.array([item for item, count in Counter(labels).items() if count > 1])
  if len(dups) > 0:
    rows = [idx for idx in range(p.shape[0]) if labels[idx] in dups]
    cols = [idx for idx in range(p.shape[1]) if idx not in not_dups]
    if len(rows) == p.shape[0]:
      winner = np.argmax(p[:,dups[0]])
      rows = [idx for idx in rows if idx != winner]
      cols = [idx for idx in cols if idx != dups[0]]
    p_new = p[rows,:]
    p_new = p_new[:,cols]
    sub_labels = assign_permutation(p_new)
    labels[rows] = np.array(cols)[sub_labels]
  return labels

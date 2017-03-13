import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.cross_decomposition import PLSRegression

import numpy as np

from IPython import embed


def make_plots(m,data,colors,names,sample_size=4,ux=0,remove_mean=False,res_out='',cat=True,title=None):
    inds_sup_train = np.random.choice(data['X'].shape[0],size=sample_size)
    inds_sup_valid = np.random.choice(data['X_valid'].shape[0],size=sample_size)
    inds_train_x = np.random.choice(data['X_'].shape[0],size=sample_size)
    inds_train_y = np.random.choice(data['_y'].shape[0],size=sample_size)

    if not cat:
        y = np.hstack([data['y'],1-data['y'].sum(axis=1,keepdims=True)])
        y_valid = np.hstack([data['y_valid'],1-data['y_valid'].sum(axis=1,keepdims=True)])
        y_corners = np.vstack((np.eye(data['y'].shape[1]),np.zeros(data['y'].shape[1]))).astype('float32')
    else:
        y = data['y']
        y_valid = data['y_valid']
        y_corners = np.eye(data['y'].shape[1]).astype('float32')

    if remove_mean:
        _ux = ux
    else:
        _ux = 0

    # pls_XY = PLSRegression(n_components=8,scale=False)
    # pls_XY.fit(data['X'],y)
    # pred_train_pls = pls_XY.predict(data['X'])
    # pred_train_pls = (pred_train_pls.T/np.sum(pred_train_pls,axis=1)).T
    # pred_valid_pls = pls_XY.predict(data['X_valid'])
    # pred_valid_pls = (pred_valid_pls.T/np.sum(pred_valid_pls,axis=1)).T
    # score_pred_train_pls = KL(pred_train_pls,y)
    # score_pred_valid_pls = KL(pred_valid_pls,y_valid)

    # pls_YX = PLSRegression(n_components=min(8,y.shape[1]),scale=False)
    # pls_YX.fit(y,data['X'])
    # gen_train_pls = pls_YX.predict(y)
    # gen_valid_pls = pls_YX.predict(y_valid)
    # score_gen_train_pls = L2(gen_train_pls,data['X'])
    # score_gen_valid_pls = L2(gen_valid_pls,data['X_valid'])

    pred_train = m.predict(x=data['X'],deterministic=True)
    if cat:
        pred_train = (pred_train.T/np.sum(pred_train,axis=1)).T
    else:
        pred_train = np.hstack([pred_train,1-pred_train.sum(axis=1,keepdims=True)])
    score_pred_train = KL(pred_train,y)
    pred_valid = m.predict(x=data['X_valid'],deterministic=True)
    if cat:
        pred_valid = (pred_valid.T/np.sum(pred_valid,axis=1)).T
    else:
        pred_valid = np.hstack([pred_valid,1-pred_valid.sum(axis=1,keepdims=True)])
    score_pred_valid = KL(pred_valid,y_valid)
    if m.model_type in [1,2]:
        z2_train = m.getZ2(x=data['X'],y=data['y'],deterministic=True)
        z2_valid = m.getZ2(x=data['X_valid'],y=data['y_valid'],deterministic=True)
        z2_train_mean = z2_train.mean(axis=0)
        z2_valid_mean = z2_valid.mean(axis=0)
        z2_gen_train = z2_train_mean*np.ones_like(z2_train).astype('float32')
        z2_gen_valid = z2_valid_mean*np.ones_like(z2_valid).astype('float32')
        z2_gen_endmembers = z2_train_mean*np.ones((y_corners.shape[0],z2_train.shape[1])).astype('float32')
        gen_train = _ux + m.generate(y=data['y'][inds_sup_train],z2=z2_gen_train[inds_sup_train],deterministic=True)  # true by default for non-variational, variational default is False
        gen_valid = _ux + m.generate(y=data['y_valid'][inds_sup_valid],z2=z2_gen_valid[inds_sup_valid],deterministic=True)
        endmembers = _ux + m.generate(y=y_corners,z2=z2_gen_endmembers,deterministic=True)
        if m.variational:
            endmembers_dists = []
            for idx_c, c in enumerate(y_corners):
                endmembers_dist = [_ux + m.generate(y=np.atleast_2d(c),z2=z2_gen_endmembers[idx_c:idx_c+1],deterministic=False).squeeze() for i in range(sample_size)]
                endmembers_dists += [np.asarray(endmembers_dist)]
    else:
        gen_train = _ux + m.generate(y=data['y'][inds_sup_train],deterministic=True)  # true by default for non-variational, variational default is False
        gen_valid = _ux + m.generate(y=data['y_valid'][inds_sup_valid],deterministic=True)
        endmembers = _ux + m.generate(y=y_corners,deterministic=True)
        if m.variational:
            endmembers_dists = []
            for idx_c, c in enumerate(y_corners):
                endmembers_dist = [_ux + m.generate(y=np.atleast_2d(c),deterministic=False).squeeze() for i in range(sample_size)]
                endmembers_dists += [np.asarray(endmembers_dist)]
    recon_train = _ux + m.generate(x=data['X_'][inds_train_x],deterministic=True)
    recon_sup_valid = _ux + m.generate(x=data['X_valid'][inds_sup_valid],deterministic=True)

    fs = 24

    # change xticks to be names
    p = 100
    plt.plot(p*y[inds_sup_train][0],'k',lw=2,label='Ground Truth')
    ssdgm_label = 'SSDGM ({:.3f})'.format(score_pred_train)
    plt.plot(p*pred_train[inds_sup_train][0],'r-.',lw=2,label=ssdgm_label)
    # pls_label = 'PLS ({:.3f})'.format(score_pred_train_pls)
    # plt.plot(p*pred_train_pls[inds_sup_train][0],'b-.',lw=2,label=pls_label)
    plt.plot(p*y[inds_sup_train].T,'k',lw=2)
    plt.plot(p*pred_train[inds_sup_train].T,'r-.',lw=2)
    # plt.plot(p*pred_train_pls[inds_sup_train].T,'b-.',lw=2)
    plt.title('Predicting Digits - Training Error', fontsize=fs)
    plt.ylabel('Probability(Digit i)', fontsize=fs)
    ax = plt.gca()
    ax.set_ylim((0,1*p))
    ax.set_xticks(np.arange(y.shape[1]))
    ax.set_xticklabels(names, fontsize=fs)
    ax.tick_params(axis='x',direction='out',top='off',length=10,labelsize=12)
    lgd = plt.legend(loc='center left',bbox_to_anchor=(1, 0.5))
    ax = plt.gca()
    plt.savefig(res_out+'/digit_train.png',additional_artists=[lgd],bbox_inches='tight')
    plt.close()

    plt.plot(p*y_valid[inds_sup_valid][0],'k',lw=2,label='Ground Truth')
    ssdgm_label = 'SSDGM ({:.3f})'.format(score_pred_valid)
    plt.plot(p*pred_valid[inds_sup_valid][0],'r-.',lw=2,label=ssdgm_label)
    # pls_label = 'PLS ({:.3f})'.format(score_pred_valid_pls)
    # plt.plot(p*pred_valid_pls[inds_sup_valid][0],'b-.',lw=2,label=pls_label)
    plt.plot(p*y_valid[inds_sup_valid].T,'k',lw=2)
    plt.plot(p*pred_valid[inds_sup_valid].T,'r-.',lw=2)
    # plt.plot(p*pred_valid_pls[inds_sup_valid].T,'b-.',lw=2)
    plt.title('Predicting Digits - Validation Error', fontsize=fs)
    plt.ylabel('Probability(Digit i)', fontsize=fs)
    ax = plt.gca()
    ax.set_ylim((0,1*p))
    ax.set_xticks(np.arange(y.shape[1]))
    ax.set_xticklabels(names, fontsize=fs)
    ax.tick_params(axis='x',direction='out',top='off',length=10,labelsize=12)
    lgd = plt.legend(loc='center left',bbox_to_anchor=(1, 0.5))
    ax = plt.gca()
    plt.savefig(res_out+'/digit_valid.png',additional_artists=[lgd],bbox_inches='tight')
    plt.close()

    imshow([_ux+data['X'][inds_sup_train],gen_train])
    plt.title('Generating Digits - Training Error', fontsize=fs)
    plt.savefig(res_out+'/gendigits_train.png',bbox_inches='tight')
    plt.close()

    imshow([_ux+data['X_valid'][inds_sup_valid],gen_valid])
    plt.title('Generating Digits - Validation Error', fontsize=fs)
    plt.savefig(res_out+'/gendigits_valid.png',bbox_inches='tight')
    plt.close()

    if m.variational:
        imshow(endmembers_dists)
        if title is None:
            plt.title('Generating Endmembers with Distributions', fontsize=fs)
        else:
            plt.title(title, fontsize=fs)
        plt.savefig(res_out+'/digits_dist.png',bbox_inches='tight')
        plt.close()

    imshow([endmembers])
    if title is None:
        plt.title('Generating Endmembers', fontsize=fs)
    else:
        plt.title(title, fontsize=fs)
    plt.savefig(res_out+'/digits_means.png',bbox_inches='tight')
    plt.close()

    imshow([_ux+data['X_'][inds_train_x],recon_train])
    plt.title('Reconstructing Digits - Training Error', fontsize=fs)
    plt.savefig(res_out+'/recon_train.png',bbox_inches='tight')
    plt.close()

    imshow([_ux+data['X_valid'][inds_sup_valid],recon_sup_valid])
    plt.title('Reconstructing Digits - Validation Error', fontsize=fs)
    plt.savefig(res_out+'/recon_valid.png',bbox_inches='tight')
    plt.close()

def imshow(image_list):
    N = len(image_list)
    M = image_list[0].shape[0]
    size = int(np.sqrt(image_list[0].shape[1]))
    for n,images in enumerate(image_list):
        image_list[n] = np.hstack([np.clip(image,0,1).reshape(size,size) for image in images])
        # for m,image in enumerate(images):
        #     plt.subplot(N,M,1+m+n*M,frameon=False,xticks=[],yticks=[])
        #     plt.imshow(np.clip(image,0,1).reshape(size,size),cmap='gray')
    plt.imshow(np.vstack(image_list),cmap='gray_r')
    plt.axis('off')

def KL(pred,true):
    eps = 1e-6
    _true = np.clip(true,eps,1.)
    _pred = np.clip(pred,eps,1.)
    KL = np.sum(_true*(np.log(_true)-np.log(_pred)),axis=1).mean(axis=0)
    return KL

def L2(pred,true):
    return np.linalg.norm(pred-true,axis=1).mean(axis=0)

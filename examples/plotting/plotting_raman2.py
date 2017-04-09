import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from itertools import product

from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV

import numpy as np

from IPython import embed


def make_plots(m1,m2,data1,data2,names,sample_size=10,ux=0,
               remove_mean=False,log_x=False,ylim=None,res_out='',title=None):
    assert data1['X'].shape[0] == data2['X'].shape[0]
    assert data1['X_valid'].shape[0] == data2['X_valid'].shape[0]

    inds_sup_train = np.random.choice(data1['X'].shape[0],size=sample_size)
    inds_sup_valid = np.random.choice(data1['X_valid'].shape[0],size=sample_size)

    y = np.hstack([data1['y'],1-data1['y'].sum(axis=1,keepdims=True)])
    y_valid = np.hstack([data1['y_valid'],1-data1['y_valid'].sum(axis=1,keepdims=True)])

    if remove_mean:
        _ux = ux
    else:
        _ux = 0

    if log_x:
        f = lambda x: np.exp(x)
    else:
        f = lambda x: x

    force_ylim = False
    if ylim is not None:
        force_ylim = True

    parameters = {'n_components':[4,8,20,50,100,500]}
    pls_XY = GridSearchCV(PLSRegression(scale=False), parameters)
    pls_XY.fit(data1['X'],y)
    pls_XY = pls_XY.best_estimator_
    pred_train_pls = pls_XY.predict(data1['X'])
    pred_train_pls = (pred_train_pls.T/np.sum(pred_train_pls,axis=1)).T
    pred_valid_pls = pls_XY.predict(data1['X_valid'])
    pred_valid_pls = (pred_valid_pls.T/np.sum(pred_valid_pls,axis=1)).T
    score_pred_train_pls = KL(pred_train_pls,y)
    score_pred_valid_pls = KL(pred_valid_pls,y_valid)

    pred_train_m1 = m1.predict(x=data1['X'],deterministic=True)
    pred_train_m1 = np.hstack([pred_train_m1,1-pred_train_m1.sum(axis=1,keepdims=True)])
    score_pred_train_m1 = KL(pred_train_m1,y)
    pred_valid_m1 = m1.predict(x=data1['X_valid'],deterministic=True)
    pred_valid_m1 = np.hstack([pred_valid_m1,1-pred_valid_m1.sum(axis=1,keepdims=True)])
    score_pred_valid_m1 = KL(pred_valid_m1,y_valid)

    pred_train_m2 = m2.predict(x=data2['X'],deterministic=True)
    pred_train_m2 = np.hstack([pred_train_m2,1-pred_train_m2.sum(axis=1,keepdims=True)])
    score_pred_train_m2 = KL(pred_train_m2,y)
    pred_valid_m2 = m2.predict(x=data2['X_valid'],deterministic=True)
    pred_valid_m2 = np.hstack([pred_valid_m2,1-pred_valid_m2.sum(axis=1,keepdims=True)])
    score_pred_valid_m2 = KL(pred_valid_m2,y_valid)

    fs = 24
    fs_tick = 18

    # change xticks to be names
    p = 100

    rep_best_train = np.argmin(KL(pred_train_m1,y,avg=False))
    plt.plot(p*y[rep_best_train],'k-o',lw=2,label='Ground Truth',zorder=0)
    ssdgm_label_m1 = 'UVAE (R+L) ({:.3f})'.format(score_pred_train_m1)
    plt.plot(p*pred_train_m1[rep_best_train],'r-.o',lw=2,label=ssdgm_label_m1,zorder=2)
    ssdgm_label_m2 = 'UVAE (R) ({:.3f})'.format(score_pred_train_m2)
    plt.plot(p*pred_train_m2[rep_best_train],'g-.o',lw=2,label=ssdgm_label_m2,zorder=2)
    pls_label = 'PLS ({:.3f})'.format(score_pred_train_pls)
    plt.plot(p*pred_train_pls[rep_best_train],'b--o',lw=2,label=pls_label,zorder=1)
    plt.title('Predicting Composition - Training Error', fontsize=fs)
    plt.ylabel('Composition (%)', fontsize=fs)
    ax = plt.gca()
    ax.set_ylim((0,1.05*p))
    ax.set_xticks(np.arange(y.shape[1]))
    ax.set_xticklabels(names, fontsize=fs, rotation='vertical')
    ax.tick_params(axis='x',direction='out',top='off',length=10,labelsize=fs_tick)
    ax.grid(linestyle='--',linewidth=1)
    lgd = plt.legend(loc='center left',bbox_to_anchor=(1, 0.5))
    plt.savefig(res_out+'/comp_train_rep_best.png',additional_artists=[lgd],bbox_inches='tight')
    plt.close()

    rep_mean_train = np.argmin(np.abs(KL(pred_train_m1,y,avg=False)-score_pred_train_m1))
    plt.plot(p*y[rep_mean_train],'k-o',lw=2,label='Ground Truth',zorder=0)
    ssdgm_label_m1 = 'UVAE (R+L) ({:.3f})'.format(score_pred_train_m1)
    plt.plot(p*pred_train_m1[rep_mean_train],'r-.o',lw=2,label=ssdgm_label_m1,zorder=2)
    ssdgm_label_m2 = 'UVAE (R) ({:.3f})'.format(score_pred_train_m2)
    plt.plot(p*pred_train_m2[rep_mean_train],'g-.o',lw=2,label=ssdgm_label_m2,zorder=2)
    pls_label = 'PLS ({:.3f})'.format(score_pred_train_pls)
    plt.plot(p*pred_train_pls[rep_mean_train],'b--o',lw=2,label=pls_label,zorder=1)
    plt.title('Predicting Composition - Training Error', fontsize=fs)
    plt.ylabel('Composition (%)', fontsize=fs)
    ax = plt.gca()
    ax.set_ylim((0,1.05*p))
    ax.set_xticks(np.arange(y.shape[1]))
    ax.set_xticklabels(names, fontsize=fs, rotation='vertical')
    ax.tick_params(axis='x',direction='out',top='off',length=10,labelsize=fs_tick)
    ax.grid(linestyle='--',linewidth=1)
    lgd = plt.legend(loc='center left',bbox_to_anchor=(1, 0.5))
    plt.savefig(res_out+'/comp_train_rep_mean.png',additional_artists=[lgd],bbox_inches='tight')
    plt.close()

    rep_median_train = np.argmin(np.abs(KL(pred_train_m1,y,avg=False)-np.median(KL(pred_train_m1,y))))
    plt.plot(p*y[rep_median_train],'k-o',lw=2,label='Ground Truth',zorder=0)
    ssdgm_label_m1 = 'UVAE (R+L) ({:.3f})'.format(score_pred_train_m1)
    plt.plot(p*pred_train_m1[rep_median_train],'r-.o',lw=2,label=ssdgm_label_m1,zorder=2)
    ssdgm_label_m2 = 'UVAE (R) ({:.3f})'.format(score_pred_train_m2)
    plt.plot(p*pred_train_m2[rep_median_train],'g-.o',lw=2,label=ssdgm_label_m2,zorder=2)
    pls_label = 'PLS ({:.3f})'.format(score_pred_train_pls)
    plt.plot(p*pred_train_pls[rep_median_train],'b--o',lw=2,label=pls_label,zorder=1)
    plt.title('Predicting Composition - Training Error', fontsize=fs)
    plt.ylabel('Composition (%)', fontsize=fs)
    ax = plt.gca()
    ax.set_ylim((0,1.05*p))
    ax.set_xticks(np.arange(y.shape[1]))
    ax.set_xticklabels(names, fontsize=fs, rotation='vertical')
    ax.tick_params(axis='x',direction='out',top='off',length=10,labelsize=fs_tick)
    ax.grid(linestyle='--',linewidth=1)
    lgd = plt.legend(loc='center left',bbox_to_anchor=(1, 0.5))
    plt.savefig(res_out+'/comp_train_rep_median.png',additional_artists=[lgd],bbox_inches='tight')
    plt.close()

    plt.plot(p*y[inds_sup_train][0],'k',lw=2,label='Ground Truth')
    ssdgm_label_m1 = 'UVAE (R+L) ({:.3f})'.format(score_pred_train_m1)
    plt.plot(p*pred_train_m1[inds_sup_train][0],'r-.',lw=2,label=ssdgm_label_m1)
    ssdgm_label_m2 = 'UVAE (R) ({:.3f})'.format(score_pred_train_m2)
    plt.plot(p*pred_train_m2[inds_sup_train][0],'g-.',lw=2,label=ssdgm_label_m2)
    pls_label = 'PLS ({:.3f})'.format(score_pred_train_pls)
    plt.plot(p*pred_train_pls[inds_sup_train][0],'b-.',lw=2,label=pls_label)
    plt.plot(p*y[inds_sup_train].T,'k',lw=2)
    plt.plot(p*pred_train_m1[inds_sup_train].T,'r-.',lw=2)
    plt.plot(p*pred_train_m2[inds_sup_train].T,'g-.',lw=2)
    plt.plot(p*pred_train_pls[inds_sup_train].T,'b-.',lw=2)
    plt.title('Predicting Composition - Training Error', fontsize=fs)
    plt.ylabel('Composition (%)', fontsize=fs)
    ax = plt.gca()
    ax.set_ylim((0,1.05*p))
    ax.set_xticks(np.arange(y.shape[1]))
    ax.set_xticklabels(names, fontsize=fs, rotation='vertical')
    ax.tick_params(axis='x',direction='out',top='off',length=10,labelsize=fs_tick)
    ax.grid(linestyle='--',linewidth=1)
    lgd = plt.legend(loc='center left',bbox_to_anchor=(1, 0.5))
    plt.savefig(res_out+'/comp_train.png',additional_artists=[lgd],bbox_inches='tight')
    plt.close()

    rep_best_valid = np.argmin(KL(pred_valid_m1,y,avg=False))
    plt.plot(p*y[rep_best_valid],'k-o',lw=2,label='Ground Truth',zorder=0)
    ssdgm_label_m1 = 'UVAE (R+L) ({:.3f})'.format(score_pred_valid_m1)
    plt.plot(p*pred_valid_m1[rep_best_valid],'r-.o',lw=2,label=ssdgm_label_m1,zorder=2)
    ssdgm_label_m2 = 'UVAE (R) ({:.3f})'.format(score_pred_valid_m2)
    plt.plot(p*pred_valid_m2[rep_best_valid],'g-.o',lw=2,label=ssdgm_label_m2,zorder=2)
    pls_label = 'PLS ({:.3f})'.format(score_pred_valid_pls)
    plt.plot(p*pred_valid_pls[rep_best_valid],'b--o',lw=2,label=pls_label,zorder=1)
    plt.title('Predicting Composition - Validation Error', fontsize=fs)
    plt.ylabel('Composition (%)', fontsize=fs)
    ax = plt.gca()
    ax.set_ylim((0,1.05*p))
    ax.set_xticks(np.arange(y.shape[1]))
    ax.set_xticklabels(names, fontsize=fs, rotation='vertical')
    ax.tick_params(axis='x',direction='out',top='off',length=10,labelsize=fs_tick)
    ax.grid(linestyle='--',linewidth=1)
    lgd = plt.legend(loc='center left',bbox_to_anchor=(1, 0.5))
    plt.savefig(res_out+'/comp_valid_rep_best.png',additional_artists=[lgd],bbox_inches='tight')
    plt.close()

    rep_mean_valid = np.argmin(np.abs(KL(pred_valid_m1,y,avg=False)-score_pred_valid_m1))
    plt.plot(p*y[rep_mean_valid],'k-o',lw=2,label='Ground Truth',zorder=0)
    ssdgm_label_m1 = 'UVAE (R+L) ({:.3f})'.format(score_pred_valid_m1)
    plt.plot(p*pred_valid_m1[rep_mean_valid],'r-.o',lw=2,label=ssdgm_label_m1,zorder=2)
    ssdgm_label_m2 = 'UVAE (R) ({:.3f})'.format(score_pred_valid_m2)
    plt.plot(p*pred_valid_m2[rep_mean_valid],'g-.o',lw=2,label=ssdgm_label_m2,zorder=2)
    pls_label = 'PLS ({:.3f})'.format(score_pred_valid_pls)
    plt.plot(p*pred_valid_pls[rep_mean_valid],'b--o',lw=2,label=pls_label,zorder=1)
    plt.title('Predicting Composition - Validation Error', fontsize=fs)
    plt.ylabel('Composition (%)', fontsize=fs)
    ax = plt.gca()
    ax.set_ylim((0,1.05*p))
    ax.set_xticks(np.arange(y.shape[1]))
    ax.set_xticklabels(names, fontsize=fs, rotation='vertical')
    ax.tick_params(axis='x',direction='out',top='off',length=10,labelsize=fs_tick)
    ax.grid(linestyle='--',linewidth=1)
    lgd = plt.legend(loc='center left',bbox_to_anchor=(1, 0.5))
    plt.savefig(res_out+'/comp_valid_rep_mean.png',additional_artists=[lgd],bbox_inches='tight')
    plt.close()

    rep_median_valid = np.argmin(np.abs(KL(pred_valid_m1,y,avg=False)-np.median(KL(pred_valid_m1,y))))
    plt.plot(p*y[rep_median_valid],'k-o',lw=2,label='Ground Truth',zorder=0)
    ssdgm_label_m1 = 'UVAE (R+L) ({:.3f})'.format(score_pred_valid_m1)
    plt.plot(p*pred_valid_m1[rep_median_valid],'r-.o',lw=2,label=ssdgm_label_m1,zorder=2)
    ssdgm_label_m2 = 'UVAE (R) ({:.3f})'.format(score_pred_valid_m2)
    plt.plot(p*pred_valid_m2[rep_median_valid],'g-.o',lw=2,label=ssdgm_label_m2,zorder=2)
    pls_label = 'PLS ({:.3f})'.format(score_pred_valid_pls)
    plt.plot(p*pred_valid_pls[rep_median_valid],'b--o',lw=2,label=pls_label,zorder=1)
    plt.title('Predicting Composition - Validation Error', fontsize=fs)
    plt.ylabel('Composition (%)', fontsize=fs)
    ax = plt.gca()
    ax.set_ylim((0,1.05*p))
    ax.set_xticks(np.arange(y.shape[1]))
    ax.set_xticklabels(names, fontsize=fs, rotation='vertical')
    ax.tick_params(axis='x',direction='out',top='off',length=10,labelsize=fs_tick)
    ax.grid(linestyle='--',linewidth=1)
    lgd = plt.legend(loc='center left',bbox_to_anchor=(1, 0.5))
    plt.savefig(res_out+'/comp_valid_rep_median.png',additional_artists=[lgd],bbox_inches='tight')
    plt.close()

    plt.plot(p*y_valid[inds_sup_valid][0],'k',lw=2,label='Ground Truth')
    ssdgm_label_m1 = 'UVAE (R+L) ({:.3f})'.format(score_pred_valid_m1)
    plt.plot(p*pred_valid_m1[inds_sup_valid][0],'r-.',lw=2,label=ssdgm_label_m1)
    ssdgm_label_m2 = 'UVAE (R) ({:.3f})'.format(score_pred_valid_m2)
    plt.plot(p*pred_valid_m2[inds_sup_valid][0],'g-.',lw=2,label=ssdgm_label_m2)
    pls_label = 'PLS ({:.3f})'.format(score_pred_valid_pls)
    plt.plot(p*pred_valid_pls[inds_sup_valid][0],'b-.',lw=2,label=pls_label)
    plt.plot(p*y_valid[inds_sup_valid].T,'k',lw=2)
    plt.plot(p*pred_valid_m1[inds_sup_valid].T,'r-.',lw=2)
    plt.plot(p*pred_valid_m2[inds_sup_valid].T,'g-.',lw=2)
    plt.plot(p*pred_valid_pls[inds_sup_valid].T,'b-.',lw=2)
    plt.title('Predicting Composition - Validation Error', fontsize=fs)
    plt.ylabel('Composition (%)', fontsize=fs)
    ax = plt.gca()
    ax.set_ylim((0,1.05*p))
    ax.set_xticks(np.arange(y.shape[1]))
    ax.set_xticklabels(names, fontsize=fs, rotation='vertical')
    ax.tick_params(axis='x',direction='out',top='off',length=10,labelsize=fs_tick)
    ax.grid(linestyle='--',linewidth=1)
    lgd = plt.legend(loc='center left',bbox_to_anchor=(1, 0.5))
    plt.savefig(res_out+'/comp_valid.png',additional_artists=[lgd],bbox_inches='tight')
    plt.close()

def KL(pred,true,avg=True):
    eps = 1e-6
    _true = np.clip(true,eps,1.)
    _pred = np.clip(pred,eps,1.)
    KL = np.sum(_true*(np.log(_true)-np.log(_pred)),axis=-1)
    if avg:
        return KL.mean()
    else:
        return KL

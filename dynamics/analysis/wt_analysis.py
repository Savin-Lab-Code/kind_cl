# monitors loss and basic metrics of an RNN optimizatio

from os.path import exists
import pickle
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.linear_model import LinearRegression
from scipy.stats import ranksums, wilcoxon
import statsmodels.api as sm
from dynamics.analysis import state_analysis as sta
from dynamics.utils.utils import CPU_Unpickler, opsbase, mwa, displ


# functions that calculate optimal state values and wait times---------------------
def topt_fun(R, ops):
    # uses optimal wait time formula to find time beyond which you should opt out
    pnocatch = (1 - ops['pcatch'])
    topt = ops['lambda'] * (np.log(R * pnocatch) - np.log(-ops['w'] * ops['lambda']))
    # allow for possible negative values with large time penalties
    return np.max([0, topt])


def V_pi_t(t, R, ops):
    # estimated reward earned for offer R, given a policy of waiting until t
    lam = ops['lambda']
    tex = lam * (1 - np.exp(-t / lam))
    Pt = (1 - np.exp(-t / lam))
    V_nocatch = (1 - ops['pcatch']) * (Pt * R + ops['w'] * tex + ops['Roo'] * np.exp(-t / lam))
    V_catch = ops['pcatch'] * (t * ops['w'] + ops['Roo'])

    V = V_nocatch + V_catch
    return V


def V_pi_topt(_, R, ops):
    topt = topt_fun(R, ops)
    V = V_pi_t(topt, R, ops)
    return V


# the data-driven code----------------------------------------
def parsesimulation(dat, ops, allreg=True):
    """
    generates wait time data for different volume, and different blocks
    will also calculate linear regression for either different blocks or just mixed blocks (useful for catch training)
    :param dat: (dict) output from session simulation
    :param ops: (dict) options for simulation parmareters
    :param allreg: (bool) for doing regressions on all blocks (True) or not (False)
    :return: wait times by block and volume, regression results, counts of action types, and reward rate
    """
    # extract outputs

    warnings.filterwarnings("ignore", message="divide by zero*")
    warnings.filterwarnings("ignore", message="invalid value encountered*")
    warnings.filterwarnings("ignore", message="Mean of empty slice")  # loops over all vols in each block, gives 0 mean
    warnings.filterwarnings("ignore", message="Degrees of freedom*")  # same, but for std
    warnings.filterwarnings("ignore", message="Creating an ndarray*")
    warnings.filterwarnings("ignore", message="kurtosistest only valid*")
    warnings.filterwarnings("ignore", message="omni_normtest is not valid*")
    warnings.filterwarnings("ignore", message="Sample size too small for normal approximation*")
    warnings.filterwarnings("ignore", message="Exact p-value calculation does not work if there are ties*")

    dl = ops['displevel']  # the verbosity level for print statements

    outcomes = np.array(dat['outcomes'])
    iscatch = np.array(dat['iscatch'])
    rewvols = np.array(dat['rewvols'])  # rewarded vol on each trial
    wt_all = np.array(dat['wt'])*ops['dt']  # wait time, s
    blocks = np.array(dat['blocks'])
    if 'w_all' not in dat:
        rho_all = -np.nanmean(np.array(ops['w']), axis=0) / ops['dt']
    else:
        rho_all = -np.nanmean(np.array(dat['w_all']), axis=0) / ops['dt']

    Vvec = ops['Vvec']
    wt_catch = wt_all[iscatch]
    rewvol_catch = rewvols[iscatch]

    # get average and SEM WT
    wt = np.array([np.mean(wt_catch[rewvol_catch == k]) for k in Vvec])
    wt_sem = np.array(
        [np.std(wt_catch[rewvol_catch == k] /
                np.sqrt(np.sum(rewvol_catch == k))) for k in Vvec])

    if ops['useblocks']:
        # get average and SEM WT
        mask_block_mixed = blocks[iscatch] == 0
        wt_mixed = np.array([np.nanmean(wt_catch[(rewvol_catch == k) & mask_block_mixed]) for k in Vvec])
        wt_mixed_sem = np.array(
            [np.std(wt_catch[rewvol_catch == k] /
                    np.sqrt(np.sum((rewvol_catch == k) & mask_block_mixed))) for k in Vvec])

        mask_block_high = blocks[iscatch] == 1
        wt_high = np.array([np.nanmean(wt_catch[(rewvol_catch == k) & mask_block_high]) for k in Vvec])
        wt_high_sem = np.array(
            [np.std(wt_catch[rewvol_catch == k] /
                    np.sqrt(np.sum((rewvol_catch == k) & mask_block_high))) for k in Vvec])

        mask_block_low = blocks[iscatch] == 2
        wt_low = np.array([np.nanmean(wt_catch[(rewvol_catch == k) & mask_block_low]) for k in Vvec])
        wt_low_sem = np.array(
            [np.std(wt_catch[rewvol_catch == k] /
                    np.sqrt(np.sum((rewvol_catch == k) & mask_block_low))) for k in Vvec])

        # calculate rank sum and sign rank between high and low
        high20 = np.array(wt_catch[(rewvol_catch == 20) & mask_block_high])
        low20 = np.array(wt_catch[(rewvol_catch == 20) & mask_block_low])
        if len(high20) > 5 and len(low20) > 5:
            p = ranksums(high20, low20).pvalue
        else:
            displ('high vs. low comparison: not enough data for ranksum', 4, dl)
            p = np.nan
        nsamp = np.min([len(high20), len(low20)])
        if nsamp > 5:
            try:
                p_sr = wilcoxon(high20[:nsamp], low20[:nsamp]).pvalue
            except ValueError:
                displ('the wilcoxon rank test failed because of maybe a tie/identical vals. here are the vals ', 4, dl)
                displ('high wait times', 4, dl)
                displ(high20[:nsamp], 4, dl)
                displ('low wait times', 4, dl)
                displ(low20[:nsamp])
                displ('setting to nan', 4, dl)
                p_sr = np.nan

        else:
            p_sr = np.nan
            displ('high vs. low comparison: not enough data for sign rank', 4, dl)

        # get slope, coeff, and R^2 per fit

        # do the log-transform of volumes here. no need before this
        X = np.expand_dims(np.log2(rewvol_catch), axis=1)

        reg = LinearRegression().fit(X, wt_catch)
        R2 = reg.score(X, wt_catch)
        slope = reg.coef_
        b = reg.intercept_
        # check stuf with statsmodels
        X2 = sm.add_constant(X)
        est = sm.OLS(wt_catch, X2)
        est2 = est.fit()
        try:
            p_all = est2.summary2().tables[1]['P>|t|']['x1']  # p value of coeff
        except KeyError:
            displ('the linear regression using stats methods did something wnoky. not saving p values', 4, dl)
            p_all = np.nan

        trange = slope*(np.max(X)-np.min(X))  # wait time range
        try:
            X_mixed = np.expand_dims(np.log2(rewvol_catch[mask_block_mixed]), axis=1)
            reg_mixed = LinearRegression().fit(X_mixed, wt_catch[mask_block_mixed])
            R2_mixed = reg_mixed.score(X_mixed, wt_catch[mask_block_mixed])
            slope_mixed = reg_mixed.coef_
            b_mixed = reg_mixed.intercept_

            X2 = sm.add_constant(X_mixed)
            est = sm.OLS(wt_catch[mask_block_mixed], X2)
            est2 = est.fit()
        except ValueError:
            displ('mixed block issue with regression', 4, dl)
            slope_mixed = np.nan
            b_mixed = np.nan
            R2_mixed = np.nan

        try:
            p_mixed = est2.summary2().tables[1]['P>|t|']['x1']
            trange_mixed = slope_mixed * (np.max(X_mixed) - np.min(X_mixed))
        except KeyError:
            displ('soemthing failed in stats model estimation of pvalue for mixed blocks. making nan', 4, dl)
            p_mixed = np.nan
            trange_mixed = np.nan
        except ValueError:
            displ('looks like you maybe didnt have enough mixed block catch trials to do this regression?', 4, dl)
            trange_mixed = np.nan
            p_mixed = np.nan

        if allreg:
            X_high = np.expand_dims(np.log2(rewvol_catch[mask_block_high]), axis=1)
            try:
                reg_high = LinearRegression().fit(X_high, wt_catch[mask_block_high])
                R2_high = reg_high.score(X_high, wt_catch[mask_block_high])
                slope_high = reg_high.coef_
                b_high = reg_high.intercept_

                X2 = sm.add_constant(X_high)
                est = sm.OLS(wt_catch[mask_block_high], X2)
                est2 = est.fit()
                trange_high = slope_high * (np.max(X_high) - np.min(X_high))

            except ValueError:
                displ("issue in high regression", 4, dl)
                R2_high = np.nan
                slope_high = np.nan
                b_high = np.nan
                trange_high = np.nan

            try:
                p_high = est2.summary2().tables[1]['P>|t|']['x1']
            except KeyError:
                displ('somethign wrong with stats model regression for high blocks. making nan', 4, dl)
                p_high = np.nan

        else:
            b_high = None
            slope_high = None
            R2_high = None
            p_high = None
            trange_high = None

        if allreg:
            X_low = np.expand_dims(np.log2(rewvol_catch[mask_block_low]), axis=1)
            try:
                reg_low = LinearRegression().fit(X_low, wt_catch[mask_block_low])
                R2_low = reg_low.score(X_low, wt_catch[mask_block_low])
                slope_low = reg_low.coef_
                b_low = reg_low.intercept_
                X2 = sm.add_constant(X_low)
                est = sm.OLS(rewvol_catch[mask_block_low], X2)
                est2 = est.fit()
                trange_low = slope_low * (np.max(X_low) - np.min(X_low))
            except ValueError:
                displ('not enough data for low regression', 4, dl)
                R2_low = np.nan
                slope_low = np.nan
                b_low = np.nan
                trange_low = np.nan

            try:
                p_low = est2.summary2().tables[1]['P>|t|']['x1']
            except KeyError:
                displ('something wrong with stats models regression for low. making nan', 4, dl)
                p_low = np.nan

        else:
            b_low = None
            slope_low = None
            R2_low = None
            p_low = None
            trange_low = None

        wt_dict = {'wt': wt, 'wt_mixed': wt_mixed, 'wt_high': wt_high, 'wt_low': wt_low,
                   'wt_sem': wt_sem, 'wt_mixed_sem': wt_mixed_sem, 'wt_high_sem': wt_high_sem, 'wt_low_sem': wt_low_sem,
                   'p_ranksum': p, 'p_signrank': p_sr}

        linreg = {'b': b, 'b_mixed': b_mixed, 'b_high': b_high, 'b_low': b_low,
                  'm': slope, 'm_mixed': slope_mixed, 'm_high': slope_high, 'm_low': slope_low,
                  'R2': R2, 'R2_mixed': R2_mixed, 'R2_high': R2_high, 'R2_low': R2_low,
                  'p': p_all, 'p_high': p_high, 'p_mixed': p_mixed, 'p_low': p_low,
                  'trange': trange, 'trange_high': trange_high, 'trange_low': trange_low,
                  'trange_mixed': trange_mixed}

    else:

        # get slope, coeff, and R^2 per fit
        if ops['pcatch'] == 0:
            wt_use = wt_all
            rewvol_use = rewvols
        else:
            wt_use = wt_catch
            rewvol_use = rewvol_catch

        X = np.expand_dims(np.log2(rewvol_use), axis=1)
        reg = LinearRegression().fit(X, wt_use)
        R2 = reg.score(X, wt_use)
        slope = reg.coef_
        b = reg.intercept_
        # check stuf with statsmodels
        X2 = sm.add_constant(X)
        est = sm.OLS(wt_use, X2)
        est2 = est.fit()
        p_all = est2.summary2().tables[1]['P>|t|']['x1']  # p value of coeff
        trange = slope * (np.max(X) - np.min(X))  # wait time range

        wt_dict = {'wt': wt, 'wt_sem': wt_sem}
        linreg = {'b': b, 'm': slope, 'R2': R2, 'p': p_all, 'trange': trange}

    # get number of optouts, win
    n_win = np.sum(outcomes == 1) / ops['ns']
    n_oo = np.sum(outcomes == 0) / ops['ns']
    n_vio = np.sum(outcomes == -1) / ops['ns']

    numdict = {'win': n_win, 'optout': n_oo, 'vio': n_vio}

    return wt_dict, linreg, numdict, rho_all, ops


# TODO: I think the deprecated
def extractloss(fname):
    """I am tired of kernels dying because of huge files.
    this will extract loss and loss_aux from .dat files
    and put in .loss files"""

    L = {}
    if exists(fname + '.dat'):
        dat = CPU_Unpickler(open(fname + '.dat', 'rb')).load()
        L['loss'] = dat['loss']
        L['loss_aux'] = dat['loss_aux']
    return L


# TODO I think this deprecated
def loaddata_loss(datapath, num, nmax=51):
    """
    will load in .loss files and concatenate
    :param datapath:
    :param num:
    :param nmax:
    :return:
    """

    # list for for saved stuff
    L_catch = []  # main loss
    Laux_catch = []  # auxiliary losses
    L_block = []
    Laux_block = []

    for k in range(1, nmax):
        print(k, end='\r')
        fname = datapath + 'rnn_curric_' + str(num) + '_catch_' + str(k)
        if exists(fname + '.loss'):
            dat = pickle.load(open(fname + '.loss', 'rb'))
            L_catch.append(dat['loss'])
            Laux_catch.append(dat['loss_aux'])

        fname = datapath + 'rnn_curric_' + str(num) + '_block_' + str(k)
        if exists(fname + '.loss'):
            dat = pickle.load(open(fname + '.loss', 'rb'))
            L_block.append(dat['loss'])
            Laux_block.append(dat['loss_aux'])

    return L_catch, Laux_catch, L_block, Laux_block


def flatten_loss(X, wt):
    """makes loss into long vector"""
    trial_idx = np.cumsum(wt)
    nbatch = len(X)
    ns = len(X[0])
    nt = (nbatch - 1) * ns + len(X[-1])
    X_flat = np.zeros((nt,))
    ind = 0
    for j in range(nbatch):
        X_flat[ind:ind + ns] = np.array(X[j])
        ind += ns

    return X_flat, trial_idx


# TODO: this might be deprecated
def loaddata_stats(datapath, num, nmax=51):
    # will load the statistics of all files in a folder and extract the following
    # loss, reward rate, wt slope, adaptation,

    # list for for saved stuff
    L_catch = []  # main loss
    Laux_catch = []  # auxiliary losses
    rho_catch = []  # reward rate
    stat_catch = []  # statistics list that has things like wait times and regression data

    L_block = []
    Laux_block = []
    rho_block = []
    stat_block = []

    ops = opsbase()
    dt = ops['dt']

    for k in range(1, nmax):
        print(k, end='\r')
        fname = datapath + 'rnn_curric_' + str(num) + '_catch_' + str(k)
        if exists(fname + '.stats'):
            dat_stat = pickle.load(open(fname + '.stats', 'rb'))
            dat = CPU_Unpickler(open(fname + '.dat', 'rb')).load()
            L_catch.append(dat['loss'])
            Laux_catch.append(dat['loss_aux'])
            rho_catch.append(np.array(dat['rewardrate_pergradstep']) / dt)
            stat_catch.append(dat_stat)

        fname = datapath + 'rnn_curric_' + str(num) + '_block_' + str(k)
        if exists(fname + '.stats'):
            dat_stat = pickle.load(open(fname + '.stats', 'rb'))
            dat = CPU_Unpickler(open(fname + '.dat', 'rb')).load()
            L_block.append(dat['loss'])
            Laux_block.append(dat['loss_aux'])
            rho_block.append(np.array(dat['rewardrate_pergradstep']) / dt)
            stat_block.append(dat_stat)

    return L_catch, Laux_catch, rho_catch, stat_catch, L_block, Laux_block, rho_block, stat_block


def plotloss_single(Xcatch, Xblock, name='L', win=40, ind=None):
    """
     will flatten and plot a given loss type. uses output of loaddat_stats
    :param Xcatch:
    :param Xblock:
    :param name:
    :param win:
    :param ind: if ind is supplied, then it is a nested loss function from an auxiliary loss
    :return:
    """

    Xcatch_flat = []
    Xblock_flat = []
    xticks_catch = []
    xticks_block = []

    for k in range(len(Xcatch)):
        if ind is None:
            Xcatch_flat_k = mwa(np.array(Xcatch[k]).flatten(), win)  # total loss

        else:
            Xcatch_flat_k = mwa(np.array([m[ind] for m in Xcatch[k]]).flatten(), win)  # auxiliary loss
        xticks_catch.append(len(Xcatch_flat_k))
        Xcatch_flat.extend(Xcatch_flat_k)

    for k in range(len(Xblock)):
        if ind is None:
            Xblock_flat_k = mwa(np.array(Xblock[k]).flatten(), win)  # total loss

        else:
            Xblock_flat_k = mwa(np.array([m[ind] for m in Xblock[k]]).flatten(), win)  # auxiliary loss
        xticks_block.append(len(Xblock_flat_k))
        Xblock_flat.extend(Xblock_flat_k)

    # plot over all training. add tick marks to show where 10k trials started and ended
    ns1 = len(Xcatch_flat)
    ns2 = len(Xblock_flat)
    print([ns1, ns2])
    xticks_cumul = np.cumsum(np.array(xticks_catch))
    xticks_cumul_block = xticks_cumul[-1] + np.cumsum(np.array(xticks_block))
    ymax = np.max([np.max(Xcatch_flat), np.max(Xblock_flat)]) * 1.3
    ymin = np.min([np.min(Xcatch_flat), np.min(Xblock_flat)]) * 1.3
    for k in xticks_cumul:
        plt.vlines(k, ymin=ymin, ymax=ymax, color='gray')
    for k in xticks_cumul_block:
        plt.vlines(k, ymin=ymin, ymax=ymax, color='gray')

    plt.plot(range(ns1), Xcatch_flat, label='catch training')
    plt.plot(range(ns1, ns1 + ns2), Xblock_flat, label='block training')

    plt.title('moving win average of ' + name + ' (' + str(win) + ' grad steps win)')
    plt.legend()
    plt.xlabel('grad steps (smoothed)')
    plt.ylabel(name)

    plt.show()

    return Xcatch_flat, Xblock_flat


# TODO: deprecated
def plotloss_all(L_catch, Laux_catch, rho_catch, stat_catch, L_block, Laux_block, rho_block, stat_block):
    """
    aggregate function that will plot all of the relevatnt things tracked over each chunk of data
    :param L_catch:
    :param Laux_catch:
    :param rho_catch:
    :param stat_catch:
    :param L_block:
    :param Laux_block:
    :param rho_block:
    :param stat_block:
    :return:
    """
    ns = 500
    # L_catch, Laux_catch, rho_catch, stat_catch, L_block, Laux_block, rho_block, stat_block = loaddata_stats(
    # datapath, nmax = 51)

    # plot specific ones
    plotloss_single(L_catch, L_block, name='total loss', win=40, ind=None)
    plotloss_single(Laux_catch, Laux_block, name='policy loss', win=40, ind=0)
    plotloss_single(Laux_catch, Laux_block, name='value loss', win=40, ind=1)
    plotloss_single(Laux_catch, Laux_block, name='kind loss', win=40, ind=2)
    plotloss_single(Laux_catch, Laux_block, name='entropy loss', win=40, ind=3)

    # handle reward rate

    plotloss_single(rho_catch, rho_block, name='reward rate', win=40, ind=None)

    # extract things like slope, intercept, log(p), adaptation
    pvals_catch = [np.log10(s[1]['p']) * np.ones((ns,)) for s in stat_catch]
    slopes_catch = [s[1]['m'] * np.ones((ns,)) for s in stat_catch]
    intercepts_catch = [s[1]['b'] * np.ones((ns,)) for s in stat_catch]
    pvals_block = [np.log10(s[1]['p_mixed']) * np.ones((ns,)) for s in stat_block]
    slopes_block = [s[1]['m_mixed'] * np.ones((ns,)) for s in stat_block]
    intercepts_block = [s[1]['b_mixed'] * np.ones((ns,)) for s in stat_block]
    adapt_20_ratios_block = [s[0]['wt_low'][2] / s[0]['wt_high'][2] * np.ones((ns,)) for s in stat_block]
    adapt_20_ratios_catch = np.zeros((len(intercepts_catch),))

    plotloss_single(slopes_catch, slopes_block, name='slope', win=1, ind=None)
    plotloss_single(pvals_catch, pvals_block, name='log(p)', win=1, ind=None)
    plotloss_single(intercepts_catch, intercepts_block, name='intercept (s)', win=1, ind=None)

    plotloss_single(adapt_20_ratios_catch, adapt_20_ratios_block, name='wt ratio', win=1, ind=None)


def modelagnostic_test(dat, dt=0.05, useRidge=True, mixedonly=False, usecrossterm=True, zscore=False):
    """
    checks if wait time on 20ul trials, conditioned on prev trial, has significant diffs. if so, then model-free
    also does linear regression of trial history on wait time and looks for significant weights
    :param dat: dat struct from session run
    :param dt: time resolution, used since data counts in # samples, not in time
    :param useRidge: (bool) for using Ridge (L2) or Lasso (L1) regularized regression
    :param mixedonly: (bool) use mixed block and catch only (True) for the history regression?
    :param usecrossterm: (bool) use the reward x block cross term (True) or not (False)
    :param zscore: (bool) should you z-score the wait times for the anlaysis? default is false for back-compatibility
    :return: wait times, regression weights, regression p values
    """
    block = np.array(dat['blocks'])
    rew = np.array(dat['rewvols'])  # don't do log volume since only used for masking
    iscatch = np.array(dat['iscatch'])
    prevrew = np.concatenate(([0], rew[:-1]), axis=0)
    wt = np.array(dat['wt'])*dt

    """ To assess wait time sensitivity to previous offers, 
    we focused on 20μL catch trials in mixed blocks only. 
    We z-scored the wait times of these trials separately."""
    mask20 = (rew == 20) & (iscatch) & (block == 0)
    if zscore:
        wt = (wt-np.nanmean(wt[mask20]))/np.nanstd(wt[mask20])

    mask_lt = (rew == 20) & (iscatch) & (block == 0) & (prevrew < 20)
    mask_gt = (rew == 20) & (iscatch) & (block == 0) & (prevrew > 20)
    # mask_norm = (rew == 20) & (iscatch) & (block == 0)

    wt_lt = wt[mask_lt]
    wt_gt = wt[mask_gt]

    # rank sum test, if there are enough samples. chose 5 samples arbitrarily. having 0 is the big issue
    if sum(mask_lt) > 5 and sum(mask_gt) > 5:
        try:
            p = ranksums(wt_lt, wt_gt).pvalue
        except ValueError:
            print('rank sum test will fail. look like all values are the same (likely timeouts)')
            p = np.nan

    else:
        print('skipping rank sum test. not enough samples')
        p = np.nan

    # sign rank test
    nsamp = np.min([len(wt_lt), len(wt_gt)])
    if nsamp > 5:
        try:
            p_sr = wilcoxon(wt_lt[:nsamp], wt_gt[:nsamp]).pvalue
        except ValueError:
            print('sign rank test will fail. look like all values are the same (likely timeouts)')
            p_sr = np.nan
    else:
        print('skipping sign rank test. not enough samples')
        p_sr = np.nan

    # linear regression of current wait time on reward offer

    rewvols = np.log2(np.array(dat['rewvols']))  # do in log volume
    wt = np.array(dat['wt'])
    if mixedonly:
        m = (iscatch) & (block == 0)
    else:
        m = iscatch

    if zscore:
        wt = (wt - np.nanmean(wt[m]))/np.nanstd(wt[m])

    Y = wt[m]

    ns = np.sum(m)
    ntrialback = 6

    X = np.zeros((ns, ntrialback + 1))
    X[:, 0] = rewvols[m]
    for k in range(ntrialback):
        pvol_k = np.insert(rewvols[:-(k + 1)], 0, np.zeros((k + 1,)))
        X[:, k + 1] = pvol_k[m]

    # running gridsearch
    alphavec = [1e-4, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
    if useRidge:
        m_hist, _, rs_hist, _, _ = sta.ridge_wgridsearch(X, Y, alphavec=alphavec)
    else:
        m_hist, _, rs_hist, _, _ = sta.lasso_wgridsearch(X, Y, alphavec=alphavec)

    # calculate pvalues
    X2 = sm.add_constant(X)
    est = sm.OLS(Y, X2)
    est2 = est.fit()
    pvec_hist = [est2.summary2().tables[1]['P>|t|']['x' + str(k)] for k in range(1, ntrialback+2)]

    # do same regression, but include current block as coefficient
    # remap block to something ordinal

    # regardless of usemixed, you need all blocks here
    m = iscatch
    ns = np.sum(m)
    Y = wt[m]  # zscore boolen handled from above
    block = np.array(dat['blocks'])
    block[block == 0] = 10  # mixed
    block[block == 1] = 20  # high
    block[block == 2] = -10  # low
    block[block == -10] = 3  # new low
    block[block == 10] = 2  # new mixed
    block[block == 20] = 1  # new high

    Xnew = np.zeros((ns, ntrialback + 3))
    Xnew[:, 0] = rewvols[m]
    for k in range(ntrialback):
        # pvol_k = np.insert(rewvols[:-(k + 1)], 0, np.zeros((k + 1,)))  # maybe I didnt do it as log-V with block?
        pvol_k = np.insert(rewvols[:-(k + 1)], 0, np.zeros((k + 1,)))  #
        Xnew[:, k + 1] = pvol_k[m]
    Xnew[:, -2] = block[m]
    Xnew[:, -1] = block[m] * rewvols[m]

    if usecrossterm:
        nvals = 9
    else:
        Xnew = Xnew[:, :-1]
        nvals = 8

    # running gridsearch
    alphavec = [1e-4, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
    if useRidge:
        m_block, _, rs_block, _, _ = sta.ridge_wgridsearch(Xnew, Y, alphavec=alphavec)
    else:
        m_block, _, rs_block, _, _ = sta.lasso_wgridsearch(Xnew, Y, alphavec=alphavec)

    # calculate pvalues
    X2 = sm.add_constant(Xnew)
    est = sm.OLS(Y, X2)
    est2 = est.fit()
    try:
        pvec_block = [est2.summary2().tables[1]['P>|t|']['x' + str(k)] for k in range(1, nvals+1)]
    except KeyError:
        print('issue with regression')
        pvec_block = np.nan*np.ones(len(range(1, nvals+1)))

    return [p, p_sr, wt_lt, wt_gt], [m_hist, rs_hist, pvec_hist], [m_block, rs_block, pvec_block]

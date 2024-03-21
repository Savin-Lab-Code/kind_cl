# analysis of wait time behavior

import numpy as np
import warnings
from sklearn.linear_model import LinearRegression
from scipy.stats import ranksums, wilcoxon
import statsmodels.api as sm
from dynamics.analysis import state_analysis as sta
from dynamics.utils.utils import displ


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
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    warnings.filterwarnings("ignore", message="Degrees of freedom*")
    warnings.filterwarnings("ignore", message="Creating an ndarray*")
    warnings.filterwarnings("ignore", message="kurtosistest only valid*")
    warnings.filterwarnings("ignore", message="omni_normtest is not valid*")
    warnings.filterwarnings("ignore", message="Sample size too small for normal approximation*")
    warnings.filterwarnings("ignore", message="Exact p-value calculation does not work if there are ties*")

    dl = ops['displevel']  # the verbosity level for print statements

    outcomes = np.array(dat['outcomes'])
    iscatch = np.array(dat['iscatch'])
    rewvols = np.log2(np.array(dat['rewvols']))  # LOG-rewarded vol on each trial
    wt_all = np.array(dat['wt'])*ops['dt']  # wait time, s
    blocks = np.array(dat['blocks'])
    if 'w_all' not in dat:
        rho_all = -np.nanmean(np.array(ops['w']), axis=0) / ops['dt']
    else:
        rho_all = -np.nanmean(np.array(dat['w_all']), axis=0) / ops['dt']

    Vvec = np.log2(ops['Vvec'])

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
        X = np.expand_dims(rewvol_catch, axis=1)
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
            X_mixed = np.expand_dims(rewvol_catch[mask_block_mixed], axis=1)
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
            X_mixed = None

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
            X_high = np.expand_dims(rewvol_catch[mask_block_high], axis=1)
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
                displ('something wrong with stats model regression for high blocks. making nan', 4, dl)
                p_high = np.nan

        else:
            b_high = None
            slope_high = None
            R2_high = None
            p_high = None
            trange_high = None

        if allreg:
            X_low = np.expand_dims(rewvol_catch[mask_block_low], axis=1)
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

        X = np.expand_dims(rewvol_use, axis=1)
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


def modelagnostic_test(dat, dt=0.05, useRidge=True, mixedonly=False, usecrossterm=True):
    """
    checks if wait time on 20ul trials, conditioned on prev trial, has significant diffs. if so, then model-free
    also does linear regression of trial history on wait time and looks for significant weights
    :param dat: dat struct from session run
    :param dt: time resolution, used since data counts in # samples, not in time
    :param useRidge: (bool) for using Ridge (L2) or Lasso (L1) regularized regression
    :param mixedonly: (bool) use mixed block and catch only (True) for the history regression?
    :param usecrossterm: (bool) use the reward x block cross term (True) or not (False)
    :return: wait times, regression weights, regression p values
    """
    block = np.array(dat['blocks'])
    rew = np.array(dat['rewvols'])
    iscatch = np.array(dat['iscatch'])
    prevrew = np.concatenate(([0], rew[:-1]), axis=0)
    wt = np.array(dat['wt'])

    # TODO removed parans. affected?
    mask_lt = rew == 20 & iscatch & block == 0 & prevrew < 20
    mask_gt = rew == 20 & iscatch & block == 0 & prevrew > 20

    wt_lt = wt[mask_lt] * dt
    wt_gt = wt[mask_gt] * dt

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
    rewvols = np.log2(np.array(dat['rewvols']))
    wt = np.array(dat['wt'])
    if mixedonly:
        m = iscatch & block == 0
    else:
        m = iscatch
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
    Y = wt[m]
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

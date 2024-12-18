import numpy as np
from scipy import interpolate
import torch
<<<<<<< HEAD


# TODO: needed? reusable for building states for dynamics?
def cond_psth(state_flat, mask, tdict, begin_idx=0, end_idx=60, omit_nexttrial=False, omit_iti=False):
    """
    will create trial averaed responses for a given trial type, starting at tstart, ending at tend
    :param state_flat: (list) flattened state list from flatten_state()
    :param mask: (list) of booleans for which trials to use
    :param tdict:  (dict) of indices for trial events of interest. typically trial start or reward (ie, next trial)
    :param begin_idx: (int) how many indices after trial_idx to start psth. use negative numbers for before event
    :param end_idx: (int) how many indices after trial_idx to end psth
    :param omit_nexttrial: (bool) should nans be used to omit data from next trials and avoid bleedover?
    :param omit_iti: (bool) should nans be used to omit the reward/opt-out epoch?
    :return: (ndarray) of size (N,) for -averaged mean and sem of N neurons
    """

    trial_idx = tdict['start']
    iti_idx = tdict['iti']  # the start of the reward/opt-out epoch

    tlen = end_idx - begin_idx
    nhidden = state_flat.shape[1]
    tm = np.nonzero(mask)[0]  # trial mask
    ntrials = len(tm)

    temp = np.zeros((ntrials - 1, tlen, nhidden))

    if omit_nexttrial:  # nan out anything from next trial
        for m in range(1, ntrials - 1):
            tstart = trial_idx[tm[m]] + begin_idx
            tstop_all = trial_idx[tm[m]] + end_idx  # possible full set of data
            if omit_iti:
                tstop_real = iti_idx[tm[m]]-1  # -1 omits the reward time bin
            else:
                tstop_real = trial_idx[tm[m + 1]]  # start of next trial

            if tstop_real > tstop_all:  # take all data
                temp[m, :, :] = state_flat[tstart:tstop_all, :]
            else:
                dtemp = tlen - (tstop_all - tstop_real)
                temp[m, :dtemp, :] = state_flat[tstart:tstop_real, :]
                temp[m, dtemp:, :] = np.nan

        response_mean = np.nanmean(temp, axis=0)
        response_sem = np.nanstd(temp, axis=0) / np.sqrt(ntrials - 1)

    else:  # use data from potentially next trial
        for m in range(1, ntrials - 1):
            tstart = trial_idx[tm[m]] + begin_idx
            tstop_all = trial_idx[tm[m]] + end_idx  # possible full set of data

            temp[m, :, :] = state_flat[tstart:tstop_all, :]
        response_mean = np.nanmean(temp, axis=0)
        response_sem = np.nanstd(temp, axis=0) / np.sqrt(ntrials - 1)

    return response_mean, response_sem


=======
from sklearn.decomposition import PCA
from scipy.io import savemat


# TODO: needed? reusable for states for dynamics
def flatten_state(X, wt, nhidden):
    """
    will take a list of state organized by gradient steps, flatten, gives idx of trial starts
    :param X: list of ns step lists that contain the hidden state. assuming here they are LSTMs
    :param wt: wait times (in timesteps) of each trial
    :param nhidden:
    :return:
    """
    trial_idx = np.cumsum(wt)
    nbatch = len(X)
    ns = len(X[0])
    nt = (nbatch - 1) * ns + len(X[-1])
    H_flat = np.zeros((nt, nhidden))
    ind = 0
    for j in range(nbatch):
        for k in range(ns):
            H_flat[ind + k, :] = X[j][k][0]  # index 0 is hidden state. index 1 is cell state
        ind += ns

    return H_flat, trial_idx


def warpstate_wITI(state_flat, events_idx, Ttrial=None):
    """
    warps state activity to be same length across trials, and accomodates variable ITI
    :param state_flat:
    :param events_idx: (ndarray) size ntrial x nevents , including end event
    :param Ttrial: (list of int) for duration, in timesteps, for each trial
    :return:
    """

    nevent = events_idx.shape[1]
    ntrials = events_idx.shape[0]
    Tmedians = np.array([np.median(events_idx[:, k] - events_idx[:, 0]) for k in range(nevent)]).astype(int)
    # set length of trial
    if Ttrial is not None:
        Tmedians = Tmedians/Tmedians[-1]*Ttrial
        Tmedians = Tmedians.astype(int)

    ntwarped = Tmedians[-1] * ntrials  # total flattened timepoints
    nhidden = state_flat.shape[-1]  # number of states
    state_warped = np.zeros((ntwarped, nhidden))
    Tratio = Tmedians / Tmedians[-1]  # proportion of warped trial that epoch should occupy
    twarp = [np.linspace(Tratio[k - 1], Tratio[k], Tmedians[k]) for k in range(1, nevent)]  # warped time vecs

    ind = 0
    events_warped = []
    for k in range(ntrials):
        events_k = [ind]  # relative event idx for the warped states
        for m in range(nevent - 1):
            idx1 = events_idx[k, m]
            idx2 = events_idx[k, m + 1]

            d1 = state_flat[idx1:idx2 + 1, :]
            tvec = np.linspace(Tratio[m], Tratio[m + 1], d1.shape[0])
            f = interpolate.interp1d(tvec, d1, axis=0)
            d1_warped = f(twarp[m])
            state_warped[ind:ind + Tmedians[m + 1], :] = d1_warped
            events_k.append(ind + Tmedians[m + 1])

        events_warped.append(events_k)
        ind += Tmedians[-1]

    return state_warped, events_warped


def sflat2statedict(sflat, nlayer=2, ngate=2):
    """
    parses flattened state into it a dictionary of its sublayers and subgates. removes redundant dimensions
    @param sflat: (np.ndarray) of size nt x nlayer x ngate x [numhiden  of 1x1xnumhidden]
    @param nlayer: (int) number of layers
    @param ngate:  (int) number of gates for the RNN. LSTM=2, GRU and RNN = 1
    @return: (dict) with keys layer#gate#, each of which is np.ndarray of size (ntxnumhidden)
    """

    statenames = [['layer' + str(j + 1) + 'gate' + str(k + 1) for k in range(ngate)] for j in range(nlayer)]
    sdict = {}
    for j in range(nlayer):
        for k in range(ngate):
            if ngate == 1:  # edge case. inner dim index will not exist for this
                if nlayer == 1:  # both indices will not exist for this either
                    if len(sflat[0].shape) == 3:
                        sdict[statenames[j][k]] = np.array([m[0, 0, :] for m in sflat])
                    else:
                        sdict[statenames[j][k]] = np.array([m[:] for m in sflat])
                else:
                    if len(sflat[0][0].shape) == 3:
                        sdict[statenames[j][k]] = np.array([m[j][0, 0, :] for m in sflat])
                    else:
                        sdict[statenames[j][k]] = np.array([m[j][:] for m in sflat])
            else:
                if len(sflat[0][0][0].shape) == 3:
                    sdict[statenames[j][k]] = np.array([m[j][k][0, 0, :] for m in sflat])
                else:
                    sdict[statenames[j][k]] = np.array([m[j][k][:] for m in sflat])

    return sdict


def postprocess4dynamics(sflat, behdict, warpstate=True, Ttrial=None, zeromean=False, epoch='wait', nlayer=2, ngate=2):
    """
    will extract cell and hidden states for each trial.
    :param sflat: (ndarray) flattened neural activity, from parse.get_trialtimes
    :param behdict: (dict) behavioral data extracted from rdict of simulation
    :param warpstate: (bool) warp the neural activity to standardize trial length?
    :param Ttrial: (int) length of trial in timesteps, if you want to warp. helpful for pca
    :param zeromean: (bool) if true, subtract the mean rate of each neuron, but based on mean during epoch
    :param epoch: (str) if zeromean is true, subtract the mean based on this epoch: 'wait', 'iti', 'start'
    :param nlayer: (int) number of layers in the network
    :param ngate: (int) number of gates output for each layer of the network. LSTM=2, GRU and RNN = 1
    :return: flattened state arrays by region and type (cell, hidden), and updated, masked behdict
    """

    offers = behdict['offers']
    outcomes = behdict['outcomes']
    blocks = behdict['blocks']
    tdict = behdict['tdict']
    task = behdict['task']
    nstate = (nlayer+1)*(ngate+1)  # how many total states

    # name ordering convention: all gates for each layer, then iterate to next layer
    sdict = sflat2statedict(sflat, nlayer, ngate)
    statenames_flat = list(sdict.keys())

    # find trial event times. handle edge cases for number of trials
    nx = np.min([len(tdict['end']), len(tdict['start']), len(tdict['iti']), len(offers)])
    tdict['start'] = tdict['start'][:nx]
    tdict['iti'] = tdict['iti'][:nx] - 1  # omit a timestep to avoid when reward arrives
    tdict['end'] = tdict['end'][:nx]
    wt = tdict['iti'] - tdict['start']
    iti = tdict['end'] - tdict['iti']

    # remove outliers: ultra-short wait times and itis, and ultra-long wait times
    if task == 'task':
        mask = (wt > 5) & (iti > 4) & (wt < 300)
    else:
        mask = (wt > 5) & (wt < 300)
    trialstart_idx = tdict['start'][mask]
    wt = wt[mask] - 1  # omits the timestep just before reward
    offers_masked = offers[:nx][mask]
    blocks_masked = blocks[:nx][mask]
    outcomes_masked = outcomes[:nx][mask]
    trials_masked = np.argwhere(mask)[:nx, 0]
    T = np.max(wt)

    # mean-subtract, if wanted
    if zeromean:
        for j in statenames_flat:
            numhidden = sdict[j].shape[1]  # number of units in each layer
            unitmeans = np.zeros(numhidden)  # order as in statenames_flat: e.g., ofc_h, ofc_c, str_h, str_c

            if epoch == 'wait':
                e1 = tdict['start']+1
                e2 = tdict['iti']

            elif epoch == 'iti':
                e1 = tdict['iti'][:-1]
                e2 = tdict['start'][2:]

            elif epoch == 'start':
                e1 = tdict['start']
                e2 = tdict['start']+1

            else:
                print('wrong epoch type supplied to postprocess4dynamics')
                e1 = None
                e2 = None

            # getting mean of subset of data without blowing up memory isn't as easy I'd hoped with std. methods
            for k in range(numhidden):
                # meank = np.array([0, 0, 0, 0])
                meank = 0
                ind = 0
                for m in range(len(e1)):
                    meank += np.nansum(sdict[j][e1[m]:e2[m], k])
                    ind += e2[m] - e1[m]
                meank = meank / ind
                unitmeans[k] = meank

            # subtract
            sdict[j] -= unitmeans

    # warp, if necessary
    if warpstate:
        print('warping states')
        # only using data between start and iti
        event_mat = np.stack((tdict['start'][mask], tdict['iti'][mask])).T

        events_warped = None
        for n in range(nstate):
            sflat_warped, events_warped = warpstate_wITI(sdict[statenames_flat[n]], event_mat, Ttrial=Ttrial)
            sdict[statenames_flat[n]] = sflat_warped

        events_warped = np.array(events_warped)
        trialstart_idx_warped = events_warped[:, 0]
        T = events_warped[0, -1] - events_warped[0, 0]

        trialstart_idx = np.array(trialstart_idx_warped)
        wt = T * np.ones(trialstart_idx.shape).astype(int)  # move to median value for all wapred

    behdict['offers_masked'] = offers_masked
    behdict['outcomes_masked'] = outcomes_masked
    behdict['blocks_masked'] = blocks_masked
    behdict['trialstart_idx'] = trialstart_idx
    behdict['T'] = T
    behdict['wt'] = wt
    behdict['trials_masked'] = trials_masked
    behdict['iti'] = iti
    behdict['outlier_trials'] = np.argwhere(mask)[:, 0]
    behdict['goodtrial_mask'] = np.argwhere(mask == 0)[:, 0]

    return sdict, behdict


def extract_data_epoch(statedict, tdict, isgoodmask, epoch='wait'):
    """
    makes dictionary from statedict that only includes data from a particular epoch
    :param statedict: (dict of np.ndarray) of state and cell states by region
    :param tdict: (dict of np.ndarray) for timing of epochs in entire data in statedict
    :param isgoodmask: (np.array of bool) mask of trials to use
    :param epoch: (str) which epoch in trial to isolate
    :return:
    """

    statenames_flat = list(statedict.keys())
    nstate = len(statenames_flat)
    dat2fit = {}
    dattmp = {}

    if len(statedict[statenames_flat[0]].shape) == 4:
        print('legacy data issue. squeezing sflat')
        for m in range(nstate):
            dattmp[statenames_flat[m]] = np.squeeze(np.squeeze(statedict[statenames_flat[m]]))
    else:
        for m in range(nstate):
            dattmp[statenames_flat[m]] = statedict[statenames_flat[m]]

    tstart = np.array(tdict['start'])[isgoodmask]  # start index of filtered trials' start
    iti = np.array(tdict['iti'])[isgoodmask]  # time between reward/opt out and start of next trial
    T = np.array(tdict['end'])[isgoodmask]

    # set events of interest
    if epoch == 'wait':
        ev_s = tstart + 1
        ev_len = iti - ev_s
    elif epoch == 'iti':
        ev_s = iti
        ev_len = T - ev_s
    elif epoch == 'start':
        ev_s = tstart
        ev_len = np.ones(len(ev_s)).astype(int)
    elif epoch == 'all':
        # grab all data
        ev_s = [0]
        ev_len = [T[-1]]
    else:
        print('incorrect epoch type supplied to condpsth_bycelltype')
        ev_s = None
        ev_len = None

    ns = len(ev_s)  # number of trials

    # concatenate epcosh from all trials onto a single axis
    for m in statenames_flat:
        num_hidden = statedict[m].shape[1]  # number of hidden units for this layer
        samps_m = np.array([]).reshape((0, num_hidden))

        for k in range(ns):
            samps_m = np.concatenate((samps_m, statedict[m][ev_s[k]: ev_s[k] + ev_len[k], :]), axis=0)

        dat2fit[m] = samps_m

    return dat2fit


def concatstates(statedict):
    """
    will combine gates for each layer for LSTMs. will simply pass for GRU and LSTM
    @param statedict: (dict) of each layer+gate type
    @return: (dict) of concatenated gates, by layer
    """

    statenames_flat = list(statedict.keys())
    statelist = []

    # get number of layers and number of gates for each layer
    layertype = np.array([int(k.split('layer')[-1].split('gate')[0]) for k in statenames_flat])
    layers = np.sort(np.unique(layertype))
    names = ['layer'+str(k) for k in layers]

    # parse data2fit by combining gates
    for k in layers:
        # find gates for this layer type
        layer_k_idx = np.argwhere(layertype == k)[:, 0]  # indices of gates for this layer
        names_k = np.array(statenames_flat)[layer_k_idx]
        gatenumbers = np.array([int(m.split('gate')[-1]) for m in names_k])  # gate number for these indices

        layer_k_idx = layer_k_idx[np.argsort(gatenumbers)]  # sorted indices to sort gates by 0,1,2, etc.

        ngate_k = len(layer_k_idx)
        # numhidden_k = dat2fit[statenames_flat[layer_k_idx[0]]].shape[1]
        datlist_k = statedict[statenames_flat[layer_k_idx[0]]]

        for j in range(1, ngate_k):
            datlist_k = np.concatenate((datlist_k, statedict[statenames_flat[layer_k_idx[j]]]), axis=1)

        statelist.append(datlist_k)

    return dict(zip(names, statelist))


def dr_transform(dat2fit, transformtype='PCA'):
    """
    dimensionality transform
    @param dat2fit: dict of np.ndarrays of size nt x nhidden, where nt is # timepoints and nhidden is # hidden units
    @param transformtype: (str) hanlde to decide what kind of transform. 'PCA', 'dPCA' supported so far
    @return:
    """

    dr_list = []
    dim_list = []

    tempdict = concatstates(dat2fit)  # combine all gates of given layers
    dat2fit_list = list(tempdict.values())
    nlayer = len(dat2fit_list)

    if transformtype == 'PCA':
        for k in range(nlayer):
            pca_k = PCA()
            pca_k.fit(dat2fit_list[k])
            dr_list.append(pca_k)
            d_k = np.argwhere(np.cumsum(pca_k.explained_variance_ratio_) > 0.90)[0, 0]
            print('dim, layer '+str(k)+': ' + str(d_k+1))
            dim_list.append(d_k)

    elif transformtype == 'deltaPCA':
        for k in range(nlayer):
            datlist_k = dat2fit_list[k][1:, :]-dat2fit_list[k][:-1, :]
            pca_k = PCA()
            pca_k.fit(datlist_k)
            dr_list.append(pca_k)
            d_k = np.argwhere(np.cumsum(pca_k.explained_variance_ratio_) > 0.90)[0, 0]
            print('dim, layer ' + str(k) + ': ' + str(d_k+1))
            dim_list.append(d_k)

    else:
        print("no other transform types supported yet")
        dr_list = None
        dim_list = None

    return dr_list, dim_list


def states2matlab(rd, savename):
    """
    will package up the statedict and behavioral data into a matlab-friendly format that is general
    @param rd: (dict) from dyn.dynamics_preprocess_suite
    @param savename: (str) name to save
    @return:
    """

    sd = concatstates(rd['statedict'])  # concatenate all gates of a given layer into state. starts dict

    sd['outcomes'] = rd['behdict']['outcomes']
    sd['offers'] = np.exp(rd['behdict']['offers'])
    sd['blocks'] = rd['behdict']['blocks']

    tdict = rd['behdict']['tdict']
    sd['t_start'] = tdict['start']
    sd['t_iti'] = tdict['iti']
    sd['t_end'] = tdict['end']

    savemat(savename, sd)

    return sd


# some summary-level and pca-specific code---------------------------------------------------------------------

# TODO: deprecated, but there is currently no psth code in my codebase
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
def condpsth_bycelltype(vmask, statedict, behdict, nsamps=0, mask_marginal=None, sampdat=True, epoch='wait'):
    """
    calculates average activity for trial types, but divided into the cell types and regions in the network
    :param vmask:
    :param statedict:
    :param behdict:
    :param nsamps:
    :param mask_marginal:
    :param sampdat:
    :param epoch:
    :return:
    """

<<<<<<< HEAD
    # things that come from parsing the data with postprocess4dynamics
=======
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
    sflat_ofc_h = statedict['sflat_ofc_h']
    sflat_ofc_c = statedict['sflat_ofc_c']
    sflat_str_h = statedict['sflat_str_h']
    sflat_str_c = statedict['sflat_str_c']

    if len(sflat_ofc_h.shape) == 4:
        print('legacy data issue. squeezing sflat')
        sflat_ofc_h = np.squeeze(np.squeeze(sflat_ofc_h))
        sflat_ofc_c = np.squeeze(np.squeeze(sflat_ofc_c))
        sflat_str_h = np.squeeze(np.squeeze(sflat_str_h))
        sflat_str_c = np.squeeze(np.squeeze(sflat_str_c))

    trialstart_idx = behdict['trialstart_idx']  # start index of filtered trials' start
    wt = behdict['wt']  # time between start and reward/opt-out
    iti = behdict['iti']  # time between reward/opt out and start of next trial
    T = behdict['T']
    nV = len(vmask)
    num_hidden = sflat_ofc_h.shape[1]

    # set events of interest
    if epoch == 'wait':
        ev_s = trialstart_idx
        ev_len = wt
    elif epoch == 'iti':
        print('in iti condpsth')
        ev_s = trialstart_idx + wt
        ev_len = iti
    elif epoch == 'start':
        print('in start condpsth')
        ev_s = trialstart_idx
        ev_len = np.ones(len(wt)).astype(int)
    else:
        print('incorrect epoch type supplied to condpsth_bycelltype')
        ev_s = None
        ev_len = None

    # all of the conditional psths,s by region and state type
    psth_ofc_h = np.zeros((nV, T, num_hidden))
    psth_ofc_c = np.zeros((nV, T, num_hidden))
    psth_str_h = np.zeros((nV, T, num_hidden))
    psth_str_c = np.zeros((nV, T, num_hidden))

    se_ofc_h = np.zeros((nV, T, num_hidden))
    se_ofc_c = np.zeros((nV, T, num_hidden))
    se_str_h = np.zeros((nV, T, num_hidden))
    se_str_c = np.zeros((nV, T, num_hidden))

    # number of trials of that type
    nsvec = []

    # samples of each type
    psth_ofc_h_samps = np.nan * np.ones((nV, nsamps, T, num_hidden))
    psth_ofc_c_samps = np.nan * np.ones((nV, nsamps, T, num_hidden))
    psth_str_h_samps = np.nan * np.ones((nV, nsamps, T, num_hidden))
    psth_str_c_samps = np.nan * np.ones((nV, nsamps, T, num_hidden))

    for m in range(nV):

        # isolate trials of that type, find where they start, based upon which epoch you care about
        t_idx = ev_s[np.array(vmask[m])]
        tlen = ev_len[vmask[m]]

        ns = len(t_idx)  # number of trials of that type
        nsvec.append(ns)

        # create the big matrix of trials of that type for eventual nanmean
        psth_ofc_h_base = np.nan * np.ones((ns, T, num_hidden))
        psth_ofc_c_base = np.nan * np.ones((ns, T, num_hidden))
        psth_str_h_base = np.nan * np.ones((ns, T, num_hidden))
        psth_str_c_base = np.nan * np.ones((ns, T, num_hidden))

        for k in range(ns):

            psth_ofc_h_base[k, :tlen[k], :] = sflat_ofc_h[t_idx[k]:t_idx[k] + tlen[k], :]
            psth_ofc_c_base[k, :tlen[k], :] = sflat_ofc_c[t_idx[k]:t_idx[k] + tlen[k], :]
            psth_str_h_base[k, :tlen[k], :] = sflat_str_h[t_idx[k]:t_idx[k] + tlen[k], :]
            psth_str_c_base[k, :tlen[k], :] = sflat_str_c[t_idx[k]:t_idx[k] + tlen[k], :]

            if k < nsamps:
                psth_ofc_h_samps[m, k, :tlen[k], :] = sflat_ofc_h[t_idx[k]:t_idx[k] + tlen[k], :]
                psth_ofc_c_samps[m, k, :tlen[k], :] = sflat_ofc_c[t_idx[k]:t_idx[k] + tlen[k], :]
                psth_str_h_samps[m, k, :tlen[k], :] = sflat_str_h[t_idx[k]:t_idx[k] + tlen[k], :]
                psth_str_c_samps[m, k, :tlen[k], :] = sflat_str_c[t_idx[k]:t_idx[k] + tlen[k], :]

        psth_ofc_h[m, :, :] = np.nanmean(psth_ofc_h_base, axis=0)
        psth_ofc_c[m, :, :] = np.nanmean(psth_ofc_c_base, axis=0)
        psth_str_h[m, :, :] = np.nanmean(psth_str_h_base, axis=0)
        psth_str_c[m, :, :] = np.nanmean(psth_str_c_base, axis=0)

        se_ofc_h[m, :, :] = np.nanstd(psth_ofc_h_base, axis=0)/np.sqrt(ns)
        se_ofc_c[m, :, :] = np.nanstd(psth_ofc_c_base, axis=0)/np.sqrt(ns)
        se_str_h[m, :, :] = np.nanstd(psth_str_h_base, axis=0)/np.sqrt(ns)
        se_str_c[m, :, :] = np.nanstd(psth_str_c_base, axis=0)/np.sqrt(ns)

    # any remaining nans go to zero
    psth_ofc_h[np.isnan(psth_ofc_h)] = 0.0
    psth_ofc_c[np.isnan(psth_ofc_c)] = 0.0
    psth_str_h[np.isnan(psth_str_h)] = 0.0
    psth_str_c[np.isnan(psth_str_c)] = 0.0

    # handle the data for fitting pca
    # samples for pca. nV of them
    if sampdat:
        samps2fit_ofc_h = np.array([]).reshape((0, num_hidden))
        samps2fit_ofc_c = np.array([]).reshape((0, num_hidden))
        samps2fit_str_h = np.array([]).reshape((0, num_hidden))
        samps2fit_str_c = np.array([]).reshape((0, num_hidden))
        if mask_marginal is None:
            mask_marginal = vmask[-1]

        # isolate trials of that type, find where they start
        t_idx = trialstart_idx[mask_marginal]
<<<<<<< HEAD
        # TODO: check m usage
        tlen = wt[vmask[m]]
=======
        tlen = wt[mask_marginal]
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
        ns = len(t_idx)  # number of trials of that type
        i0 = 1  # buffer around trial start to avoid impulse inputs

        for k in range(ns):
            samps2fit_ofc_h = np.concatenate((samps2fit_ofc_h,
<<<<<<< HEAD
                                             sflat_ofc_h[t_idx[k] + i0:t_idx[k] + tlen[k], :]), axis=0)
            samps2fit_ofc_c = np.concatenate((samps2fit_ofc_c,
                                             sflat_ofc_c[t_idx[k] + i0:t_idx[k] + tlen[k], :]), axis=0)
            samps2fit_str_h = np.concatenate((samps2fit_str_h,
                                             sflat_str_h[t_idx[k] + i0:t_idx[k] + tlen[k], :]), axis=0)
            samps2fit_str_c = np.concatenate((samps2fit_str_c,
                                             sflat_str_c[t_idx[k] + i0:t_idx[k] + tlen[k], :]), axis=0)
    else:
        print('no samples being used for pca. using trial averages to highlight temporal variability')
=======
                                              sflat_ofc_h[t_idx[k] + i0:t_idx[k] + tlen[k], :]), axis=0)
            samps2fit_ofc_c = np.concatenate((samps2fit_ofc_c,
                                              sflat_ofc_c[t_idx[k] + i0:t_idx[k] + tlen[k], :]), axis=0)
            samps2fit_str_h = np.concatenate((samps2fit_str_h,
                                              sflat_str_h[t_idx[k] + i0:t_idx[k] + tlen[k], :]), axis=0)
            samps2fit_str_c = np.concatenate((samps2fit_str_c,
                                              sflat_str_c[t_idx[k] + i0:t_idx[k] + tlen[k], :]), axis=0)
    else:
        print('no samples being used to pca. using trial averages to highlith temporarl variability')
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
        samps2fit_ofc_h = None
        samps2fit_ofc_c = None
        samps2fit_str_h = None
        samps2fit_str_c = None

    psths = {'ofc_h': psth_ofc_h, 'ofc_c': psth_ofc_c, 'str_h': psth_str_h, 'str_c': psth_str_c,
             'se_ofc_h': se_ofc_h, 'se_ofc_c': se_ofc_c, 'se_str_h': se_str_h, 'se_str_c': se_str_c}
<<<<<<< HEAD
=======
    # TODO: are samps needed anymore? I don't think that code really uses this. it's redundant with dat2fit
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
    samps = {'ofc_h': psth_ofc_h_samps, 'ofc_c': psth_ofc_c_samps, 'str_h': psth_str_h_samps, 'str_c': psth_str_c_samps}
    dat2fit = {'ofc_h': samps2fit_ofc_h, 'ofc_c': samps2fit_ofc_c, 'str_h': samps2fit_str_h, 'str_c': samps2fit_str_c}

    return dat2fit, psths, nsvec, samps


<<<<<<< HEAD
def trial_samples(state_flat, mask, tdict, nsamps, begin_idx=0, end_idx=60,
                  omit_nexttrial=False, omit_iti=False, samplist=None):
    """
    grabs specific trials rather than averaging
    :param state_flat: (list) flattened state list from flatten_state()
    :param mask: (list) or booleans for which trials to use
    :param tdict:  (dict) of lists oindices for trial events of interest. typically trial start or reward
    :param nsamps: (int) number of trial samples to use
    :param begin_idx: (int) how many indices after trial_idx to start psth. use negative numbers for before event
    :param end_idx: (int) how many indices after trial_idx to end psth
    :param omit_nexttrial: (bool) should nans be used to omit data from next trials and avoid bleedover?
    :param omit_iti: (bool) should nans be used to omit the reward/opt-out epoch?
    :param samplist: (list) of trial indices to grab. if None, randomly choose
    :return:
    """

    trial_idx = tdict['start']
    iti_idx = tdict['iti']  # the start of the reward/opt-out epoch

    tlen = end_idx - begin_idx
    nhidden = state_flat.shape[1]
    Xsamps = []

    # find identity of samples
    nt_k = np.sum(mask)
    if samplist is None:
        samp_idx = np.random.choice(range(nt_k), nsamps)  # choose a set of true indices
    else:
        samp_idx = samplist

    idx = np.nonzero(mask)[0][samp_idx]  # location within trial_idx of samples. need this for checking next trials

    for ii in idx:

        temp = np.zeros((tlen, nhidden))

        if omit_nexttrial:  # nan out anything from next trial

            tstart = trial_idx[ii] + begin_idx
            tstop_all = trial_idx[ii] + end_idx  # possible full set of data
            if omit_iti:
                tstop_real = iti_idx[ii]-1  # -1 omits rewarded time bin
            else:
                tstop_real = trial_idx[ii+1]  # start of next trial

            if tstop_real > tstop_all:  # take all data
                temp = state_flat[tstart:tstop_all, :]
            else:
                dtemp = tlen - (tstop_all - tstop_real)
                temp[:dtemp, :] = state_flat[tstart:tstop_real, :]
                temp[dtemp:, :] = np.nan

        else:  # use data from potentially next trial
            tstart = trial_idx[ii] + begin_idx
            tstop_all = trial_idx[ii] + end_idx  # possible full set of data
            temp = state_flat[tstart:tstop_all, :]

        Xsamps.append(temp)

    return Xsamps, idx


def warpstate_wITI(state_flat, events_idx, Ttrial=None):
    """
    warps state activity to be same length across trials, and accomodates variable ITI
    :param state_flat: (ndarray) size nt x N, for nt time points, N neurons?
    :param events_idx: (ndarray) size ntrial x nevents , including end event
    :param Ttrial: (int) length of warped trial, in timesteps. if None, use median trial length
    :return:
    """

    nevent = events_idx.shape[1]
    ntrials = events_idx.shape[0]
    Tmedians = np.array([np.median(events_idx[:, k] - events_idx[:, 0]) for k in range(nevent)]).astype(int)
    # set length of trial
    if Ttrial is not None:
        Tmedians = Tmedians/Tmedians[-1]*Ttrial
        Tmedians = Tmedians.astype(int)

    ntwarped = Tmedians[-1] * ntrials  # total flattened timepoints
    nhidden = state_flat.shape[-1]  # number of states
    state_warped = np.zeros((ntwarped, nhidden))
    Tratio = Tmedians / Tmedians[-1]  # proportion of warped trial that epoch should occupy
    twarp = [np.linspace(Tratio[k - 1], Tratio[k], Tmedians[k]) for k in range(1, nevent)]  # warped time vecs

    ind = 0
    events_warped = []
    for k in range(ntrials):
        events_k = [ind]  # relative event idx for the warped states
        for m in range(nevent - 1):
            idx1 = events_idx[k, m]
            idx2 = events_idx[k, m + 1]

            d1 = state_flat[idx1:idx2 + 1, :]
            tvec = np.linspace(Tratio[m], Tratio[m + 1], d1.shape[0])
            f = interpolate.interp1d(tvec, d1, axis=0)
            d1_warped = f(twarp[m])
            state_warped[ind:ind + Tmedians[m + 1], :] = d1_warped
            events_k.append(ind + Tmedians[m + 1])

        events_warped.append(events_k)
        ind += Tmedians[-1]

    return state_warped, events_warped


def postprocess4dynamics(sflat, behdict, warpstate=True, Ttrial=None, zeromean=False, epoch='wait'):
    """
    will extract cell and hidden states for each trial.
    :param sflat: (ndarray) flattened neural activity, from parse.get_trialtimes
    :param behdict: (dict) behavioral data extracted from rdict of simulation
    :param warpstate: (bool) warp the neural activity to standardize trial length?
    :param Ttrial: (int) length of trial in timesteps, if you want to warp. helpful for pca
    :param zeromean: (bool) if true, subtract the mean rate of each neuron, but based on mean during epoch
    :param epoch: (str) if zeromean is true, subtract the mean based on this epoch: 'wait', 'iti', 'start'
    :return: flattened state arrays by region and type (cell, hidden), and updated, masked behdict
    """

    offers = behdict['offers']
    outcomes = behdict['outcomes']
    blocks = behdict['blocks']
    tdict = behdict['tdict']
    task = behdict['task']

    sflat_ofc_h = np.array([k[0][0][:]for k in sflat])
    sflat_ofc_c = np.array([k[0][1][:] for k in sflat])
    sflat_str_h = np.array([k[1][0][:] for k in sflat])
    sflat_str_c = np.array([k[1][1][:] for k in sflat])

    # find trial event times. handle edge cases for number of trials
    nx = np.min([len(tdict['end']), len(tdict['start']), len(tdict['iti']), len(offers)])
    tdict['start'] = tdict['start'][:nx]
    tdict['iti'] = tdict['iti'][:nx] - 1  # omit a timestep to avoid when reward arrives
    tdict['end'] = tdict['end'][:nx]
    wt = tdict['iti'] - tdict['start']
    iti = tdict['end'] - tdict['iti']

    # remove outliers: ultra-short wait times and itis, and ultra-long wait times
    if task == 'task':
        mask = (wt > 5) & (iti > 4) & (wt < 300)
    else:
        mask = (wt > 5) & (wt < 300)
    trialstart_idx = tdict['start'][mask]
    wt = wt[mask] - 1  # omits the timestep just before reward
    offers_masked = offers[:nx][mask]
    blocks_masked = blocks[:nx][mask]
    outcomes_masked = outcomes[:nx][mask]
    trials_masked = np.argwhere(mask)[:nx, 0]
    T = np.max(wt)

    # mean-subtract, if wanted
    if zeromean:
        numhidden = sflat_ofc_h.shape[1]  # number of units in each layer
        unitmeans = np.zeros((numhidden, 4))  # means will be orders as ofc_h, ofc_c, str_h, str_c

        if epoch == 'wait':
            e1 = tdict['start']+1
            e2 = tdict['iit']

        elif epoch == 'iti':
            e1 = tdict['iit'][:-1]
            e2 = tdict['start'][2:]

        elif epoch == 'start':
            e1 = tdict['start']
            e2 = tdict['start']+1

        else:
            print('wrong epoch type supplied to postprocess4dynamics')
            e1 = None
            e2 = None

        # getting mean of subset of data without blowing up memory isn't as easy I'd hoped with std. methods
        for k in range(numhidden):
            meank = np.array([0, 0, 0, 0])
            ind = 0
            for m in range(len(e1)):
                meank[0] += sflat_ofc_h[e1[m]:e2[m], k]
                meank[1] += sflat_ofc_c[e1[m]:e2[m], k]
                meank[2] += sflat_str_h[e1[m]:e2[m], k]
                meank[3] += sflat_str_c[e1[m]:e2[m], k]
                ind += e2[m] - e1[m]
            meank = meank / ind
            unitmeans[k, :] = meank

        # subtract
        sflat_ofc_h -= unitmeans[:, 0]
        sflat_ofc_c -= unitmeans[:, 1]
        sflat_str_h -= unitmeans[:, 2]
        sflat_str_c -= unitmeans[:, 3]

    # warp, if necessary
    if warpstate:
        print('warping states')

        # only use data between start and iti
        event_mat = np.stack((tdict['start'][mask], tdict['iti'][mask])).T

        sflat_ofc_h, events_warped = warpstate_wITI(sflat_ofc_h, event_mat, Ttrial=Ttrial)
        sflat_ofc_c, _ = warpstate_wITI(sflat_ofc_c, event_mat, Ttrial=Ttrial)
        sflat_str_h, _ = warpstate_wITI(sflat_str_h, event_mat, Ttrial=Ttrial)
        sflat_str_c, _ = warpstate_wITI(sflat_str_c, event_mat, Ttrial=Ttrial)

        events_warped = np.array(events_warped)
        trialstart_idx_warped = events_warped[:, 0]
        T = events_warped[0, -1] - events_warped[0, 0]

        trialstart_idx = np.array(trialstart_idx_warped)
        wt = T * np.ones(trialstart_idx.shape).astype(int)  # move to median value for all wapred

    statedict = {'sflat_ofc_h': sflat_ofc_h, 'sflat_ofc_c': sflat_ofc_c,
                 'sflat_str_h': sflat_str_h, 'sflat_str_c': sflat_str_c}

    behdict['offers_masked'] = offers_masked
    behdict['outcomes_masked'] = outcomes_masked
    behdict['blocks_masked'] = blocks_masked
    behdict['trialstart_idx'] = trialstart_idx
    behdict['T'] = T
    behdict['wt'] = wt
    behdict['trials_masked'] = trials_masked
    behdict['iti'] = iti

    return statedict, behdict


def pca_samples(sflat, tdict, vmask, pca_use, idx=0):
    """
    calculates pca for samples given by vmask.
    :param sflat: (list) of neural activities by timepoint
    :param tdict: (dict) of epoch times fdor each trial
    :param vmask: (list) of bool vals for different conditions
    :param pca_use: (PCA) object used to perform PCA
=======
# TODO: currently deprecated, but could be useful for later anlaysis,.
def pca_samples(sflat, tdict, vmask, pca_use, keys, idx=0):
    """
    calculates pca for samples given by vmask. This is a very common operation, so I hope that
    I reuse this code...
    :param sflat: (list) of neural activities by timepoint
    :param tdict: (dict) of epoch times fdor each trial
    :param vmask: (list) of bool vals for different conditions
    :param pca_use: (PCA) object that has been fit to data from the task
    :param keys: (list) key names for vmask
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
    :param idx: (int) 0 for OFC, or 1 for STR
    :return: (dict) of pca-transformed values for each samples in vmask conditions
    """

    # do this for both kinds
    # create the samples
    sampsdict = {}

    ind = 0

<<<<<<< HEAD
=======
    # check that number of vmask entries and number of keys is correct
    assert len(vmask) == len(keys), "keys for vmask are not correct. check previous code for error"

>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
    for k in range(len(vmask)):  # loop over condition
        trial_idx_list = vmask[k]
        sampsvec = []

        for j in range(len(trial_idx_list)):
<<<<<<< HEAD
=======
            # TODO: make the start and end epochs inputs
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
            idxs = tdict['start'][trial_idx_list[j]]
            idxf = tdict['iti'][trial_idx_list[j]]
            if idxs == idxf:  # dont transform 1-timestep trials
                continue
            else:
                state = []
                for m in range(idxs, idxf):
                    h = [sflat[m][0][0], sflat[m][1][0]]
                    c = [sflat[m][0][1], sflat[m][1][1]]
<<<<<<< HEAD
=======
                    # sm = np.concatenate((np.squeeze(h[idx].detach().numpy()),
                    #                     np.squeeze(c[idx].detach().numpy())), axis=0)
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
                    if type(h[idx]) is torch.Tensor:
                        sm = np.concatenate((np.squeeze(np.squeeze(h[idx][0, 0, :].cpu().detach().numpy())),
                                             np.squeeze(np.squeeze(c[idx][0, 0, :].cpu().detach().numpy()))), axis=0)
                    else:
                        sm = np.concatenate((h[idx], c[idx]), axis=0)

                    state.append(sm)
                state = np.array(state)
                sampsvec.append(state)
<<<<<<< HEAD
        sampsdict[k] = sampsvec
=======
        sampsdict[keys[k]] = sampsvec
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
        ind += 1

    # transform all samples with pca
    samps_pcdict = {}

    for k in sampsdict.keys():
        samps_pcdict[k] = [pca_use.transform(m) for m in sampsdict[k]]

    return samps_pcdict, sampsdict

# code for structruing data for different uses, like behavioral modeling, state analysis, etc.

import datetime
from copy import deepcopy
import pickle
import numpy as np
from scipy.io import savemat
from dynamics.utils.utils import CPU_Unpickler, opsbase
from dynamics.process.rnn import wt_envs
import dynamics.analysis.wt_analysis as wta
import dynamics.vis.wt_vis as wt_vis
import dynamics.utils.utils as utils
import torch
from os.path import exists
import gc
import json
from sklearn.cluster import DBSCAN as dbscan
import warnings


# TODO: should I run that other preprocessing step before this, the one that mimics the rat pipeline?
def generate_rattrial_rnn(loadname, savename,
                          date_base=str(datetime.date.today())+' ', ops=None, ntrials=None):
    """
    create a .mat file that can be used with the behavioral model
    :param savename_base: (str) savename path + preamble.
    :param date_base: date of simulation. if unknown, use current date
    :param ops: wt_reinforce ops
    :param ntrials: (list) number of trials per session
    :return: simplified structure that is way smaller
    """
    if ops is None:
        ops = opsbase()


    if type(loadname) is str:

        if not exists(loadname):
            print('data does not exist')

        if loadname.split('.')[-1]=='dat':
            print('loading legacy data')
            with open(loadname, 'rb') as f:
                #a = CPU_Unpickler(f).load()
                a = pickle.load(f)
        elif loadname.split('.')[-1] == 'json':
            with open(loadname, 'r') as f:
                a = json.load(f)
    elif type(loadname) is dict:
        print('supplied data, not filename. using data directroy')
        a = loadname


    pcatch = ops['pcatch']
    dt = ops['dt']
    tau = ops['lam']
    wt_thresh = dt*(np.nanmean(a['wt']) + 2*np.nanstd(a['wt']))

    ns = len(a['rewvols'])  # number trials. needed for proper shape

    # create fake sessions with correct number of trials to mimic rat session
    if ntrials is None:
        subsess = 40 * 5  # number of trials in 5-block portion
        nsess = 10  # hard-coded fake paritioning into sessions
        ns_t = np.floor((ns / nsess) / subsess) * subsess  # number of trials per session

        ntrials = []
        date = np.zeros((nsess, 11), dtype='str')
        for k in range(nsess - 1):
            ntrials.append(ns_t)
            date[k, :] = date_base

        ntrials.append(ns - (nsess - 1) * ns_t)  # edge case. last session has potentially more
        ntrials = np.reshape(np.array(ntrials), (nsess, -1))
    else:
        nsess = len(ntrials)
        ntrials = np.array(ntrials)
        date = np.zeros((nsess, 11), dtype='str')
        for k in range(nsess):
            date[k, :] = date_base

    reward = np.reshape(a['rewvols'], (ns, -1))
    block = np.reshape(a['blocks'], (ns, -1))+1  # andrew's behavioral model indexs blocks as [1,2,3]
    # hacky way to handle this right now. some historical code might include reward delay
    if 'reward_delay' in a:
        reward_delay = np.reshape(a['reward_delay'], (ns, -1))
    else:
        reward_delay = np.reshape([k['rewardDelay'] for k in a['env'].trials], (ns, -1))
    prob_catch = pcatch * np.ones(reward.shape)
    wait_time = np.reshape(dt * np.array(a['wt']), (ns, -1)) + dt  # shift everything by dt to be >0

    date[-1, :] = date_base

    hits = np.reshape(a['outcomes'], (ns, -1)) == 1
    optout = np.reshape(a['outcomes'], (ns, -1)) == 0
    vios = np.reshape(a['outcomes'], (ns, -1)) == -1
    catch = np.reshape(a['iscatch'], (ns, -1))

    # ITI is used in the selection criteria of some things. make "bad" data in the 99th percentile
    # so that those things don't pull too much out of your code
    ITI = ops['dt'] * np.ones(reward.shape)
    ITI[a['wt'] > wt_thresh] = 100  # should make these in the 99th percentile?

    # stuff not yet relevant in my sim
    rt = np.nan * np.ones(reward.shape)

    wait_for_poke = np.nan * np.ones(reward.shape)
    # optout.astype(float)

    # TODO: if this throws type error, try np.float instead of float
    A = {'reward': reward, 'date': date, 'block': block, 'reward_delay': reward_delay, 'prob_catch': prob_catch,
         'wait_time': wait_time, 'ntrials': ntrials, 'hits': hits.astype(bool), 'optout': optout.astype(bool),
         'vios': vios.astype(bool), 'catch': catch.astype(bool),
         'rt': rt, 'ITI': ITI, 'wait_for_poke': wait_for_poke, 'tau': tau, 'pcatch': pcatch,
         'wait_thresh': wt_thresh, 'seed': ops['seed']}
    AA = {'A': A}

    savemat(savename, AA, do_compression=False)

    return A


# loading and parsing scripts
# TODO: eventually remove this. only have code that loads the .stats file and runs analysis code from that
def load_dat4stats(fname):
    """
    loads .stats data containing regression and wait time information from a training run.
    preprocess loading for  parse_simulation in anlysis
    :param fname:
    :return:
    """
    dat_stat = pickle.load(open(fname, 'rb'))
    fits = dat_stat[1]
    nums = dat_stat[2]
    wt_dict = dat_stat[0]

    if len(dat_stat) > 3:
        wtlist = dat_stat[3]
        reg_hist = dat_stat[4]
        reg_block = dat_stat[5]
    else:
        wtlist = None
        reg_hist = None
        reg_block = None

    if 'p_mixed' in fits:
        pval = fits['p_mixed']
        slope = fits['m_mixed']
    else:
        pval = fits['p']
        slope = fits['m']

    if 'b_mixed' in fits:
        b = fits['b_mixed']
        adapt_20_ratio = wt_dict['wt_high'][2] / wt_dict['wt_low'][2]
        wtrange = wt_dict['wt_mixed'][-1] - wt_dict['wt_mixed'][0]
    else:  # for catch blocks
        b = []
        adapt_20_ratio = []
        wtrange = []

    return pval, slope, adapt_20_ratio, nums, wtrange, b, wtlist, reg_hist, reg_block


#TODO: find which figures use this. I think it is for the behavioral model
def format2singlesess(datlist, tokeep):
    """
    joines multiple training sessions into 1
    :param datlist: (list) of dat dicts from session runs
    :param tokeep: (list) of booleans for which datafiles to keep
    :return:
    """
    dat_filt = {}
    dat_filt['blocks'] = []
    dat_filt['rewvols'] = []
    dat_filt['iscatch'] = []
    dat_filt['outcomes'] = []
    dat_filt['wt'] = []
    dat_filt['rewards_pertrial'] = []
    #dat_filt['w_all'] = []
    dat_filt['reward_delay'] = []

    for k in range(len(datlist)):
        if tokeep[k]:
            dat_filt['blocks'].extend(datlist[k]['blocks'])
            dat_filt['rewvols'].extend(datlist[k]['rewvols'])
            dat_filt['reward_delay'].extend(datlist[k]['reward_delay'])
            dat_filt['iscatch'].extend(datlist[k]['iscatch'])
            dat_filt['outcomes'].extend(datlist[k]['outcomes'])
            dat_filt['wt'].extend(datlist[k]['wt'])
            dat_filt['rewards_pertrial'].extend(datlist[k]['rewards_pertrial'])
            # dat_filt['w_all'].extend(datlist[k]['w_all'])

    return dat_filt

# TODO: find which usess this. might be for aggregating
def format2sess(dat, useblocks=True, blockspersess=5):
    """
    makes the data look like the rat pipeline. breaks into small sessions, checks for linear sensitivity
    This is used on RNN data when doing further analysis, like model-agnostic testing
    :param dat: (dict) from session simulation
    :param useblocks: (bool) if true, this data has block structure
    :return:
    """

    ops = opsbase()
    if useblocks:
        ops['useblocks'] = True
    else:
        ops['useblocks'] = False

    # 0. fitlering out > 1.5 sd
    if sum(dat['iscatch']) > 0:
        wt_catch = np.array(dat['wt'])[dat['iscatch']] * ops['dt']
        wt_mean = np.mean(wt_catch)
        wt_std = np.std(wt_catch)
        wt_cutoff = wt_mean + 1.5 * wt_std
    else:  # no-catch data
        wt_cutoff = np.max([dat['wt']])*ops['dt']
    keepinds = np.nonzero(np.array(dat['wt']) * ops['dt'] <= wt_cutoff)[0]

    # 1. break into sessions pieces
    block = np.array(dat['blocks'])
    bdiff = np.nonzero(np.diff(block))[0]
    # TODO: decide how to parse with ain put parameter
    if useblocks:
        sessinds = list(bdiff[4::blockspersess])
        sessinds.insert(0, 0)
    else:
        blen = 40  # make this up for things without block structure
        ns = len(block)
        sessinds = list(range(0, ns, blen*blockspersess))

    datlist = []
    for k in range(len(sessinds) - 1):

        # filter out any bad data
        useinds = keepinds[(keepinds >= sessinds[k]) & (keepinds < sessinds[k + 1])]

        dat_k = {}
        dat_k['blocks'] = np.array(dat['blocks'])[useinds]
        dat_k['rewvols'] = np.array(dat['rewvols'])[useinds]
        dat_k['reward_delay'] = np.array(dat['reward_delay'])[useinds]
        dat_k['iscatch'] = np.array(dat['iscatch'])[useinds]
        dat_k['outcomes'] = np.array(dat['outcomes'])[useinds]
        dat_k['wt'] = np.array(dat['wt'])[useinds]
        dat_k['rewards_pertrial'] = np.array(dat['rewards_pertrial'])[useinds]
        dat_k['wt'] = np.array(dat['wt'])[useinds]
        # dat_k['w_all'] = np.array(dat['w_all'])[useinds]

        # exclude if there is not a catch trial for every block type. only true for last stage
        if useblocks:
            mask0 = np.sum(dat_k['blocks'][dat_k['iscatch']] == 0) > 0
            mask1 = np.sum(dat_k['blocks'][dat_k['iscatch']] == 1) > 0
            mask2 = np.sum(dat_k['blocks'][dat_k['iscatch']] == 2) > 0

            if mask0 & mask1 & mask2:
                datlist.append(dat_k)
        else:
            datlist.append(dat_k)

    # run statistics. extract p value for model
    outputlist = []
    ops['displevel'] = 3  # avoids seeing warnings about regression issues
    try:
        for k in datlist:
            outputlist.append(wta.parsesimulation(k, ops, allreg=False))
    except ValueError:  # for no-catch trials
        outputlist = []

    # if p value is sig, and slope is positive, then include
    if useblocks:
        tokeep = [(m[1]['p_mixed'] < 0.05) & (m[1]['m_mixed'] > 0) for m in outputlist]
    else:
        if len(outputlist) > 0:  # for catch sessions
            tokeep = [(m[1]['p'] < 0.05) & (m[1]['m'] > 0) for m in outputlist]
        else:
            tokeep = [True for _ in datlist]

    return datlist, tokeep

# TODO: needed?
def wt_partialdata(dat, partitions):
    """
    break a larger dat struct into pieces and calculate wt curves.
    :param dat:  typical data struct
    :param partitions: list of start and stop idx of gradient steps to cut data
    :return: wait time data for each partition
    """
    timings = np.ceil(np.cumsum(np.array(dat['wt'])) / 500)
    pvec_idx = [[np.argmin(np.abs(timings - k[0])), np.argmin(np.abs(timings - k[1]))] for k in partitions]

    dat_k = deepcopy(dat)
    outcomes_old = deepcopy(dat['outcomes'])
    iscatch_old = deepcopy(dat['iscatch'])
    rewvols_old = deepcopy(dat['rewvols'])
    wt_old = deepcopy(dat['wt'])
    blocks_old = deepcopy(dat['blocks'])

    wtdict_list = []
    linreg_list = []
    numdict_list = []
    rho_all_list = []

    for k in pvec_idx:
        dat['outcomes'] = outcomes_old[k[0]:k[1]]
        dat['iscatch'] = iscatch_old[k[0]:k[1]]
        dat['rewvols'] = rewvols_old[k[0]:k[1]]
        dat['wt'] = wt_old[k[0]:k[1]]
        dat['blocks'] = blocks_old[k[0]:k[1]]

        ops = opsbase()  # needed for things like dt I think?

        wt_dict, linreg, numdict, rho_all, ops = wta.parsesimulation(dat_k, ops)
        wtdict_list.append(wt_dict)
        linreg_list.append(linreg)
        numdict_list.append(numdict)
        rho_all_list.append(rho_all)

        wt_vis.plotblocks(wt_dict, idx=1, name=k)
        wt_vis.plotsensitivity(wt_dict, idx=1, name=k)

    return wtdict_list, linreg_list, numdict_list, rho_all_list


def get_trialtimes(dat, includestate=True):
    """
    returns a dictionary of indices for trial start, end, and beginning of the ITI. from a task run
    :param dat:
    :param includestate: (bool) flatten state? sometimes no state info in dat...
    :return: (ndarray) flattened state, (ndarray) flattened inputs, and (dict) of indices for each trial event
    """
    ismultiregion = dat['ops']['ismultiregion']  # is the a multiregion network with stacked inputs?
    s = dat['states']  # state, including cell state
    inp = dat['inputs']  # inputs to RNN
    wt = dat['wt']
    ns = len(wt)
    iti = dat['env'].iti[:ns]  # iti for each trial
    inputcase = dat['ops']['inputcase']

    sflat = []
    inpflat = []

    # TODO: handle multiregion inputs. already done I thinj?

    for k in range(len(inp)):
        inpflat.extend(inp[k])
    if includestate:
        for k in range(len(inp)):
            sflat.extend(s[k])

    # pull important parts. assume that

    if ismultiregion:
        # for now, assume that offer and reward is present in first unit
        unit = 0
        inpflat_simple = np.array([k[unit][0, 0, :].detach().numpy() for k in inpflat])
        idx_start = utils.getstart_from_OFCinput(inpflat_simple,inputcase=inputcase)
        idx_start = np.argwhere(idx_start == 1)[:,0]

        #offers = [k[unit][0, 0, 0] for k in inpflat]
        #offers = [utils.getoffers_from_OFCinput(k[unit][0, 0, :], inputcase=inputcase, alltime=True) for k in inpflat]
    else:
        #offers = [k[0, 0, 0] for k in inpflat]
        inpflat_simple = np.array([k[0, 0, :].detach().numpy() for k in inpflat])
        idx_start = utils.getstart_from_OFCinput(inpflat_simple, inputcase=inputcase)
        idx_start = np.argwhere(idx_start == 1)[:, 0]

        #offers = [utils.getoffers_from_OFCinput(k[0, 0, :], inputcase=inputcase, alltime=True) for k in inpflat]


    # for a given trial, get start, beginiti, and end times
    #idx_start = np.argwhere(np.array(offers) > 0)

    idx_end = idx_start[1:] - 1
    idx_start = idx_start[:-1]
    # idx_end = np.concatenate((idx_end,np.array([len(offers)])),axis = 0)
    idx_rew = np.cumsum(np.array(dat['wt']) + iti) - iti  # beginning of iti, when reward hits as input

    envtype = dat['ops']['envtype']
    if envtype == 'wt_env_wITI_partial_batchedreward':
        # true ITI starts after deterministic reward period
        idx_iti = idx_rew + dat['ops']['iti_determ']
    else:
        # for back-compatibility, keep the same
        idx_iti = idx_rew


    tdict = {'start': idx_start, 'iti': idx_iti, 'rew':idx_rew, 'end': idx_end}

    return sflat, inpflat, tdict


def batchsamples_kind_allstate(net, inputs, si, device=torch.device('cpu')):
    """
    code to propagate and flatten data from kindergarten. analogue of wt_reinfoce_cont
    take a target network, run some samples through it, but save each state. no gradient calc
    The state will be flattened to preserve temporal depenencies, meaning batches are concatenated
    :param net: netowrk
    :param inputs: inputs to network
    :param si: initial state
    :param device: device
    :return outputs: produced outputs from network
    :return state: hidden state of network
    :return net: network. TODO needed?
    """

    batchsize, nsteps, _ = inputs.shape
    # parse inputs, mostly for multiregion
    inputs = net.parseiputs(inputs)

    # states = [si]
    # outputs = [None]
    # preds_supervised = [None]

    states = []
    outputs = []  # supervised learning outputs
    outputs2 = []  # activations for prediction
    # preds_supervised = []
    # preeds_activations = []

    sk = si

    for k in range(nsteps):

        inputs_k = [torch.unsqueeze(m[:, k, :], axis=1) for m in inputs]

        output_k = net(inputs_k, sk, isITI=False,
                       device=device)  # assume that all models will augment action space for ITI
        # paction_all = output['policy']  # prob of actions
        sk = output_k['state']  # state of network
        # this will switch the batch and timestep dimensions in states
        if batchsize > 1:  # a reordering issue for > 1, and a squeeze issues for ==1
            sk = [tuple([torch.unsqueeze(torch.squeeze(mm), axis=0) for mm in m]) for m in sk]

        # value = output['value']  # value of state
        pred_supervised_k = output_k['pred_supervised']  # supervised output predicitons for kindergarten
        pred_activations_k = output_k['activations']  # prediction activations for block or trial inference

        # needed?
        outputs_rs = torch.reshape(pred_supervised_k, (batchsize, 1, -1))
        outputs.append(outputs_rs.detach().numpy())  # migh squeeze the t dim later?

        outputs2_rs = torch.reshape(pred_activations_k, (batchsize, 1, -1))
        outputs2.append(outputs2_rs.detach().numpy())  # migh squeeze the t dim later?

        states.append(sk)

    # detach states.
    if isinstance(si[0], tuple):  # multiregion
        [[[k.detach() for k in sk] for sk in m] for m in states]
    elif isinstance(si, tuple):  # LSTM
        [[k.detach() for k in m] for m in states]
    else:  # vanilla RNN, GRU
        [m.detach for m in states]

    # if batchsize if > 1, resize inputs, outputs, states here
    if batchsize > 1:
        print('handling batchsize > 1. reshaping')
        outputsnew = []
        outputsnew2 = []

        statesnew = []

        for j in range(batchsize):
            # batch and timestep dims were swapped above
            skj = [[tuple([torch.unsqueeze(cell[:, j, :], dim=1).detach().numpy() for cell in reg]) for reg in st] for
                   st in states]
            statesnew.extend(skj)
            outputsnew.extend([np.squeeze(k[j, :]) for k in outputs])
            outputsnew2.extend([np.squeeze(k[j, :]) for k in outputs2])

        outputs = np.array(outputsnew)
        outputs2 = np.array(outputsnew2)
        states = statesnew
        # reshape inputs as well. only ofc one. used for finding trial starts
        # inputs = [torch.reshape(k, (1, batchsize * nsteps, -1)) for k in inputs]
    else:
        # this is a patch, but just detach the state for now.
        statesnew = [[tuple([torch.unsqueeze(cell[:, 0, :], dim=1).detach().numpy() for cell in reg])
                      for reg in st]
                     for st in states]
        states = statesnew

    # find trial start and end based on offer positions
    idx_start = []
    idx_end = []
    for j in range(batchsize):
        offers = inputs[0][j, :, 0].detach().numpy()
        idx_start_j = np.argwhere(offers > 0)[:, 0]
        idx_end_j = np.concatenate((idx_start_j[1:] - 1, np.array([len(offers) - 1])), axis=0).astype(int)

        idx_start_j += nsteps * j  # account for each batch
        idx_end_j += nsteps * j
        idx_start.extend(list(idx_start_j))
        idx_end.extend(list(idx_end_j))

    # squeeze
    idx_start = np.squeeze(idx_start)
    idx_end = np.squeeze(idx_end)

    tdict = {'start': idx_start, 'end': idx_end, 'iti': idx_end}

    return outputs, outputs2, states, tdict

def extract_outputs_startend(dat):
    """
    pulls events of interest at trial start and end(meaning iti) times. like predictions, pi, V
    @param dat: (object) or (dict) structure from 10k files from training
    @return:
    """

    _, _, tdict = get_trialtimes(dat, includestate=False)
    ts_list = tdict['start']
    te_list = tdict['iti']

    # pi, grab at trial end, then flatten
    pi_tmp = [[dat['pi'][0][j][0, 0] for j in m] for m in te_list]
    pi_end = [item for m in pi_tmp for item in m]

    # values: grab at trial start, then flatten
    val_tmp = [[dat['values'][k][j][0] for j in ts_list[k]] for k in range(len(ts_list))]
    val_start = [item for m in val_tmp for item in m]

    # predictions: grab at trial start
    pred_tmp1 = [[dat['preds'][k][j][0, 0] for j in ts_list[k]] for k in range(len(ts_list))]
    p1 = [item for m in pred_tmp1 for item in m]
    pred_tmp2 = [[dat['preds'][k][j][0, 1] for j in ts_list[k]] for k in range(len(ts_list))]
    p2 = [item for m in pred_tmp2 for item in m]
    pred_tmp3 = [[dat['preds'][k][j][0, 2] for j in ts_list[k]] for k in range(len(ts_list))]
    p3 = [item for m in pred_tmp3 for item in m]
    preds_start = np.array([p1, p2, p3]).T

    return pi_end, val_start, preds_start


def joinbeh(fname, fname_global, overwrite=False):

    if exists(fname_global):
        print('loading global data')
        with open(fname_global, 'r') as f:
            alldat_list = json.load(f)
    else:
        print('master behavioral data file does not exist. creating empty list for it')
        alldat_list = [{'name': 'name'}]

    # if already in the structure. just return. no need to load data
    fname_short = fname.split('/')[-1]
    namelist = [j['name'] for j in alldat_list]
    prevsaved = fname_short in namelist

    if not prevsaved or overwrite:
        if exists(fname):
            print('loading data')
            if fname.split('.')[-1] == 'json':
                with open(fname, 'r') as f:
                    dat = json.load(f)
            elif fname.split('.')[-1] == 'dat':
                dat = CPU_Unpickler(open(fname, 'rb')).load()

            if overwrite:  # must pop original file from list
                print('removing old entries and overwriting')
                fname_short = fname.split('/')[-1]
                names = [j['name'] for j in alldat_list]
                indices = [i for i, x in enumerate(names) if x == fname_short]
                indices.reverse()# pop in reverse order
                for j in indices:
                    alldat_list.pop(j)
        else:
            print('sample behavioral file doesnt exist. aborting')
            return None
    else:
        print('data exists in master file. returning')
        return None

    print('extracting simplified behavioral data')
    fname_short = fname.split('/')[-1]
    behdat = {'name': fname_short,
              'rewvols': [int(m) for m in dat['rewvols']],
              'iscatch': [bool(m) for m in dat['iscatch']],
              'blocks': [int(m) for m in dat['blocks']],
              'outcomes': [bool(m) for m in dat['outcomes']],
              'wt': [int(m) for m in dat['wt']],
              'rewards_pertrial': [float(m) for m in dat['rewards_pertrial']],
              'reward_delay': [float(m) for m in dat['reward_delay']],
              'rewardrate_pergradstep': [float(m) for m in dat['rewardrate_pergradstep']]}
    #behdat['pi_end'] = [float(m) for m in pi_end]
    #behdat['val_start'] = [float(m) for m in val_start]
    #behdat['preds_start'] = [[float(r) for r in m] for m in preds_start]


    print('checking/adding to main data')
    if not exists(fname_global):
        # this overwrites the dummy handle [{'name':'name'}] when file does not exist entry
        alldat_list = [behdat]
    else:
        print('adding to list')
        alldat_list.append(behdat)
        # resave
        print('resaving')

    with open(fname_global, 'w') as f:
        json.dump(alldat_list, f)

    del behdat
    del dat
    gc.collect()

    return None


# TODO: probably remove. but not until after analysis code is made to extract beginning and end events, pred, pi,
def extract_behavior(fname, fname_global, alldat_list=None):
    """removes the relevant behavior bits from a dat struct already saved to file
        inputs:
        fname: file to load with single RNN dat

    """

    if exists(fname_global):
        print('loading global data')
        if alldat_list is None:
            with open(fname_global, 'r') as f:
                alldat_list = json.load(f)
        else:
            print('alldat_list already supplied')
    else:
        print('behavior data does not exist. creating empty list for it')
        alldat_list = [{'name': 'name'}]

    # if already in the structure. just return. no need to load data
    fname_short = fname.split('/')[-1]
    for j in alldat_list:
        if j['name'] == fname_short:
            print('I am in here. returning')
            return None
    else:

        print('loading data')
        if exists(fname):
            if fname.split('.')[-1] == 'dat':
                dat = CPU_Unpickler(open(fname, 'rb')).load()
            elif fname.split('.')[-1] == 'json':
                with open(fname, 'r') as f:
                    dat = json.load(f)
            else:
                print('incorrect file type for raw file for aggreggating')
        else:
            print('file doesnt exist. aborting')
            return None
        # TODO: grab from here.... to......
        print('trimming data')
        ts_list = []  # indices in each minibatch for a trial start
        te_list = []  # indices in each minibatch for a reward delivery
        w = dat['ops']['w']
        for k in range(len(dat['inputs'])):
            inp_offer = np.array([m[0][0, 0, 0].detach().numpy() for m in dat['inputs'][k]])
            inp_rew = np.array([m[0][0, 0, 2].detach().numpy() for m in dat['inputs'][k]])

            on_idx = np.argwhere(inp_offer > 0)[:, 0]
            off_idx_tmp = np.argwhere(inp_rew != w)[:, 0]
            off_idx_start = np.argwhere(np.diff(off_idx_tmp) != 1)[:, 0] + 1
            off_idx = off_idx_tmp[off_idx_start[:-1]] - 1  # reward input is timestep forward. want decision time

            ts_list.append(on_idx)
            te_list.append(off_idx)

        # simplify the probability of wait at reward/optout, value at trial start,
        # and prediciton activations at start

        # pi, grab at trial end, then flatten
        pi_tmp = [[dat['pi'][0][j][0, 0] for j in m] for m in te_list]
        pi_end = [item for m in pi_tmp for item in m]

        # values: grab at trial start, then flatten
        val_tmp = [[dat['values'][k][j][0] for j in ts_list[k]] for k in range(len(ts_list))]
        val_start = [item for m in val_tmp for item in m]

        # predictions: grab at trial start
        pred_tmp1 = [[dat['preds'][k][j][0, 0] for j in ts_list[k]] for k in range(len(ts_list))]
        p1 = [item for m in pred_tmp1 for item in m]
        pred_tmp2 = [[dat['preds'][k][j][0, 1] for j in ts_list[k]] for k in range(len(ts_list))]
        p2 = [item for m in pred_tmp2 for item in m]
        pred_tmp3 = [[dat['preds'][k][j][0, 2] for j in ts_list[k]] for k in range(len(ts_list))]
        p3 = [item for m in pred_tmp3 for item in m]
        preds_start = np.array([p1, p2, p3]).T
        # TODO: to here

        reward_delay = np.array([k['rewardDelay'] for k in dat['env'].trials])

        behdat = {}
        behdat['rewvols'] = [int(m) for m in dat['rewvols']]
        behdat['iscatch'] = [bool(m) for m in dat['iscatch']]
        behdat['blocks'] = [int(m) for m in dat['blocks']]
        behdat['outcomes'] = [bool(m) for m in dat['outcomes']]
        behdat['wt'] = [int(m) for m in dat['wt']]
        behdat['rewards_pertrial'] = [float(m) for m in dat['rewards_pertrial']]
        behdat['reward_delay'] = [float(m) for m in reward_delay]
        behdat['pi_end'] = [float(m) for m in pi_end]
        behdat['val_start'] = [float(m) for m in val_start]
        behdat['preds_start'] = [[float(r) for r in m] for m in preds_start]
        fname_short = fname.split('/')[-1]
        behdat['name'] = fname_short

        print('checking/adding to main data')
        if not exists(fname_global):
            print('global data doesnt exist yet. creating: ' + fname_global)
            alldat_list = [behdat]
        else:
            print('adding to list')
            alldat_list.append(behdat)
            # resave
            print('resaving')

        with open(fname_global, 'w') as f:
            json.dump(alldat_list, f)

        del(behdat)
        del(dat)
        gc.collect()

        return None


def behdat2json(num, datadir):
    """takes a behavioral .dat file and converts and saves to a .json file. MUCH smaller
    """

    # TODO: this is a temporary piece of code that might be omitted in later commits
    fname = datadir + 'rnn_' + str(num) + '_allbeh'
    print(fname)

    with open(fname + '.dat', 'rb') as f:
        beh = pickle.load(f)

    # print('converting to json friendly')
    newbeh = []
    for k in beh:
        n = {}

        n['rewvols'] = [int(m) for m in k['rewvols']]
        n['iscatch'] = [bool(m) for m in k['iscatch']]
        n['blocks'] = [int(m) for m in k['blocks']]
        n['outcomes'] = [int(m) for m in k['outcomes']]
        n['wt'] = [int(m) for m in k['wt']] # look up
        n['rewards_pertrial'] = [float(m) for m in k['rewards_pertrial']]
        n['reward_delay'] = [float(m) for m in k['reward_delay']]
        n['pi_end'] = [float(m) for m in k['pi_end']]
        n['val_start'] = [float(m) for m in k['val_start']]
        n['preds_start'] = [[float(r) for r in m] for m in k['preds_start']]
        n['name'] = k['name']

        # k['reward_delay'] = list(k['reward_delay'])

        newbeh.append(n)

    # print('saving')
    with open(fname + '.json', 'w') as f:
        json.dump(newbeh, f)

    # print('cleaning up')
    del beh
    del newbeh
    gc.collect()

    # print('done')
    return None

def rec(d):
    """
    little recursion helper to flatten ragged lists
    @param d:
    @return:
    """

    if type(d) is list or type(d) is np.ndarray or type(d) is tuple:
        if type(d) is np.ndarray and d.ndim == 0:
            if type(d) is torch.Tensor:
                ans = d.cpu().detach().numpy().tolist()
            else:
                ans = np.array(d).tolist()
        else:
            ans = [rec(k) for k in d]
    elif type(d) is dict:  # a list of dicts
        ans = dict2json(d)
    else:
        if type(d) is torch.Tensor:
            ans = d.cpu().detach().numpy().tolist()
        else:
            ans = np.array(d).tolist()
    return ans

def stats2json(fname, newname):

    with open(fname,'rb') as f:
        dat_stat = pickle.load(f)

    newdat_0 = dict2json(dat_stat[0])
    newdat_1 = dict2json(dat_stat[1])
    newdat_2 = dict2json(dat_stat[2])

    newdat_3 = []
    newdat_3.append(float(dat_stat[3][0]))
    newdat_3.append(float(dat_stat[3][1]))
    newdat_3.append(dat_stat[3][2].tolist())
    newdat_3.append(dat_stat[3][3].tolist())

    newdat_4 = []
    newdat_4.append(dat_stat[4][0].tolist())
    newdat_4.append(dat_stat[4][1])
    newdat_4.append([ float(k) for k in dat_stat[4][2] ])

    newdat_5 = []
    newdat_5.append(dat_stat[5][0].tolist())
    newdat_5.append(dat_stat[5][1])
    newdat_5.append([ float(k) for k in dat_stat[5][2] ])

    newdat = [newdat_0, newdat_1, newdat_2, newdat_3, newdat_4, newdat_5]

    with open(newname, 'w') as f:
        json.dump(newdat, f)


def dict2json(d):
    """
    converts a dictionary entries into json-friendly format. generic code
    @param ops:
    @return:
    """

    for k in d.keys():
        if type(d[k]) is dict:
            d[k] = dict2json(d[k])
        if type(d[k]) is np.bool_:
            d[k] = bool(d[k])
        if type(d[k]) is np.float64:
            d[k] = float(d[k])
        elif type(d[k]) is np.int64:
            d[k] = int(d[k])
        if type(d[k]) is torch.Tensor:
            d[k] = d[k].cpu().detach().numpy().tolist()
        elif type(d[k]) is np.ndarray or type(d[k]) is list:
            d[k] = rec(d[k])
            #warnings.filterwarnings("error")
            #try:
            #    d[k] = np.array(d[k])
            #    d[k] = d[k].tolist()
            #except:
            #    print('nested lists or nd arrays might be ragged. converting manually')
            #    d[k] = rec(d[k])
            #warnings.resetwarnings()



            ## account for just one level of nesting
            #if type(d[k][0]) is np.float64:
            #    d[k] = [float(m) for m in d[k]]
            #elif type(d[k][0]) is np.int64:
            #    d[k] = [int(m) for m in d[k]]
            #else:
            #    d[k] = list(d[k])

    return d

# check the dat ops struct to try and get the dat2json code to work
def dat2json(dat, save_tdvars=True):
    """
    code to convert a raw .dat file saved via pickle to a json-friendly format.
            This saves with much less memory requirements for my pytorch data
    :param dat: (dict) holding the large dat data. if str, open fil
    :param save_tdvars: (bool) if true, save time. dep. variables: state, preds, outcomes, etc.
    :return:
    """

    newdat = {}
    # high-level things
    newdat['tmesh'] = [float(m) for m in dat['tmesh']]
    # newdat['env'] = dat['env'] #TODO: handle this stuff
    newdat['state_reset_idx'] = [int(k) for k in dat['state_reset_idx']]

    # ops has a few fields that need to be massaged
    newdat['ops'] = dict2json(dat['ops'])

    # env is a class, so convert to dict first
    newdat['env'] = dict2json(vars(dat['env']))
    # save classtype metadata
    classtype = str(dat['env'].__class__).split('.')[-1].split('\'')[0]
    newdat['env']['classtype'] = classtype

    # trial-level things
    newdat['rewvols'] = [int(m) for m in dat['rewvols']]
    newdat['iscatch'] = [bool(m) for m in dat['iscatch']]
    newdat['blocks'] = [int(m) for m in dat['blocks']]
    newdat['outcomes'] = [int(m) for m in dat['outcomes']]
    newdat['wt'] = [int(m) for m in dat['wt']]
    newdat['rewards_pertrial'] = [float(m) for m in dat['rewards_pertrial']]
    newdat['reward_delay'] = [float(m) for m in dat['reward_delay']]
    newdat['w_all'] = [float(k) for k in dat['w_all']]

    # gradient-level things.
    newdat['loss'] = [float(k[0]) for k in dat['loss']]
    newdat['loss_aux'] = [[float(m) for m in k] for k in dat['loss_aux']]
    newdat['rewardrate_pergradstep'] = [float(m) for m in dat['rewardrate_pergradstep']]
    newdat['annealed_weights'] = [[float(m) for m in k] for k in dat['annealed_weights']]

    # TODO THESE GRAD-LEVEL THINGS NOT TESTED. not used in new saves, so not tested
    if len(dat['grad_all']) > 0:
        newdat['grad_all'] = [float(k) for k in dat['grad_all']]
    else:
        newdat['grad_all'] = []

    if len(dat['theta']) > 0:
        newdat['theta'] = [float(k) for k in dat['theta']]
    else:
        newdat['theta'] = []

    # timestep-level things: [grad][timestep][dims]
    if save_tdvars:
        # states: ngrad, ntime, nregion, ncell, 1,1, nneur. squeeze out nuiscance ones
        for k in ['actions','values', 'rewards', 'preds', 'outputs_supervised', 'pi']:
            newdat[k] = np.squeeze(np.array(dat[k])).tolist()

        # if trainingonly=True or trackstate = False: this is empty
        if len(dat['states']) == 0:
            print('no state info. ')
            newdat['states'] = []
        elif type(dat['states'][0][0][0][0]) is torch.Tensor:
            print('old state')
            newdat['states'] = [[[[p.cpu().detach().numpy().astype(float).tolist() for p in n]for n in m] for m in k] for k in
                                dat['states']]
            #handle extra dims
            newdat['states'] = np.squeeze(np.squeeze(np.array(newdat['states']))).tolist()
            print('done w old state')
        else:
            newdat['states'] = np.squeeze(np.squeeze(np.array(dat[k])).tolist())

        if type(dat['inputs'][0][0]) is list:  # a multi-region network
            newdat['inputs'] = [[[n.cpu().detach().numpy().astype(float).tolist() for n in m] for m in k] for k in dat['inputs']]
        elif type(dat['inputs'][0][0]) is torch.Tensor:
            newdat['inputs'] = [[m.cpu().detach().numpy().astype(float).tolist() for m in k] for k in dat['inputs']]
            newdat['inputs'] = np.squeeze(np.squeeze(np.array(newdat['inputs']))).tolist()

    return newdat


def json2dat(d):
    # converts json back into dat

    # these are fields that have arrays as an entry for each list. add them back in
    expand_keys = ['loss']
    newd = {}
    for k in d.keys():
        # print(k)
        if k == 'ops':
            newd[k] = json2dat(d[k])
        if k == 'env':
            if 'classtype' in d[k]:
                classtype = d[k].pop('classtype')
            else:  # an older save file. assume
                classtype = 'wt_env_wITI_batchedreward'

            env = wt_envs.envdict[classtype](ops=newd['ops'])
            env.__dict__.update(d[k])
            newd[k] = env

        elif k == 'inputs':
            newd[k] = [[[torch.tensor(np.array(m)) for m in k] for k in j] for j in d[k]]
        elif k == 'loss':
            newd[k] = np.expand_dims(np.expand_dims(np.array(d[k]), axis=-1), axis=-1)
        elif type(d[k]) is list:
            newd[k] = np.array(d[k])
        else:
            newd[k] = d[k]

    return newd



def KEconvert2matlab(fname, dosave=True, eps=3):
    """
    extracts the fixed and slow points from KE minization file, clusters with DBSCAN,
    then converts to a matlab file
    :param fname: file name of kinetic energy minima located by minimization
    :param dosave: if true, saves .mat file. otherwise just returns dict
    :param eps: the DBscan parameter
    :return: (dict) of coordinates, schur modes, eigenvalues, etc.
    """
    # print(fname)
    #warnings.filterwarnings("ignore", message="Trying to unpickle estimator PCA*")
    if not exists(fname):
        print('data file does not exist')
        return None

    if fname.split('.')[-1] == 'dat':
        #print('loading legacy .dat data')
        with open(fname, 'rb') as f:
            dat = pickle.load(f)

    #elif fname.split('.')[-1] == 'json':
    #    print('loading data')
    #    with open(fname, 'r') as f:
    #        dat = json.load(f)
    #print(dat.keys())

    # format everything into a matlab-friendly thing
    ns = len(dat['output'])
    #crds = np.array([[dat['output'][k][0][0], dat['output'][k][0][1]] for k in range(ns)])  # assume 2d
    crds = np.array([dat['output'][k][0] for k in range(ns)])
    KE = np.array([dat['output'][k][1] for k in range(ns)])
    jac_pc = np.array([dat['output'][k][2][0] for k in range(ns)])
    evals_full = np.array([dat['output'][k][2][1] for k in range(ns)])

    # filter the data into slow and fast points, then cluster to get the effecgtive points
    # create cutoffs for fixed points and slow points. filter data
    mask_fixed = KE < 0.0001

    # define slow points that only change magnitude < 5% across duration of trial
    lam = 2.5
    dt = 0.05
    eps_norm = 0.05
    eps_fp = 1e-3
    sdiffnorm = 2 * np.sqrt(KE) * lam / (np.linalg.norm(crds, axis=1))
    mask_slow = sdiffnorm < eps_norm

    mask = (mask_fixed) | (mask_slow)
    # potentially get ride of this
    mask = np.ones(mask.shape).astype(bool)

    crds_masked = crds[mask, :]
    KE_masked = KE[mask]
    jac_pc_masked = jac_pc[mask, :, :]
    evals_full_masked = evals_full[mask, :]
    sdiffnorm_masked = sdiffnorm[mask]
    isfixed_masked = KE_masked < 0.0001
    ns_masked = sum(mask)

    clustering = dbscan(eps=eps, min_samples=10).fit(crds_masked)
    labs = np.unique(clustering.labels_)
    labels = clustering.labels_

    #

    savedict = {'ns': ns_masked, 'crds': crds_masked, 'KE': KE_masked, 'jac_pc': jac_pc_masked,
                'evals_full_masked': evals_full, 'statediff': sdiffnorm_masked, 'labels': labels,
                'isfixed': isfixed_masked}
    if dosave:
        savemat(fname.split('.')[0] + '.mat', savedict)
    return savedict


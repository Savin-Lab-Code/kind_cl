<<<<<<< HEAD
# code for structruing beh data for different uses, like behavioral modeling, state analysis, etc.
=======
# code for structruing data for different uses, like behavioral modeling, state analysis, etc.
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)

import datetime
import pickle
import numpy as np
from scipy.io import savemat
from dynamics.utils.utils import CPU_Unpickler, opsbase
from dynamics.process.rnn import wt_envs
<<<<<<< HEAD
=======
import dynamics.utils.utils as utils
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
import torch
from os.path import exists
import gc
import json
from sklearn.cluster import DBSCAN as dbscan


<<<<<<< HEAD
def generate_rattrial_rnn(loadname, savename,
                          date_base=str(datetime.date.today())+' ', ops=None, ntrials=None):
    """
    create a .mat file that can be used with the Constantinople lab Matlab codebase
    :param loadname: (str) data file to be loaded .dat or .json type
    :param savename: (str) savename path + preamble.
=======
# TODO: should I run that other preprocessing step before this, the one that mimics the rat pipeline?
def generate_rattrial_rnn(loadname, savename,
                          date_base=str(datetime.date.today())+' ', ops=None, ntrials=None):
    """
    create a .mat file that can be used with the behavioral model
    :param savename_base: (str) savename path + preamble.
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
    :param date_base: date of simulation. if unknown, use current date
    :param ops: wt_reinforce ops
    :param ntrials: (list) number of trials per session
    :return: simplified structure that is way smaller
    """
    if ops is None:
        ops = opsbase()

    if not exists(loadname):
        print('data does not exist')

<<<<<<< HEAD
    if loadname.split('.')[-1] == 'dat':
        print('loading legacy data')
        with open(loadname, 'rb') as f:
=======
    if loadname.split('.')[-1]=='dat':
        print('loading legacy data')
        with open(loadname, 'rb') as f:
            #a = CPU_Unpickler(f).load()
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
            a = pickle.load(f)
    elif loadname.split('.')[-1] == 'json':
        with open(loadname, 'r') as f:
            a = json.load(f)

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

<<<<<<< HEAD
=======
    # TODO: get ntrials as input
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
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
<<<<<<< HEAD
    # optout.astype(float)

=======

    # TODO: if this throws type error, try np.float instead of float
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
    A = {'reward': reward, 'date': date, 'block': block, 'reward_delay': reward_delay, 'prob_catch': prob_catch,
         'wait_time': wait_time, 'ntrials': ntrials, 'hits': hits.astype(bool), 'optout': optout.astype(bool),
         'vios': vios.astype(bool), 'catch': catch.astype(bool),
         'rt': rt, 'ITI': ITI, 'wait_for_poke': wait_for_poke, 'tau': tau, 'pcatch': pcatch,
         'wait_thresh': wt_thresh, 'seed': ops['seed']}
    AA = {'A': A}

    savemat(savename, AA, do_compression=False)

    return A


<<<<<<< HEAD
=======
# loading and parsing scripts
# TODO: eventually remove this. only have code that loads the .stats file and runs analysis code from that
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
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
<<<<<<< HEAD
=======
    inputcase = dat['ops']['inputcase']
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)

    sflat = []
    inpflat = []

<<<<<<< HEAD
=======
    # TODO: handle multiregion inputs. already done I thinj?

>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
    for k in range(len(inp)):
        inpflat.extend(inp[k])
    if includestate:
        for k in range(len(inp)):
            sflat.extend(s[k])

    # pull important parts. assume that
<<<<<<< HEAD
    if ismultiregion:
        # for now, assume that offer and reward is present in first unit
        unit = 0
        offers = [k[unit][0, 0, 0] for k in inpflat]
        # rews = [k[unit][0, 0, 2] for k in inpflat]
    else:
        offers = [k[0, 0, 0] for k in inpflat]
        # rews = [k[0, 0, 2] for k in inpflat]

    # for a given trial, get start, beginiti, and end times
    idx_start = np.argwhere(np.array(offers) > 0)
    idx_end = idx_start[1:, 0] - 1
    idx_start = idx_start[:-1, 0]
    # idx_end = np.concatenate((idx_end,np.array([len(offers)])),axis = 0)
=======

    if ismultiregion:
        # for now, assume that offer and reward is present in first unit
        unit = 0
        inpflat_simple = np.array([k[unit][0, 0, :].detach().numpy() for k in inpflat])
        idx_start = utils.getstart_from_OFCinput(inpflat_simple, inputcase=inputcase)
        idx_start = np.argwhere(idx_start == 1)[:, 0]

    else:
        inpflat_simple = np.array([k[0, 0, :].detach().numpy() for k in inpflat])
        idx_start = utils.getstart_from_OFCinput(inpflat_simple, inputcase=inputcase)
        idx_start = np.argwhere(idx_start == 1)[:, 0]

    # for a given trial, get start, beginiti, and end times
    idx_end = idx_start[1:] - 1
    idx_start = idx_start[:-1]
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
    idx_rew = np.cumsum(np.array(dat['wt']) + iti) - iti  # beginning of iti, when reward hits as input

    tdict = {'start': idx_start, 'iti': idx_rew, 'end': idx_end}

    return sflat, inpflat, tdict


def batchsamples_kind_allstate(net, inputs, si, device=torch.device('cpu')):
    """
    code to propagate and flatten data from kindergarten. analogue of wt_reinfoce_cont
    take a target network, run some samples through it, but save each state. no gradient calc
    :param net: netowrk
    :param inputs: inputs to network
    :param si: initial state
    :param device: device
    :return outputs: produced outputs from network
    :return state: hidden state of network
<<<<<<< HEAD
    :return net: network.
=======
    :return net: network. TODO needed?
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
    """

    batchsize, nsteps, _ = inputs.shape
    # parse inputs, mostly for multiregion
    inputs = net.parseiputs(inputs)
    states = []
    outputs = []  # supervised learning outputs
    outputs2 = []  # activations for prediction

    sk = si

    for k in range(nsteps):

        inputs_k = [torch.unsqueeze(m[:, k, :], dim=1) for m in inputs]

        output_k = net(inputs_k, sk, isITI=False,
                       device=device)  # assume that all models will augment action space for ITI
        sk = output_k['state']  # state of network
        if batchsize > 1:  # a reordering issue for > 1, and a squeeze issues for ==1
            sk = [tuple([torch.unsqueeze(torch.squeeze(mm), dim=0) for mm in m]) for m in sk]

<<<<<<< HEAD
        pred_supervised_k = output_k['pred_supervised']  # supervised output predicitons for kindergarten
        pred_activations_k = output_k['activations']  # prediction activations for block or trial inference

        outputs_rs = torch.reshape(pred_supervised_k, (batchsize, 1, -1))
        outputs.append(outputs_rs.detach().numpy())

        outputs2_rs = torch.reshape(pred_activations_k, (batchsize, 1, -1))
        outputs2.append(outputs2_rs.detach().numpy())
=======
        # value = output['value']  # value of state
        pred_supervised_k = output_k['pred_supervised']  # supervised output predicitons for kindergarten
        pred_activations_k = output_k['activations']  # prediction activations for block or trial inference

        # needed?
        outputs_rs = torch.reshape(pred_supervised_k, (batchsize, 1, -1))
        outputs.append(outputs_rs.detach().numpy())  # migh squeeze the t dim later?

        outputs2_rs = torch.reshape(pred_activations_k, (batchsize, 1, -1))
        outputs2.append(outputs2_rs.detach().numpy())  # migh squeeze the t dim later?
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)

        states.append(sk)

    # detach states.
    if isinstance(si[0], tuple):  # multiregion
        [[[k.detach() for k in sk] for sk in m] for m in states]
    elif isinstance(si, tuple):  # LSTM
        [[k.detach() for k in m] for m in states]
    else:  # vanilla RNN, GRU
<<<<<<< HEAD
        _ = [m.detach for m in states]
=======
        [m.detach() for m in states]
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)

    # if batchsize if > 1, resize inputs, outputs, states here
    if batchsize > 1:
        print('handling batchsize > 1. reshaping')
        outputsnew = []
        outputsnew2 = []

        statesnew = []
        for j in range(batchsize):
            skj = [[tuple([torch.unsqueeze(cell[:, j, :], dim=1) for cell in reg]) for reg in st] for st in states]
            statesnew.extend(skj)
            outputsnew.extend([np.squeeze(k[j, :]) for k in outputs])
            outputsnew2.extend([np.squeeze(k[j, :]) for k in outputs2])

        outputs = np.array(outputsnew)
        outputs2 = np.array(outputsnew2)
        states = statesnew
        # reshape inputs as well. only ofc one. used for finding trial starts
        inputs = [torch.reshape(k, (1, batchsize * nsteps, -1)) for k in inputs]

    # This flattens things similar to get_trialtimes. need to find trial starts
    offers = inputs[0][0, :, 0]
    idx_start = np.argwhere(offers > 0)[0, :]
    idx_end = np.concatenate((idx_start[1:] - 1, np.array([len(offers) - 1])), axis=0).astype(int)

    # squeeze
    idx_start = np.squeeze(idx_start)
    idx_end = np.squeeze(idx_end)

    tdict = {'start': idx_start, 'end': idx_end, 'iti': idx_end}

    return outputs, outputs2, states, tdict


def joinbeh(fname, fname_global, overwrite=False):
<<<<<<< HEAD
    """
    creates a behavior-only data file for an entire RNN, over training
    @param fname: (str) potential file to load
    @param fname_global: (str) aggregated filename to append
    @param overwrite: (bool) should you overwrite data if it exists in fname_global?
    @return:
    """
=======
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)

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
            else:
<<<<<<< HEAD
                print('unsupported data type: ' + fname.split('.')[-1])
=======
                print('error in data file name: '+ fname)
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
                dat = None

            if overwrite:  # must pop original file from list
                print('removing old entries and overwriting')
                fname_short = fname.split('/')[-1]
                names = [j['name'] for j in alldat_list]
                indices = [i for i, x in enumerate(names) if x == fname_short]
<<<<<<< HEAD
                indices.reverse()  # pop in reverse order
=======
                indices.reverse()# pop in reverse order
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
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

<<<<<<< HEAD
=======

>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
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


<<<<<<< HEAD
=======
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

>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
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


def dict2json(d):
    """
    converts a dictionary entries into json-friendly format. generic code
<<<<<<< HEAD
    @param d: dictionary
    @return:
=======
    @param d: (dict) of results, likely from simulation
    @return: (dict) with values that json-friendly for saving
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
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

    return d

<<<<<<< HEAD

=======
# check the dat ops struct to try and get the dat2json code to work
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
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
<<<<<<< HEAD
=======
    # newdat['env'] = dat['env'] #TODO: handle this stuff
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
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
<<<<<<< HEAD
        for k in ['actions', 'values', 'rewards', 'preds', 'outputs_supervised', 'pi']:
=======
        for k in ['actions','values', 'rewards', 'preds', 'outputs_supervised', 'pi']:
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
            newdat[k] = np.squeeze(np.array(dat[k])).tolist()

        # if trainingonly=True or trackstate = False: this is empty
        if len(dat['states']) == 0:
            print('no state info. ')
            newdat['states'] = []
        elif type(dat['states'][0][0][0][0]) is torch.Tensor:
            print('old state')
<<<<<<< HEAD
            newdat['states'] = [[[[p.cpu().detach().numpy().astype(float).tolist() for p in n]for n in m]
                                 for m in k] for k in dat['states']]
            # handle extra dims
=======
            newdat['states'] = [[[[p.cpu().detach().numpy().astype(float).tolist() for p in n]for n in m] for m in k] for k in
                                dat['states']]
            #handle extra dims
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
            newdat['states'] = np.squeeze(np.squeeze(np.array(newdat['states']))).tolist()
            print('done w old state')
        else:
            newdat['states'] = np.squeeze(np.squeeze(np.array(dat[k])).tolist())

        if type(dat['inputs'][0][0]) is list:  # a multi-region network
<<<<<<< HEAD
            newdat['inputs'] = [[[n.cpu().detach().numpy().astype(float).tolist() for n in m] for m in k] for
                                k in dat['inputs']]
=======
            newdat['inputs'] = [[[n.cpu().detach().numpy().astype(float).tolist() for n in m] for m in k] for k in dat['inputs']]
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
        elif type(dat['inputs'][0][0]) is torch.Tensor:
            newdat['inputs'] = [[m.cpu().detach().numpy().astype(float).tolist() for m in k] for k in dat['inputs']]
            newdat['inputs'] = np.squeeze(np.squeeze(np.array(newdat['inputs']))).tolist()

    return newdat


def json2dat(d):
    # converts json back into dat

    # these are fields that have arrays as an entry for each list. add them back in
    newd = {}
    for k in d.keys():
        # print(k)
        if k == 'ops':
            newd[k] = json2dat(d[k])
        if k == 'env':
            classtype = d[k].pop('classtype')
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


<<<<<<< HEAD
=======

>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
def KEconvert2matlab(fname, dosave=True, eps=3):
    """
    extracts the fixed and slow points from KE minization file, clusters with DBSCAN,
    then converts to a matlab file
    :param fname: file name of kinetic energy minima located by minimization
    :param dosave: if true, saves .mat file. otherwise just returns dict
    :param eps: the DBscan parameter
    :return: (dict) of coordinates, schur modes, eigenvalues, etc.
    """

    if not exists(fname):
        print('data file does not exist')
        return None

    if fname.split('.')[-1] == 'dat':
        with open(fname, 'rb') as f:
            dat = pickle.load(f)

    # format everything into a matlab-friendly thing
    ns = len(dat['output'])
    crds = np.array([dat['output'][k][0] for k in range(ns)])
    KE = np.array([dat['output'][k][1] for k in range(ns)])
    jac_pc = np.array([dat['output'][k][2][0] for k in range(ns)])
    evals_full = np.array([dat['output'][k][2][1] for k in range(ns)])

    # filter the data into slow and fast points, then cluster to get the effecgtive points
    # create cutoffs for fixed points and slow points. filter data
    mask_fixed = KE < 0.0001

    # define slow points that only change magnitude < 5% across duration of trial
    lam = 2.5
    eps_norm = 0.05
    sdiffnorm = 2 * np.sqrt(KE) * lam / (np.linalg.norm(crds, axis=1))
    mask_slow = sdiffnorm < eps_norm
<<<<<<< HEAD

    mask = mask_fixed | mask_slow
=======
    mask = (mask_fixed) | (mask_slow)
    # potentially get rid of this
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
    mask = np.ones(mask.shape).astype(bool)

    crds_masked = crds[mask, :]
    KE_masked = KE[mask]
    jac_pc_masked = jac_pc[mask, :, :]
    sdiffnorm_masked = sdiffnorm[mask]
    isfixed_masked = KE_masked < 0.0001
    ns_masked = sum(mask)

    clustering = dbscan(eps=eps, min_samples=10).fit(crds_masked)
    labels = clustering.labels_

    savedict = {'ns': ns_masked, 'crds': crds_masked, 'KE': KE_masked, 'jac_pc': jac_pc_masked,
                'evals_full_masked': evals_full, 'statediff': sdiffnorm_masked, 'labels': labels,
                'isfixed': isfixed_masked}
    if dosave:
        savemat(fname.split('.')[0] + '.mat', savedict)
    return savedict

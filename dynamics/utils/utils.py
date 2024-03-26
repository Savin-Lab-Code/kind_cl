# simple helper function for tiered output
import psutil
import torch
import numpy as np
import gc
import pickle
import io
import configparser
import ast
import json


def getfnames(num, s_idx, tphase, idx, dbase='/scratch/dh148/dynamics/results/rnn/ac/20231003/',
              reg_idx=0, block='mixed', epoch='wait'):
    """
    gives filenames for important RNN files
    @param num: (int) RNN number
    @param s_idx: (int) curriculum type. 0 = full_cl, 1 = nok_cl, 2 = nok_nocl. others below
    @param tphase: (int) stage of training. 6=freeze, 5=block, 4=catch, 3=nocatch, 2=pred, 1=hard , 0=simple
    @param idx: (int) phase withint training stage. typically 0-10 for nocatch-block, but nok_cl and nok_cl go higher
    @param dbase: (str) base directory for results.
    @param reg_idx: (int) RNN region (0=ofc, 1 = str). used for KE min filename and flowfield filename
    @param block: (str) block type (mixed, high, low). used for KE min filename and flowfield filename
    @param epoch: (str) part of tril ('wait','start','iti'). used for KE min filename and flowfield filename
    @return:
    """

    subdirlist = ['full_cl/', 'nok_cl/', 'nok_nocl/', 'pkind_mem/', 'pkind_count/', 'pkind_int/', 'pkind_pred/',
                  'sham_wkind/', 'sham_wokind/']

    # important directories
    datadir_dat = dbase + subdirlist[s_idx]  # primary subdirecotry for that curric type
    savedir_KE = dbase + subdirlist[s_idx] + 'dynamics/KEmin_constrained/'  # for kinetic energy min file
    savedir_flows = dbase + subdirlist[s_idx] + 'dynamics/flows/'  # where flow field .mat files stored
    savedir_stats = dbase + subdirlist[s_idx] + str(num) + '/'  # for stats based on simulation of 10k trials

    # lambda functions for each type of training
    fgen = lambda NUM, IDX, BASE, SESS, S_IDX: datadir_dat + str(NUM) + '/' + BASE + str(
        num) + '_' + SESS + '_' + str(IDX)

    fname_funs = [
        lambda NUM, IDX, S_IDX: datadir_dat + str(NUM) + '/' + 'rnn_kindergarten_' + str(NUM) + '_simple',
        lambda NUM, IDX, S_IDX: datadir_dat + str(NUM) + '/' + 'rnn_kindergarten_' + str(NUM) + '_int_0_' + str(
            IDX),
        lambda NUM, IDX, S_IDX: datadir_dat + str(NUM) + '/' + 'rnn_pred_' + str(NUM) + '_' + str(idx),
        lambda NUM, IDX, S_IDX: fgen(NUM, IDX, 'rnn_curric_', 'nocatch', S_IDX),
        lambda NUM, IDX, S_IDX: fgen(NUM, IDX, 'rnn_curric_', 'catch', S_IDX),
        lambda NUM, IDX, S_IDX: fgen(NUM, IDX, 'rnn_curric_', 'block', S_IDX),
        lambda NUM, IDX, S_IDX: fgen(NUM, IDX, 'rnn_curric_', 'block', S_IDX) + '_freeze'
        ]

    # important filenames. model file, original training data, simulated actiivty, stats file , parent beh

    # model file
    modelname = fname_funs[tphase](num, idx, s_idx) + '.model'
    model_init = modelname.split('rnn_')[0]+'rnn_init_'+str(num)+ '.model'

    # behavioral and network activity parent file
    if s_idx < 2:  # full_cl and nok_cl used 10k trials. nok_nocl and partial use 1k from frozen network
        # aggregated behavioral data file
        fname_behdat_fun = datadir_dat + 'rnn_' + str(num) + '_allbeh.json'
        # stats file saved during that stage of training
        statsname = savedir_stats + modelname.split('/')[-1].split('.')[0] + '.stats'
        # neural activity data. also 'name' handle in bulk behavioral file
        savename = modelname.split('.')[0] + '.json'
    else:
        fname_behdat_fun = datadir_dat + 'rnn_' + str(num) + '_allbeh_1k.json'
        statsname = savedir_stats + modelname.split('/')[-1].split('.')[0] + '_1k.stats'
        savename = modelname.split('.')[0] + '_1k.json'

    # dynamics
    base_ke = modelname.split('/')[-1].split('.')[0]
    kemin_name = savedir_KE + 'kemin_' + base_ke + 'reg_' + str(reg_idx) + '_' + block + '_' + epoch + '.dat'
    flowname = savedir_flows + 'flowfields' + base_ke + 'reg_' + str(reg_idx) + '_' + block + '_' + epoch + '.mat'
    flowname_boutique = flowname.split('.mat')[0]+'_boutique.mat'  # uses custom PC lims based on KEmin

    d = {'model': modelname, 'init':model_init, 'stats': statsname, 'allbeh': fname_behdat_fun, 'dat': savename,
         'ke': kemin_name, 'flow': flowname, 'flow_boutique': flowname_boutique}

    return d


def retrieve_behdat(fname_behdat, fname):
    """ helper code to lookup the dat in the behavioral data containing training data
    for entire RNN.
    if fname_behda is a list, just search it (for faster lookup)
    if a filename, load the file."""

    fname_base = fname.split('/')[-1]
    dat = None

    if type(fname_behdat) is list:  # easier to recylce
        datlist = fname_behdat
    else:
        datlist = json.load(open(fname_behdat, 'r'))

    for j in datlist:
        if j['name'] == fname_base:
            dat = j
            break

    return dat


def opsbase():
    """
    default options used across many aspects of the wait time simulations
    :return:
    """

    # set a numpy random seed for reproducibility
    np.random.seed(101)
    ops = {}

    # learning and task stuff
    ops['gamma'] = 1.0  # the temporal discounting factor
    ops['alpha'] = 0.0001  # the learning rate for meta-RL
    ops['lam'] = 2.5  # the mean parameter for exponential distribution of reward delays
    ops['pcatch'] = 0.15  # the catch probability. turn off for now
    ops['ctmax'] = 500  # number of timesteps in a gradient batch, in non-episodic run.
    ops['blocklength'] = 40  # how many trials in a block
    ops['freeze'] = False  # should weights be frozen and not trained?
    ops['state_resetfreq'] = 160  # frequency (in trials) that state should be reset. maybe blocklen*3
    ops['p_statereset'] = 1.0  # if < 1, use probabilistic state reset. this is prob. of reset at each trial
    ops['prog_stop'] = False  # use progress-based monitoring in curriculum learning
    ops['envtype'] = 'wt_env_wITI_batchedreward'  # what kind of environment to use. envs.py
    ops['iti_determ'] = 1  # dterministic ITI. number of steps between trials.
    ops['iti_random'] = 20  # option for stochstic ITI, uniform from 0 to iti_random. overrides iti_determ
    ops['iti_type'] = 'uniform'  # the type of distribution to pull from. uniform or gaussian currently
    ops['pblock'] = 0.5  # probability of block transition after blocklength trials

    # basic behavior
    ops['trainonly'] = True  # record output on all the sessions (False), or just train, but save at 100k updates(True)
    ops['tracktheta'] = False  # keep track of network parameters over training
    ops['trackstate'] = False  # keeps track of network state over training. LOTS OF DATA
    ops['plot'] = False  # will you plot during training?
    ops['device'] = 'cpu'  # what device will pytorch run on?

    # reward terms
    ops['Vvec'] = [5, 10, 20, 40, 80]  # rewarded volume.
    ops['Vvec_low'] = [5, 10, 20]  # for low blocks.
    ops['Vvec_high'] = [20, 40, 80]  # for high blocks.
    ops['Ract'] = -2.0  # the penalty for opting out.
    ops['w'] = -0.05  # the wait time penalty
    ops['win'] = 20  # number of trials for moving average estimation of reward rate
    ops['R_vio'] = ops['w'] * 20  # penalty for a violation trial. for now, just a normal 20step waiting penalty

    # trial and time stuff
    ops['seed'] = 101  # seed to training. used in setting trials for maain task, and in kindergen?
    ops['ns'] = 10000  # how many trials per experiment/session
    ops['T'] = 30  # longest time before "violation" happens
    ops['dt'] = 0.05  # time bins, in s
    ops['blocklen'] = 40  # size of block length
    ops['useblocks'] = True  # use block structure or not?
    ops['displevel'] = 4  # verbosity level. prints every 1000 trials. =5 prints every trial
    ops['inputcase'] = 11  # type of trial input
    ops['batchsize'] = 1  # how big is task batchsize. for not useful except of gpu saturation
    ops['snr'] = [0.0, 0.0, 0.0, 0.0]  # the signal to noise ratio on the inputs. variance of noise
    ops['snr_hidden'] = 0.0  # varaince of noise on hidden units
    ops['p_vio'] = 0.0  # rate of violation trials
    ops['T_vio'] = 0.1  # violation freq

    # loss, network type
    ops['rnntype'] = 'LSTM'  # what type of RNN unit. RNN, LSTM, GRU
    ops['dropout'] = 0.0  # what proportion of RNN-to-linear weights are dropped at each call
    ops['modeltype'] = 'RNNModel_multiregion_2allt'  # what kind of network to use
    ops['costtype'] = 'loss_actorcritic_regularized_9'  # what is the loss?
    ops['ismultiregion'] = True  # is the network a multiregion network?
    ops['rank'] = [None, None]  # if positive, is a low-rank connectivity matrix

    # cost-specific stuff
    ops['kindergarten_prediction'] = False  # the prediction using batches
    ops['nocatch_force'] = False  # force specific action in non-catch training.
    ops['alpha_rho'] = 0.3  # learning rate for reward rate with rho. for contin.case.
    ops['alpha_rho_av'] = 0.05  # learning rate for reward rate in average reward rate cost.
    ops['lambda_policy'] = 1.0  # weight on policy.
    ops['lambda_value'] = 0.00001  # weight on value loss.
    ops['lambda_entropy'] = 0.05  # weigth on entropy loss.
    ops['lambda_pred'] = 0.5  # weight on block prediction loss.
    ops['anneal_type'] = 'none'  # decides if cost weights should be annealed. only option, 'round_linear_kindergarten'

    # kindergarten stuff-----------
    ops['gamma_kindergarten'] = 0.0001  # learning rate for kindergarten
    ops['lambda_supervised'] = 10.0  # weighting of supervised loss in loss function.
    ops['seed_kindergarten'] = 0  # random seed for simulating. should be set each iteration
    ops['lossinds_kindergarten'] = [0, 1, 2]  # which indices of target to train on
    ops['batchsize_kindergarten'] = 1000  # batchsize of the kindergarten training
    ops['stages_kindergarten'] = ['simple', 'intermediate']  # which stages of kindergarten to do.
    ops['nepoch_kindergarten'] = 10000  # number of training epochs, if optim for set num epochs
    ops['stopargs_int_kindergarten'] = 0.001  # the minimum required value to hit, if optim till hit lower bound
    ops['base_epoch_kind'] = 1000  # how many epochs until a save

    # simple training
    ops['nsteps_simple_kindergarten'] = 20
    ops['nsteps_train_simple_kindergarten'] = 20  # sets size of minibatches in training. default train in one step
    # intermediate training
    ops['nsteps_list_kindergarten'] = [20, 25]  # step size of each trial in supervised loss
    ops['nsteps_list_int'] = np.random.randint(20, 100, 10)
    ops['highvartrials_kind'] = False  # use a high variability of trials in intermediate?
    ops['nsteps_train_int'] = np.sum(ops['nsteps_list_int'])
    ops['stoparg_simple'] = 30.0
    ops['stoptype_simple'] = 'converge'

    # prediction + kind training.
    ops['pred_stoparg'] = 30.0
    ops['pred_stoptype'] = 'converge'
    ops['beta_pred_kindpred'] = 0.5  # weight on prediciton loss DURING KINDERGARTEN only
    ops['beta_supervised_kindpred'] = 10.0  # the weight of kindergarten during training prediction
    ops['lossinds_supervised_kindpred'] = [0, 1, 2]  # which indices of kindergarten to simult. train with prediction
    ops['batchsize_pred'] = 20
    ops['ntrials_pred'] = 400  # how many trials per epoch
    ops['gamma_pred'] = 0.001  # learning rate
    ops['nepoch_pred'] = 10000  #
    ops['base_epoch_pred'] = 10  # how many epochs until a period model and data save.
    ops['pred_updatefun'] = 'update'

    # sham delay 2 match task
    ops['kindergarten_d2m'] = False
    ops['d2m_p'] = 0.5  # proportion of trials that are match trials
    ops['lambda_d2m'] = 0.0
    ops['beta_d2m_kindd2m'] = 1.0  #
    ops['base_epoch_d2m'] = 100  # how many epochs until a period model and data save.
    ops['gamma_d2m'] = 0.001  # learning rate
    ops['beta_pred_d2m'] = 0.0
    ops['beta_supervised_kindd2m'] = 0.0
    ops['batchsize_d2m'] = 1000
    ops['tmin_d2m'] = 0.1
    ops['tmax_d2m'] = 5
    ops['nepoch_d2m'] = 10000  #
    ops['d2m_updatefun'] = 'update_both'
    ops['lossinds_supervised_kindd2m'] = [1, 2]  # only include counting, integration task
    ops['d2m_stoptype'] = 'lowlim'
    ops['d2m_stoparg'] = 0.01

    # photostim options
    ops['pstim_prop'] = 0.4  # proporation of rewarded trials that are stimulated
    ops['pstim_amp'] = 1.0  # amplitude of stimulation

    return ops


def setinputs(v, k, w, r, a, block=0, nt=600, inputcase=1, device=None, snr=None, auxinps=None):
    """ chooses what inputs will be used in RNN
    v: volume
    k: time index (as integer index, ideally normed to go from 0 to 1)
    w: reward rate
    r: reward on last time step
    a: action on last time step. as integer (0) wait (1) optout
    block: what block are you in
    nt: max number of time points
    inputcase: what type of input will be used
    device: where will comp happen
    snr: (float) signal to noise ratio. sets variance of noise
    auxinps: (list? dict?) of extra inputs, relevant to auxiliary tasks"""

    if device is None:
        device = torch.device('cpu')

    if inputcase == 11:  # generic form for continous, non-episodic MDP.
        if isinstance(snr, list):  # independent signal to noise ratio for each input dim
            noise = np.array([k*np.random.normal(size=1)[0] for k in snr])
        else:
            noise = snr*np.random.normal(size=(4,))
        noise[0] = np.max([0, noise[0]])  # avoid negative numbers

        if k == 0:
            inputs_nn = torch.tensor([np.log(v), 1.0, r, a], device=device).requires_grad_(False)
            inputs = torch.tensor([np.log(v+noise[0]), 1.0, r+noise[2], a], device=device).requires_grad_(False)
        else:
            inputs_nn = torch.tensor([0.0, 0.0, r, a], device=device).requires_grad_(False)
            inputs = torch.tensor([0.0, 0.0, r+noise[2], a], device=device).requires_grad_(False)

    elif inputcase == 15:  # can run the delay to match task
        s1_sham = auxinps[0]  # stimulus 1 for the d2m task
        s2_sham = auxinps[1]  # stimulus 2 for the d2m task
        if isinstance(snr, list):  # independent signal to noise ratio for each input dim
            noise = np.array([k*np.random.normal(size=1)[0] for k in snr])
        else:
            noise = snr*np.random.normal(size=(6,))
        noise[0] = np.max([0, noise[0]])  # avoid negative numbers

        if k == 0:
            inputs_nn = torch.tensor([np.log(v), 1.0, r, a, s1_sham, s2_sham], device=device).requires_grad_(False)
            inputs = torch.tensor([np.log(v+noise[0]), 1.0, r+noise[2], a, s1_sham, s2_sham],
                                  device=device).requires_grad_(False)
        else:
            inputs_nn = torch.tensor([0.0, 0.0, r, a, s1_sham, s2_sham], device=device).requires_grad_(False)
            inputs = torch.tensor([0.0, 0.0, r+noise[2], a, s1_sham, s2_sham],
                                  device=device).requires_grad_(False)

    else:  # default to all constant
        inputs_nn = torch.tensor([np.log(v), k / nt, w], device=device).requires_grad_(False)
        noise = 0*inputs_nn
        inputs = inputs_nn + noise

    return inputs, inputs_nn


# configuration parser
def parse_configs(fname, uniquedict=None, ops=None):
    """
    take a configuraiton file and updates the ops struct, as well as extracts extra things for training
    :param fname: config file name
    :param uniquedict: (dict) of name keys and datatype values to extract from config. None=look for ones 4 fulltrain()
    :param ops: ops structure to overwrite with new configs. if None, use default
    :return:
    """
    cfg = configparser.ConfigParser()
    cfg.optionxform = lambda option: option  # override default behavior to make keys lc
    cfg.read(fname)

    # look for unique options that are not found in opsbase, and save those in a separate dictionary
    # where the value of the key is the datatype
    if uniquedict is None:
        uniquedict = {'netseed': int, 'device': str, 'savedir': str, 'modelname': str, 'nn': int, 'bias': float,
                      'nrounds1': int, 'nrounds2': int, 'nrounds3': int,
                      'nrounds1_s': int, 'nrounds2_s': int, 'nrounds3_s': int,
                      'usekindergarten': bool, 'roundstart_pred': int, 'adam_fname': str}

    uniqueops = {}
    for k in uniquedict:
        if k not in cfg['args']:
            print('uniqueops: '+k + ' not defined in config. using default')
            if k == 'adam_fname':
                uniqueops[k] = ''
            elif k == 'roundstart_pred':
                uniqueops[k] = 0
        else:
            if len(cfg['args'][k]) > 0 and cfg['args'][k][0] == '[':  # you are a list
                uniqueops[k] = ast.literal_eval(cfg['args'][k])
            elif uniquedict[k] is int:
                uniqueops[k] = cfg['args'].getint(k)
            elif uniquedict[k] is float:
                uniqueops[k] = cfg['args'].getfloat(k)
            elif uniquedict[k] is bool:
                uniqueops[k] = cfg['args'].getboolean(k)
            elif uniquedict[k] is list or uniquedict[k] is np.ndarray:
                uniqueops[k] = ast.literal_eval(cfg['args'][k])
            else:
                uniqueops[k] = cfg['args'][k]

    # grab the options themselves. overwrite with options from config file
    if ops is None:
        ops = opsbase()

    for key in cfg['args']:
        if key in ops:
            # first check if it is an array
            if len(cfg['args'][key]) > 0 and cfg['args'][key][0] == '[':  # you are a list! treat as a list
                ops[key] = ast.literal_eval(cfg['args'][key])
            elif type(ops[key]) is int or type(ops[key]) is np.int64:
                ops[key] = cfg['args'].getint(key)
            elif type(ops[key]) is float or type(ops[key]) is np.float64:
                ops[key] = cfg['args'].getfloat(key)
            elif type(ops[key]) is bool:
                ops[key] = cfg['args'].getboolean(key)
            elif type(ops[key]) is list or type(ops[key]) is np.ndarray:
                ops[key] = ast.literal_eval(cfg['args'][key])
            else:
                ops[key] = cfg['args'][key]

    # handle number of neurons for multiregion
    if ops['ismultiregion']:
        ops['nn'] = ast.literal_eval(cfg['args']['nn'])

    return ops, uniqueops


class CPU_Unpickler(pickle.Unpickler):
    """
    can unpickle objects that were originally on a GPU and loaded onto cpu
    """
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


def displ(text, printlevel=3, givenlevel=3):
    """
    prints message if givenlevel >= printlevel
    :param text: message to be printed
    :param printlevel: required level to print.
    :param givenlevel: supplied level.
    :return:
    """

    if givenlevel >= printlevel:
        print(text)

    return


def memcheck(device, clearmem=False):
    """
    checks cpu and gpu ram, empties cache if requested
    :param device:
    :param clearmem:
    :return:
    """

    # look for gpu ram
    if device.type == 'cuda':
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r - a  # free inside reserved
        gpuram = np.array([t, r, a, f])/1e9
    else:
        gpuram = np.array([0, 0, 0, 0])

    # cpu ram and usage
    cpuram = np.array(psutil.virtual_memory())[[0, 1, 3, 4]]/1e9
    psutil.virtual_memory()

    if clearmem:
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return cpuram, gpuram


def mwa(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

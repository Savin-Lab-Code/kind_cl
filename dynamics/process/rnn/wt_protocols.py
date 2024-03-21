# training protocols for the wait-time RNN
import numpy as np
import torch
import pickle
from dynamics.process.rnn import wt_kindergarten, wt_costs, wt_reinforce_cont_new, wt_pred, parse, wt_nets
from dynamics.utils import utils
from dynamics.analysis import wt_analysis as wta
from dynamics.process.rnn import wt_delay2match as wt_d2m
from dynamics.utils.utils import opsbase
import gc
import json


def savecurric(fname, net, updater, dat, ops, domodel=True, dostats=True, dodat=True,
               deldat=False, doMAT=False):
    """
    easy save code for saving models and progress for curriculum training. just nicer
    @param fname: file name to save
    @param net: network
    @param updater: optimizer
    @param dat: data to save
    @param ops: options
    @param domodel: (bool) save model?
    @param dostats: (bool) save stats
    @param dodat: (bool) save dat
    @param deldat:
    @param doMAT:
    @return:
    """
    output = None

    trainonly = ops['trainonly']

    if domodel:
        torch.save(net.state_dict(), fname + '.model')  # save model
        torch.save(updater.state_dict(), fname + '.adam')  # save model

    if dostats and not trainonly:
        wt_dict, linreg, numdict, rho_all, ops = wta.parsesimulation(dat, ops)
        if doMAT:
            wtlist, reg_hist, reg_block = wta.modelagnostic_test(dat, dt=ops['dt'])
            pickle.dump([wt_dict, linreg, numdict, wtlist, reg_hist, reg_block], open(fname + '.stats', 'wb'))
        else:
            pickle.dump([wt_dict, linreg, numdict], open(fname + '.stats', 'wb'))
        output = [linreg, wt_dict]

    if dodat and not trainonly:
        print('saving:')
        dat_json = parse.dat2json(dat)
        with open(fname+'.json', 'w') as f:
            json.dump(dat_json, f)

        print('done saving')

    if deldat:
        del dat

    # garbage collect and clear cuda memory at each save
    gc.collect()
    if ops['device'] == 'cuda':
        print('clearing cuda cache at end of 10k session')
        torch.cuda.empty_cache()

    return output


def training_kindergarten(net, ops=None, savename='rnn_kindergarten', device=torch.device('cpu'),
                          savelog=False, optim_fname=''):
    """
    runs through the kindergarten training protocol to precondition the RNN
    :param net: (RNNModelSupervised) network from wt_reinforce
    :param savename: (str) base name of save file
    :param device: where code is run
    :param ops: options for handling the run.
    :param savelog: if savelog=true, save the logs. can be large files though. default to no
    :param optim_fname: (str) if anything but '', will load the adam state. if so, will set fname to '' to not overwrite
    :return:
    """

    # set up the basic options
    if ops is None:
        ops = opsbase()

    seed = ops['seed_kindergarten']
    np.random.seed(seed)
    batchsize = ops['batchsize_kindergarten']
    nsteps_simple = ops['nsteps_simple_kindergarten']
    nsteps_train_simple = ops['nsteps_train_simple_kindergarten']
    nsteps_int_list = ops['nsteps_list_int']
    nsteps_train_int = ops['nsteps_train_int']
    gamma = ops['gamma_kindergarten']
    nepoch = ops['nepoch_kindergarten']
    stopargs_int = ops['stopargs_int_kindergarten']
    stages = ops['stages_kindergarten']
    stoparg_simple = ops['stoparg_simple']
    stoptype_simple = ops['stoptype_simple']

    # size of inputs
    if ops['ismultiregion']:
        din = np.max(net.din)  # for multiregion
    else:
        din = net.din  # input size

    if device.type == 'cuda':
        net.cuda()  # send to gpu

    # train up to do all simple tasks in sequence-------------------------
    print('training on simple task')

    # create training data
    lossind_global = ops['lossinds_kindergarten']  # should be in order of preferred training. ideally [0,1,2]
    lossind_global.reverse()  # sets to [2,1,0]
    lossind_list = [lossind_global[k:] for k in range(len(lossind_global))]  # sets multiple stages of training
    lossind_list.reverse()  # puts them in order
    lossind_global.reverse()  # return
    print('task inds for training:')
    print(lossind_list)

    ltot_train_list = []
    gradnorm_list = []

    if 'simple' in stages:
        ltot_train_list_simple = []
        gradnorm_list_simple = []

        updater = torch.optim.Adam(net.parameters(), lr=gamma)
        outptdict = None  # initialize
        if optim_fname != '':
            updater.load_state_dict(torch.load(optim_fname, map_location=device.type))
            print('loading previous adam state')
            optim_fname = ''

        for j in lossind_list:
            print('training on output dims:' + str(j))
            lossinds = j

            # create simple training data
            inputs, targets, _, _ = wt_kindergarten.trainingdata_simple(
                seed=seed, nsteps=nsteps_simple, batchsize=batchsize, din=din)

            si = net.begin_state(batchsize=batchsize, device=device)
            # [k.detach() for k in si]  # LSTMs have tuples
            # track the state. do it as list comprehension for LSTMS
            if isinstance(si[0], tuple):  # multiregion LSTM
                [[m.detach() for m in sk] for sk in si]
            elif isinstance(si, tuple) or isinstance(si, list):  # LSTM or mr RNN
                [sk.detach() for sk in si]
            else:  # vanilla RNN and GRU
                si.detach()

            # train, round 1
            ltot_train, grad_norm, outptdict = wt_kindergarten.train(net, updater, si, inputs, targets, device=device,
                                                                     nepoch=nepoch, nsteps=nsteps_train_simple,
                                                                     lossinds=lossinds, stoptype=stoptype_simple,
                                                                     stopargs=stoparg_simple)

            ltot_train_list_simple.append(ltot_train)
            gradnorm_list_simple.append(grad_norm)

        ltot_train_list.append(ltot_train_list_simple)
        gradnorm_list.append(gradnorm_list_simple)

        print('saving')
        torch.save(net.state_dict(), savename + '_simple.model')
        torch.save(updater.state_dict(), savename + '_simple.adam')

        if savelog:
            savedict = {'ltot_train_list': ltot_train_list, 'gradnorm_list': gradnorm_list, 'outptdict': outptdict}
            savedat_json = parse.dict2json(savedict)
            with open(savename+'_simple.json', 'w') as f:
                json.dump(savedat_json, f)

    # train on intermediate task-------------------------------------
    if 'intermediate' in stages:
        print('training on intermediate task: concatenation of 2 trials')

        updater = torch.optim.Adam(net.parameters(), lr=gamma)
        if optim_fname != '':
            updater.load_state_dict(torch.load(optim_fname, map_location=device.type))
            print('loading previous adam state')
            optim_fname = ''

        lossind_list = [lossind_global]  # all of the ones at once
        print('task inds for training:')
        print(lossind_list)
        print('trial sizes for intermediate training:')
        print(nsteps_int_list)

        # track across the entire set of training
        ltot_train_list_int = []
        gradnorm_list_int = []

        ind_idx = 0
        for j in lossind_list:
            print('training on output dims:' + str(j))
            lossinds = j

            # memory, and integration targets and inputs
            inputs_int, targets_int, _, _ = wt_kindergarten.trainingdata_intermediate(
                seed=seed + 10, nsteps_list=nsteps_int_list, batchsize=batchsize, din=din,
                highvartrials=ops['highvartrials_kind'])

            si = net.begin_state(batchsize=batchsize, device=device)
            if isinstance(si[0], tuple):  # multiregion LSTM
                [[m.detach() for m in sk] for sk in si]
            elif isinstance(si, tuple) or isinstance(si, list):  # LSTM or mr RNN
                [sk.detach() for sk in si]
            else:  # vanilla RNN and GRU
                si.detach()

            base_epoch = ops['base_epoch_kind']
            nrounds = int(np.ceil(nepoch / base_epoch))  # number of potential rounds

            for k in range(nrounds):

                # train, round 1
                ltot_train, grad_norm, outpts = wt_kindergarten.train(net, updater, si, inputs_int, targets_int,
                                                                      device=device, nepoch=base_epoch,
                                                                      nsteps=nsteps_train_int,
                                                                      lossinds=lossinds, stoptype='lowlim',
                                                                      stopargs=stopargs_int)

                ltot_train_list_int.append(ltot_train)
                gradnorm_list_int.append(grad_norm)
                # add nsteps_list and batchsize to outputs
                outpts['batchsize'] = batchsize
                outpts['nsteps'] = nsteps_int_list

                print('saving')
                torch.save(net.state_dict(), savename + '_int_'+str(ind_idx)+'_'+str(k)+'.model')
                torch.save(net.state_dict(), savename + '_int_' + str(ind_idx) + '_final.model')
                torch.save(updater.state_dict(), savename + '_int_' + str(ind_idx) + '_' + str(k) + '.adam')
                torch.save(updater.state_dict(), savename + '_int_' + str(ind_idx) + '_' + str(k) + '_final.adam')
                if savelog:
                    savedict = {'ltot_train_list': ltot_train, 'gradnorm_list': grad_norm,
                                'outptdict': outpts}
                    savedat_json = parse.dict2json(savedict)
                    with open(savename + '_int_'+str(ind_idx)+'_'+str(k)+'.json', 'w') as f:
                        json.dump(savedat_json, f)
                if outpts['isdone']:
                    break
            ind_idx += 1
        # add intermediate trainign to all training
        ltot_train_list.append(ltot_train_list_int)
        gradnorm_list.append(gradnorm_list_int)

    return net, ltot_train_list, gradnorm_list, optim_fname


def training_prediction(net, ops, savename='rnn_pred_', device=torch.device('cpu'), savelog=False,
                        roundstart=0, optim_fname=''):
    """
    runs through one-step prediction with batched, supervised learning
    :param net: (RNNModelSupervised) network from wt_reinforce
    :param ops: (dict) the options dictionary controlling simulation
    :param savename: (str) base name of save file
    :param device: where code is run
    :param ops: options for handling the run.
    :param savelog: if savelog=true, save the logs. can be large files though. default to no
    :param roundstart: (int) where to start training. used for restarts
    :param optim_fname: (str) optimizer fname. if '', don't load. if not, load, then set to '' to stop overwriting
    :return:
    """

    seed = ops['seed_kindergarten']
    np.random.seed(seed)
    batchsize = ops['batchsize_pred']
    ntrials_pred = ops['ntrials_pred']  # how many trials
    gamma = ops['gamma_pred']
    nepoch = ops['nepoch_pred']

    # decide on which update function to use
    updatefun_name = ops['pred_updatefun']
    if updatefun_name == 'update':  # loss at all time points
        updatefun = wt_pred.update
    elif updatefun_name == 'update_trialstart':  # loss only at trial start
        updatefun = wt_pred.update_trialstart
    else:
        updatefun = wt_pred.update

    if device.type == 'cuda':
        net.cuda()  # send to gpu

    # create training data
    ltot_train_list = []
    gradnorm_list = []

    # train, round 1: no opt-outs, no catch, no block
    print('prediction training')
    p_optout = 0.3  # optout rate
    pcatch = ops['pcatch']  # original pcatch
    ops['pcatch'] = 0.0  # catch trial rate
    inputs, targets, _, _, _, _ = wt_pred.trainingdata(ops, seed=seed, ntrials=ntrials_pred, batchsize=batchsize,
                                                       useblocks=True, p_optout=p_optout)
    print('done with datagen')
    nsteps = inputs.shape[1]  # total number of training steps

    # train
    updater = torch.optim.Adam(net.parameters(), lr=gamma)
    if optim_fname != '':
        updater.load_state_dict(torch.load(optim_fname, map_location=device.type))
        print('loading previous adam state')
        optim_fname = ''

    # make sure stop argument is treated correctly
    if ops['pred_stoptype'] == 'converge':
        ops['pred_stoparg'] = int(ops['pred_stoparg'])
    print('beginning train')

    si = net.begin_state(batchsize=batchsize, device=device)
    # determine number of subepochs. save ever 10 epochs
    base_epoch = ops['base_epoch_pred']
    nrounds = int(np.ceil(nepoch/base_epoch))  # number of potential rounds
    for j in range(roundstart, nrounds):

        ltot_train, grad_norm, outpts = wt_pred.train(ops, net, updater, si, inputs, targets, device=device,
                                                      nepoch=base_epoch, nsteps=nsteps, stoptype=ops['pred_stoptype'],
                                                      stopargs=ops['pred_stoparg'], updatefun=updatefun)
        # appead lossssea and gradients
        ltot_train_list.append(ltot_train)
        gradnorm_list.append(grad_norm)
        print('saving')
        torch.save(net.state_dict(), savename + '_'+str(j+1)+'.model')
        torch.save(net.state_dict(), savename + '_final.model')
        torch.save(updater.state_dict(), savename + '_' + str(j + 1) + '.adam')
        torch.save(updater.state_dict(), savename + '_final.adam')
        if savelog:
            savedict = {'ltot_train_list': ltot_train_list, 'gradnorm_list': gradnorm_list,
                        'outptdict': outpts}
            savedat_json = parse.dict2json(savedict)
            with open(savename + '_'+str(j+1)+'.json', 'w') as f:
                json.dump(savedat_json, f)

        if outpts['isdone']:
            break

    ops['pcatch'] = pcatch  # return value

    return net, ltot_train_list, gradnorm_list, optim_fname


def training_sham(net, ops, savename='rnn_d2m_', device=torch.device('cpu'), savelog=False,
                  roundstart=0, optim_fname=''):
    """
    runs through sham tasks (i.e., delay to match) with other kindergarten tasks included
    :param net: (RNNModelSupervised) network from wt_reinforce
    :param ops: (dict) the options dictionary controlling simulation
    :param savename: (str) base name of save file
    :param device: where code is run
    :param ops: options for handling the run.
    :param savelog: if savelog=true, save the logs. can be large files though. default to no
    :param roundstart: (int) where to start training. used for restarts
    :param optim_fname: (str) if '', do nothing. else, load optim state, then set to '' to prevent overwriting
    :return:
    """

    seed = ops['seed_kindergarten']
    np.random.seed(seed)
    batchsize = ops['batchsize_d2m']
    nt = int(ops['tmax_d2m']/ops['dt'])  # how many timesteps per samples
    gamma = ops['gamma_d2m']
    nepoch = ops['nepoch_d2m']

    if device.type == 'cuda':
        net.cuda()  # send to gpu

    # create training data
    ltot_train_list = []
    gradnorm_list = []

    # train, round 1: no opt-outs, no catch, no block
    print('sham training')
    inputs, targets, _, _, _ = wt_d2m.trainingdata(ops, seed=seed, T=nt, batchsize=batchsize)
    print('done with datagen')
    nsteps = inputs.shape[1]  # total number of training steps

    # train
    print('end-point training only')
    updatefun = wt_d2m.update_end
    updater = torch.optim.Adam(net.parameters(), lr=gamma)
    if optim_fname != '':
        print('loading previous adam state')
        updater.load_state_dict(torch.load(optim_fname, map_location=device.type))
        optim_fname = ''
    # make sure stop argument is treated correctly
    if ops['d2m_stoptype'] == 'converge':
        ops['d2m_stoparg'] = int(ops['d2m_stoparg'])
    print('beginning train')

    si = net.begin_state(batchsize=batchsize, device=device)
    # determine number of subepochs. save ever 10 epochs
    base_epoch = ops['base_epoch_d2m']
    nrounds = int(np.ceil(nepoch/base_epoch))  # number of potential rounds
    for j in range(roundstart, nrounds):

        ltot_train, grad_norm, outpts = wt_d2m.train(ops, net, updater, si, inputs, targets, device=device,
                                                     nepoch=base_epoch, nsteps=nsteps, stoptype=ops['d2m_stoptype'],
                                                     stopargs=ops['d2m_stoparg'], updatefun=updatefun)

        # appead lossssea and gradients
        ltot_train_list.append(ltot_train)
        gradnorm_list.append(grad_norm)
        print('saving')
        torch.save(net.state_dict(), savename + 'endpt_'+str(j+1)+'.model')
        torch.save(net.state_dict(), savename + 'endpt_final.model')
        torch.save(updater.state_dict(), savename + 'endpt_' + str(j + 1) + '.adam')
        torch.save(updater.state_dict(), savename + 'endpt_final.adam')
        if savelog:
            savedict = {'ltot_train_list': ltot_train_list, 'gradnorm_list': gradnorm_list,
                        'outptdict': outpts}
            savedat_json = parse.dict2json(savedict)
            with open(savename + '_endpt_'+str(j+1)+'.json', 'w') as f:
                json.dump(savedat_json, f)

        if outpts['isdone']:
            break

    # train
    print('all-time and endpoint training')
    updatefun = wt_d2m.update_both
    updater = torch.optim.Adam(net.parameters(), lr=gamma)
    if optim_fname != '':
        print('loading previous adam state')
        updater.load_state_dict(torch.load(optim_fname, map_location=device.type))
        optim_fname = ''
    # make sure stop argument is treated correctly
    if ops['d2m_stoptype'] == 'converge':
        ops['d2m_stoparg'] = int(ops['d2m_stoparg'])
    print('beginning train')

    si = net.begin_state(batchsize=batchsize, device=device)
    # determine number of subepochs. save ever 10 epochs
    base_epoch = ops['base_epoch_d2m']
    nrounds = int(np.ceil(nepoch / base_epoch))  # number of potential rounds
    for j in range(roundstart, nrounds):

        ltot_train, grad_norm, outpts = wt_d2m.train(ops, net, updater, si, inputs, targets, device=device,
                                                     nepoch=base_epoch, nsteps=nsteps,
                                                     stoptype=ops['d2m_stoptype'],
                                                     stopargs=ops['d2m_stoparg'], updatefun=updatefun)

        # appead lossssea and gradients
        ltot_train_list.append(ltot_train)
        gradnorm_list.append(grad_norm)
        print('saving')
        torch.save(net.state_dict(), savename + 'both_' + str(j + 1) + '.model')
        torch.save(net.state_dict(), savename + 'both_final.model')
        torch.save(updater.state_dict(), savename + 'both_' + str(j + 1) + '.adam')
        torch.save(updater.state_dict(), savename + 'both_final.adam')
        if savelog:
            savedict = {'ltot_train_list': ltot_train_list, 'gradnorm_list': gradnorm_list,
                        'outptdict': outpts}
            savedat_json = parse.dict2json(savedict)
            with open(savename + '_both_' + str(j + 1) + '.json', 'w') as f:
                json.dump(savedat_json, f)

        if outpts['isdone']:
            break

    return net, ltot_train_list, gradnorm_list, optim_fname


def curriculumtraining(net, ops, savename='curric', device=torch.device('cpu'),
                       nrounds=None, nrounds_start=None, savetmp=False, optim_fname=''):
    """
    do curriculum training to do no cath trials, catch trials, no block
    can do either episodic, or continuous-form training
    :return:
    """

    # choose cost function(s)
    costfun = wt_costs.name2fun[ops['costtype']]
    optimizer = torch.optim.Adam(net.parameters(), lr=ops['alpha'])
    if optim_fname != '':
        print('loading previous adam state')
        optimizer.load_state_dict(torch.load(optim_fname, map_location=device.type))
        optim_fname = ''

    if nrounds is None:
        nrounds = [50, 50, 50]  # how many rounds of each curriculum to perform
    if nrounds_start is None:
        nrounds_start = [0, 0, 0]  # index to start iterations of each curriculum stage
    print(nrounds)
    print(nrounds_start)

    freeze_global = ops['freeze']  # freeze weights?

    # round 1------------------------------
    print('round 1: no catch trials')
    pcatch = ops['pcatch']
    print('pcatch for later stages:')
    print(pcatch)
    ops['pcatch'] = 0.00
    ops['useblocks'] = False

    for k in range(nrounds_start[0], nrounds[0]):  # train for 1 million, save every 100 k
        print(k)

        # if ((k + 1) % 10 == 0 or k == 0) and savetmp:
        #  try saving all data and see how it goes
        ops['trainonly'] = False

        returndict_nocatch = wt_reinforce_cont_new.session_torch_cont(ops, net, optimizer, costfun, device)
        fname = savename + '_nocatch_' + str(int((k + 1)))
        savecurric(fname, net, optimizer, returndict_nocatch, ops,
                   domodel=True, dostats=True, dodat=True, deldat=True)

        del returndict_nocatch
        gc.collect()
        if ops['device'] == 'cuda':
            print('clearing cuda cache at end of 10k nocatch session')
            torch.cuda.empty_cache()

    # round 2: catch trials
    print('round 2: catch trials')
    ops['nocatch_force'] = None  # turns orcing a wait decision off
    ops['pcatch'] = pcatch
    ops['useblocks'] = False
    # initialize

    for k in range(nrounds_start[1], nrounds[1]):
        print(k)

        # if ((k + 1) % 10 == 0 or k == 0) and savetmp:
        ops['trainonly'] = False

        returndict_catch = wt_reinforce_cont_new.session_torch_cont(ops, net, optimizer, costfun, device)

        # save, including statistics
        fname = savename + '_catch_' + str(int((k + 1)))
        output = savecurric(fname, net, optimizer, returndict_catch, ops,
                            domodel=True, dostats=True, dodat=True, deldat=True, doMAT=False)

        # early stopping?
        if ops['prog_stop']:
            linreg = output[0]
            if linreg['p'] < 0.01 and linreg['m'] > 0:
                print('p value for wait time slope significant, and slope is positive. stopping')
                break

        # delete again
        del output
        del returndict_catch
        gc.collect()
        if ops['device'] == 'cuda':
            print('clearing cuda cache at end of 10k catch session')
            torch.cuda.empty_cache()

    # round 3: block structure
    print('round 3: block struture')
    ops['nocatch_force'] = None
    ops['pcatch'] = pcatch
    ops['useblocks'] = True

    # annealing schedule. For now, assume piecewise constant over ns trials, linear decay
    lambda_actor = ops['lambda_policy']
    lambda_critic = ops['lambda_value']
    lambda_ent = ops['lambda_entropy']
    lambda_pred = ops['lambda_pred']
    lambda_kind = ops['lambda_supervised']

    nr_sum = nrounds[2] - nrounds_start[2]
    avec_default = np.ones((nr_sum,))
    avec_actor = lambda_actor * avec_default
    avec_critic = lambda_critic * avec_default
    avec_pred = lambda_pred * avec_default
    avec_ent = lambda_ent * avec_default
    avec_kind = lambda_kind * avec_default

    if ops['anneal_type'] == 'round_linear_kindergarten':  # linear decay, at each round of ns samps, linear decay
        avec_kind = np.linspace(lambda_kind, 0, nr_sum)
    elif ops['anneal_type'] == 'round_linear_kindandpred':
        avec_kind = np.linspace(lambda_kind, 0, nr_sum)
        avec_pred = np.linspace(lambda_pred, 0, nr_sum)

    anneal_ind = 0  # keeps track of anneal index. used mostly for restarts
    for k in range(nrounds_start[2], nrounds[2]):
        print(k)

        ops['trainonly'] = False
        # set annealed
        ops['lambda_policy'] = avec_actor[anneal_ind]
        ops['lambda_value'] = avec_critic[anneal_ind]
        ops['lambda_pred'] = avec_pred[anneal_ind]
        ops['lambda_entropy'] = avec_ent[anneal_ind]
        ops['lambda_supervised'] = avec_kind[anneal_ind]

        returndict_block = wt_reinforce_cont_new.session_torch_cont(ops, net, optimizer, costfun, device)
        ops['freeze'] = True

        # save
        fname = savename + '_block_' + str(int((k + 1)))
        _ = savecurric(fname, net, optimizer, returndict_block, ops,
                       domodel=True, dostats=True, dodat=True, deldat=True, doMAT=True)

        # delete again
        del returndict_block
        gc.collect()
        if ops['device'] == 'cuda':
            print('clearing cuda cache at end of 10k block session')
            torch.cuda.empty_cache()

        if ((k + 1) % 10 == 0 or k == 0) and savetmp:
            print('freeze round')
            returndict_block_fr = wt_reinforce_cont_new.session_torch_cont(ops, net, optimizer, costfun, device)
            ops['freeze'] = freeze_global
            fname = savename + '_block_' + str(int((k + 1)))+'_freeze'
            _ = savecurric(fname, net, optimizer, returndict_block_fr, ops,
                           domodel=True, dostats=True, dodat=True, deldat=True, doMAT=True)

            # delete again
            del returndict_block_fr
            gc.collect()
            if ops['device'] == 'cuda':
                print('clearing cuda cache at end of 10k frozen block session')
                torch.cuda.empty_cache()

        anneal_ind += 1

    gc.collect()
    return net, optim_fname


def sim_task(modelname, configname=None, simtype='task', task_seed=101):
    """
    will simulate for a given task
    for heavier dynamics processing
    :param modelname: (str) full path to model to use for simulation
    :param configname: (str) path to .cfg file. make sure trainonly=False, trackstate=True for sim task.
    :param simtype: (str) 'task', 'inf','kind' for different types of simulations
    :param task_seed: (int) seed to numpy generator for generating trials
    :return:
    """

    # avoid having exact same trial list for every RNN. let task_seed be their RNN number for unique list
    np.random.seed(task_seed)

    # set basic args args
    if configname is None:
        ops = utils.opsbase()
        ops['device'] = 'cpu'
        ops['displevel'] = 5
        ops['freeze'] = True
        ops['trainonly'] = False
        ops['trackstate'] = True
        ops['useblocks'] = True
        ops['nocatch_force'] = False
        ops['pcatch'] = 0.2
        ops['ns'] = 1000
        numhidden = [256, 256]
    else:
        ops, uniqueops = utils.parse_configs(configname)
        numhidden = uniqueops['nn']

    costfun = wt_costs.name2fun[ops['costtype']]
    device = torch.device('cpu')

    # load model
    nd = wt_nets.netdict[ops['modeltype']]  # dictionary of models and size params
    netfun = nd['net']
    netseed = 101
    net = netfun(din=nd['din'], dout=nd['dout'], num_hiddens=numhidden, seed=netseed, rnntype=ops['rnntype'],
                 snr=ops['snr_hidden'], dropout=ops['dropout'], rank=ops['rank'], inputcase=nd['inputcase'])
    net.load_state_dict(torch.load(modelname, map_location=device.type))
    optimizer = torch.optim.Adam(net.parameters(), lr=ops['alpha'])

    if simtype == 'task':

        rdict = wt_reinforce_cont_new.session_torch_cont(ops, net, optimizer, costfun, device)

    elif simtype == 'kind':
        batchsize = 100
        nsteps_list = [30]
        nsteps = sum(nsteps_list)
        inp, targ, V, rewards = wt_kindergarten.trainingdata_intermediate(seed=101,
                                                                          nsteps_list=nsteps_list,
                                                                          batchsize=batchsize)
        inputs = torch.Tensor(inp).to(device)
        si = net.begin_state(batchsize=batchsize, device=device)

        outpt_supervised, preds, sflat, tdict = parse.batchsamples_kind_allstate(net, inputs, si)

        # reshape inp to handle batch size
        inp = np.reshape(inp, (batchsize * nsteps, -1))
        blocks = np.zeros((batchsize * nsteps,))
        offers = inp[inp[:, 0] > 0, 0]

        targnew = []
        for j in range(batchsize):
            targnew.extend(np.squeeze(targ[j, :, :]))
        targ = np.array(targnew)

        # create a behdict out of these things!
        behdict = {'tdict': tdict, 'preds': preds, 'blocks': blocks, 'offers': offers, 'targs': targ,
                   'outcomes': 0*np.ones(offers.shape).astype(int)}

        rdict = {'behdict': behdict, 'outpt_supervised': outpt_supervised, 'sflat': sflat, 'inp': inp}

    elif simtype == 'inf':
        batchsize = 1
        ntrials = 200
        inp, targ, blocks, oo_idx, oo_fast_idx, catch_idx = wt_pred.trainingdata(
            ops, seed=101, ntrials=ntrials, batchsize=batchsize, useblocks=True, p_optout=0.0)
        blocks = np.array(np.squeeze(blocks))

        inputs = torch.Tensor(inp).to(device)
        si = net.begin_state(batchsize=batchsize, device=device)

        outpt, outpt2, sflat, tdict = parse.batchsamples_kind_allstate(net, inputs, si)
        inp = np.squeeze(inp)
        offers = np.array(inp[inp[:, 0] > 0, 0])
        outpt = np.array([np.squeeze(k[0, 0, :]) for k in outpt])
        preds = np.array([np.squeeze(k[0, 0, :]) for k in outpt2])

        # create a behdict out of these things!
        behdict = {'tdict': tdict, 'preds': preds, 'blocks': blocks, 'offers': offers, 'targs': targ,
                   'outcomes': 0*np.ones(offers.shape).astype(int)}

        rdict = {'behdict': behdict, 'outpt_supervised': outpt, 'sflat': sflat, 'inp': inp}

    else:
        print('going to sim other types here. not supported yet')
        rdict = None

    return rdict

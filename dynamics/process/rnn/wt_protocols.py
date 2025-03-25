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
from datetime import datetime
# from dynamics.process.rnn import parse
# import tracemalloc
# import sys
# from dynamics.utils import profiler
# from dynamics.utils.utils import memcheck, compress_save
# import warnings
from datetime import datetime



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
        # attributes of models will NOT be saved with torch.save. important ones must be saved separately
        if hasattr(net,'projection_neurons'):
            proj_neurons = np.sort(net.projection_neurons).tolist()
            with open(fname + '_model_attributes.json', 'w') as f:
                json.dump({'projection_neurons': proj_neurons}, f)

    if dostats and not trainonly:
        wt_dict, linreg, numdict, rho_all, ops = wta.parsesimulation(dat, ops)
        if doMAT:
            # following the true pipeline can leave zero data during training because of only keeping 1.5 std from mean
            # as well as looking for good data in all blocks. so omit this parse from the quick-and-dirty test
            # datlist, tokeep = parse.format2sess(dat)
            # dat_filt = parse.format2singlesess(datlist, tokeep) # for now dont use the preprocessing step. buggy
            wtlist, reg_hist, reg_block = wta.modelagnostic_test(dat, dt=ops['dt'])
            pickle.dump([wt_dict, linreg, numdict, wtlist, reg_hist, reg_block], open(fname + '.stats', 'wb'))
            # make them into json objects to avoid any further numpy insanity
            parse.stats2json(fname + '.stats', fname + '_stats.json')
            # del datlist, dat_filt
        else:
            pickle.dump([wt_dict, linreg, numdict], open(fname + '.stats', 'wb'))
            parse.stats2json(fname + '.stats', fname + '_stats.json')
        output = [linreg, wt_dict]

    if dodat and not trainonly:
        print('saving:')
        dat_json = parse.dat2json(dat)
        with open(fname+'.json', 'w') as f:
            json.dump(dat_json, f)

        # pickle.dump(dat, open(fname + '.dat', 'wb'))  # save data changed 19 october. hoping to retro save as .gz
        # compress_save(dat, fname + '.dat'+'.gz')
        print('done saving')

    if deldat:
        # memory checks
        # device = torch.device(ops['device'])
        # print('collect 1 start:' + str(datetime.now()))
        # print(memcheck(device, clearmem=True))
        del dat

        # print('collect 1 stopt:' + str(datetime.now()))
        # print(memcheck(device, clearmem=False))

    # garbage collect and clear cuda memory at each save
    gc.collect()
    if ops['device'] == 'cuda':
        print('clearing cuda cache at end of 10k session')
        torch.cuda.empty_cache()

    return output


def training_curric_general(net, stagelist, ops=None, savedir=None, device=torch.device('cpu'), optim_fname=''):
    """
    a general curriculum training routine to do curriculum of tasks in sequence.
    @param net: (torch.NN) neural network to optimize
    @param stagelist: (list) (or nested lists) of CL_Stage objects
    @param ops: (dict) options for the optimization
    @param savedir: (str) location for saving, if needed
    @param device: (torch.device) object for what device trains network. 'cpu' or 'cuda', usually
    @param optim_fname: (str) name of previous optimizer state, if reloading from an earlier save
    @return:
    """

    # set up the basic options
    if ops is None:
        ops = opsbase()

    # check that savedir has been set, if needed
    assert (sum([k.savefreq > 0 for k in stagelist]) > 0) and (savedir is not None), ("savedir must "
                                                                                      "be set if CL  savefreq > 1")

    seed = ops['seed_kindergarten']
    np.random.seed(seed)

    nstages = len(stagelist)

    # output list to report at end of entire training
    losslist = []  # will be a nested list of len nstages

    # begin iterating over each stage
    for k in range(nstages):
        clstage = stagelist[k]
        print('beginning stage ' + str(k))

        # check if savefreq has been set correctly. avoid division errors in mod calc
        if clstage.savefreq == 0 or clstage.savefreq is None:
            clstage.savefreq = clstage.maxepoch + 1

        assert type(clstage.tasklist) is list, "tasks in CL_stage objects must be lists. even one task per stage."
        ntasks_k = len(clstage.tasklist)
        loss_k = torch.tensor(np.inf)  # initialize at high val
        loss_best = loss_k
        converge_ind = 0  # number of epochs since improvement

        # loss and gradients over trainign for individual stage
        loss_stage = []
        grad_stage = []

        # set the optimizer. load previous adam state on first task if provided
        updater = torch.optim.Adam(net.parameters(), lr=stagelist[k].learningrate)
        if optim_fname != '' and k == 0:
            print('loading previous adam state')
            updater.load_state_dict(torch.load(optim_fname, map_location=device.type))

        saveidx = 0  # index for saving, since each stage could have different savefreq
        for epoch in range(clstage.maxepoch):

            loss_epoch = []
            outputs_epoch = []
            for j in range(ntasks_k):
                task = clstage.tasklist[j]

                # initialize state. detach so not part of grad calc
                si = net.begin_state(batchsize=task.batchsize, device=device)
                if isinstance(si[0], tuple):  # multiregion LSTM
                    [[m.detach() for m in sk] for sk in si]
                elif isinstance(si, tuple) or isinstance(si, list):  # LSTM or mr RNN
                    [sk.detach() for sk in si]
                else:  # vanilla RNN and GRU
                    si.detach()

                # calc loss
                loss_j, outputs_j = task.loss(net, ops, si, batchsize=task.batchsize, seed=epoch*(k+1))
                loss_epoch.append(loss_j*task.weight)
                outputs_epoch.append(outputs_j)

            # update
            updater.zero_grad()
            loss = torch.stack(loss_epoch[:]).sum()
            loss.backward()
            updater.step()

            # calc gradient
            abs_sum = 0
            for p in net.parameters():
                # some losses don't use entire network, and will have None gradients
                if p.grad is not None:
                    abs_sum += abs(np.sum(p.grad.cpu().numpy()))

            # abs_sum = np.sum([abs(np.sum(k.grad.cpu().numpy())) for k in net.parameters()])
            grad_stage.append(abs_sum)

            loss_detached = [k.detach().cpu().numpy() for k in loss_epoch]
            loss_detached.insert(0, loss.detach().cpu().numpy())  # first index is total loss
            loss_stage.append(loss_detached)

            # print losses
            if (epoch + 1) % 10 == 0:
                print(loss_detached, end='\r')

            # decide to save?
            if (epoch + 1) % clstage.savefreq == 0:
                print('saving network and adam state')

                savename = savedir + clstage.name + '_'+str(saveidx)
                torch.save(net.state_dict(), savename + '.model')
                torch.save(updater.state_dict(), savename + '.adam')

                if clstage.savelog:
                    print('saving training data, gradient, outputs')
                    savedict = {'loss': loss_detached, 'grad': abs_sum,
                                'outputlist': outputs_epoch, 'clstage': clstage.__dict__}
                    savedat_json = parse.dict2json(savedict)
                    with open(savename + '.json', 'w') as f:
                        json.dump(savedat_json, f)

                saveidx += 1

            # do a convergence stopping test
            if clstage.convtype == 'converge':
                if loss <= loss_best:
                    loss_best = loss
                    converge_ind = 0
                else:
                    converge_ind += 1

                if converge_ind > clstage.criteria:
                    print('stopping stage early. no improvement compared to average of last ' +
                          str(clstage.criteria) + 'epochs')
                    savename = savedir + clstage.name + '_final'
                    torch.save(net.state_dict(), savename + '.model')
                    torch.save(updater.state_dict(), savename + '.adam')
                    break

            elif clstage.convtype == 'lowlim':
                if loss < clstage.criteria:
                    print('stopping stage  early. hit a lower limit for stopping')
                    savename = savedir + clstage.name + '_final'
                    torch.save(net.state_dict(), savename + '.model')
                    torch.save(updater.state_dict(), savename + '.adam')
                    break

        # add to losses at end of stage
        losslist.append(loss_stage)

    return losslist


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
    gamma = ops['gamma_kindergarten']  # TODO: will I need to make a learning-rate schedule for this?
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

    # lossind_list = [[0], [0, 1], [0, 1, 2]]
    ltot_train_list = []
    gradnorm_list = []
    updater = None

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
                seed=seed, nsteps=nsteps_simple, batchsize=batchsize, din=din, ops=ops)

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
            with open(savename + '_simple.json', 'w') as f:
                json.dump(savedat_json, f)
            # pickle.dump([ltot_train_list, gradnorm_list, outptdict], open(savename + '_simple.log', 'wb'))

        # del ltot_train_list, gradnorm_list

    # train on intermediate task-------------------------------------
    if 'intermediate' in stages:
        print('training on intermediate task: concatenation of 2 trials')

        updater = torch.optim.Adam(net.parameters(), lr=gamma)
        if optim_fname != '':
            updater.load_state_dict(torch.load(optim_fname, map_location=device.type))
            print('loading previous adam state')
            optim_fname = ''

        # change dimensions of trainign data. now timesteps is long, to concatenate trials
        # must have many trials and blocks to learn about structure
        # lossind_list = [[0, 1, 2, 3]]
        # lossind_list = [[0, 1, 2]]
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
                highvartrials=ops['highvartrials_kind'], ops=ops)

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
    if updatefun_name == 'update':
        updatefun = wt_pred.update
    elif updatefun_name == 'update_trialstart':
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

    # TODO: add general predicion for 2 state
    if ops['numstates_pred'] == 3:
        inputs, targets, _, _, _, _ = wt_pred.trainingdata(ops, seed=seed, ntrials=ntrials_pred, batchsize=batchsize,
                                                           useblocks=True, p_optout=p_optout)
    elif ops['numstates_pred'] == 5:
        vdict = {0: [2, 4, 8],
                 1: [2, 4, 8, 16],
                 2: [2, 4, 8, 16, 32, 64, 128],
                 3: [16, 32, 64, 128],
                 4: [32, 64, 128]}
        statetrans_fun = wt_pred.transition_5state
        stateops = {'vdict': vdict, 'statetrans_fun': statetrans_fun}

        inputs, targets, _, _, _, _ = wt_pred.trainingdata_general(ops, seed=seed, ntrials=ntrials_pred,
                                                                  batchsize=batchsize,
                                                                  useblocks=True, p_optout=p_optout, stateops=stateops)
    elif ops['numstates_pred'] == 2:
        vdict = {0: [2, 4, 8], 1: [8, 16, 32]}
        statetrans_fun = wt_pred.transition_2state
        stateops = {'vdict': vdict, 'statetrans_fun':statetrans_fun}

        inputs, targets, _, _, _, _ = wt_pred.trainingdata_general(ops, seed=seed, ntrials=ntrials_pred,
                                                                   batchsize=batchsize,
                                                                   useblocks=True, p_optout=p_optout, stateops=stateops)
    else:
        print('wrong type of prediction training data specified. check numstates_pred (2,3[wt task], 5)')
        inputs = None
        targets = None

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
    inputs, targets, _, _,_ = wt_d2m.trainingdata(ops, seed=seed, T=nt, batchsize=batchsize)
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

    # beginning to clean up curriculum training
    # tracemalloc.start(5)

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
        print(datetime.now())

        # if ((k + 1) % 10 == 0 or k == 0) and savetmp:
        #  try saving all data and see how it goes
        ops['trainonly'] = False

        returndict_nocatch = wt_reinforce_cont_new.session_torch_cont(ops, net, optimizer, costfun, device)
        # save
        fname = savename + '_nocatch_' + str(int((k + 1)))
        savecurric(fname, net, optimizer, returndict_nocatch, ops,
                   domodel=True, dostats=True, dodat=True, deldat=True)

        # delete again
        # print(memcheck(device))
        del returndict_nocatch
        gc.collect()
        if ops['device'] == 'cuda':
            print('clearing cuda cache at end of 10k nocatch session')
            torch.cuda.empty_cache()
        # print(memcheck(device))

    # round 2: catch trials
    print('round 2: catch trials')
    ops['nocatch_force'] = None  # turns orcing a wait decision off
    ops['pcatch'] = pcatch
    ops['useblocks'] = False

    for k in range(nrounds_start[1], nrounds[1]):
        print(k)
        print(datetime.now())

        # if ((k + 1) % 10 == 0 or k == 0) and savetmp:
        ops['trainonly'] = False

        returndict_catch = wt_reinforce_cont_new.session_torch_cont(ops, net, optimizer, costfun, device)

        # save, including statistics
        fname = savename + '_catch_' + str(int((k + 1)))
        output = savecurric(fname, net, optimizer, returndict_catch, ops,
                            domodel=True, dostats=True, dodat=True, deldat=True, doMAT=False)

        # delete again
        # print(memcheck(device))
        del output
        del returndict_catch
        gc.collect()
        if ops['device'] == 'cuda':
            print('clearing cuda cache at end of 10k catch session')
            torch.cuda.empty_cache()
        # print(memcheck(device))

        # early stopping?
        if ops['prog_stop']:
            linreg = output[0]
            if linreg['p'] < 0.01 and linreg['m'] > 0:
                print('p value for wait time slope significant, and slope is positive. stopping')
                break

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

        # if ((k + 1) % 10 == 0 or k == 0) and savetmp:

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
        output = savecurric(fname, net, optimizer, returndict_block, ops,
                            domodel=True, dostats=True, dodat=True, deldat=True, doMAT=True)

        # delete again
        # print(memcheck(device))
        del output
        del returndict_block
        gc.collect()
        if ops['device'] == 'cuda':
            print('clearing cuda cache at end of 10k block session')
            torch.cuda.empty_cache()
        # print(memcheck(device))

        if ((k + 1) % 10 == 0 or k == 0) and savetmp:
            print('freeze round')
            returndict_block_fr = wt_reinforce_cont_new.session_torch_cont(ops, net, optimizer, costfun, device)
            ops['freeze'] = freeze_global
            fname = savename + '_block_' + str(int((k + 1)))+'_freeze'
            _ = savecurric(fname, net, optimizer, returndict_block_fr, ops,
                           domodel=True, dostats=True, dodat=True, deldat=True, doMAT=True)

            # delete again
            # print(memcheck(device))
            del returndict_block_fr
            gc.collect()
            if ops['device'] == 'cuda':
                print('clearing cuda cache at end of 10k frozen block session')
                torch.cuda.empty_cache()
            # print(memcheck(device))

        anneal_ind += 1

        # early stopping?
        if ops['prog_stop']:
            wt_dict = output[1]
            # is there significant adaptaiton for 20 uL?
            wtmix_ulim = wt_dict['wt_mixed'][2] + 2 * wt_dict['wt_mixed_sem'][2]
            wtmix_llim = wt_dict['wt_mixed'][2] - 2 * wt_dict['wt_mixed_sem'][2]
            if wt_dict['wt_high'][2] > wtmix_ulim and wt_dict['wt_low'][2] < wtmix_llim:
                print('block adaptation for 20uL is signiticant to 2 SEM. stopping')
                break

    gc.collect()
    return net, optim_fname


def sim_task(modelname, configname=None, simtype='task', task_seed = 101):
    """
    will simulate for a given task
    for heavier dynamics processing
    :param modelname: (str) full path to model to use for simulation
    :param configname: (str) path to .cfg file. make sure trainonly=False, trackstate=True for sim task.
    :param simtype: (str) 'task', 'inf','kind' for different types of simulations
    :return:
    """

    # avoid having exact same trial list for every RNN. let task_seed be their RNN number for unique list
    np.random.seed(task_seed)

    # TODO: set the nsteps list or ns, or etc. as an option

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

    basename = modelname.split('/')[-1].split('.')[0]  # the prefix for saved data

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
        nsteps_list=[30, 20, 10]
        #nsteps_list = [30]
        ntrials = len(nsteps_list)
        nsteps = sum(nsteps_list)
        inp, targ, V, rewards = wt_kindergarten.trainingdata_intermediate(seed=101,
                                                                          nsteps_list=nsteps_list,
                                                                          batchsize=batchsize,
                                                                          ops=ops)
        inputs = torch.Tensor(inp).to(device)
        targets = torch.Tensor(targ).to(device)
        si = net.begin_state(batchsize=batchsize, device=device)
        device = torch.device('cpu')

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
        otypes = np.unique(offers)
        behdict = {'tdict': tdict, 'preds': preds, 'blocks': blocks, 'offers': offers, 'targs': targ,
                   'outcomes': 0*np.ones(offers.shape).astype(int)}

        rdict = {'behdict': behdict, 'outpt_supervised': outpt_supervised, 'sflat': sflat, 'inp': inp, 'ops':ops}


    elif simtype == 'inf':
        batchsize = 1
        ntrials = 200
        inp, targ, blocks, oo_idx, oo_fast_idx, catch_idx = wt_pred.trainingdata(
            ops, seed=101, ntrials=ntrials, batchsize=batchsize, useblocks=True, p_optout=0.0)
        blocks = np.array(np.squeeze(blocks))

        inputs = torch.Tensor(inp).to(device)
        targets = torch.Tensor(targ).to(device)
        si = net.begin_state(batchsize=batchsize, device=device)
        device = torch.device('cpu')

        outpt, outpt2, sflat, tdict = parse.batchsamples_kind_allstate(net, inputs, si)
        inp = np.squeeze(inp)
        offers = np.array(inp[inp[:, 0] > 0, 0])
        outpt = np.array([np.squeeze(k[0, 0, :]) for k in outpt])
        preds = np.array([np.squeeze(k[0, 0, :]) for k in outpt2])

        # create a behdict out of these things!
        otypes = np.unique(offers)
        behdict = {'tdict': tdict, 'preds': preds, 'blocks': blocks, 'offers': offers, 'targs': targ,
                   'outcomes': 0*np.ones(offers.shape).astype(int)}

        rdict = {'behdict': behdict, 'outpt_supervised': outpt, 'sflat': sflat, 'inp': inp, 'ops':ops}


    else:
        print('going to sim other types here. not supported yet')

    return rdict

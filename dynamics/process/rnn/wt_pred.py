# code to learn one-step prediction via supervised training. auxiliary cost code
# very similar to wt_kindergarten in some respects, and should probably consolidate at some point

import numpy as np
import torch
from torch import nn


# generate training data
def trainingdata(ops, seed=101, ntrials=400, batchsize=20, useblocks=False, p_optout=0.0):
    """
    training set to  perform block inference
    :param ops: ops dictionary with bits about task
    :param seed: random seed for trials
    :param ntrials: number of trials per sample
    :param batchsize: number of independent samples
    :param useblocks: use block structure or not
    :param p_optout: (float) proportion of trials in this data that are optout trials
    :return:
    """
    np.random.seed(seed)

    vdict = {0: [5, 10, 20, 40, 80],
             1: [20, 40, 80],
             2: [5, 10, 20]}
    ntmax = int(ops['T'] / ops['dt']) + 1
    tmesh = np.linspace(0, ops['T'], ntmax)  # time bins

    # set the master lists about optouts, rewards, delays, etc. add some padded extra trials for last batch
    vols = [np.random.choice(vdict[0], size=(batchsize, ntrials)),
            np.random.choice(vdict[1], size=(batchsize, ntrials)),
            np.random.choice(vdict[2], size=(batchsize, ntrials))]  # volume of water reward
    vols = np.array(vols)

    delays = np.random.exponential(scale=ops['lam'], size=(batchsize, ntrials))  # how long until reward arrives

    Ract = ops['Ract']  # optout penalty
    T = ops['T']  # max time. catch trial delay

    # determine which blocks will be used
    blocklen = ops['blocklength']
    blocks_all = np.nan * np.ones((batchsize, ntrials))
    for k in range(batchsize):
        ind = 0
        prevblock = 2
        blocks = [0]

        for j in range(1, ntrials):
            if useblocks:
                # what block were you previously
                block = blocks[-1]

                if ind < blocklen:
                    ind += 1
                    blocks.append(block)
                elif np.random.binomial(1, ops['pblock']) == 0:  # a failed, probabalistic block transition
                    ind += 1
                    blocks.append(block)
                else:  # block transition
                    ind = 0
                    if block == 2 or block == 1:
                        blocks.append(0)
                    elif block == 0 and prevblock == 2:
                        blocks.append(1)
                    elif block == 0 and prevblock == 1:
                        blocks.append(2)

                    prevblock = block
            else:
                blocks.append(0)

        blocks_all[k, :] = np.array(blocks)

    # assign volumes based on which blocks should be used. also determine catch trials, optouts, quick optouts
    oo_idx = []  # which trials are opt out
    pfast = 0.2  # 20% of optouts will be fast
    oo_fast_idx = []  # which trialsa are fast opt outs, opt out on first time point
    catch_idx = []  # which trials are catch, delay is T
    blocks_all = blocks_all.astype(int)
    vols_use = np.nan * np.ones((batchsize, ntrials))
    rews_use = np.nan * np.ones((batchsize, ntrials))
    actions_use = np.zeros((batchsize, ntrials))

    for k in range(batchsize):
        # decide on opt outs and catch trials
        oo_trials = np.random.choice(range(ntrials), replace=False, size=(int(np.floor(p_optout * ntrials))))
        oo_idx.append(oo_trials)
        oo_fast_idx.append(np.random.choice(oo_trials, replace=False, size=(int(np.floor(pfast * len(oo_trials))))))
        catch_idx.append(np.random.choice(range(ntrials), replace=False, size=(int(np.floor(ops['pcatch'] * ntrials)))))

        for j in range(ntrials):
            vols_use[k, j] = vols[blocks_all[k, j], k, j]

        # update rewards
        rews_use[k, :] = vols_use[k, :]
        rews_use[k, catch_idx[-1]] = 0.0  # catch trials get no reward. overwrite
        rews_use[k, oo_trials] = Ract  # opt outs get pantly. overwrite, including optint out on a catch trial

        # update delays
        delays[k, catch_idx[-1]] = T
        delays[k, oo_fast_idx[-1]] = ops['dt']

        # actions
        actions_use[k, oo_trials] = 1

    # format: set delays into index format, choose volumes
    trial_lens = np.digitize(delays, bins=tmesh)
    trial_lens_cumul = np.cumsum(trial_lens, axis=1)  # counts trial length. + 1 for zeroindexing

    # build inputs and outputs
    ns_all = np.min(np.sum(trial_lens, axis=1))  # smallest batch slice, in timesteps
    if ops['inputcase'] == 11:
        ninp = 4
    elif ops['inputcase'] == 15:
        ninp = 6
    else:
        print('input case not supported for inference kindergarten: '+ops['inputcase'])
        ninp = None

    inputs = np.nan*np.ones((batchsize, ns_all, ninp))  # hardcoded for inputs case 11
    targets = np.nan*np.ones((batchsize, ns_all, 4))  # rew offer, count time, rew hist, block pred

    for k in range(batchsize):
        # initialize
        ns_k = np.sum(trial_lens[k, :])  # number of timesteps in that batch
        input_k = np.zeros((1, ns_k, ninp))
        target_k = np.zeros((1, ns_k, 4))

        # start and stop of each trial
        tstart = np.concatenate(([0], trial_lens_cumul[k, :-1]), axis=0)
        tstop = trial_lens_cumul[k, :]-1

        # set inputs and outputs.
        input_k[0, tstart, 0] = np.log(vols_use[k, :])
        input_k[0, tstart, 1] = 1
        input_k[0, :, 2] = ops['w']
        input_k[0, tstop, 2] += rews_use[k, :]
        input_k[0, tstop, 3] = actions_use[k, :]
        # use opt outs? not yet

        # outputs. first 3 are original kindergarten, last is block prediction
        for j in range(ntrials):
            rew = input_k[0, tstart[j]:tstop[j]+1, 2]  # rewards on that trial
            t_k = range(1, trial_lens[k, j]+1)  # +1 shift from 0
            target_k[0, tstart[j]:tstop[j]+1, 0] = np.log(vols_use[k, j])  # recall log volume
            target_k[0, tstart[j]:tstop[j]+1, 1] = np.array(t_k) / 100  # keeps on same order of mag
            target_k[0, tstart[j]:tstop[j]+1, 2] = np.cumsum(rew) / np.array(t_k)
            target_k[0, tstart[j]:tstop[j]+1, 3] = blocks_all[k, j]

        # trim edges
        inputs[k, :] = input_k[0, :ns_all, :]
        targets[k, :] = target_k[0, :ns_all, :]

    return inputs, targets, blocks_all, oo_idx, oo_fast_idx, catch_idx


# the supervised learning bits for predicting, updating, and full training in kindergarten
# duplicate from wt_kindergarten
def batchsamples(net, inputs, si, device=torch.device('cpu')):
    """
    take a target network, run some samples through it
    :param net: netowrk
    :param inputs: inputs to network
    :param si: initial state
    :param device: torch device
    :return outputs: produced outputs from network
    :return state: hidden state of network
    :return net: network.
    """

    batchsize, nsteps, _ = inputs.shape
    # parse inputs, mostly for multiregion
    inputs = net.parseiputs(inputs)

    output = net(inputs, si, isITI=False, device=device)  # assume that all models will augment action space for ITI
    state = output['state']  # state of network
    pred = output['activations']  # activations for the block prediction
    outputs_pred = torch.reshape(pred, (batchsize, nsteps, -1))
    outputs_pred.requires_grad_(True)
    outputs_supervised = output['pred_supervised']
    outputs_supervised = torch.reshape(outputs_supervised, (batchsize, nsteps, -1))
    outputs_supervised.requires_grad_(True)
    outputs = [outputs_supervised, outputs_pred]

    # detach states
    if isinstance(state[0], tuple):  # multiregion
        [[k.detach() for k in sk] for sk in state]
        [[k.detach() for k in sk] for sk in si]
    elif isinstance(state, tuple) or isinstance(state, list):  # LSTM or mr RNN
        [k.detach() for k in state]
        [k.detach() for k in si]
    else:  # vanilla RNN, GRU
        state.detach()
        si.detach()

    return outputs, state, net


def update(updater, outputs, targets,  lambda_kind, lambda_pred, _, lossinds=None):
    """
    calculates loss and updates weights via an optimizer. includes a portion for prediction AND kindergarten
    :param updater: torch updater, like Adam
    :param outputs: outputs predicted by network
    :param targets: target outputs
    :param lambda_kind: weight on kindergarten portion
    :param lambda_pred: weight on prediction
    :param _: (np.ndarray) inputs to network. used to find when trial start is. not used here
    :param lossinds: (list) which indices of output to use to MSE loss on  kindergarten
    :return: loss_np value of loss for that minibatch
    """

    if lossinds is None:
        lossinds = [0, 1, 2]

    loss_supervised = nn.MSELoss()
    loss_pred = nn.CrossEntropyLoss()

    # inlude kindergarten loss and prediction. weighted accordingly. HARDCODED INDICES
    lossinds_kind = lossinds
    lossinds_pred = 3

    outputs_supervised = outputs[0]
    outputs_pred = outputs[1]
    dout = outputs_pred.shape[-1]
    outputs_pred = torch.reshape(outputs_pred, (-1, dout))  # cross entropy has specific input dim requirements
    targets_supervised = targets[:, :, lossinds_kind]
    targets_pred = torch.reshape(targets[:, :, lossinds_pred].long(),  (-1,))

    lval_kind = loss_supervised(outputs_supervised[:, :, lossinds_kind], targets_supervised)
    lval_pred = loss_pred(outputs_pred, targets_pred)
    lval = lambda_kind*lval_kind + lambda_pred*lval_pred

    if updater is not None:  # just use to calculate loss, like for mixed loss funs in other trainings
        updater.zero_grad()  # updaters accumulate gradients. zero out
        lval.backward()
        # grad_clipp(net,100) #clip the gradient to prevent explosion. other built-in methods?
        updater.step()
        loss_np = lval.detach().cpu().numpy()  # the value of the loss from that batch

        return loss_np, lval_pred.detach().cpu().numpy(), lval_kind.detach().cpu().numpy()
    else:  # return the tesor form
        return lval, lval_pred, lval_kind


def update_trialstart(updater, outputs, targets,  lambda_kind, lambda_pred, inp, lossinds=None):
    """
    calculates loss and updates weights via an optimizer, but only at trial start
    :param updater: torch updater, like Adam
    :param outputs: outputs predicted by network
    :param targets: target outputs
    :param lambda_kind: weight on kindergarten portion
    :param lambda_pred: weight on prediction
    :param inp: (np.ndarray) inputs, used to find where trial start is
    :param lossinds: (list) which indices of output to use to MSE loss on  kindergarten
    :return: loss_np value of loss for that minibatch
    """

    if lossinds is None:
        lossinds = [0, 1, 2]
    loss_supervised = nn.MSELoss()
    loss_pred = nn.CrossEntropyLoss()

    # inlude kindergarten loss and prediction. weighted accordingly. HARDCODED INDICES
    lossinds_kind = lossinds
    lossinds_pred = 3

    offers = inp[0, :, 0]
    offer_idx = np.squeeze(np.argwhere(offers > 0))

    outputs_supervised = outputs[0]
    outputs_pred = outputs[1]
    dout = outputs_pred.shape[-1]
    outputs_pred = torch.reshape(outputs_pred[:, offer_idx, :], (-1, dout))  # specific input dim requirements
    targets_supervised = targets[:, :, lossinds_kind]
    targets_pred = torch.reshape(targets[:, offer_idx, lossinds_pred].long(),  (-1,))

    lval_kind = loss_supervised(outputs_supervised[:, :, lossinds_kind], targets_supervised)
    lval_pred = loss_pred(outputs_pred, targets_pred)
    lval = lambda_kind*lval_kind + lambda_pred*lval_pred

    if updater is not None:  # just use to calculate loss, like for mixed loss funs in other trainings
        updater.zero_grad()  # updaters accumulate gradients. zero out
        lval.backward()
        # grad_clipp(net,100) #clip the gradient to prevent explosion. other built-in methods?
        updater.step()
        loss_np = lval.detach().cpu().numpy()  # the value of the loss from that batch

        return loss_np, lval_pred.detach().cpu().numpy(), lval_kind.detach().cpu().numpy()
    else:  # return the tesor form
        return lval, lval_pred, lval_kind


# duplicate of kindergarten.
def train(ops, net, updater, si, inp, targ, device=torch.device('cpu'), nepoch=100, nsteps=100,
          stoptype='lowlim', stopargs=0.01, updatefun=update):
    """
    update over multiple epochs. each epoch is a pass over the training data.
    currently designed to update gradient every nsteps=100 steps
    :param ops: (dict) simulation options. alway sgood to have around
    :param net: network
    :param updater: optimizers
    :param si: initial hidden state
    :param inp: inputs to network
    :param targ: target outputs of network
    :param device: which device will training happen on
    :param nepoch: number of training epochs
    :param nsteps: number of timesteps per sample
    :param stoptype: (str) 'lowlim' for hitting a lower limit of loss, or 'converge' to monitor imporvement over epochs
    :param stopargs: (int) argument to handle each stoptype. 'lowlim': gives stop value, 'converge': num epochs back
    :param updatefun: update function, like Adam
    :return:
    """

    beta_kind = ops['beta_supervised_kindpred']
    beta_pred = ops['beta_pred_kindpred']
    lossinds = ops['lossinds_supervised_kindpred']

    ltot_train = []  # loss from all of the batches
    grad_norm = []  # normalized gradient each step. to see optimization activity

    inputs = torch.Tensor(inp).to(device)
    targets = torch.Tensor(targ).to(device)

    loss_best = np.inf
    checkind = 0  # initialze the check index for saturating performance
    outptdict = {'nsteps': nsteps}

    for epoch in range(nepoch):
        # reset hidden state each time via state=None.
        state = si
        ind = 0
        nbatch = int(inputs.shape[1] - nsteps + 1)  # run overlapping batches
        ltot_train_k = []
        lpred_k = []
        grad_norm_k = []

        for k in range(nbatch):
            inputs_k = inputs[:, ind:ind + nsteps, :]
            targets_k = targets[:, ind:ind + nsteps, :]

            # generate samples
            outputs, state, net = batchsamples(net, inputs_k, state, device=device)

            # update
            loss_k, lval_pred, lval_kind = updatefun(updater, outputs, targets_k, beta_kind, beta_pred, inp,
                                                     lossinds)
            ltot_train_k.append(loss_k)
            lpred_k.append(lval_pred)

            # track gradient
            abs_sum = np.sum([abs(np.sum(k.grad.cpu().numpy())) for k in net.parameters()])
            grad_norm_k.append(abs_sum)

            ind += 1

        ltot_train.append(ltot_train_k)
        grad_norm.append(grad_norm_k)

        outptdict['epoch'] = epoch
        outptdict['isdone'] = False

        if (epoch + 1) % 10 == 0:
            print([np.mean(ltot_train_k), np.mean(lpred_k)])

        # do a convergence stopping test
        if stoptype == 'converge':
            loss_av = np.mean(lpred_k)

            if loss_av <= loss_best:
                loss_best = loss_av
                checkind = 0
            else:
                checkind += 1

            if checkind > stopargs:
                print('stopping early. no improvement compared to average of last ' + str(stopargs) + 'epochs')
                outptdict['isdone'] = True
                break
        elif stoptype == 'lowlim':
            loss_av = np.mean(lpred_k)

            if loss_av < stopargs:
                print('stopping early. hit a lower limit for stopping')
                outptdict['isdone'] = True
                break

    return ltot_train, grad_norm, outptdict

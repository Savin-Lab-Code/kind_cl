# preconditioning of RNN with supervised learning of simpler tasks before wait-time

import numpy as np
import torch
from torch import nn


<<<<<<< HEAD
# generate training area for kindergarten tasks. supervised learning ones only
=======
# the Kindergarten training area. supervised learning to pre-condition an RNN------------
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
def trainingdata_simple(seed=101, nsteps=20, batchsize=1000, din=4, ops=None):
    """
    training set to i)remember input, ii)integrate time iii) integrate reward signal
    :param seed: (int) random seed for trials
    :param nsteps: (int) number of time steps per sample
    :param batchsize: (int) number of independent samples
    :param din: (int) size of input space
    :param ops: (dict)
    :return:
    """
    np.random.seed(seed)

    # volume inputs to remember
    Ract = -0.1  # opt-out penal
<<<<<<< HEAD
    V = np.random.choice([5, 10, 20, 40, 80], replace=True, size=(batchsize,))
=======
    vals = ops['kind_vals']  # the default kindergarten values. can overwrite from default to make more general
    V = np.random.choice(vals, replace=True, size=(batchsize,))
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
    # "rewards" that should be averaged in a moving window
    rewards = np.random.rand(batchsize, nsteps)
    # make about 10% of the trials have an opt-out instead
    oo_idx = np.random.choice(range(batchsize), replace=False,  size=(int(np.floor(0.1*batchsize))))
    rewards[:, -1] = rewards[:, -1] + V  # add reward at last timestep
    rewards[oo_idx, -1] = rewards[oo_idx, -1] - V[oo_idx] + Ract  # remove reward, add opt-out penalty

    if ops is None:
        din = 4
    else:
        if ops['inputcase'] == 11:
            din = 4
        elif ops['inputcase'] == 15:
            din = 6
<<<<<<< HEAD
=======
        elif ops['inputcase'] == 17:
            din = 8
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
    inputs = np.zeros((batchsize, nsteps, din))
    outputs = np.zeros((batchsize, nsteps, 3))

    for j in range(batchsize):
        # INPUTS
<<<<<<< HEAD
        inputs[j, 0, 0] = np.log(V[j])
        inputs[j, 0, 1] = 1  # trial start
        inputs[j, :, 2] = rewards[j, :]  # rewards

        inputs[j, :, 3] = 0  # actions. wait until reward,
        if j in oo_idx:  # an opt-out trial
            inputs[j, -1, 3] = 1
=======
        if ops['inputcase'] == 17:
            if V[j] == 5:
                inputs[j, 0, 0] = 1  # Trial offer #use if statements to add multiple diemensions
            elif V[j] == 10:
                inputs[j, 0, 1] = 1
            elif V[j] == 20:
                # Shows that the channel for 20 is being used only when the volume reward is 20
                # print(V[j])
                # print("The input volume is 20")
                inputs[j, 0, 2] = 1
                # Shows that index is one
                # print(inputs[j,0,2])
            elif V[j] == 40:
                inputs[j, 0, 3] = 1
            elif V[j] == 80:
                inputs[j, 0, 4] = 1
        else:
            inputs[j, 0, 0] = np.log(V[j])

        # adjust final indices based on inputsize
        inputs[j, 0, din-3] = 1  # trial start
        inputs[j, :, din-2] = rewards[j, :]  # rewards

        inputs[j, :, din-1] = 0  # actions. wait until reward,
        if j in oo_idx:  # an opt-out trial
            inputs[j, -1, din-1] = 1
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)

        # OUTPUTS
        outputs[j, :, 0] = np.log(V[j])  # recall log volume
        outputs[j, :, 1] = np.array(range(nsteps))/100  # remember time. factor of 10 keeps on same order of mag.
        outputs[j, :, 2] = np.cumsum(rewards[j, :])/(np.array(range(nsteps))+1)

    return inputs, outputs, V, rewards


def trainingdata_intermediate(seed=101, nsteps_list=None, batchsize=1000, din=4, highvartrials=False, ops=None):
    """
    concatenate simple training to make harder supervised data. but the order within each batch dim
    :param seed: (int) random seed
    :param nsteps_list: (list) of number of steps for each "trial". length of list is  number of trials
    :param batchsize: (int) size of minibatch independent samples
    :param din: (int) size of input space
<<<<<<< HEAD
    :param highvartrials: (bool) generate trials with high variance in trial lengths across the minibatch?
    :param ops: (dict) with details about simulation
=======
    :param highvartrials: (bool) use high variance in trial types across the batch?
    :param ops: (dict) the training ops dictionary
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
    :return:
    """
    if nsteps_list is None:
        nsteps_list = [20, 25]
    np.random.seed(seed)

<<<<<<< HEAD
=======
    if ops is None:
        print('ops struct not defined in intermediate kindergarten. might have incorrect model specs')

>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
    inputs = []
    outputs = []
    V = []
    rewards = []
    #  intermetdiate loop ones
    inputs_m = []
    outputs_m = []

    n = len(nsteps_list)
<<<<<<< HEAD

=======
    # nstep = sum(nsteps_list)

    # TODO: not super effective, but maybe keep
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
    if highvartrials:
        # generate unique sequences for each batch, then clip to smallest sequence to make a proper minibatch
        nsteps_list_all = np.random.randint(20, 100, (n, batchsize))
        nsum = np.sum(nsteps_list_all, axis=0)
        nmin = np.min(nsum)
        nmin_idx = np.argmin(nsum)
        csum = np.cumsum(nsteps_list_all, axis=0)

        # find where to clip all other batches
        clip_idx = []
        for j in range(batchsize):
            if j == nmin_idx:
                clip_idx.append(n - 1)
            else:
                clip_idx.append(np.argwhere(csum[:, j] > nmin)[0, 0] - 1)

        newmaxes = np.array([csum[clip_idx[j], j] for j in range(batchsize)])
        tdiff = nmin - newmaxes
        nsteps_list_all_tmp = []
        for j in range(batchsize):
            nsteps_list_all_tmp.append(nsteps_list_all[0:clip_idx[j] + 1, j])
            nsteps_list_all_tmp[-1][-1] += tdiff[j]

        nsteps_list_all = nsteps_list_all_tmp
<<<<<<< HEAD
    else:
        nsteps_list_all = None
=======
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)

    for k in range(batchsize):

        if highvartrials:  # a unique trial list for each minibatch
            nsteps_list_m = nsteps_list_all[k]
            n = len(nsteps_list_m)
        else:  # for same trials in each minibatch, but scrambled in order
            nsteps_list_m = np.random.choice(nsteps_list, n, replace=False)

<<<<<<< HEAD
        # initialize out of loop
        V_m = None
        rewards_m = None

=======
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
        for m in range(n):  # loop over nsteps. scramble on each batch
            nsteps = nsteps_list_m[m]
            inputs_k, outputs_k, V_k, rewards_k = trainingdata_simple(seed + 100*k + m, nsteps, batchsize=1, din=din,
                                                                      ops=ops)
            V_k = np.expand_dims(V_k, axis=0)  # make a 'batch'-like dimension
            V_m = []
            rewards_m = []
            if m == 0:
                inputs_m = inputs_k
                outputs_m = outputs_k
            else:
                inputs_m = np.concatenate((inputs_m, inputs_k), axis=1)
                outputs_m = np.concatenate((outputs_m, outputs_k), axis=1)

            V_m.append(V_k)
            rewards_m.append(rewards_k)

        if k == 0:
            inputs = inputs_m
            outputs = outputs_m
        else:
            inputs = np.concatenate((inputs, inputs_m), axis=0)
            outputs = np.concatenate((outputs, outputs_m), axis=0)

        V.append(V_m)
        rewards.append(rewards_m)

    return inputs, outputs, V, rewards


# the supervised learning bits for predicting, updating, and full training in kindergarten
def batchsamples(net, inputs, si, device=torch.device('cpu')):
    """
    take a target network, run some samples through it
    :param net: netowrk
    :param inputs: inputs to network
    :param si: initial state
    :param device: device
    :return outputs: produced outputs from network
    :return state: hidden state of network
<<<<<<< HEAD
    :return net: network
=======
    :return net: network. TODO needed?
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
    """

    batchsize, nsteps, _ = inputs.shape

    # parse inputs, mostly for multiregion
    inputs = net.parseiputs(inputs)

    output = net(inputs, si, isITI=False, device=device)  # assume that all models will augment action space for ITI
    state = output['state']  # state of network
    pred_supervised = output['pred_supervised']  # supervised output predicitons for kindergarten

    outputs = torch.reshape(pred_supervised, (batchsize, nsteps, -1))
    outputs.requires_grad_(True)

    # detach states.
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


def update(updater, outputs, targets, lossinds=None):
    """
    calculates loss and updates weights via an optimizer
    :param updater: torch updater, like Adam
    :param outputs: outputs predicted by network
    :param targets: target outputs
    :param lossinds: which indices of output to use to MSE loss
    :return: loss_np value of loss for that minibatch
    """

    if lossinds is None:
        lossinds = [0, 1, 2]

    loss = nn.MSELoss()
    lval = loss(outputs[:, :, lossinds], targets[:, :, lossinds])

    if updater is not None:  # just use to calculate loss, like for mixed loss funs in other trainings
        updater.zero_grad()  # updaters accumulate gradients. zero out
        lval.backward()
        updater.step()
        loss_np = lval.detach().cpu().numpy()  # the value of the loss from that batch

        return loss_np
    else:  # return the tesor form
        return lval


<<<<<<< HEAD
def train(net, updater, si, inp, targ, device=torch.device('cpu'), nepoch=100, nsteps=100, lossinds=None,
          stoptype='lowlim', stopargs=0.01, win=3, chkptepoch=0):
=======
def update_individual(updater, outputs, targets, lossinds=None):
    """
    calculates loss and updates weights via an optimizer, but calculates loss individually.
    to prevent issues in training, it will not allow an updater
    :param updater: torch updater, like Adam
    :param outputs: outputs predicted by network
    :param targets: target outputs
    :param lossinds: which indices of output to use to MSE loss
    :return: loss_np value of loss for that minibatch
    """

    if lossinds is None:
        lossinds = [0, 1, 2]

    loss = nn.MSELoss()
    lval_list = [loss(outputs[:, :, k], targets[:, :, k]) for k in lossinds]

    if updater is not None:  # just use to calculate loss, like for mixed loss funs in other trainings
        print('update_individual doesnt support direct parameter updating. used for aggregated loss functions')
    else:  # return the tesor form
        return lval_list


def train(net, updater, si, inp, targ, device=torch.device('cpu'), nepoch=100, nsteps=100, lossinds=None,
          stoptype='lowlim', stopargs=0.01):
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
    """
    update over multiple epochs. each epoch is a pass over the training data.
    currently designed to update gradient every nsteps=100 steps
    :param net: network
    :param updater: optimizers
    :param si: initial hidden state
    :param inp: inputs to network
    :param targ: target outputs of network
    :param device: which device will training happen on
    :param nepoch: number of training epochs
    :param nsteps: number of timesteps per sample
    :param lossinds: indeices of output to use in loss
    :param stoptype: (str) 'lowlim' for hitting a lower limit of loss, or 'converge' to monitor imporvement over epochs
    :param stopargs: (int) argument to handle each stoptype. 'lowlim': gives stop value, 'converge': num epochs back
<<<<<<< HEAD
    :param win: (int) kindergarten-specific param for how many trials should be averaged. NOT USED
    :param chkptepoch: (int) frequency, in epochs, to run checkpoint plotting fucntion. 0= no plotting. NOT USED
=======
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
    :return:
    """

    if lossinds is None:
        lossinds = [0, 1, 2]

    ltot_train = []  # loss from all of the batches
    grad_norm = []  # normalized gradient each step. to see optimization activity

    inputs = torch.Tensor(inp).to(device)
    targets = torch.Tensor(targ).to(device)

    loss_best = np.inf
    checkind = 0  # initialze the check index for saturating performance

    outptdict = {}

    for epoch in range(nepoch):

        # reset hidden state each time via state=None.
        state = si
        ind = 0
        nbatch = int(inputs.shape[1] - nsteps + 1)  # run overlapping batches
        ltot_train_k = []
        grad_norm_k = []

        for k in range(nbatch):
            inputs_k = inputs[:, ind:ind + nsteps, :]
            targets_k = targets[:, ind:ind + nsteps, :]

            # generate samples
            outputs, state, net = batchsamples(net, inputs_k, state, device)

            # update
            loss_k = update(updater, outputs, targets_k, lossinds)
            ltot_train_k.append(loss_k)

            # track gradient
            abs_sum = np.sum([abs(np.sum(k.grad.cpu().numpy())) for k in net.parameters()])
            grad_norm_k.append(abs_sum)

            # ind += nsteps
            ind += 1

        ltot_train.append(ltot_train_k)
        grad_norm.append(grad_norm_k)

        outptdict['epoch'] = epoch
        outptdict['isdone'] = False

        if (epoch + 1) % 10 == 0:
            print(np.mean(ltot_train_k), end='\r')

        # do a convergence stopping test
        if stoptype == 'converge':
            loss_av = np.mean(ltot_train_k)

            if loss_av <= loss_best:
                loss_best = loss_av
<<<<<<< HEAD
=======
                # params_best = net.parameters()
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
                checkind = 0
            else:
                checkind += 1

            if checkind > int(stopargs):
                print('stopping early. no improvement compared to average of last ' + str(stopargs) + 'epochs')
                outptdict['isdone'] = True
                break
        elif stoptype == 'lowlim':
            loss_av = np.mean(ltot_train_k)

            if loss_av < stopargs:
                print('stopping early. hit a lower limit for stopping')
                outptdict['isdone'] = True
                break

    return ltot_train, grad_norm, outptdict

# cost functions for wait time tasks
# import numpy as np
import torch
<<<<<<< HEAD
from dynamics.process.rnn import wt_kindergarten
from torch import nn
=======
from dynamics.process.rnn import wt_kindergarten, wt_pred
from torch import nn
# from torch.distributions.categorical import Categorical
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
import numpy as np


def loss_actorcritic_minh(inp, ops):
    """
    Minh 2016 form of episodic actor-critic loss. handles sampling of value differently than Sutton Barto form,
    by using reward samples to approximate value function. similar to REINFORCE, but requires either
    end of episodes, or batches of experiences for unrolling.
    This code assumes that data ends in a terminal state
    :param inp: input dict. requires rewards, v, log_probs
    :param ops: standard ops dict. requires gamma, device
    :return:
    """

    rewards = inp['rewards']  # actual reward received at each time
    values = inp['values']  # state-value of estimate of chosen action
    log_probs = inp['log_probs']  # policy for each action
    gamma = ops['gamma']  # discount factor

    lambda_value = ops['lambda_value']

    if inp['done']:  # was a terminal state
        nt_j = len(rewards)
        R = 0  # assumes final state was terminal state
    else:
        nt_j = len(rewards)-1
        R = values[-1]
    loss_policy = torch.zeros(size=(nt_j,))
    loss_value = torch.zeros(size=(nt_j,))

    for k in range(nt_j-1, -1, -1):
        R = rewards[k] + gamma*R
        rpe = R - values[k]
        loss_policy[k] = -log_probs[k] * rpe.detach()
        loss_value[k] = rpe*rpe  # in Minh they used rpe^2 for some reason

    return [loss_policy.mean(), lambda_value*loss_value.mean()], []


def loss_actorcritic_minh_iti(inp, ops):
    """
    Same form, but only grabs non-ITI data, and keeps loss for all time points. going to see how gradient looks
    :param inp: input dict. requires rewards, v, log_probs
    :param ops: standard ops dict. requires gamma, device
    :return:
    """

    rewards = inp['rewards']  # actual reward received at each time
    values = inp['values']  # state-value of estimate of chosen action
    log_probs = inp['log_probs']  # policy for each action
    gamma = ops['gamma']  # discount factor
    actions = inp['actions']  # used to decide which steps were ITI

    lambda_value = ops['lambda_value']

    if inp['done']:  # was a terminal state
        nt_j = len(rewards)
        R = 0  # assumes final state was terminal state
    else:
        nt_j = len(rewards)-1
        R = values[-1]
    loss_policy = torch.zeros(size=(nt_j,))
    loss_value = torch.zeros(size=(nt_j,))

    for k in range(nt_j-1, -1, -1):
        R = rewards[k] + gamma*R
        rpe = R - values[k]
        loss_policy[k] = -log_probs[k] * rpe.detach()
        loss_value[k] = rpe*rpe  # in Minh they used rpe^2 for some reason

    # decide what are ITI timesteps. anything with a forced iti action (==2) is considered ITI
    goodt = []
    for k in range(nt_j-1, -1, -1):
<<<<<<< HEAD
=======
        # goodt.append(torch.not_equal(actions[k], 2)) original form
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
        if torch.not_equal(actions[k], 2):
            goodt.append(k)
    goodt.reverse()

    loss_policy_good = loss_policy[goodt]
    loss_value_good = loss_value[goodt]

<<<<<<< HEAD
    return [loss_policy_good.mean(), lambda_value*loss_value_good.mean()], []


def loss_kindergarten(inp, ops):
    """
    will calculate a supervised loss signal from target signals from kindergarten
    :param inp: input dictionary. requires net, batchsize, lossinds, nsteps_list, seed
    :param ops: standard ops dict. requires device
=======
    # for debugging purposes. keep all time indices and look at outputs
    # loss_policy_tmp = loss_policy[goodt]
    # loss_value_tmp = loss_value[goodt]
    # return [loss_policy, lambda_value*loss_value], [loss_policy_tmp,loss_value_tmp ], goodt

    return [loss_policy_good.mean(), lambda_value*loss_value_good.mean()], []


def loss_kindergarten(inp, ops, updatefun=wt_kindergarten.update):
    """
    will calculate a supervised loss signal from target signals from kindergarten
    :param inp: (dict) input dictionary. requires net, batchsize, lossinds, nsteps_list, seed
    :param ops: (dict) standard ops dict. requires device
    :param updatefun: (fun) fun to calculate loss function. use .update_individual for unique weights for each loss
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
    :return:
    """

    net = inp['net']
    device = torch.device(ops['device'])
    batchsize = ops['batchsize_kindergarten']  # size of batch
    lossinds = ops['lossinds_kindergarten']  # which indices from supervised output to include in loss
    nsteps_list = ops['nsteps_list_kindergarten']  # size of each 'trial' , as a vector
    seed = ops['seed_kindergarten']  # random seed

    if ops['ismultiregion']:
        din = np.max(net.din)  # for multiregion
    else:
        din = net.din  # input size

    # which indices of the kindergarten signal do you want to use for training
    if lossinds is None:
        lossinds = [0, 1]
    # how many timesteps for each "trial" in the list of training data
    if nsteps_list is None:
        nsteps_list = [20, 25]

    # generate training data
<<<<<<< HEAD
=======
    # TODO: this must work with multiregion or make a new method
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
    inputs_sup, targets_sup, _, _ = wt_kindergarten.trainingdata_intermediate(
        seed=seed, nsteps_list=nsteps_list, batchsize=batchsize, din=din, ops=ops)
    inputs_supervised = torch.Tensor(inputs_sup).to(device)
    targets_supervised = torch.Tensor(targets_sup).to(device)

    # calculate
    si_supervised = net.begin_state(batchsize=batchsize, device=device)
    # track the state. do it as list comprehension for LSTMS
    if isinstance(si_supervised[0], tuple):  # TODO: does this work with GRU and vanilla RNN?
        [[k.detach() for k in sk] for sk in si_supervised]
    else:
        [k.detach() for k in si_supervised]
    # [k.detach() for k in si_supervised]
    outputs_supervised, _, _ = wt_kindergarten.batchsamples(net, inputs_supervised, si_supervised, device)

<<<<<<< HEAD
    # calculate MSE.
    supervised_loss = wt_kindergarten.update(updater=None, outputs=outputs_supervised,
=======
    # calculate MSE. will be list if updatefun = wt_kindergarten.update_individual
    supervised_loss = updatefun(updater=None, outputs=outputs_supervised,
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
                                             targets=targets_supervised, lossinds=lossinds)

    return supervised_loss


def loss_prediction_prob_block(inp, ops):
    """
    cross-entropy loss of current block, at current offer

    :param inp: input dictionary. requires net, batchsize, lossinds, nsteps_list, seed
    :param ops: standard ops dict. requires device
    :return:
    """

<<<<<<< HEAD
    nstep = ops['ctmax']  # unroll width
=======
    nstep = ops['ctmax']  # unroll width TODO: this is problematic for true episodic RL where ctmax = -1
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)

    # extract last inputs and reshape into a good tensor
    # find where trial start is
    if ops['ismultiregion']:
        reg_idx = 0  # assume that ALL regions receive the trial start input at index 1, but pick one for now
        start_idx = torch.tensor([inp['inputs'][k][reg_idx][0, 0, 1] for k in range(nstep)], device=ops['device'])
    else:
        start_idx = torch.tensor([inp['inputs'][k][0, 0, 1] for k in range(nstep)], device=ops['device'])
    idx = torch.nonzero(start_idx)
    ntrial = len(idx)

    targets = torch.zeros((ntrial,), device=ops['device'], dtype=torch.int64)
    nclass = len(inp['preds'][idx[0]][0, :])
    s_preds = torch.zeros((ntrial, nclass), device=ops['device'])  # activations, not probabilities
    p_preds = torch.zeros((ntrial, nclass), device=ops['device'])  # probabilities
    block = torch.tensor(inp['blocks'], device=ops['device'])
    for k in range(ntrial):
        targets[k] = block[idx[k]]
        s_preds[k, :] = inp['preds'][idx[k]][0, :]  # this asks to "predict block, given current trial ofer"
        p_preds[k, :] = nn.Softmax(dim=0)(s_preds[k, :])

    loss = nn.CrossEntropyLoss()
    lval = loss(s_preds, targets)

    return lval, [p_preds, targets]  # keeps same type of output as other main costs, so usable direclty in training


def loss_prediction_prob_block_allt(inp, ops):
    """
    cross-entropy loss of current block, at all time

    :param inp: input dictionary. requires net, batchsize, lossinds, nsteps_list, seed
    :param ops: standard ops dict. requires device
    :return:
    """
<<<<<<< HEAD
=======

    # nstep = ops['ctmax']  # unroll width TODO: this is problematic for true episodic RL where ctmax = -1
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
    nstep = len(inp['inputs'])  # number of timepoints

    targets = torch.tensor(inp['blocks'], device=ops['device'], dtype=torch.int64)
    nclass = len(inp['preds'][0][0, :])
    s_preds = torch.zeros((nstep, nclass), device=ops['device'])  # activations, not probabilities
    p_preds = torch.zeros((nstep, nclass), device=ops['device'])  # probabilities

    for k in range(nstep):
        s_preds[k, :] = inp['preds'][k][0, :]  # this asks to "predict block, given current trial ofer"
        p_preds[k, :] = nn.Softmax(dim=0)(s_preds[k, :])

    loss = nn.CrossEntropyLoss()
    lval = loss(s_preds, targets)

    return lval, [p_preds, targets]  # keeps same type of output as other main costs, so usable direclty in training


<<<<<<< HEAD
=======
def loss_prediction_general_allt(inp, ops):
    """
    generalized version of loss_prediction_prob_block_allt. doesn't use wt task to set probs,
    but rather simulates from either 2,3 or 5 state statespace

    :param inp: input dictionary. requires net
    :param ops: standard ops dict. requires device, batchsize, lossinds, nsteps_list, seed
    :return:
    """

    seed = ops['seed_kindergarten']  # TODO: will this be the same data each time then??
    np.random.seed(seed)
    batchsize = ops['batchsize_pred']
    ntrials_pred = ops['ntrials_pred']  # how many trials
    device = torch.device(ops['device'])
    net = inp['net']
    p_optout = 0.3  # optout rate

    if device.type == 'cuda':
        net.cuda()  # send to gpu

    # TODO: add general predicion for 2 state
    if ops['numstates_pred'] == 3:
        inputs, targets, _, _, _, _ = wt_pred.trainingdata(ops, seed=seed, ntrials=ntrials_pred,
                                                           batchsize=batchsize,
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
        stateops = {'vdict': vdict, 'statetrans_fun': statetrans_fun}

        inputs, targets, _, _, _, _ = wt_pred.trainingdata_general(ops, seed=seed, ntrials=ntrials_pred,
                                                                  batchsize=batchsize,
                                                                  useblocks=True, p_optout=p_optout, stateops=stateops)
    else:
        print('wrong type of prediction training data specified. check numstates_pred (2,3[wt task], 5)')
        inputs = None
        targets = None

    # make predictions. send things to correct device
    si = net.begin_state(batchsize=batchsize, device=device)
    inputs = torch.Tensor(inputs).to(device)
    targets = torch.Tensor(targets).to(device)
    outputs, state, net = wt_pred.batchsamples(net, inputs, si, device=device)

    # do inference across entire trial. this is the code from wt_pred.update
    loss_pred = nn.CrossEntropyLoss()
    outputs_pred = outputs[1]
    dout = outputs_pred.shape[-1]
    print('504')
    print(outputs_pred.shape)

    outputs_pred = torch.reshape(outputs_pred, (-1, dout))  # cross entropy has specific input dim requirements
    lossinds_pred = ops['lossinds_pred']  # TODO is this a thing?
    targets_pred = torch.reshape(targets[:, :, lossinds_pred].long(), (-1,))

    lval = loss_pred(outputs_pred, targets_pred)

    nstep = outputs_pred.shape[0]
    p_preds = nn.Softmax(dim = 1)(outputs_pred)
    print('wt_costs 502')
    print(p_preds.shape)

    return lval, [p_preds, targets]  # keeps same type of output as other main costs, so usable direclty in training


>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
def loss_d2m_allt(inp, ops):
    """
    cross-entropy loss of delay to match task. all time points

    :param inp: input dictionary. requires net, batchsize, lossinds, nsteps_list, seed
    :param ops: standard ops dict. requires device
    :return:
    """

    nstep = len(inp['inputs'])  # number of timepoints
    s1 = np.zeros((nstep,))
    s2 = np.zeros((nstep,))

    # create target
    if ops['inputcase'] == 15:
        for j in range(nstep):
            s1[j] = inp['inputs'][j][1][0, 0, -2].cpu().detach().numpy()
            s2[j] = inp['inputs'][j][1][0, 0, -1].cpu().detach().numpy()

        targets = np.array(s1 != s2).astype(int)+1
        targets[s2 == 0] = 0
        targets = torch.tensor(targets, device=ops['device'], dtype=torch.int64)
    else:
        print('this input case is not supported with this loss. check inputcase in ops')
        targets = None

    nclass = len(inp['outputs_sham'][0][0, :])
    s_preds = torch.zeros((nstep, nclass), device=ops['device'])  # activations, not probabilities
    p_preds = torch.zeros((nstep, nclass), device=ops['device'])  # probabilities

    for k in range(nstep):
        s_preds[k, :] = inp['outputs_sham'][k][0, :]  # this asks to "predict block, given current trial ofer"
        p_preds[k, :] = nn.Softmax(dim=0)(s_preds[k, :])

    loss = nn.CrossEntropyLoss()
    lval = loss(s_preds, targets)

    return lval, [p_preds, targets]  # keeps same type of output as other main costs, so usable direclty in training


def loss_d2m_end(inp, ops):
    """
    cross-entropy loss of current delay to match/non-match, but only at second stimulus pres

    :param inp: input dictionary. requires net, batchsize, lossinds, nsteps_list, seed
    :param ops: standard ops dict. requires device
    :return:
    """
    nstep = len(inp['inputs'])  # number of timepoints
    s1 = np.zeros((nstep,))
    s2 = np.zeros((nstep,))

    # create target, extract when stim 2 is nonzero
    if ops['inputcase'] == 15:
        for j in range(nstep):
            s1[j] = inp['inputs'][j][1][0, 0, -2].cpu().detach().numpy()
            s2[j] = inp['inputs'][j][1][0, 0, -1].cpu().detach().numpy()
        s2_idx = np.argwhere(s2 > 0)
        targets = np.array(s1[s2_idx] != s2[s2_idx]).astype(int) + 1
        targets = torch.tensor(targets, device=ops['device'], dtype=torch.int64)

    else:
        print('this input case is not supported with this loss. check inputcase in ops')
<<<<<<< HEAD
        targets = None
        s2_idx = None
=======
        s2_idx = np.nan
        targets = None
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)

    if len(s2_idx) > 0:  # for single entries, squeeze is weird. just select indices
        s2_idx = s2_idx[0, :]
        targets = targets[0, :]

    idx = torch.tensor(s2_idx, device=ops['device'], dtype=torch.int64)

    if len(s2_idx) > 0:
        ntrial = len(idx)

        nclass = len(inp['outputs_sham'][0][0, :])
        s_preds = torch.zeros((ntrial, nclass), device=ops['device'])  # activations, not probabilities
        p_preds = torch.zeros((ntrial, nclass), device=ops['device'])  # probabilities
        for k in range(ntrial):
            s_preds[k, :] = inp['outputs_sham'][idx[k]][0, :]  # this asks to "predict block, given current trial ofer"
            p_preds[k, :] = nn.Softmax(dim=0)(s_preds[k, :])

        loss = nn.CrossEntropyLoss()
        lval = loss(s_preds, targets)
    else:
        lval = torch.tensor(0)
        p_preds = np.nan
        targets = np.nan

    return lval, [p_preds, targets]  # keeps same type of output as other main costs, so usable direclty in training


def loss_d2m(inp, ops):
    """
<<<<<<< HEAD
    full cross-entropy loss of current delay to match/non-match. equally weights endpt and waitint
=======

    full cross-entropy loss of current delay to match/non-match. equally weights endpt and waitint

>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
    :param inp: input dictionary. requires net, batchsize, lossinds, nsteps_list, seed
    :param ops: standard ops dict. requires device
    :return:
    """

    lval_allt, preds_and_targs = loss_d2m_allt(inp, ops)
    lval_end, _, = loss_d2m_end(inp, ops)

<<<<<<< HEAD
    # primarily weight getting endpoitn loss correct, but regularize with not being weird in between samples
=======
    # TODO: decide on the ratio
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
    lval = 0.1*lval_allt + 0.9*lval_end

    return lval, preds_and_targs


# combination costs----------------------------------------------------------------------------------

<<<<<<< HEAD

def loss_actorcritic_regularized_9(inp, ops):
    """
    primary loss function for multi-region networks. mnih RL loss + kind + ent + all-time block prediction
=======
def loss_actorcritic_regularized_9(inp, ops):
    """
    like regularize_7, but all-time block prediction
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
    :param inp: input dict. requires rewards, log_probs, v, net, batchsize, lossinds, nsteps_list, seed
    :param ops: standard ops dict. requires gamma, lambda_supvervised, device
    :return:
    """
    # REINFORCE params
    [L_policy, L_value], _ = loss_actorcritic_minh_iti(inp, ops)

    entropy = inp['entropy']

    # kindergarten params
    lambda_supervised = ops['lambda_supervised']
    L_kindergarten = loss_kindergarten(inp, ops)

    L_ent = -ops['lambda_entropy']*entropy/ops['ctmax']  # entropy is a summed value over timesteps. normalize

    # prediction, but only on blocks
    if ops['useblocks']:
        L_pred, _ = loss_prediction_prob_block_allt(inp, ops)
    else:
        L_pred = torch.tensor(0.0, device=ops['device'], requires_grad=False)
    lambda_pred = ops['lambda_pred']

    Lall = ops['lambda_policy']*L_policy + L_value + lambda_supervised * L_kindergarten + L_ent + lambda_pred * L_pred

    return [Lall], [L_policy, L_value, L_kindergarten, L_ent, L_pred]


def loss_actorcritic_regularized_10(inp, ops):
    """
    like regularize_9, but with sham delay to match loss
    :param inp: input dict. requires rewards, log_probs, v, net, batchsize, lossinds, nsteps_list, seed
    :param ops: standard ops dict. requires gamma, lambda_supvervised, device
    :return:
    """
    # REINFORCE params
    [L_policy, L_value], _ = loss_actorcritic_minh_iti(inp, ops)

    entropy = inp['entropy']

    # kindergarten params
    lambda_supervised = ops['lambda_supervised']
    L_kindergarten = loss_kindergarten(inp, ops)

    L_ent = -ops['lambda_entropy']*entropy/ops['ctmax']  # entropy is a summed value over timesteps. normalize

    # prediction, but only on blocks
    if ops['useblocks']:
        L_pred, _ = loss_prediction_prob_block_allt(inp, ops)
    else:
        L_pred = torch.tensor(0.0, device=ops['device'], requires_grad=False)
    lambda_pred = ops['lambda_pred']

    # delay to match
    lambda_d2m = ops['lambda_d2m']
    L_d2m, _ = loss_d2m(inp, ops)

    Lall = (ops['lambda_policy']*L_policy + L_value) + (lambda_supervised * L_kindergarten + L_ent
                                                        + lambda_pred * L_pred + lambda_d2m * L_d2m)

    return [Lall], [L_policy, L_value, L_kindergarten, L_ent, L_pred, L_d2m]


<<<<<<< HEAD
# simple dictionary to associate name with function
name2fun = {'loss_actorcritic_minh': loss_actorcritic_minh,
=======
def loss_actorcritic_regularized_11(inp, ops):
    """
    like regularize_9, but generic block prediction

    :param inp: input dict. requires rewards, log_probs, v, net, batchsize, lossinds, nsteps_list, seed
    :param ops: standard ops dict. requires gamma, lambda_supvervised, device
    :return:
    """
    # REINFORCE params
    [L_policy, L_value], _ = loss_actorcritic_minh_iti(inp, ops)

    entropy = inp['entropy']

    # kindergarten params
    lambda_supervised = ops['lambda_supervised']
    L_kindergarten = loss_kindergarten(inp, ops)

    L_ent = -ops['lambda_entropy']*entropy/ops['ctmax']  # entropy is a summed value over timesteps. normalize

    # prediction, but only on blocks
    if ops['useblocks']:
        L_pred, _ = loss_prediction_general_allt(inp, ops)
    else:
        L_pred = torch.tensor(0.0, device=ops['device'], requires_grad=False)
    lambda_pred = ops['lambda_pred']

    Lall = ops['lambda_policy']*L_policy + L_value + lambda_supervised * L_kindergarten + L_ent + lambda_pred * L_pred

    return [Lall], [L_policy, L_value, L_kindergarten, L_ent, L_pred]


def loss_actorcritic_regularized_12(inp, ops):
    """
    like regularize_9, (all-time block inference), but separates losses for mem, count, int
    :param inp: input dict. requires rewards, log_probs, v, net, batchsize, lossinds, nsteps_list, seed
    :param ops: standard ops dict. requires gamma, lambda_supvervised, device
    :return:
    """
    # REINFORCE params
    [L_policy, L_value], _ = loss_actorcritic_minh_iti(inp, ops)

    entropy = inp['entropy']

    # kindergarten params
    # list of losses
    L_kindergarten_list = loss_kindergarten(inp, ops, updatefun=wt_kindergarten.update_individual)

    L_ent = -ops['lambda_entropy']*entropy/ops['ctmax']  # entropy is a summed value over timesteps. normalize

    # prediction, but only on blocks
    if ops['useblocks']:
        L_pred, _ = loss_prediction_prob_block_allt(inp, ops)
    else:
        L_pred = torch.tensor(0.0, device=ops['device'], requires_grad=False)
    lambda_pred = ops['lambda_pred']

    # do the losses for each loss individually
    lambda_supervised_list = ops['lambda_supervised_list']
    nsup = len(lambda_supervised_list)
    L_kindergarten = lambda_supervised_list[0]*L_kindergarten_list[0]
    for k in range(1,nsup):
        L_kindergarten = L_kindergarten + lambda_supervised_list[k]*L_kindergarten_list[k]

    Lall = ops['lambda_policy']*L_policy + L_value + L_kindergarten + L_ent + lambda_pred * L_pred

    # make L_aux list
    L_aux = [L_policy, L_value]
    for k in range(nsup):
        L_aux.append(L_kindergarten_list[k])
    L_aux.append(L_ent)
    L_aux.append(L_pred)

    return [Lall], L_aux


# simple dictionary to associate name with function
name2fun = {
            'loss_actorcritic_minh': loss_actorcritic_minh,
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
            'loss_actorcritic_minh_iti': loss_actorcritic_minh_iti,
            'loss_prediction_prob_block_allt': loss_prediction_prob_block_allt,
            'loss_kindergarten': loss_kindergarten,
            'loss_prediction_prob_block': loss_prediction_prob_block,
            'loss_d2m_allt': loss_d2m_allt,
            'loss_d2m_end': loss_d2m_end,
            'loss_d2m': loss_d2m,
            'loss_actorcritic_regularized_9': loss_actorcritic_regularized_9,
<<<<<<< HEAD
            'loss_actorcritic_regularized_10': loss_actorcritic_regularized_10
=======
            'loss_actorcritic_regularized_10': loss_actorcritic_regularized_10,
            'loss_actorcritic_regularized_11': loss_actorcritic_regularized_11,
            'loss_actorcritic_regularized_12': loss_actorcritic_regularized_12
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
            }

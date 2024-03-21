# another auxiliary task: delay to match/non-match
import numpy as np
import torch
from torch import nn
import dynamics.process.rnn.wt_pred as wt_pred


# generate training data
def trainingdata(ops, seed=101, T=100, batchsize=20):
    """
    training set to  perform the delay to match/non-match task
    :param ops: ops dictionary with bits about task
    :param seed: random seed for trials
    :param T: number of timesteps per trial
    :param batchsize: number of independent samples
    :return:
    """
    np.random.seed(seed)
    Tmin = int(ops['tmin_d2m']/ops['dt'])  # minimum number of samples before second stim arrives

    # have 5 different inputs, simialr to inputs of main taskk
    offers = [5, 10, 20, 40, 80]
    inp1_offer = np.random.choice(offers, size=(batchsize,))
    psame = ops['d2m_p']  # proporation of trials that will be match trials
    inp2_offer = np.random.choice(offers, size=(batchsize,))
    sametrials = np.squeeze(np.argwhere(np.random.binomial(1, psame, batchsize) == 1))
    inp2_offer[sametrials] = inp1_offer[sametrials]

    # change the coincindetal matches on non-match trials
    for k in range(batchsize):
        if (not np.isin(k, sametrials)) and (inp2_offer[k] == inp1_offer[k]):
            offers_tmp = [5, 10, 20, 40, 80]
            offers_tmp.pop(offers_tmp.index(inp1_offer[k]))
            inp2_offer[k] = np.random.choice(offers_tmp, size=(1,))

    # delay betweens samples
    delays = np.random.randint(low=Tmin, high=T, size=batchsize)

    # inputs
    inp1 = np.zeros((batchsize, T))
    inp2 = np.zeros((batchsize, T))
    inp1[:, 0] = inp1_offer
    for k in range(batchsize):
        inp2[k, delays[k]] = inp2_offer[k]
    inputs = np.stack((inp1, inp2), axis=2)

    # make targets
    targets = np.zeros((batchsize, T, 1)).astype(int)
    for k in range(batchsize):
        if np.isin(k, sametrials):
            targets[k, delays[k]] = 1
        else:
            targets[k, delays[k]] = 2

    # concatenate with prediction inputs and targets to be size-consistent with inputs/outputs
    ntrial_sham = int(T*ops['dt']/ops['lam'])*10  # mult by factor of 10 to make sure enough data
    inputs_pad, targets_pad, _, _, _, _ = wt_pred.trainingdata(ops, seed=101,
                                                               ntrials=ntrial_sham, batchsize=batchsize,
                                                               useblocks=False, p_optout=0.0)

    inputs = np.concatenate((inputs_pad[:, :T, :4], inputs), axis=2)
    targets = np.concatenate((targets_pad[:, :T, :], targets), axis=2)

    return inputs, targets, inp1_offer, inp2_offer, sametrials


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
    :return net: network
    """

    batchsize, nsteps, _ = inputs.shape
    # parse inputs, mostly for multiregion. TODO: this inputtype is hardcoded.
    inputs = net.parseiputs(inputs, inputcase=15)

    output = net(inputs, si, isITI=False, device=device)  # assume that all models will augment action space for ITI
    state = output['state']  # state of network
    # parse the outputs. not all of them will be needed, unless training on kind+pred+d2m
    # but, good to have it written to support that
    pred = output['activations']  # activations for the block prediction
    outputs_pred = torch.reshape(pred, (batchsize, nsteps, -1))
    outputs_pred.requires_grad_(True)
    outputs_supervised = output['pred_supervised']
    outputs_supervised = torch.reshape(outputs_supervised, (batchsize, nsteps, -1))
    outputs_supervised.requires_grad_(True)
    outpt_d2m = output['output_sham']
    outputs_d2m = torch.reshape(outpt_d2m, (batchsize, nsteps, -1))
    outputs_d2m.requires_grad_(True)

    outputs = [outputs_supervised, outputs_pred, outputs_d2m]

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


def update(updater, outputs, targets, lambda_d2m, lambda_kind, lambda_pred, _, lossinds=None):
    """
    calculates loss and updates weights via an optimizer. includes a portion for d2m, prediction, kindergarten
    :param updater: torch updater, like Adam
    :param outputs: outputs predicted by network
    :param targets: target outputs
    :param lambda_d2m, weight on sham delay 2 match task
    :param lambda_kind: weight on kindergarten portion
    :param lambda_pred: weight on prediction
    :param _: (np.ndarray) inputs to network. not used here
    :param lossinds: (list) which indices of output to use to MSE loss on  kindergarten
    :return: loss_np value of loss for that minibatch
    """

    # default loss indices
    if lossinds is None:
        lossinds = [0, 1, 2]

    loss_supervised = nn.MSELoss()
    loss_pred = nn.CrossEntropyLoss()
    loss_d2m = nn.CrossEntropyLoss()

    # inlude kindergarten loss and prediction. weighted accordingly. HARDCODED INDICES
    lossinds_kind = lossinds
    lossinds_pred = 3
    lossinds_d2m = 4

    outputs_supervised = outputs[0]
    outputs_pred = outputs[1]
    outputs_d2m = outputs[2]
    dout_pred = outputs_pred.shape[-1]
    outputs_pred = torch.reshape(outputs_pred, (-1, dout_pred))  # cross entropy has specific input dim requirements
    dout_d2m = outputs_d2m.shape[-1]
    outputs_d2m = torch.reshape(outputs_d2m, (-1, dout_d2m))
    targets_supervised = targets[:, :, lossinds_kind]
    targets_pred = torch.reshape(targets[:, :, lossinds_pred].long(),  (-1,))
    targets_d2m = torch.reshape(targets[:, :, lossinds_d2m].long(), (-1,))

    lval_kind = loss_supervised(outputs_supervised[:, :, lossinds_kind], targets_supervised)
    lval_pred = loss_pred(outputs_pred, targets_pred)
    lval_d2m = loss_d2m(outputs_d2m, targets_d2m)

    lval = lambda_kind*lval_kind + lambda_pred*lval_pred + lambda_d2m*lval_d2m

    if updater is not None:  # just use to calculate loss, like for mixed loss funs in other trainings
        updater.zero_grad()  # updaters accumulate gradients. zero out
        lval.backward()
        # grad_clipp(net,100) #clip the gradient to prevent explosion. other built-in methods?
        updater.step()
        loss_np = lval.detach().cpu().numpy()  # the value of the loss from that batch

        l_d2m_detach = lval_d2m.detach().cpu().numpy()
        l_pred_detach = lval_pred.detach().cpu().numpy()
        l_kind_detach = lval_kind.detach().cpu().numpy()

        return loss_np, l_d2m_detach, l_pred_detach, l_kind_detach
    else:  # return the tesor form
        return lval, lval_d2m, lval_pred, lval_kind


def update_end(updater, outputs, targets, lambda_d2m, lambda_kind, lambda_pred, inp, lossinds=None):
    """
    only trains on when the second stimulus is presented
    :param updater: torch updater, like Adam
    :param outputs: outputs predicted by network
    :param targets: target outputs
    :param lambda_d2m, weight on sham delay 2 match task
    :param lambda_kind: weight on kindergarten portion
    :param lambda_pred: weight on prediction
    :param inp: (np.ndarray) inputs, which includes stimulus for sham task
    :param lossinds: (list) which indices of output to use to MSE loss on  kindergarten
    :return: loss_np value of loss for that minibatch
    """

    # default loss indices
    if lossinds is None:
        lossinds = [0, 1, 2]

    delay_idx = np.squeeze(np.argwhere(inp[:, :, -1] > 0))[:, 1]
    offers = inp[0, :, 0]
    offer_idx = np.squeeze(np.argwhere(offers > 0))

    loss_supervised = nn.MSELoss()
    loss_pred = nn.CrossEntropyLoss()
    loss_d2m = nn.CrossEntropyLoss()

    # inlude kindergarten loss and prediction. weighted accordingly. HARDCODED INDICES
    lossinds_kind = lossinds
    lossinds_pred = 3
    lossinds_d2m = 4

    outputs_supervised = outputs[0]
    outputs_pred = outputs[1]
    batchsize_d2m = outputs[2].shape[0]
    dout_d2m = outputs[2].shape[-1]
    dout_pred = outputs_pred.shape[-1]

    outputs_d2m = torch.tensor(np.zeros((batchsize_d2m, dout_d2m)))
    targets_d2m = torch.tensor(np.zeros((batchsize_d2m, 1)))
    for k in range(len(delay_idx)):
        outputs_d2m[k, :] = outputs[2][k, delay_idx[k], :]
        targets_d2m[k] = targets[k, delay_idx[k], lossinds_d2m]
    outputs_d2m.requires_grad_(True)
    targets_d2m.requires_grad_(True)

    # select the indices needed to make into 2d
    outputs_pred = torch.reshape(outputs_pred[:, offer_idx, :], (-1, dout_pred))

    outputs_d2m = torch.reshape(outputs_d2m, (-1, dout_d2m))
    targets_supervised = targets[:, :, lossinds_kind]
    targets_pred = torch.reshape(targets[:, offer_idx, lossinds_pred].long(),  (-1,))
    targets_d2m = torch.reshape(targets_d2m.long(), (-1,))

    lval_kind = loss_supervised(outputs_supervised[:, :, lossinds_kind], targets_supervised)
    lval_pred = loss_pred(outputs_pred, targets_pred)
    lval_d2m = loss_d2m(outputs_d2m, targets_d2m)
    lval = lambda_kind*lval_kind + lambda_pred*lval_pred + lambda_d2m*lval_d2m

    if updater is not None:  # just use to calculate loss, like for mixed loss funs in other trainings
        updater.zero_grad()  # updaters accumulate gradients. zero out
        lval.backward()
        # grad_clipp(net,100) #clip the gradient to prevent explosion. other built-in methods?
        updater.step()
        loss_np = lval.detach().cpu().numpy()  # the value of the loss from that batch

        l_d2m_detach = lval_d2m.detach().cpu().numpy()
        l_pred_detach = lval_pred.detach().cpu().numpy()
        l_kind_detach = lval_kind.detach().cpu().numpy()

        return loss_np, l_d2m_detach, l_pred_detach, l_kind_detach
    else:  # return the tesor form
        return lval, lval_d2m, lval_pred, lval_kind


def update_both(updater, outputs, targets, lambda_d2m, lambda_kind, lambda_pred, inp, lossinds=None):
    """
    uses the all-time and end-point time updates together,
    to put equal weight on learning to wait, and correclty respond
    :param updater:
    :param outputs:
    :param targets:
    :param lambda_d2m:
    :param lambda_kind:
    :param lambda_pred:
    :param inp:
    :param lossinds:
    :return:
    """
    if lossinds is None:
        lossinds = [0, 1, 2]

    lval_all, lval_d2m_all, lval_pred, lval_kind = update(None, outputs, targets, lambda_d2m, lambda_kind,
                                                          lambda_pred, inp, lossinds)
    lval_end, lval_d2m_end, _, _ = update_end(None, outputs, targets, lambda_d2m, lambda_kind,
                                              lambda_pred, inp, lossinds)

    lval = 0.1*lval_all + 0.9*lval_end
    lval_d2m = 0.1*lval_d2m_all + 0.9*lval_d2m_end

    if updater is not None:  # just use to calculate loss, like for mixed loss funs in other trainings
        updater.zero_grad()  # updaters accumulate gradients. zero out
        lval.backward()
        # grad_clipp(net,100) #clip the gradient to prevent explosion. other built-in methods?
        updater.step()
        loss_np = lval.detach().cpu().numpy()  # the value of the loss from that batch

        l_d2m_detach = lval_d2m.detach().cpu().numpy()
        l_pred_detach = lval_pred.detach().cpu().numpy()
        l_kind_detach = lval_kind.detach().cpu().numpy()

        return loss_np, l_d2m_detach, l_pred_detach, l_kind_detach
    else:  # return the tesor form
        return lval, lval_d2m, lval_pred, lval_kind


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
    :param updatefun: minimizer, like Adam
    :return:
    """

    beta_kind = ops['beta_supervised_kindd2m']
    beta_pred = ops['beta_pred_d2m']
    beta_d2m = ops['beta_d2m_kindd2m']
    lossinds = ops['lossinds_supervised_kindd2m']

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
        ld2m = []
        grad_norm_k = []

        # initialize outside of loop
        lval_kind = None
        lval_d2m = None

        for k in range(nbatch):
            inputs_k = inputs[:, ind:ind + nsteps, :]
            targets_k = targets[:, ind:ind + nsteps, :]

            # generate samples
            outputs, state, net = batchsamples(net, inputs_k, state, device=device)

            # update
            loss_k, lval_d2m, lval_pred, lval_kind = updatefun(updater, outputs, targets_k, beta_d2m, beta_kind,
                                                               beta_pred, inp, lossinds)
            ltot_train_k.append(loss_k)
            lpred_k.append(lval_pred)
            ld2m.append(lval_d2m)

            # track gradient
            abs_sum = np.sum([abs(np.sum(k.grad.cpu().numpy())) for k in net.parameters()])
            grad_norm_k.append(abs_sum)

            ind += 1

        ltot_train.append(ltot_train_k)
        grad_norm.append(grad_norm_k)

        outptdict['epoch'] = epoch
        outptdict['isdone'] = False

        if (epoch + 1) % 10 == 0:
            # print(np.mean(ltot_train_k), end='\r')
            print([np.mean(ltot_train_k), np.mean(ld2m), np.mean(lpred_k), lval_kind])

        # do a convergence stopping test
        if stoptype == 'converge':
            loss_av = np.mean(lval_d2m)

            if loss_av <= loss_best:
                loss_best = loss_av
                # params_best = net.parameters()
                checkind = 0
            else:
                checkind += 1

            if checkind > stopargs:
                print('stopping early. no improvement compared to average of last ' + str(stopargs) + 'epochs')
                outptdict['isdone'] = True
                break
        elif stoptype == 'lowlim':
            loss_av = np.mean(lval_d2m)

            if loss_av < stopargs:
                print('stopping early. hit a lower limit for stopping')
                outptdict['isdone'] = True
                break

    return ltot_train, grad_norm, outptdict

# sets of generic kindergarten costs that can we used with wt_protocols.train_generic method

# these loss functions should take the general form:
# loss, outputs = task.loss(net, ops, si)
# where loss is the tensor form of the lsos, and outputs is a dictionary of floats
# the form of outputs is not set in stone yet. but probably inputs and outputs of network over time

import torch
from dynamics.process.rnn import wt_kindergarten, wt_pred
import numpy as np
from torch import nn


def loss_supervised_simple(net, ops, si, batchsize, seed=101):
    """base loss function for supervised losses. memory, count, stim int. but only for one trial
    @param net: (torch.nn) network
    @param ops: (dict) of all training options
    @param si: (list) initial state tensors for hidden units
    @param batchsize: (int) batchsize dim size, for inputs
    @param seed: (int) for random num generator
    @return:"""

    device = torch.device(ops['device'])
    lossinds = ops['lossinds_kindergarten']  # which indices from supervised output to include in loss
    nsteps_simple = ops['nsteps_simple_kindergarten']

    if ops['ismultiregion']:
        din = np.max(net.din)  # for multiregion
    else:
        din = net.din  # input size

    # generate training data
    inputs, targets, _, _ = wt_kindergarten.trainingdata_simple(
        seed=seed, nsteps=nsteps_simple, batchsize=batchsize, din=din, ops=ops)
    inputs_supervised = torch.Tensor(inputs).to(device)
    targets_supervised = torch.Tensor(targets).to(device)

    # process with RNN
    outputs_supervised, _, _ = wt_kindergarten.batchsamples(net, inputs_supervised, si, device)

    # calculate MSE for 3 tasks
    supervised_loss = wt_kindergarten.update_individual(updater=None, outputs=outputs_supervised,
                                                        targets=targets_supervised, lossinds=lossinds)

    return supervised_loss, [inputs_supervised, targets_supervised, outputs_supervised]


def loss_supervised_hard(net, ops, si, batchsize, seed=101):
    """base loss function for supervised losses. memory, count, stim int. for multiple trials
    @param net: (torch.nn) network
    @param ops: (dict) of all training options
    @param si: (list) initial state tensors for hidden units
    @param batchsize: (int) batchsize dim size, for inputs
    @param seed: (int) for random num generator
    @return:"""

    device = torch.device(ops['device'])
    lossinds = ops['lossinds_kindergarten']  # which indices from supervised output to include in loss
    nsteps_int_list = ops['nsteps_list_int']

    if ops['ismultiregion']:
        din = np.max(net.din)  # for multiregion
    else:
        din = net.din  # input size

    # generate training data
    inputs, targets, _, _ = wt_kindergarten.trainingdata_intermediate(
        seed=seed, nsteps_list=nsteps_int_list, batchsize=batchsize, din=din,
        highvartrials=ops['highvartrials_kind'], ops=ops)
    inputs_supervised = torch.Tensor(inputs).to(device)
    targets_supervised = torch.Tensor(targets).to(device)

    # process with RNN
    outputs_supervised, _, _ = wt_kindergarten.batchsamples(net, inputs_supervised, si, device)

    # calculate MSE for 3 tasks
    supervised_loss = wt_kindergarten.update_individual(updater=None, outputs=outputs_supervised,
                                                        targets=targets_supervised, lossinds=lossinds)

    return supervised_loss, [inputs_supervised, targets_supervised, outputs_supervised]


# derivative losses of that supervised loss
def loss_memory_simple(net, ops, si, batchsize, seed=101):
    """
    memory task
    @param net: (torch.nn) network
    @param ops: (dict) of all training options
    @param si: (list) initial state tensors for hidden units
    @param batchsize: (int) batchsize dim size, for inputs
    @param seed: (int) for random num generator
    @return: tensor for loss
    """
    loss, outpts = loss_supervised_simple(net, ops, si, batchsize, seed)
    return loss[0], outpts


def loss_memory_hard(net, ops, si, batchsize, seed=101):
    """
    memory task
    @param net: (torch.nn) network
    @param ops: (dict) of all training options
    @param si: (list) initial state tensors for hidden units
    @param batchsize: (int) batchsize dim size, for inputs
    @param seed: (int) for random num generator
    @return: tensor for loss
    """
    loss, outpts = loss_supervised_hard(net, ops, si, batchsize, seed)
    return loss[0], outpts


def loss_count_simple(net, ops, si, batchsize, seed=101):
    """
        counting task
        @param net: (torch.nn) network
        @param ops: (dict) of all training options
        @param si: (list) initial state tensors for hidden units
        @param batchsize: (int) batchsize dim size, for inputs
        @param seed: (int) for random num generator
        @return: tensor for loss
        """
    loss, outpts = loss_supervised_simple(net, ops, si, batchsize, seed)
    return loss[1], outpts


def loss_count_hard(net, ops, si, batchsize, seed=101):
    """
        counting task
        @param net: (torch.nn) network
        @param ops: (dict) of all training options
        @param si: (list) initial state tensors for hidden units
        @param batchsize: (int) batchsize dim size, for inputs
        @param seed: (int) for random num generator
        @return: tensor for loss
        """
    loss, outpts = loss_supervised_hard(net, ops, si, batchsize, seed)
    return loss[1], outpts


def loss_integration_simple(net, ops, si, batchsize, seed=101):
    """
        stimulus integration task
        @param net: (torch.nn) network
        @param ops: (dict) of all training options
        @param si: (list) initial state tensors for hidden units
        @param batchsize: (int) batchsize dim size, for inputs
        @param seed: (int) for random num generator
        @return: tensor for loss
        """
    loss, outpts = loss_supervised_simple(net, ops, si, batchsize, seed)
    return loss[2], outpts


def loss_integration_hard(net, ops, si, batchsize, seed=101):
    """
        stimulus integration task
        @param net: (torch.nn) network
        @param ops: (dict) of all training options
        @param si: (list) initial state tensors for hidden units
        @param batchsize: (int) batchsize dim size, for inputs
        @param seed: (int) for random num generator
        @return: tensor for loss
        """
    loss, outpts = loss_supervised_hard(net, ops, si, batchsize, seed)
    return loss[2], outpts


def loss_inference_base(net, ops, si, batchsize, seed=101):
    """
    block inference loss. calculates for both all time points, and trial start
    @param net: (torch.nn) network
    @param ops: (dict) of all training options
    @param si: (list) initial state tensors for hidden units
    @param batchsize: (int) batchsize dim size, for inputs
    @param seed: (int) for random num generator
    @return: tensor for loss
    """

    np.random.seed(seed)
    ntrials_pred = ops['ntrials_pred']  # how many trials
    device = torch.device(ops['device'])
    p_optout = 0.2  # optout rate  for

    # TODO: needed?
    if device.type == 'cuda':
        net.cuda()  # send to gpu

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
    inputs = torch.Tensor(inputs).to(device)
    targets = torch.Tensor(targets).to(device)
    outputs, state, net = wt_pred.batchsamples(net, inputs, si, device=device)

    # do inference across entire trial. this is the code from wt_pred.update
    loss_pred = nn.CrossEntropyLoss()
    outputs_pred = outputs[1]
    dout = outputs_pred.shape[-1]

    outputs_pred = torch.reshape(outputs_pred, (-1, dout))  # cross entropy has specific input dim requirements
    lossinds_pred = ops['lossinds_pred']  # TODO is this a thing?
    targets_pred = torch.reshape(targets[:, :, lossinds_pred].long(), (-1,))

    lval = loss_pred(outputs_pred, targets_pred)

    # trial start loss
    offers = inputs[0, :, 0].cpu().detach().numpy()
    outputs_pred = outputs[1]
    offer_idx = np.squeeze(np.argwhere(offers > 0))
    outputs_pred = torch.reshape(outputs_pred[:, offer_idx, :], (-1, dout))  # specific input dim requirements
    targets_pred = torch.reshape(targets[:, offer_idx, lossinds_pred].long(), (-1,))
    lval_pred_tstart = loss_pred(outputs_pred, targets_pred)

    return [lval, lval_pred_tstart], [inputs, targets_pred, outputs_pred]


def loss_inference_allt(net, ops, si, batchsize, seed = 101):
    """
    block inference loss at trial start only
    @param net: (torch.nn) network
    @param ops: (dict) of all training options
    @param si: (list) initial state tensors for hidden units
    @param batchsize: (int) batchsize dim size, for inputs
    @param seed: (int) for random num generator
    @return: tensor for loss
    """
    loss, outpts = loss_inference_base(net, ops, si, batchsize, seed)
    return loss[0], outpts


def loss_inference_trialstart(net, ops, si, batchsize, seed = 101):
    """
    block inference loss at trial start only
    @param net: (torch.nn) network
    @param ops: (dict) of all training options
    @param si: (list) initial state tensors for hidden units
    @param batchsize: (int) batchsize dim size, for inputs
    @return: tensor for loss
    """
    loss, outpts = loss_inference_base(net, ops, si, batchsize, seed)
    return loss[1], outpts


# lossdict = {'wt_costs_generic.loss_memory_simple': loss_memory_simple,
#            'wt_costs_generic.loss_memory_hard': loss_memory_hard,
#            'wt_costs_generic.loss_count_simple': loss_count_simple,
#            'wt_costs_generic.loss_count_hard': loss_count_hard,
#            'wt_costs_generic.loss_integration_simple': loss_integration_simple,
#            'wt_costs_generic.loss_integration_hard': loss_integration_hard,
#            'wt_costs_generic.loss_inference_allt': loss_inference_allt,
#            'wt_costs_generic.loss_inference_trialstart': loss_inference_trialstart
#            }

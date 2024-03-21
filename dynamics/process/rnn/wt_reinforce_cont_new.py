# continuous form of the wait time task

import torch
from torch.distributions.categorical import Categorical
from dynamics.utils.utils import displ, memcheck, setinputs
import numpy as np
import dynamics.process.rnn.wt_envs as wt_envs
import gc
import matplotlib.pyplot as plt


def simulate_torch_rnn_cont(net, state, env, nstep=100, inputcase=11, device=None, force_action=False,
                            batchsize=1, snr=0.0):
    """
    simulates across chunks of time, rather than trials. for continous environments. not episodic
    :param net: RNN.module object
    :param state: state of network
    :param env: environment object
    :param nstep: (int) number of steps per minibatch. if -1: use trials as minibatches for episodic RL
    :param inputcase: (int) which types of inputs supplied to network. in utils.py.
    :param device: device for pytorch run
    :param force_action: if supplied, forces action chosen from policy
    :param batchsize: (int) number of samples in minibatch. for us, almost always 1
    :param snr: (float, or list) signal to noise ratio for each input
    :return:
    """

    if device is None:
        device = torch.device('cpu')

    # Outputs updated at each step
    rewards = []  # GRADIENT
    pi = []  # the policy at every time step. GRADIENT
    values = []  # the output value
    actions = []  # action at each timestep. GRADIENT
    log_probs = []  # the log probability of take action
    states = []  # keep a running tally of the states
    inputs = []  # the inputs to the netwrok at each step. used for prediction
    noises = []  # the noise on the inputs
    preds = []  # predictions at each time step
    outputs_supervised = []  # supervised outputs at each time step
    outputs_sham = []  # outputs for additional, unhelpful tasks
    blocks = []  # blocks at each time step
    entropy = 0  # running tally of entropy
    isdone = None

    if nstep == -1:
        usetrials = True
        nstep = 5000  # add buffer for ITI
        N = env.n  # trial number
    else:
        usetrials = False
        N = None

    # iterate over time
    for k in range(nstep):

        # deal with inputs
        # grab set inputs: volume, timestep, reward rate, reward, action
        inp, inp_nn = setinputs(env.trial['v'], env.trial['t'], env.trial['w'],
                                env.trial['rewprev'], env.trial['aprev'],
                                env.trial['block'], nstep, inputcase, device, snr,
                                env.trial['auxinps'])

        # expand dims here. copy same inputs across batches
        inputs_tmp = torch.unsqueeze(torch.unsqueeze(inp, 0), 0)
        inputs_nn_tmp = torch.unsqueeze(torch.unsqueeze(inp_nn, 0), 0)
        inp = torch.zeros((batchsize, inputs_tmp.shape[1], inputs_tmp.shape[2]), device=device)
        inp_nn = torch.zeros((batchsize, inputs_nn_tmp.shape[1], inputs_nn_tmp.shape[2]), device=device)
        for m in range(batchsize):
            inp[m, :, :] = inputs_tmp
            inp_nn[m, :, :] = inputs_nn_tmp
        # cast to double
        inp = inp.to(torch.float32)
        inp_nn = inp_nn.to(torch.float32)

        # parse, for network-specific ordering of inputs. for multiregion
        inp = net.parseiputs(inp, inputcase)
        inp_nn = net.parseiputs(inp_nn, inputcase)

        inputs.append(inp_nn)  # save noise-free version
        # save yoursalf some space. only append if noise is real
        if isinstance(snr, list):
            if sum(snr) > 0.0:
                noises.append(inp)
        elif snr > 0.0:
            noises.append(inp)  # save noise as well

        # handle state
        if isinstance(state[0], tuple) or isinstance(state[0], list):  # a multi-region state
            [[n.to(torch.float32) for n in m] for m in state]
        else:
            [m.to(torch.float32) for m in state]

        blocks.append(env.trial['block'])

        # propagate and assign outputs
        output = net(inp, state, isITI=env.beginITI, device=device)  # assume augmented action space for ITI
        paction_all = output['policy']  # prob of actions
        state = output['state']  # state of network
        value = output['value']  # value of state
        output_supervised = output['pred_supervised']  # supervised output predicitons for kindergarten
        pred_activations = output['activations']  # prediction activations for block or trial inference
        output_sham = output['output_sham']

        # for now, only grab action of first element of batch
        paction = torch.unsqueeze(paction_all[0, :], 0)

        # track the state. do it as list comprehension for LSTMS
        if isinstance(state[0], tuple):  # multiregion LSTM
            states.append([[m.detach() for m in sk] for sk in state])
        elif isinstance(state, tuple) or isinstance(state, list):  # LSTM or mr RNN
            states.append([sk.detach() for sk in state])
        else:  # vanilla RNN and GRU
            states.append([state.detach()])

        # action
        m = Categorical(paction)

        if force_action:  # a forcing mechanism to wait. used in non-catch training, and inference
            action = torch.tensor([0], device=device)  # wait
        else:
            action = m.sample()
        logprob = m.log_prob(action)
        pi.append(paction)
        actions.append(action)
        preds.append(pred_activations)
        outputs_supervised.append(output_supervised)
        outputs_sham.append(output_sham)
        values.append(value)
        log_probs.append(logprob)
        entropy += m.entropy()

        # interact with environment
        rew_t = env.step(action)
        rewards.append(rew_t)

        if usetrials and env.n > N:
            isdone = True
            break
        else:
            isdone = False

    output = {'inputs': inputs, 'rewards': rewards, 'actions': actions, 'pi': pi, 'log_probs': log_probs,
              'values': values, 'net': net, 'states': states, 'entropy': entropy, 'done': isdone, 'preds': preds,
              'blocks': blocks, 'noises': noises, 'outputs_supervised': outputs_supervised,
              'outputs_sham': outputs_sham}

    # delete
    del states
    del inputs
    gc.collect()

    return output


def session_torch_cont(ops, net, optimizer, costfun, device):
    """
    the session-level simulation. calls minibatch simulations, does backprop, gathers outputs
    :param ops:
    :param net:
    :param optimizer: either a torch optimizer, or list of uch objects for multiregion
    :param costfun: either a loss function declaration in wt_costs.py, or list of such objects for multiregion
    :param device: torch device
    :return:
    """

    # reproducibility
    seed = ops['seed']
    np.random.seed(seed)  # for reproducibility
    torch.manual_seed(seed)  # I think this was the required random seed that I needed

    # timing and sizing stuff stuff
    tmesh = np.linspace(0, ops['T'], int(ops['T'] / ops['dt']) + 1)  # time bins
    ns = ops['ns']  # number of trials
    nparam = 0  # network  parameters
    for k in net.parameters():
        nparam += np.prod(k.shape)

    plot = ops['plot']

    # set up save vectors
    # trial-level results
    w_all = []  # the penalty term on each trial
    blocks = []  # the block type of each trial
    outcomes = []
    wt = []  # the index (of tmesh) of time waited per trial
    iscatch = []  # is a trial a catch trial
    rewvols = []  # size of volume reward
    rewards_pertrial = []  # total reward accumulated per trial
    reward_delay = []

    # every timestep results. will be a single dimension that can be parsed into trials later
    states = []  # running tally of the hidden state
    inputs = []  # keep inputs to network if keeping states
    pi = []  # policy: prob of each action
    values = []  # value of state
    actions = []  # actions taken
    rewards = []  # experienced reward each timestep

    # every optimization step results
    loss_list = []  # the losses from each trial. can themselves be a list on each trial
    loss_list_aux = []  # auxiliary losses not used to in training. like kindergarten
    theta_all = []  # copy of the parameters. in numpy format. THIS IS LARGE
    grad_all = []  # gradient norm per optimization step
    rewardrate_pergradstep = []  # average reward per unit time at each gradient step
    preds = []  # predictions, if applicable
    outputs_supervised = []  # supervised outputs, if applicable
    outputs_sham = []  # outputs for additional, unhelpful costs
    annealed_weights = []  # loss weights. will potentially change each step with annealing

    # global
    reset_idx = []  # when state resets

    # initailize a wait-time environment
    envtype = ops['envtype']
    env = wt_envs.envdict[envtype](ops)
    state = net.begin_state(device=device, batchsize=ops['batchsize'])

    # start looping over trials
    disp_thresh = 1000  # iteration frequency to plot or display stuff
    state_reset_thresh = ops['state_resetfreq']  # how often (in trials) to reset the state. 0 = persist

    while env.n < ns:

        if env.n > disp_thresh:
            displ(env.n, 4, ops['displevel'])
            # displ(memcheck(device), 4, ops['displevel'])

            if plot:  # plot has a placeholder for reward rate. shoudl I add to env?
                chkpt(loss_list, grad_all[:], env.rewards_pertrial, env.rewards_pertrial)

            disp_thresh += 1000

        displ(env.n, 5, ops['displevel'])

        # simulate a batch of time
        outputs = simulate_torch_rnn_cont(
            net, state, env, nstep=ops['ctmax'], inputcase=ops['inputcase'], device=device,
            force_action=ops['nocatch_force'], snr=ops['snr'])

        # do periodic state resetting, or probabilistic ones
        if ops['p_statereset'] < 1.0:  # probabilistic reset
            if np.random.binomial(1, ops['p_statereset']) == 1:
                state = net.begin_state(device=device, batchsize=ops['batchsize'])
                reset_idx.append(env.n)
        else:
            if ops['state_resetfreq'] > 0:
                if env.n > state_reset_thresh:
                    state = net.begin_state(device=device, batchsize=ops['batchsize'])
                    state_reset_thresh += ops['state_resetfreq']
                    reset_idx.append(env.n)
                else:
                    state = outputs['states'][-1]  # last state is first of next round
            else:
                state = outputs['states'][-1]  # last state is first of next round

        # update params, if weights not frozen
        if not ops['freeze']:
            # call loss function and step
            L, L_aux = costfun(outputs, ops)
            optimizer.zero_grad()
            loss = torch.stack(L[:]).sum()
            loss.backward()
            optimizer.step()

            # random update needed for continuous control
            if costfun.__name__ == 'loss_actorcritic_cont':
                ops['Rav'] = L_aux[0]
        else:
            L, L_aux = costfun(outputs, ops)

        # update stuff if tracking
        if not ops['trainonly']:

            # update per-gradient step stuff
            loss_list.append([a.detach().cpu().numpy() for a in L])
            loss_list_aux.append([a.detach().cpu().numpy() for a in L_aux])
            # keep anneal vals
            loss_weights = [ops['lambda_policy'], ops['lambda_value'],  ops['lambda_entropy'],
                            ops['lambda_pred'], ops['lambda_supervised'], ops['lambda_d2m']]

            annealed_weights.append(loss_weights)

            pi.append([k.detach().cpu().numpy() for k in outputs['pi']])
            if outputs['values'][0] is not None:
                values.append([k.detach().cpu().numpy() for k in outputs['values']])
            actions.append([k.detach().cpu().numpy() for k in outputs['actions']])
            rewards.append(outputs['rewards'])

            rewardrate_pergradstep.append(np.mean(outputs['rewards']))
            if outputs['preds'][0] is not None:
                preds.append([k.detach().cpu().numpy() for k in outputs['preds']])
                outputs_supervised.append([k.detach().cpu().numpy() for k in outputs['outputs_supervised']])

                test = np.array([k.detach().cpu().numpy() for k in outputs['outputs_sham']])
                if np.sum(~np.isnan(test)) > 0:
                    outputs_sham.append([k.detach().cpu().numpy() for k in outputs['outputs_sham']])

            # calculate gradient
            if not ops['freeze']:
                grad_k = 0.0
                for k in net.parameters():
                    grad_k += np.sum(k.grad.detach().cpu().numpy() ** 2)
                grad_k = np.sqrt(grad_k)
                grad_all.append(grad_k)

            # track desired things
            if ops['trackstate']:
                states.append(outputs['states'])

            inputs.append(outputs['inputs'])

            if ops['tracktheta']:
                theta = np.array([])
                for k in net.parameters():
                    theta_k = k.detach().cpu().numpy().flatten()
                    theta = np.concatenate((theta, theta_k))
                theta_all.append(theta)

        # clear from memory
        _,  gpumem = memcheck(device, clearmem=False)
        if gpumem[2] > 10.0:
            print(gpumem)
            print('memory getting large. deleting some stuff')
            del outputs
            gc.collect()
            if ops['device'] == 'cuda':
                torch.cuda.empty_cache()
            print(gpumem)

    # update trial-level outputs at end
    if not ops['trainonly']:
        blocks = [k['block'] for k in env.trials]
        outcomes = [k['outcome'] for k in env.trials]
        wt = [k['t'] for k in env.trials]
        iscatch = [k['iscatch'] for k in env.trials]
        rewvols = [k['v'] for k in env.trials]
        w_all = [k['w'] for k in env.trials]
        rewards_pertrial = env.rewards_pertrial
        reward_delay = [k['rewardDelay'] for k in env.trials]

    returndict = {
        'rewvols': rewvols, 'iscatch': iscatch, 'blocks': blocks,
        'outcomes': outcomes, 'actions': actions, 'wt': wt, 'pi': pi, 'values': values,
        'grad_all': grad_all, 'rewards': rewards, 'tmesh': tmesh, 'w_all': w_all, 'theta': theta_all,
        'loss': loss_list, 'loss_aux': loss_list_aux, 'states': states, 'inputs': inputs,
        'rewards_pertrial': rewards_pertrial, 'env': env, 'rewardrate_pergradstep': rewardrate_pergradstep,
        'reward_delay': reward_delay, 'ops': ops, 'preds': preds, 'outputs_supervised': outputs_supervised,
        'state_reset_idx': reset_idx,
        'annealed_weights': annealed_weights, 'outputs_sham': outputs_sham}

    return returndict


def chkpt(loss, gradient, rewards, wall, rows=1, cols=4, figsize=(10, 3)):
    fig, ax = plt.subplots(rows, cols, figsize=figsize)
    if len(loss[0]) == 2:  # includes policy and value loss
        ax[0].plot([ll[0] for ll in loss], 'k')
        ax[0].set_title('loss')
        ax[0].set_ylabel('policy')
        ax[0].set_ylabel('grad steps')
        axx = ax[0].twinx()
        axx.plot([ll[1] for ll in loss], 'r')
        axx.set_ylabel('value', color='red')
        [t.set_color('red') for t in axx.yaxis.get_ticklines()]
        [t.set_color('red') for t in axx.yaxis.get_ticklabels()]
    else:
        ax[0].plot([ll[0] for ll in loss], 'k')
        ax[0].set_title('loss')
        ax[0].set_xlabel('policy')

    ax[1].plot(gradient)
    ax[1].set_title('gradnorm')
    ax[1].set_xlabel('grad steps')
    ax[2].plot(rewards)
    ax[2].set_title('rewards')
    ax[2].set_xlabel('trials')
    ax[3].plot(wall)
    ax[3].set_xlabel('trials')
    ax[3].set_title('reward rate')
    plt.tight_layout()
    plt.show()

    return

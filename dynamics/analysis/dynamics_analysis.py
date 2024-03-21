import numpy as np
from dynamics.process.rnn import wt_nets, linearize
from dynamics.process.rnn import parse, parse_state
from dynamics.utils import utils
import torch
from torch import nn
from sklearn.decomposition import PCA
from scipy.io import savemat


def dynamics_preprocess_suite(modelname, rdict, simtype='task', dynops=None):
    """
    The collection of simulation/load-->save-->preprocessing-->pca things to get ready
    for heavier dynamics processing
    @param modelname: (str) .model file name
    @param rdict: (dict) from simulation of wt_protocols.sim_1k()
    @param simtype: (str) 'task','kind','inf'
    @param dynops: (dict) containing ops for how the prepcoessing shoudl go. see below for defaults
    @return:
    """

    # options for the simulation suite
    if dynops is None:
        dynops = {'numhidden': [256, 256], 'resim': False, 'configname': None, 'epoch': 'wait',
                  'zeromean': False}

    # some hard coded stuff
    configname = dynops['configname']
    numhidden = dynops['numhidden']  # number of hidden units per layer

    # set basic args
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
    else:
        ops, _ = utils.parse_configs(configname)

    device = torch.device('cpu')

    # load model
    nd = wt_nets.netdict[ops['modeltype']]  # dictionary of models and size params
    netfun = nd['net']
    netseed = 101
    net = netfun(din=nd['din'], dout=nd['dout'], num_hiddens=numhidden, seed=netseed, rnntype=ops['rnntype'],
                 snr=ops['snr_hidden'], dropout=ops['dropout'], rank=ops['rank'])
    net.load_state_dict(torch.load(modelname, map_location=device.type))

    if simtype == 'task':
        sflat, inp, tdict = parse.get_trialtimes(rdict)
        inp = np.array(
            [np.squeeze(np.squeeze(k[0].detach().numpy())) for k in inp])  # correct for using parse_inputs on inp
        blocks = np.array(rdict['blocks'])
        outcomes = np.array(rdict['outcomes'])
        offers = inp[inp[:, 0] > 0, 0]
        output = rdict['outputs_supervised']
        preds = rdict['preds']
        output_flat = []
        pred_flat = []
        for j in output:
            output_flat.extend(j)
        for j in preds:
            pred_flat.extend(j)

        behdict = {'offers': offers, 'output': output_flat, 'blocks': blocks, 'tdict': tdict,
                   'outcomes': outcomes, 'preds': pred_flat, 'task': 'task'}

        # preprocess for dynamics. flattening data and getting event timings
        print('postprocessing simulation data for dynamics')
        print('epoch ' + dynops['epoch'])
        statedict, behdict = parse_state.postprocess4dynamics(sflat, behdict, warpstate=False, Ttrial=256 * 3,
                                                              zeromean=dynops['zeromean'], epoch=dynops['epoch'])

        # build trial type masks
        print('building trial masks and getting conditional PSTHs')
        blocks_masked = behdict['blocks_masked']
        outcomes_masked = behdict['outcomes_masked']
        offers_masked = behdict['offers_masked']
        vals = np.unique(offers_masked)

        # masks for trials types.
        # Ordering........
        # 5 volumes in mixed blocks,
        # R=20 block types (M,H,L),
        # R=20 optout,
        # R=20 hit,
        # all trials
        vmask = [np.argwhere((blocks_masked == 0) & (offers_masked == k))[:, 0] for k in
                 vals]
        vmask.append(np.argwhere((blocks_masked == 0) & (offers_masked == vals[2]))[:, 0])
        vmask.append(np.argwhere((blocks_masked == 1) & (offers_masked == vals[2]))[:, 0])
        vmask.append(np.argwhere((blocks_masked == 2) & (offers_masked == vals[2]))[:, 0])
        vmask.append(np.argwhere((outcomes_masked == 0) & (offers_masked == vals[2]))[:, 0])
        vmask.append(np.argwhere((outcomes_masked == 1) & (offers_masked == vals[2]))[:, 0])
        vmask.append(np.argwhere(outcomes_masked < 10)[:, 0])  # marginal, all trials

        dat2fit, psths, nsvec, samps = parse_state.condpsth_bycelltype(vmask, statedict, behdict, nsamps=10,
                                                                       sampdat=True,
                                                                       epoch=dynops['epoch'])

    elif simtype == 'kind':

        behdict = rdict['behdict']
        tdict = behdict['tdict']
        sflat = rdict['sflat']
        inp = rdict['inp']

        # create a behdict out of these things
        otypes = np.unique(behdict['offers'])
        vmask = [np.argwhere(behdict['offers'] == k)[:, 0] for k in otypes]

        # create the dat2fit dictionary
        ns = len(sflat)
        nneur = sflat[0][0][0].shape[-1]
        ofc_h = np.zeros((ns, nneur))
        ofc_c = np.zeros((ns, nneur))
        str_h = np.zeros((ns, nneur))
        str_c = np.zeros((ns, nneur))

        for j in range(ns):
            ofc_h[j, :] = sflat[j][0][0][0, 0, :].detach().numpy()
            ofc_c[j, :] = sflat[j][0][0][0, 0, :].detach().numpy()
            str_h[j, :] = sflat[j][0][0][0, 0, :].detach().numpy()
            str_c[j, :] = sflat[j][0][0][0, 0, :].detach().numpy()

        dat2fit = {'ofc_h': ofc_h, 'ofc_c': ofc_c, 'str_h': str_h, 'str_c': str_c}
        statedict = dat2fit

    elif simtype == 'inf':
        behdict = rdict['behdict']
        tdict = behdict['tdict']
        sflat = rdict['sflat']
        inp = rdict['inp']

        # create a behdict out of these things!
        otypes = np.unique(behdict['offers'])
        vmask = [np.argwhere((behdict['offers'] == k) & (behdict['blocks'] == 0))[:, 0] for k in otypes]
        vmask.append(np.argwhere(behdict['blocks'] == 0)[:, 0])
        vmask.append(np.argwhere(behdict['blocks'] == 1)[:, 0])
        vmask.append(np.argwhere(behdict['blocks'] == 2)[:, 0])

        # create the dat2fit dictionary
        ns = len(sflat)
        nneur = sflat[0][0][0].shape[-1]
        ofc_h = np.zeros((ns, nneur))
        ofc_c = np.zeros((ns, nneur))
        str_h = np.zeros((ns, nneur))
        str_c = np.zeros((ns, nneur))

        for j in range(ns):
            ofc_h[j, :] = sflat[j][0][0][0, 0, :].detach().numpy()
            ofc_c[j, :] = sflat[j][0][0][0, 0, :].detach().numpy()
            str_h[j, :] = sflat[j][0][0][0, 0, :].detach().numpy()
            str_c[j, :] = sflat[j][0][0][0, 0, :].detach().numpy()

        dat2fit = {'ofc_h': ofc_h, 'ofc_c': ofc_c, 'str_h': str_h, 'str_c': str_c}
        statedict = dat2fit

    else:
        print('going to sim other types here. not supported yet')
        dat2fit = None
        sflat = None
        tdict = None
        vmask = None
        inp = None
        statedict = None
        behdict = None

    # do pca
    datlist_ofc = np.concatenate((dat2fit['ofc_h'], dat2fit['ofc_c']), axis=1)
    datlist_str = np.concatenate((dat2fit['str_h'], dat2fit['str_c']), axis=1)
    pca_ofc = PCA()
    pca_ofc.fit(datlist_ofc)
    dOFC = np.argwhere(np.cumsum(pca_ofc.explained_variance_ratio_) > 0.90)[0, 0]
    pca_str = PCA()
    pca_str.fit(datlist_str)
    dSTR = np.argwhere(np.cumsum(pca_str.explained_variance_ratio_) > 0.90)[0, 0]
    print('Dim')
    print(dOFC + 1)
    print(dSTR+1)

    # prepare inputs for dynamics simulations
    if simtype == 'task' or simtype == 'inf':
        keys = ['5', '10', '20', '40', '80', 'mixed', 'high', 'low', 'optout', 'hit', 'all']
    elif simtype == 'kind':
        keys = ['5', '10', '20', '40', '80']
    else:
        print('simtype not defined correclty. did you say "pred" instead of "inf" again? ')
        keys = None

    krange = range(len(keys))

    # create samples and rename keys
    samps_pcdict_ofc, sampsdict_ofc = parse_state.pca_samples(sflat, tdict, vmask, pca_use=pca_ofc, idx=0)
    samps_pcdict_str, sampsdict_str = parse_state.pca_samples(sflat, tdict, vmask, pca_use=pca_str, idx=1)
    for k in krange:
        samps_pcdict_ofc[keys[k]] = samps_pcdict_ofc.pop(k)
        samps_pcdict_str[keys[k]] = samps_pcdict_str.pop(k)
        sampsdict_ofc[keys[k]] = sampsdict_ofc.pop(k)
        sampsdict_str[keys[k]] = sampsdict_str.pop(k)

    # final output function.
    rd = {}
    rd['samps_pcdict_str'] = samps_pcdict_str
    rd['samps_pcdict_ofc'] = samps_pcdict_ofc
    rd['sampsdict_ofc'] = sampsdict_ofc
    rd['sampsdict_str'] = sampsdict_str
    rd['vmask'] = vmask
    rd['behdict'] = behdict
    rd['inp'] = inp
    rd['tdict'] = tdict
    rd['statedict'] = statedict
    rd['pca_ofc'] = pca_ofc
    rd['pca_str'] = pca_str
    rd['net'] = net
    rd['D'] = [dOFC+1, dSTR+1]

    return rd


def savedynamics(sgrads, outputs, mesh, samps_pcdict, savename, D, vmask, trials):
    """ the save function after running dynamics"""
    # format things for saving
    sgrad_norm = sgrads['norm']
    sgrad_x = sgrads['x']
    sgrad_y = sgrads['y']
    sgrad_z = sgrads['z']
    blockp_grid = outputs['blockp']
    pi_grid = outputs['pi']
    v_grid = outputs['V']
    X0 = mesh['X0']
    Y0 = mesh['Y0']
    Z0 = mesh['Z0']

    samps_mixed = samps_pcdict['mixed']
    samps_high = samps_pcdict['high']
    samps_low = samps_pcdict['low']
    samps_5 = samps_pcdict['5']
    samps_10 = samps_pcdict['10']
    samps_20 = samps_pcdict['20']
    samps_40 = samps_pcdict['40']
    samps_80 = samps_pcdict['80']
    samps_all = samps_pcdict['all']

    vdict = {'five': vmask[0], 'ten': vmask[1], 'twenty': vmask[2], 'forty': vmask[3], 'eighty': vmask[4],
             'mixed': vmask[5], 'high': vmask[6], 'low': vmask[7], 'optout': vmask[8], 'win': vmask[9],
             'all': vmask[10]}

    sdict = {'sgrad_norm': sgrad_norm, 'sgrad_x': sgrad_x, 'sgrad_y': sgrad_y, 'sgrad_z': sgrad_z,
             'X0': X0, 'Y0': Y0, 'Z0': Z0,
             'blockp': blockp_grid, 'v': v_grid, 'pi': pi_grid,
             'samps_mixed': samps_mixed, 'samps_high': samps_high, 'samps_low': samps_low,
             'samps_5': samps_5, 'samps_10': samps_10, 'samps_20': samps_20,
             'samps_40': samps_40, 'samps_80': samps_80, 'samps_all': samps_all, 'D': D,
             'vmask': vdict, 'trials': trials}

    savemat(savename, sdict)
    return None


def define_dynamics_bounds(pca, psths2transform, resvec=None, bfrac=0.1, D=2):
    """
    get the pc ranges of support. used to define KE min initial conditions and flowfield boundaries
    :param pca: pca object that was fit with psth
    :param psths2transform: sample trajectories to fit bounding box
    :param resvec: (list) of resolution for [x,y,z] dimensions. if None, calc at every integer pt
    :param bfrac: (float) proportion of range to pad on each side of PCA space
    :param D: (int). dimension to use
    :return:
    """
    dat_proj = [pca.transform(k) for k in psths2transform]

    pcranges = []

    for j in range(D):

        # determine min and max of pc space
        nj_min = np.min([int(np.min(k[:, j])) for k in dat_proj])
        nj_max = np.max([int(np.max(k[:, j])) for k in dat_proj])

        bfj = int(np.floor(bfrac * (nj_max - nj_min)))

        if resvec is None:
            pc_j = np.linspace(nj_min - bfj, nj_max + bfj, int(nj_max - nj_min) + 2 * bfj + 1)

        else:
            print('custom res for dynamics')
            pc_j = np.linspace(nj_min - bfj, nj_max + bfj, resvec[0])

        pcranges.append(pc_j)

    return pcranges


def pcadynamics(rnnuse, pca, sampsdict, rnn_input, net, reg_idx, D=2,
                nn_reg=256, mesh=None, resvec=None, bfrac=0.1):
    """ calculate the effective dynamics in PC space from the RNN.
        basic flow:
            - define PCA space for a type of neural data, then discretize PC space
            - project PC points back to neural space
            - run through RNN to get updated state
            - project back to PC space, then calculate effective gradient

        pca: (PCA) a fit pca object
        sampsdict: (dict) of list of samples to transform. list has dims (nt x nfeatures)
        rnn_input is the statci input to the RNn
        D is the dimension of PC space

        input suggestions:
        for ofc:
            rnn_input = inp[tdict['start'][13]+1,:]
        for str:
            i_ofc = np.array([4.0660,  0.2620, -0.4273])  # a mixed block input from trial 13
            rnn_input = np.concatenate( (inp[tdict['start'][13]+1,0:2], i_ofc), axis = 0)
    """

    dt = 0.05

    print('formatting samples')
    sampsall = []
    for k in sampsdict.keys():
        sampsall.extend(sampsdict[k])
    # find mean of higher order features
    proj_samps = [pca.transform(m) for m in sampsall]
    proj_samps_flat = proj_samps[0]
    for k in proj_samps[1:]:
        proj_samps_flat = np.concatenate((proj_samps_flat, k), axis=0)
    pc_means = np.nanmean(proj_samps_flat, axis=0)

    # decide if the x,y,z mesh needs to be calculated or not
    print('calculating mesh')
    if mesh is None:
        pcranges = define_dynamics_bounds(pca, sampsall, resvec=resvec, bfrac=bfrac, D=D)
        if D == 2:
            X0 = pcranges[0]
            Y0 = pcranges[1]
            Z0 = [pc_means[2]]
        else:
            X0 = pcranges[0]
            Y0 = pcranges[1]
            Z0 = pcranges[1]
    else:
        # use a predefined mesh of PC coords
        X0 = mesh[0]
        Y0 = mesh[1]
        if D == 2:
            Z0 = [pc_means[2]]
        else:
            Z0 = mesh[2]

    # make grid of PC values, and output variables
    nx0 = len(X0)
    ny0 = len(Y0)
    nz0 = len(Z0)
    sgrad_x = np.zeros((nx0, ny0, nz0))
    sgrad_y = np.zeros((nx0, ny0, nz0))
    sgrad_z = np.zeros((nx0, ny0, nz0))
    sgrad_norm = np.zeros((nx0, ny0, nz0))

    blockp_grid = np.zeros((nx0, ny0, nz0, 3))
    pi_grid = np.zeros((nx0, ny0, nz0))
    v_grid = np.zeros((nx0, ny0, nz0))

    # input to RNN
    i = torch.tensor(np.expand_dims(np.expand_dims(rnn_input, axis=0), axis=0)).to(torch.float32)

    print('number of points in each PC dim')
    print([nx0, ny0, nz0])
    # the gradient calculation
    for k1 in range(nx0):
        for k2 in range(ny0):
            print([k1, k2], end='\r')
            for k3 in range(nz0):

                # original point in PC space
                if D == 2:
                    s_tm1 = [X0[k1], Y0[k2]]
                    # s_tm1.extend(np.zeros((n_neur-2)))  # uses 0 padding
                    s_tm1.extend(pc_means[D:])  # uses PC mean in those dims
                    s_tm1 = np.array(s_tm1)
                elif D == 3:
                    s_tm1 = [X0[k1], Y0[k2], Z0[k3]]
                    # s_tm1.extend(np.zeros((n_neur - 3)))
                    s_tm1.extend(pc_means[D:])  # uses PC mean in those dims
                    s_tm1 = np.array(s_tm1)
                else:
                    print('higher dim > 3 not supported yet.')
                    s_tm1 = None

                # point in neural space
                s_temp = torch.squeeze(torch.tensor(
                    pca.inverse_transform(np.expand_dims(s_tm1, axis=0)))).to(torch.float32)

                s = (torch.unsqueeze(torch.unsqueeze(s_temp[:nn_reg], dim=0), dim=0),
                     torch.unsqueeze(torch.unsqueeze(s_temp[nn_reg:], dim=0), dim=0))  # tuple of hidden and cell
                # calculate updated state
                Y, sp = rnnuse(i, s)
                Y = Y.reshape((-1, Y.shape[-1]))

                # set output variables
                output1 = net.linear1(Y)
                pred_activations = output1[:, 0:3].detach().numpy()
                blockp_grid[k1, k2, k3, :] = pred_activations
                output2 = net.linear2(Y)
                pi_activation = output2[:, :3]
                output_policy = nn.Softmax(dim=1)(pi_activation)
                pi_grid[k1, k2, k3] = output_policy[0, 0].detach().numpy()
                v_grid[k1, k2, k3] = output2[:, 3].detach().numpy()

                # project back into pc space
                test = np.concatenate(
                    (sp[0][0, 0, :].detach().numpy(), sp[1][0, 0, :].detach().numpy()), axis=0)
                sp_proj = pca.transform(np.expand_dims(test, axis=0))
                # sp_proj = sp_proj[0, 0:D]

                sgrad = np.squeeze((sp_proj - s_tm1) / dt)
                if D == 2:
                    sgrad_x[k1, k2, k3] = sgrad[0]
                    sgrad_y[k1, k2, k3] = sgrad[1]
                    sgrad_norm[k1, k2, k3] = 0.5 * np.linalg.norm(sgrad[:D]) ** 2
                elif D == 3:
                    sgrad_x[k1, k2, k3] = sgrad[0]
                    sgrad_y[k1, k2, k3] = sgrad[1]
                    sgrad_z[k1, k2, k3] = sgrad[2]
                    sgrad_norm[k1, k2, k3] = 0.5 * np.linalg.norm(sgrad[:D]) ** 2

    if D == 2:
        sgrad_x = np.squeeze(sgrad_x)
        sgrad_y = np.squeeze(sgrad_y)
        sgrad_norm = np.squeeze(sgrad_norm)

    sgrads = {'x': sgrad_x, 'y': sgrad_y, 'z': sgrad_z, 'norm': sgrad_norm}
    outputs = {'pi': pi_grid, 'V': v_grid, 'blockp': blockp_grid}
    mesh = {'X0': X0, 'Y0': Y0, 'Z0': Z0}

    return sgrads, outputs, mesh


def KEcalc(S1, S2, net, pca_use, i, pc_means, reg_idx=0, D=2):
    """ Calculate the kinetic energy of the dynanics as well its gradient
        S1: (list) OFC state in neural space
        S2: (list) STR state in neural space
        net: (nn.module) network being analyze
        pca_use: (PCA) fit PCA object that dim. reduced dynamics
        pc_means: (list) average PC values for each dim
        reg_idx: (int) define which layer to check. 0=OFC, 1=STR
        pc_constrain: (bool) if True, calculated KE and gradient on the PC manifold"""

    # S is in D-dim PCA space. going to calclate KE in PC space
    nn_reg = 256
    dt = 0.05

    # pc_means needed to be consisten with constrained form
    del pc_means

    # project both back into neural space
    # only analyzing one region, so pad botch with same pc_means values
    s1_tm1 = list(S1)
    s1_temp = torch.tensor(s1_tm1).to(torch.float32)
    s1 = (torch.unsqueeze(torch.unsqueeze(s1_temp[:nn_reg], dim=0), dim=0),
          torch.unsqueeze(torch.unsqueeze(s1_temp[nn_reg:], dim=0), dim=0))  # tuple of hidden and cell

    s2_tm1 = list(S2)
    s2_temp = torch.tensor(s2_tm1).to(torch.float32)
    s2 = (torch.unsqueeze(torch.unsqueeze(s2_temp[:nn_reg], dim=0), dim=0),
          torch.unsqueeze(torch.unsqueeze(s2_temp[nn_reg:], dim=0), dim=0))  # tuple of hidden and cell

    if reg_idx == 0:
        rnn_use = net.rnn1
        s = s1
        s_tm1 = s1_tm1
    elif reg_idx == 1:
        rnn_use = net.rnn2
        s = s2
        s_tm1 = s2_tm1
    else:
        print('reg_idx not set correclty')
        rnn_use = None
        s_tm1 = None
        s = None

    # calculate updated state:
    # first lineraize, parse inputs and get jacobian
    h = [s1[0], s2[0]]
    c = [s1[1], s2[1]]
    net2 = linearize.LSTM_fullcopy(net, nreg=2, reg=reg_idx)

    parsed_inp = i

    _, sp = rnn_use(parsed_inp, s)
    sp_concat = np.concatenate(
        (sp[0][0, 0, :].detach().numpy(), sp[1][0, 0, :].detach().numpy()), axis=0)

    # calculate kinetic energy and jacobian
    F = np.squeeze((sp_concat - s_tm1) / dt)
    KE = 0.5 * np.linalg.norm(F) ** 2
    dFdh = net2.linearize_full(parsed_inp, h, c)
    F_jac = (dFdh - np.eye(2 * nn_reg)) / dt
    KEprime = F_jac.T @ F

    return KE, KEprime, dFdh


def KEcalc_pcconstrain(S1, S2, net, pca_use, i, pc_means, reg_idx=0, D=2):
    """ Calculate the kinetic energy of the dynanics as well its gradient, constrained to PC space
        S1: (list) OFC state projected in PC space
        S2: (list) STR state projected in PC space
        net: (nn.module) network being analyzed
        pca_use: (PCA) fit PCA object that dim. reduced dynamics
        i: (ndarray) the specific input for that layer.
        pc_means: (list) average PC values for each dim
        reg_idx: (int) define which layer to check. 0=OFC, 1=STR"""

    # S is in D-dim PCA space. going to calclate KE in PC space
    nn_reg = 256
    dt = 0.05

    # project both back into neural space

    # only analyzing one region, so pad both with same pc_means values
    s1_tm1 = list(S1)
    s1_tm1.extend(pc_means[D:])
    s1_tm1 = np.array(s1_tm1)  # PC space, but full dim
    s1_neur = torch.squeeze(torch.tensor(
        pca_use.inverse_transform(np.expand_dims(s1_tm1, axis=0)))).to(torch.float32)

    s1 = (torch.unsqueeze(torch.unsqueeze(s1_neur[:nn_reg], dim=0), dim=0),
          torch.unsqueeze(torch.unsqueeze(s1_neur[nn_reg:], dim=0), dim=0))  # tuple of hidden and cell

    s2_tm1 = list(S2)
    s2_tm1.extend(pc_means[D:])
    s2_tm1 = np.array(s2_tm1)  # PC space, but full dim
    s2_neur = torch.squeeze(torch.tensor(
        pca_use.inverse_transform(np.expand_dims(s2_tm1, axis=0)))).to(torch.float32)

    s2 = (torch.unsqueeze(torch.unsqueeze(s2_neur[:nn_reg], dim=0), dim=0),
          torch.unsqueeze(torch.unsqueeze(s2_neur[nn_reg:], dim=0), dim=0))  # tuple of hidden and cell

    if reg_idx == 0:
        rnn_use = net.rnn1
        s = s1
        s_tm1 = s1_tm1  # PC space, but full dim
    elif reg_idx == 1:
        rnn_use = net.rnn2
        s = s2
        s_tm1 = s2_tm1  # PC space, but full dim
    else:
        print('more than 2 RNN layers not supported for dynamics yet. check reg_idx input')
        rnn_use = None
        s = None
        s_tm1 = None

    # calculate updated state:

    # first lineraize, parse inputs and get jacobian
    h = [s1[0], s2[0]]
    c = [s1[1], s2[1]]
    net2 = linearize.LSTM_fullcopy(net, nreg=2, reg=reg_idx)
    # parsed_inp, _, _ = net2.parse_inputs(i, h, c)

    # process the input
    # parsed_inp = torch.tensor(np.expand_dims(np.expand_dims(i,axis=0),axis=0)).to(torch.float32)
    parsed_inp = i

    _, sp = rnn_use(parsed_inp, s)
    sp_neur = np.concatenate(
        (sp[0][0, 0, :].detach().numpy(), sp[1][0, 0, :].detach().numpy()), axis=0)  # neural space
    sp_pc = pca_use.transform(np.expand_dims(sp_neur, axis=0))  # PC space, but full dim

    # calculate kinetic energy and jacobian, neural activity space
    dFdh = net2.linearize_full(parsed_inp, h, c)
    F_jac = (dFdh - np.eye(2 * nn_reg)) / dt

    F_PC = np.squeeze((sp_pc - s_tm1) / dt)
    W = pca_use.components_[0:D]
    KE = 0.5 * np.linalg.norm(F_PC[:D]) ** 2  # only contributions from that part of phase space

    test = W @ F_jac @ W.T
    KEprime = test.T @ F_PC[:D]

    return KE, KEprime, dFdh

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
                  'zeromean': False, 'drtype': 'PCA'}

    # some hard coded stuff
    configname = dynops['configname']
    numhidden = dynops['numhidden']  # number of hidden units per layer

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
    else:
        ops, _ = utils.parse_configs(configname)

    device = torch.device('cpu')
    nlayer = len(numhidden)
    if ops['rnntype'] == 'LSTM':
        ngate = 2
    else:
        ngate = 1

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
        offers, _ = utils.getoffers_from_OFCinput(inp, ops['inputcase'])
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

        # preprocess for dynamics. flattening data,  event timings, removing outliers of v. short and v. long trials
        print('postprocessing simulation data for dynamics. finding outliers and repackaging state')
        print('epoch ' + dynops['epoch'])

        statedict, behdict = parse_state.postprocess4dynamics(sflat, behdict, warpstate=False, Ttrial=256 * 3,
                                                              zeromean=dynops['zeromean'], epoch=dynops['epoch'],
                                                              nlayer=nlayer, ngate=ngate)

        # extract the data needed for PCA fitting
        dat2fit = parse_state.extract_data_epoch(statedict, tdict, behdict['goodtrial_mask'],
                                     epoch=dynops['epoch'])


    # note: not going to save pred data. it is easy and fast to generate
    elif simtype == 'kind' or simtype == 'inf':

        behdict = rdict['behdict']
        tdict = behdict['tdict']
        outpt_supervised = rdict['outpt_supervised']
        sflat = rdict['sflat']
        inp = rdict['inp']

        # create the dat2fit dictionary
        # ns = len(sflat)
        # nneur = sflat[0][0][0].shape[-1]
        # ofc_h = np.zeros((ns, nneur))
        # ofc_c = np.zeros((ns, nneur))
        # str_h = np.zeros((ns, nneur))
        # str_c = np.zeros((ns, nneur))

        # for j in range(ns):
        #    ofc_h[j, :] = sflat[j][0][0][0, 0, :].detach().numpy()
        #    ofc_c[j, :] = sflat[j][0][0][0, 0, :].detach().numpy()
        #    str_h[j, :] = sflat[j][0][0][0, 0, :].detach().numpy()
        #    str_c[j, :] = sflat[j][0][0][0, 0, :].detach().numpy()

        # dat2fit = {'ofc_h': ofc_h, 'ofc_c': ofc_c, 'str_h': str_h, 'str_c': str_c}
        # statedict = dat2fit
        # TODO: not tested yet
        statedict = parse_state.sflat2statedict(sflat, nlayer=nlayer, ngate=ngate)
        dat2fit = statedict


    else:
        print('going to sim other types here. not supported yet')
        dat2fit = None
        tdict = None
        inp = None
        statedict = None
        behdict = None

    # do pca, or some other dimensionality transform
    dr_list, dim_list = parse_state.dr_transform(dat2fit, transformtype=dynops['drtype'])

    # final output function. TODO: figure out what is needed here
    rd = {}
    rd['behdict'] = behdict
    rd['inp'] = inp
    rd['dat2fit'] = dat2fit
    rd['tdict'] = tdict
    rd['statedict'] = statedict
    rd['dr_list'] = dr_list
    rd['D'] = dim_list
    rd['net'] = net

    return rd

def savedynamics(sgrads, outputs, mesh, savename, dr_use, D):
    """
    the save function after running dynamics
    @param sgrads: gradients along each dimension
    @param outputs: network outputs
    @param mesh: spatial mesh for low-dim representation
    @param savename: (str) savename
    @param dr_use: dimensionality-reduciton technique object (e..g, PCA)
    @param D: (int) dimensionality
    @return:
    """
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


    # the dimensionality reduciton method pieces, in matlab-friendly format
    dr_dict = {'components': dr_use.components_, 'var_explained': dr_use.explained_variance_,
               'featuremeans': dr_use.mean_, 'type': str(type(dr_use))}

    sdict = {'sgrad_norm': sgrad_norm, 'sgrad_x': sgrad_x, 'sgrad_y': sgrad_y, 'sgrad_z': sgrad_z,
             'X0': X0, 'Y0': Y0, 'Z0': Z0,
             'blockp': blockp_grid, 'v': v_grid, 'pi': pi_grid, 'D': D, 'dim_red': dr_dict}

    savemat(savename, sdict)
    return None


def define_dynamics_bounds(pca, dat2fit, resvec=None, bfrac=0.1, D=2, usepcmeans=True):
    """
    get the pc ranges
    :param pca: pca object that was fit with psth
    :param dat2fit: (ndarray) sample trajectories to fit bounding box. nt x nfet
    :param resvec: (list) of resolution for [x,y,z] dimensions. if None, calc at every integer pt
    :param bfrac: (float) proportion of range to pad on each side of PCA space
    :param D: (int). dimension to use
    :param usepcmeans: (bool). will only care about visited locations in D-dim space (true), or high dim space (false)
    :return:
    """
    dat_proj = pca.transform(dat2fit)

    pcranges = []


    for j in range(D):

        nj_min = np.min(dat_proj[:,j])
        nj_max = np.max(dat_proj[:,j])

        bfj = int(np.floor(bfrac * (nj_max - nj_min)))

        if resvec is None:
            pc_j = np.linspace(nj_min - bfj, nj_max + bfj, int(nj_max - nj_min) + 2 * bfj + 1)

        else:
            print('custom res for dynamics')
            pc_j = np.linspace(nj_min - bfj, nj_max + bfj, resvec[0])

        pcranges.append(pc_j)

    return pcranges

def chooseinputs(rd, block_idx, reg_idx, epoch, blocktype, choicetype='maxbelief'):
    """
    will decide which trial, and thus which timepoint, to use for the dynamics inputs. can support looking at
    first instance of a given block type, trial with highest belief, or others as needed
    @param rd: (dict) of simulation data
    @param block_idx: (int) or None, for a hard-coded choice of block
    @param reg_idx: (int) 0 for OFC, 1 for STR
    @param epoch: (str) 'wait','start', or 'iti' for which part of task to grab
    @param blocktype: (str) 'low' 'mixed' or 'high'
    @param choicetype: (str) method of choosing block. supports 'maxbelief' or 'first', 'stim'
    @return:
    """

    block = rd['behdict']['blocks']
    offers = rd['behdict']['offers']
    outcomes = rd['behdict']['outcomes']
    ns = len(block)
    t_idx = None  # initialize

    if block_idx is None:
        # inputs don't matter for OFC
        if reg_idx == 0:
            block_idx = 0
            if epoch == 'start':
                t_idx = rd['tdict']['start'][block_idx]
            elif epoch == 'wait':
                t_idx = rd['tdict']['start'][block_idx] + 1
            elif epoch == 'iti':
                t_idx = rd['tdict']['iti'][block_idx] + 1
            else:
                print('epoch is wrong. cant assign t_idx')
        # this is where it matters
        elif reg_idx == 1:
            if choicetype == 'maxbelief':
                if blocktype == 'mixed':
                    block_idx = np.argwhere((block[:ns] == 0) & (outcomes[:ns] == 0)
                                            & (np.exp(offers[:ns]).astype(int) == 20))

                    # find index with strongest belief on that block
                    beliefs = []
                    for j in block_idx[:, 0]:
                        t_idx = rd['tdict']['iti'][j] - 1
                        beliefs.append(rd['behdict']['preds'][t_idx][0])
                    ind = np.argmax(beliefs)
                    block_idx = block_idx[ind, 0]

                    print(block_idx)
                elif blocktype == 'high':
                    block_idx = np.argwhere((block[:ns] == 1) & (outcomes[:ns] == 0)
                                            & (np.exp(offers[:ns]).astype(int) == 20))
                    # find index with strongest belief on that block
                    beliefs = []
                    for j in block_idx[:, 0]:
                        t_idx = rd['tdict']['iti'][j] - 1
                        beliefs.append(rd['behdict']['preds'][t_idx][1])
                    ind = np.argmax(beliefs)
                    block_idx = block_idx[ind, 0]
                    print(block_idx)

                elif blocktype == 'low':

                    block_idx = np.argwhere((block[:ns] == 2) & (outcomes[:ns] == 0)
                                            & (np.exp(offers[:ns]).astype(int) == 20))
                    # find index with strongest belief on that block
                    beliefs = []
                    for j in block_idx[:, 0]:
                        t_idx = rd['tdict']['iti'][j] - 1
                        beliefs.append(rd['behdict']['preds'][t_idx][2])
                    ind = np.argmax(beliefs)
                    block_idx = block_idx[ind, 0]
                    print(block_idx)
                else:
                    print('blocktype not specified, or not part of main task. defaulting')
                    block_idx = 0

            elif choicetype == 'first':
                block_unique = np.unique(np.array(block))
                if 0 in block_unique:
                    mixedblock = np.argwhere(block == 0)[0, 0]
                else:
                    mixedblock = None
                if 1 in block_unique:
                    highblock = np.argwhere(block == 1)[0, 0]
                else:
                    highblock = None
                if 2 in block_unique:
                    lowblock = np.argwhere(block == 2)[0, 0]
                else:
                    lowblock = None

                block_idx_dict = {'low': lowblock, 'mixed': mixedblock, 'high': highblock}
                block_idx = block_idx_dict[blocktype]
            else:
                print('choicetype provided is not supported. use either "maxbelief" or "first". provided: '+choicetype)
                block_idx = None

            if epoch == 'start':
                t_idx = rd['tdict']['start'][block_idx]
            elif epoch == 'wait':
                t_idx = rd['tdict']['iti'][block_idx] - 1  # try using KE points right at opt out
            elif epoch == 'iti':
                t_idx = rd['tdict']['iti'][block_idx] + 1
            else:
                print('epoch not speicified correctly. cant assign t_idx')
                t_idx = None

            # t_idx = rd['tdict']['start'][block_idx] + 1
        else:
            print('reg_idx not defined correctly')
            t_idx = None
            block_idx = None
    else:
        if epoch == 'wait':
            t_idx = rd['tdict']['iti'][block_idx] - 1
        elif epoch == 'start':
            t_idx = rd['tdict']['start'][block_idx]
        elif epoch == 'iti':
            t_idx = rd['tdict']['iti'][block_idx]+1
        else:
            print('something wrong in epoch ')
            t_idx = None

    print('using blocktype: ' + blocktype + '. trial is ' + str(block_idx))
    return t_idx


def pcadynamics(rnnuse, pca, dat2fit, rnn_input, net, D=2,
                nn_reg=256, mesh=None, resvec=None, bfrac=0.1):
    """ calculate the effecgive dynamics in PC space from the RNN. only handles 1-3D right now.
        basic flow:
            -define PCA space for a type of neural data, then discretize PC space
            - project PC points back to neural space
            - run through RNN to get updated state
            - project back to PC space, then calculate effective gradient

        @param rnnuse: (tf.NN) sub-network/layer used to calculate dynamics
        @param pca: (PCA) a fit pca object
        @param dat2fit: (np.ndarray) of shape (nt, 2*nn) in neural space. nt timepoints, nn num units. x2 for LSTM
        @param rnn_input: (np.ndarrays) for the input into RNN
        @param net: (tf.NN) full RNN network
        @param D: (int) dimensionlaity of dynamics to calculate. NOTE: can differ from data dimensionality
        @param nn_reg: (int) number of units per region. used to help grab correct piece of data from RNN
        @param mesh: (list of floats) for D-dim values in PC space of over which to calulate dynamics. None= calc mesh
        @param resvec: (list of ints) complementary param to mesh, provides res along each D dimension, but not bounds
        @param bfrac: (float) if calculating mesh on the fly, the fraction of PC space to pad out from max PC bounds

    """

    assert (D > 0) and (D < 4), "Dimensionality must be 1, 2 or 3. higher dims not supported"

    dt = 0.05  # if more behdat needed, i'll make it an input. otherwise hardcode

    print('formatting samples')
    proj_samps = pca.transform(dat2fit)
    pc_means = np.nanmean(proj_samps, axis=0)

    # decide if the x,y,z mesh needs to be calculated or not. pad out to 3D, regardless of D
    if mesh is None:
        print('calculating mesh')
        print(dat2fit.shape)
        pcranges = define_dynamics_bounds(pca, dat2fit, resvec=resvec, bfrac=bfrac, D=D)
        X0 = pcranges[0]
        if D == 1:
            Y0 = [pc_means[1]]
            Z0 = [pc_means[2]]
        elif D == 2:
            Y0 = pcranges[1]
            Z0 = [pc_means[2]]
        else:
            Y0 = pcranges[1]
            Z0 = pcranges[2]
    else:
        X0 = mesh[0]
        if D == 1:
            Y0 = [pc_means[1]]
            Z0 = [pc_means[2]]
        elif D == 2:
            Y0 = mesh[1]
            Z0 = [pc_means[2]]
        else:
            Y0 = mesh[1]
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

    print([nx0, ny0, nz0])
    # the gradient calculation
    for k1 in range(nx0):
        for k2 in range(ny0):
            print([k1, k2], end='\r')
            for k3 in range(nz0):

                # original point in PC space
                if D == 1:
                    s_tm1 = [X0[k1]]
                    s_tm1.extend(pc_means[D:])  # uses PC mean in those dims
                    s_tm1 = np.array(s_tm1)
                elif D == 2:
                    s_tm1 = [X0[k1], Y0[k2]]
                    s_tm1.extend(pc_means[D:])  # uses PC mean in those dims
                    s_tm1 = np.array(s_tm1)
                elif D == 3:
                    s_tm1 = [X0[k1], Y0[k2], Z0[k3]]
                    s_tm1.extend(pc_means[D:])  # uses PC mean in those dims
                    s_tm1 = np.array(s_tm1)

                # point in neural space
                s_temp = torch.squeeze(torch.tensor(
                    pca.inverse_transform(np.expand_dims(s_tm1, axis=0)))).to(torch.float32)

                s = wt_nets.flat2gates(s_temp, rnnuse)
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
                test = wt_nets.gates2flat(sp, rnnuse)
                test = test.detach().numpy()
                sp_proj = pca.transform(np.expand_dims(test, axis=0))

                sgrad = np.squeeze((sp_proj - s_tm1) / dt)
                if D == 1:
                    sgrad_x[k1, k2, k3] = sgrad[0]
                elif D == 2:
                    sgrad_x[k1, k2, k3] = sgrad[0]
                    sgrad_y[k1, k2, k3] = sgrad[1]
                else:
                    sgrad_x[k1, k2, k3] = sgrad[0]
                    sgrad_y[k1, k2, k3] = sgrad[1]
                    sgrad_z[k1, k2, k3] = sgrad[2]

                sgrad_norm[k1, k2, k3] = 0.5 * np.linalg.norm(sgrad[:D]) ** 2

    # squeeze out things that were actually calculated
    if D == 1:
        sgrad_x = np.squeeze(sgrad_x)
    elif D == 2:
        sgrad_x = np.squeeze(sgrad_x)
        sgrad_y = np.squeeze(sgrad_y)
        sgrad_norm = np.squeeze(sgrad_norm)

    sgrads = {'x': sgrad_x, 'y': sgrad_y, 'z': sgrad_z, 'norm': sgrad_norm}
    outputs = {'pi': pi_grid, 'V': v_grid, 'blockp': blockp_grid}
    mesh = {'X0': X0, 'Y0': Y0, 'Z0': Z0}

    return sgrads, outputs, mesh


# do this in neural space and pc space

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


def KEcalc_pcconstrain(S1, S2, net, pca_use, i, pc_means, reg_idx=0, D=2, usepcmeans=True):
    """ Calculate the kinetic energy of the dynanics as well its gradient, constrained to PC space
        S1: (list) OFC state projected in PC space
        S2: (list) STR state projected in PC space
        net: (nn.module) network being analyzed
        pca_use: (PCA) fit PCA object that dim. reduced dynamics
        i: (ndarray) the specific input for that layer.
        pc_means: (list) average PC values for each dim
        reg_idx: (int) define which layer to check. 0=OFC, 1=STR
        D: (ind) dimensionaliaty of low-dim PC space
        usepcmeans: (bool) if true, use >D diensions of pca respons as mean pca response in pc_means

        note: if usepcmeans is false, then S1 and S2 should be the full response in PC space. same # as units"""

    # S is in D-dim PCA space. going to calclate KE in PC space
    nn_reg = 256
    dt = 0.05

    # project both back into neural space

    # only analyzing one region, so pad both with same pc_means values
    s1_tm1 = list(S1)
    # TODO: augment this to have option for >D pc space being used
    if usepcmeans:
        s1_tm1.extend(pc_means[D:])
    else:
        assert len(s1_tm1) == nn_reg*2, "usepcmeans=False. S1 S2 must have same length as 2xnumber of units: 512?"

    s1_tm1 = np.array(s1_tm1)  # PC space, but full dim
    s1_neur = torch.squeeze(torch.tensor(
        pca_use.inverse_transform(np.expand_dims(s1_tm1, axis=0)))).to(torch.float32)

    s1 = (torch.unsqueeze(torch.unsqueeze(s1_neur[:nn_reg], axis=0), axis=0),
          torch.unsqueeze(torch.unsqueeze(s1_neur[nn_reg:], axis=0), axis=0))  # tuple of hidden and cell

    s2_tm1 = list(S2)
    if usepcmeans:
        s2_tm1.extend(pc_means[D:])
    else:
        assert len(s2_tm1) == nn_reg * 2, "usepcmeans=False. S2 must have same length as 2xnumber of units: 512?"

    s2_tm1 = np.array(s2_tm1)  # PC space, but full dim
    s2_neur = torch.squeeze(torch.tensor(
        pca_use.inverse_transform(np.expand_dims(s2_tm1, axis=0)))).to(torch.float32)

    s2 = (torch.unsqueeze(torch.unsqueeze(s2_neur[:nn_reg], axis=0), axis=0),
          torch.unsqueeze(torch.unsqueeze(s2_neur[nn_reg:], axis=0), axis=0))  # tuple of hidden and cell

    if reg_idx == 0:
        rnn_use = net.rnn1
        s = s1
        s_tm1 = s1_tm1  # PC space, but full dim
        s_neur = s1_neur.detach().numpy()  # neural space
    elif reg_idx == 1:
        rnn_use = net.rnn2
        s = s2
        s_tm1 = s2_tm1  # PC space, but full dim
        s_neur = s2_neur.detach().numpy()  # neural space

    # calculate updated state:

    # first lineraize, parse inputs and get jacobian
    h = [s1[0], s2[0]]
    c = [s1[1], s2[1]]
    net2 = linearize.LSTM_fullcopy(net, nreg=2, reg=reg_idx)
    # parsed_inp, _, _ = net2.parse_inputs(i, h, c)

    # process the input
    parsed_inp = i

    _, sp = rnn_use(parsed_inp, s)
    sp_neur = np.concatenate(
        (sp[0][0, 0, :].detach().numpy(), sp[1][0, 0, :].detach().numpy()), axis=0)  # neural space
    sp_pc = pca_use.transform(np.expand_dims(sp_neur, axis=0))  # PC space, but full dim

    # calculate kinetic energy and jacobian, neural activity space
    dFdh = net2.linearize_full(parsed_inp, h, c)
    F_jac = (dFdh - np.eye(2 * nn_reg)) / dt

    F_PC = np.squeeze((sp_pc - s_tm1) / dt)
    F_neur = np.squeeze((sp_neur - s_neur) / dt)

    if usepcmeans:
        KE = 0.5 * np.linalg.norm(F_PC[:D]) ** 2  # only contributions from that part of phase space
        W = pca_use.components_[0:D]
        test = W @ F_jac @ W.T
        KEprime = test.T @ F_PC[:D]
    else:
        KE = 0.5 * np.linalg.norm(F_PC) ** 2
        W = pca_use.components_[0:2*nn_reg]
        test = W @ F_jac @ W.T
        KEprime = test.T @ F_PC


    return KE, KEprime, dFdh


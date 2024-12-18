# code to calculate the flow fields and stats for each network

# code to extend the plotted region of flow fields, then convert to matlab
import pickle
import json
import torch
import numpy as np
from dynamics.process.rnn import parse, wt_protocols, parse_state
from dynamics.analysis import dynamics_analysis as dyn
from dynamics.utils import utils


def genflowfields(modelname, dataname, kemin_name, savename, flowops, overwrite=None):
    """
    loads the data of interest, calculate flow fields. Will calculate a 1D or  2D field, will save true D in .mat file
    @param modelname: (str) name of rnn model parameters file
    @param dataname: (str) loaded data for geneaerling the PC bounds on these flowfields
    @param kemin_name: (st) savename of kinetic energy results
    @param savename: (str) savename of flowfield
    @param flowops: (dict) of handling of how simulation will run. see below
    @param overwrite: (dict) of potential options to short-circuit standard choices for inputs, data2fit. see below
    @return:
    """

    # overwrite features can short-circut some of the default things
    if overwrite is None:
        D_overwrite = None
        dat2fit = None
        block_idx = None
        rnn_input = None
    else:
        # some of these can be None as well
        D_overwrite = overwrite['D']
        dat2fit = overwrite['dat2fit']
        rnn_input = overwrite['rnn_input']
        block_idx = overwrite['block_idx']

    # immediately check if overwritten D is too large
    if D_overwrite is not None:
        assert D_overwrite < 3, "overwritten D value too large. must be 1 or 2"

    reg_idx = flowops['reg_idx']  # 0 = OFC, 1 = STR
    tphase = flowops['tphase']  # see taskdict, below
    epoch = flowops['epoch']  # 'start','wait','reward'
    blocktype = flowops['blocktype']  # 'low','mixed','high. used only for STR
    verbose = flowops['verbose']  # print optimizaiton results as it goes?
    useKEbounds = flowops['useKEbounds']  # if true, load KE min file to calculate effective PC range
    choicetype = flowops['choicetype']  # how to choose block. either "maxbelief" or "first"
    configname = flowops['configname']  # config file

    ops, _ = utils.parse_configs(configname)
    numhidden = ops['nn']  # list of number hidden units
    numhidden_reg = numhidden[reg_idx]  # number of hidden units in layer of choice

    taskdict = {0: 'kind', 1: 'kind', 2: 'inf', 3: 'task', 4: 'task', 5: 'task'}
    dynops = {'numhidden': numhidden, 'resim': False, 'configname': configname, 'epoch': epoch,
              'zeromean': False, 'block_idx': None, 'drtype': 'PCA'}

    print('loading/simulating task data')
    if taskdict[tphase] == 'task':
        if dataname.split('.')[-1] == 'json':
            with open(dataname, 'r') as f:
                dat_json = json.load(f)
                dat = parse.json2dat(dat_json)
        elif dataname.split('.')[-1] == 'dat':
            print('loading legacy .dat data')
            with open(dataname, 'rb') as f:
                dat = pickle.load(f)
    else:
        print('simulating data')
        print(taskdict[tphase])
        dat = wt_protocols.sim_task(modelname, configname=configname, simtype=taskdict[tphase])

    print('preprocessing data')
    rd = dyn.dynamics_preprocess_suite(modelname, dat, simtype=taskdict[tphase], dynops=dynops)
    # get outlier trials that were exluded, so you can piece together matlab file better
    if dat2fit is None:
        dat2fit = rd['dat2fit']  # data used in the dr transform

    print('saving trajectory data. NOT the same as dat2fit for low-dim transofmr if provided')
    savename_trajec = savename.replace('.mat', '_trajectory.mat')
    _ = parse_state.states2matlab(rd, savename_trajec)

    # get dim red transform and layer-specific data for setting bounds for the flowfield visualization
    dr_use = rd['dr_list'][reg_idx]
    tempdict = parse_state.concatstates(dat2fit)
    dat2fit = tempdict['layer' + str(reg_idx + 1)]  # all gates of a layer concatenated together

    # choose which RNN in the network will be studied. single region supported so far
    if reg_idx == 0:
        rnnuse = rd['net'].rnn1
    elif reg_idx == 1:
        rnnuse = rd['net'].rnn2
    else:
        print('only region 1 (reg_idx=0) and region 2 (reg_idx=1) are currently supported for flow field')
        rnnuse = None

    if rnn_input is None:
        print('choosing inputs')
        t_idx = dyn.chooseinputs(rd, block_idx, reg_idx, epoch, blocktype, choicetype=choicetype)
        # some inputs require some checking, like maggie's opt-stim input
        if choicetype == 'stim':
            # find a trial that was opto-stimulated,
            assert len(rd['inp'][0, :]) == 5, 'input for stim case is incorrect. should be length 5'
            nt = rd['inp'].shape[0]
            stim_input_idx = np.argwhere(np.array([rd['inp'][k, -1] for k in range(nt)]) != 0)[0, 0]
            rnn_input = rd['inp'][stim_input_idx, :]
        else:
            rnn_input = rd['inp'][t_idx, :]
        print(rnn_input)
        i_ofc = np.expand_dims(np.expand_dims(rd['behdict']['preds'][t_idx], axis=0), axis=0)
        print(i_ofc)

        print('parsing inputs to be correct for given layer')
        if reg_idx == 0:
            i = torch.tensor(rnn_input).to(torch.float32)
        elif reg_idx == 1:
            inputcase = dat['ops']['inputcase']
            rnn_input = np.expand_dims(
                                       np.expand_dims(utils.getSTRinp_from_OFCinput(
                                                      rnn_input, inputcase=inputcase), axis=0), axis=0)
            i = np.concatenate((rnn_input, i_ofc), axis=2)
            i = torch.tensor(np.squeeze(np.squeeze(i))).to(torch.float32)
        else:
            print('cant set input, i. reg_idx 0 and reg_idx = 1 currently only regions suppored.')
            i = None
            dr_use = None
            rnnuse = None

        rnn_input = i
    else:
        print('rnn_input provided to function. bypasing')

    # load the the KE min results and find the full range, then redecide on it. call is 'mesh'
    if useKEbounds:
        if verbose:
            print('using KE min to find good bounds for flowfields')

        kedat1 = pickle.load(open(kemin_name, 'rb'))
        kedat = parse.KEconvert2matlab(kemin_name, dosave=False)

        # should D be overwritten? do some checks:
        print('dimensionality of KE data and simulation data, respectively')
        print(len(kedat1['crds']))
        print(len(kedat1['crds'][0]))
        D_KE = kedat1['crds'][0][6]
        D_dat = rd['D'][reg_idx]

        if D_KE != D_dat:
            print('true data dimensionality differs from one used in KE calc. using KE dim for flowfields')

        if D_overwrite is None:
            D = D_KE
        else:
            D = D_overwrite
            if D_overwrite != D_KE:
                print('overwriting dimensionality with value that differs from one used in KE')
                print('this might cause subtle shifts in located KE fixed/slow points vs how incidated in flowfields')
            else:
                print('overwriting dimenisionality. matches KE dim')

        assert D < 3, "D is too large. must be 1 or 2"

        # find min and max range required to incorporate all found fixed/slow points
        labs = np.unique(kedat['labels'][kedat['labels'] > -1])
        labs_idx = [np.argwhere(kedat['labels'] == k)[0, 0] for k in labs]

        xvals_fp = [kedat['crds'][k][0] for k in labs_idx]
        xvals_base = [k[0][0] for k in kedat1['crds']]
        xmin = np.min([np.min(xvals_base), np.min(xvals_fp)])
        xmax = np.max([np.max(xvals_base), np.max(xvals_fp)])
        nx = 100
        xvals = np.linspace(xmin * 1.2, xmax * 1.2, nx)

        if D == 1:
            mesh = [xvals]
        elif D > 1 and D_KE > 1:
            yvals_fp = [kedat['crds'][k][1] for k in labs_idx]
            yvals_base = [k[0][1] for k in kedat1['crds']]
            ymin = np.min([np.min(yvals_base), np.min(yvals_fp)])
            ymax = np.max([np.max(yvals_base), np.max(yvals_fp)])
            ny = 100
            yvals = np.linspace(ymin * 1.2, ymax * 1.2, ny)

            mesh = [xvals, yvals]
        elif D > 1 and D_KE == 1:
            # and odd use case, but if original dynamics are 1D, but you want to visualize in 2D
            print('odd use case that original dynamics are 1D, but you want to see in 2D. calc 2D mesh')
            mesh_tmp = dyn.define_dynamics_bounds(dr_use, dat2fit, resvec=[50, 50], D=D)

            mesh = [xvals, mesh_tmp[1]]
        else:
            mesh = None  # should not reach

        resvec = None
    else:
        if verbose:
            print('using PC activity to define bounds')  # this happens inside the pcadynamics() function

        # check how D is handled
        D_dat = rd['D'][reg_idx]

        if D_overwrite is None:
            D = D_dat
        else:
            D = D_overwrite

        assert D < 3, "D is too large. must be 1 or 2"

        if D == 1:  # higher density for 1D
            resvec = [100]
        else:
            resvec = [50, 50]

        mesh = None

    # try running the PCA dynamics.
    if verbose:
        print('running dynamics')

    net = rd['net']
    sgrads, outputs, mesh = dyn.pcadynamics(rnnuse, dr_use, dat2fit, rnn_input, net,
                                            D=D, resvec=resvec, mesh=mesh, nn_reg=numhidden_reg)

    if verbose:
        print('saving: '+savename)

    dyn.savedynamics(sgrads, outputs, mesh, savename, dr_use, rd['D'][reg_idx])
    savename_weights = modelname.replace('.model', '_weights.mat')
    _ = utils.networkweights2matlab(net, savename_weights)

    return sgrads, outputs, mesh

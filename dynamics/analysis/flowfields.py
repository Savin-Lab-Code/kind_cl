# code to calculate the flow fields and stats for each network

# code to extend the plotted region of flow fields, then convert to matlab
import pickle
import json
import torch
import numpy as np
from dynamics.process.rnn import parse, wt_protocols
from dynamics.analysis import dynamics_analysis as dyn


def genflowfields(modelname, dataname, kemin_name, savename, flowops, D=2, block_idx=None):
    """
    loads the data of interest, calculate flow fields. Will calculate a 2D field, but will save true D in .mat file
    @param modelname:
    @param dataname:
    @param kemin_name:
    @param savename: savename of flowfield
    @param flowops:
    @param D: (int) dimensionality of flow field D = 1 or D = 2 right now please
    @param block_idx (int) specific trial number in dataname .json file to use for STR's block inputs, if applicable
    @return:
    """

    reg_idx = flowops['reg_idx']  # 0 = OFC, 1 = STR
    tphase = flowops['tphase']  # see taskdict, below
    epoch = flowops['epoch']  # 'start','wait','reward'
    blocktype = flowops['blocktype']  # 'low','mixed','high. used only for STR
    verbose = flowops['verbose']  # print optimizaiton results as it goes?
    useKEbounds = flowops['useKEbounds']  # if true, load KE min file to calculate effective PC range
    bd = flowops['basedirectory']  # where the config files live for simulation. ~/projects/dynamics/analysis

    tphase_dict = {0: bd + 'block.cfg', 1: bd + 'block.cfg', 2: bd + 'block.cfg',
                   3: bd + 'nocatch.cfg', 4: bd + 'catch.cfg', 5: bd + 'block.cfg'}
    taskdict = {0: 'kind', 1: 'kind', 2: 'inf', 3: 'task', 4: 'task', 5: 'task'}
    dynops = {'numhidden': [256, 256], 'resim': False, 'configname': tphase_dict[tphase], 'epoch': epoch,
              'zeromean': False, 'block_idx': None}

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
        dat = wt_protocols.sim_task(modelname, configname=tphase_dict[tphase], simtype=taskdict[tphase])

    print('preprocessing data')

    rd = dyn.dynamics_preprocess_suite(modelname, dat, simtype=taskdict[tphase], dynops=dynops)

    block = rd['behdict']['blocks']
    offers = rd['behdict']['offers']
    outcomes = rd['behdict']['outcomes']
    ns = len(block)

    print('choosing inputs')
    t_idx = None  # default index
    if block_idx is None:
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
        elif reg_idx == 1:
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
            if epoch == 'start':
                t_idx = rd['tdict']['start'][block_idx]
            elif epoch == 'wait':
                t_idx = rd['tdict']['iti'][block_idx] - 1  # try using KE points right at opt out
            elif epoch == 'iti':
                t_idx = rd['tdict']['iti'][block_idx] + 1
            else:
                print('epoch not specified correctly. cant assign t_idx')
                t_idx = None

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
            print('epoch not specified correctly. cant assign t_idx')
            t_idx = None

    print('using blocktype: ' + blocktype + '. trial is ' + str(block_idx))

    rnn_input = rd['inp'][t_idx, :]
    print(rnn_input)
    i_ofc = np.expand_dims(np.expand_dims(rd['behdict']['preds'][t_idx], axis=0), axis=0)
    print(i_ofc)

    print('parsing regions-specific stuff')
    if reg_idx == 0:
        pca_use = rd['pca_ofc']
        rnnuse = rd['net'].rnn1
        sampsdict_use = rd['sampsdict_ofc']
        i = torch.tensor(rnn_input).to(torch.float32)

    elif reg_idx == 1:
        pca_use = rd['pca_str']
        rnnuse = rd['net'].rnn2
        sampsdict_use = rd['sampsdict_str']
        rnn_input = np.expand_dims(np.expand_dims(rnn_input[:2], axis=0), axis=0)
        print('input check')
        print(rnn_input.shape)
        print(i_ofc.shape)
        i = np.concatenate((rnn_input, i_ofc), axis=2)
        i = torch.tensor(np.squeeze(np.squeeze(i))).to(torch.float32)
    else:
        print('reg_idx not defined correctly. cant set input, i')
        i = None
        sampsdict_use = None
        pca_use = None
        rnnuse = None

    rnn_input = i

    print('calculating pc_means')
    sampsall = []
    for k in sampsdict_use.keys():
        sampsall.extend(sampsdict_use[k])
    # find mean of higher order features
    proj_samps = [pca_use.transform(m) for m in sampsall]
    proj_samps_flat = proj_samps[0]
    for k in proj_samps[1:]:
        proj_samps_flat = np.concatenate((proj_samps_flat, k), axis=0)

    # load the the KE min results and find the full range, then redecide on it. call is 'mesh'
    if useKEbounds:
        if verbose:
            print('using KE min to find good bounds for flowfields')

        kedat1 = pickle.load(open(kemin_name, 'rb'))
        kedat = parse.KEconvert2matlab(kemin_name, dosave=False)

        labs = np.unique(kedat['labels'][kedat['labels'] > -1])
        labs_idx = [np.argwhere(kedat['labels'] == k)[0, 0] for k in labs]
        xvals_fp = [kedat['crds'][k][0] for k in labs_idx]
        xvals_base = [k[0][0] for k in kedat1['crds']]
        if rd['D'][reg_idx] == 1:
            yvals_fp = xvals_fp
            yvals_base = xvals_base
        else:
            yvals_fp = [kedat['crds'][k][1] for k in labs_idx]
            yvals_base = xvals_base

        xmin = np.min([np.min(xvals_base), np.min(xvals_fp)])
        ymin = np.min([np.min(yvals_base), np.min(yvals_fp)])

        xmax = np.max([np.max(xvals_base), np.max(xvals_fp)])
        ymax = np.max([np.max(yvals_base), np.max(yvals_fp)])

        nx = 100
        xvals = np.linspace(xmin*1.2, xmax*1.2, nx)
        yvals = np.linspace(ymin*1.2, ymax*1.2, nx)

        mesh = [xvals, yvals]
        resvec = None
    else:
        if verbose:
            print('using PC activity to define bounds')  # this will happen inside the pcadynamics() function
        mesh = None

        if rd['D'][reg_idx] == 1:  # higher density for 1D
            resvec = [100, 20, 20]
        else:
            resvec = [50, 50, 20]

    # try running the PCA dynamics.
    if verbose:
        print('running dynamics')

    net = rd['net']
    sgrads, outputs, mesh = dyn.pcadynamics(rnnuse, pca_use, sampsdict_use, rnn_input, net, reg_idx=reg_idx,
                                            D=D, resvec=resvec, mesh=mesh)

    if verbose:
        print('saving: '+savename)
    if reg_idx == 0:
        dyn.savedynamics(sgrads, outputs, mesh, rd['samps_pcdict_ofc'], savename, rd['D'][reg_idx], rd['vmask'],
                         rd['behdict']['trials_masked'])
    else:
        dyn.savedynamics(sgrads, outputs, mesh, rd['samps_pcdict_str'], savename, rd['D'][reg_idx], rd['vmask'],
                         rd['behdict']['trials_masked'])

    return sgrads, outputs, mesh, rd['samps_pcdict_ofc']

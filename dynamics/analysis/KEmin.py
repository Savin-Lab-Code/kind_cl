# find fixed points, core code

import sys
import pickle
import json
import ast

import torch
import numpy as np
from scipy.optimize import minimize
from dynamics.analysis import dynamics_analysis as dyn
from dynamics.process.rnn import wt_protocols, parse
from dynamics.utils import utils


def run_min(x):
    """ the function being runin parallell. a minimization"""

    # hard-coded for speed
    s0 = x[0]
    net = x[1]
    pca_use = x[2]
    i = x[3]
    pc_means = x[4]
    reg_idx = x[5]
    D = x[6]
    constrained = x[7]
    verbose = x[8]
    usepcmeans = x[9]

    if verbose:
        if usepcmeans:
            print(s0)
            # print(s0)
            # print(s0.shape)
        else:
            print(s0[0])
            # print(s0.shape)

    # constrained = True

    if constrained:  # do the PCA-constrained minimization
        if reg_idx == 0:
            KEfun = lambda xx: dyn.KEcalc_pcconstrain(xx, s0, net, pca_use, i, pc_means, reg_idx=reg_idx, D=D,
                                                      usepcmeans=usepcmeans)[0]
            KEgradfun = lambda xx: dyn.KEcalc_pcconstrain(xx, s0, net, pca_use, i, pc_means, reg_idx=reg_idx, D=D,
                                                          usepcmeans=usepcmeans)[1]
            KEjacfun = lambda xx: dyn.KEcalc_pcconstrain(xx, s0, net, pca_use, i, pc_means, reg_idx=reg_idx, D=D,
                                                         usepcmeans=usepcmeans)[2]
        elif reg_idx == 1:
            KEfun = lambda xx: dyn.KEcalc_pcconstrain(s0, xx, net, pca_use, i, pc_means, reg_idx=reg_idx, D=D,
                                                      usepcmeans=usepcmeans)[0]
            KEgradfun = lambda xx: dyn.KEcalc_pcconstrain(s0, xx, net, pca_use, i, pc_means, reg_idx=reg_idx, D=D,
                                                          usepcmeans=usepcmeans)[1]
            KEjacfun = lambda xx: dyn.KEcalc_pcconstrain(s0, xx, net, pca_use, i, pc_means, reg_idx=reg_idx, D=D,
                                                         usepcmeans=usepcmeans)[2]
        else:
            print('reg_idx defined wrong')
            KEfun = None
            KEgradfun = None
            KEjacfun = None

        test = minimize(KEfun, s0, jac=KEgradfun)
        x = test['x']
        x_D = x

    else:
        s0.extend(pc_means[D:])
        s0 = np.array(s0)
        s0 = torch.squeeze(torch.tensor(
            pca_use.inverse_transform(np.expand_dims(s0, axis=0)))).to(torch.float32)
        if reg_idx == 0:
            KEfun = lambda xx: dyn.KEcalc(xx, s0, net, pca_use, i, pc_means, reg_idx=reg_idx, D=D)[0]
            KEgradfun = lambda xx: dyn.KEcalc(xx, s0, net, pca_use, i, pc_means, reg_idx=reg_idx, D=D)[1]
            KEjacfun = lambda xx: dyn.KEcalc(xx, s0, net, pca_use, i, pc_means, reg_idx=reg_idx, D=D)[2]

        elif reg_idx == 1:
            KEfun = lambda xx: dyn.KEcalc(s0, xx, net, pca_use, i, pc_means, reg_idx=reg_idx)[0]
            KEgradfun = lambda xx: dyn.KEcalc(s0, xx, net, pca_use, i, pc_means, reg_idx=reg_idx)[1]
            KEjacfun = lambda xx: dyn.KEcalc(s0, xx, net, pca_use, i, pc_means, reg_idx=reg_idx)[2]
        else:
            print('reg_idx not defined correctly')
            KEfun = None
            KEgradfun = None
            KEjacfun = None

        test = minimize(KEfun, s0, jac=KEgradfun)
        x = test['x']
        x_D = pca_use.transform(x.reshape(1, -1))[0, :D]

    # print a summary

    Kf = KEfun(x_D)
    jac = KEjacfun(x_D)
    W = pca_use.components_[:D]
    jac_pc = W @ jac @ W.T
    if verbose:
        Ks = KEfun(s0)
        print([Ks, Kf])

    return x_D, Kf, [jac_pc, np.linalg.eig(jac)[0]]


def KEmin(modelname, dataname, KEops, D_overwrite=None, bd=None):
    """
    The kinetic energy minimization suite. laods 1k.json data, preprocesses to get pca and samples, then runs KE min
    @param modelname: (str) model file to load network
    @param dataname: (str) 1_k.json file of previously simulated data
    @param KEops: (dict) of options for the minimization. mostly subselecting epoch and trial types. see below
    @param D_overwrite: (int) should dimension of the minimization be chnaged? if not None, then use this value
    @param bd: (str) base directory where config files live. if None, assume on cluster
    @return: list of coordinates, kinetic energy, jacobians, and eigenvalues
    """

    reg_idx = KEops['reg_idx']  # 0 = OFC, 1 = STR
    tphase = KEops['tphase']  # see taskdict, below
    epoch = KEops['epoch']  # 'start','wait','reward'
    blocktype = KEops['blocktype']  # 'low','mixed','high. used only for STR
    # num_proc = KEops['num_proc']  # number of processors for the MPI map. only 1 works on the cluster...
    constrained = KEops['constrained']  # do full minimizaiton
    verbose = KEops['verbose']  # print optimizaiton results as it goes?
    choicetype = KEops['choicetype']  # how to choose the trial to get time_idx. 'maxbelief' and 'first"
    usepcmeans = KEops['usepcmeans']  # bool. if false, use actual pc locations in >D space. if true, use pceman resp

    # each phase of training might have different options. these config files have them hard coded
    if bd is None:
        bd = '/home/dh148/projects/dynamics/dynamics/analysis/'
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
        dat = wt_protocols.sim_task(modelname, configname=tphase_dict[tphase], simtype=taskdict[tphase])

    print('preprocessing data')
    rd = dyn.dynamics_preprocess_suite(modelname, dat, simtype=taskdict[tphase], dynops=dynops)

    if D_overwrite is None:
        D = rd['D'][reg_idx]  # the dimensionality of the PC space
    else:
        D = D_overwrite

    block_idx = dynops['block_idx']

    print('choosing inputs')
    t_idx = dyn.chooseinputs(rd, block_idx, reg_idx, epoch, blocktype, choicetype=choicetype)
    print('using blocktype: ' + blocktype + '. trial is ' + str(block_idx))

    rnn_input = rd['inp'][t_idx, :]
    i_ofc = np.expand_dims(np.expand_dims(rd['behdict']['preds'][t_idx], axis=0), axis=0)

    print('parsing regions-specific stuff')
    if reg_idx == 0:
        pca_use = rd['pca_ofc']
        # rnnuse = rd['net'].rnn1
        sampsdict_use = rd['sampsdict_ofc']
        i = torch.tensor(rnn_input).to(torch.float32)

    elif reg_idx == 1:
        pca_use = rd['pca_str']
        sampsdict_use = rd['sampsdict_str']
        inputcase = dat['ops']['inputcase']
        # rnn_input = np.expand_dims(np.expand_dims(rnn_input[:2], axis=0), axis=0)
        rnn_input = np.expand_dims(
            np.expand_dims(utils.getSTRinp_from_OFCinput(
                rnn_input, inputcase=inputcase), axis=0), axis=0)
        i = np.concatenate((rnn_input, i_ofc), axis=2)
        i = torch.tensor(np.squeeze(np.squeeze(i))).to(torch.float32)
    else:
        print('reg_idx not defined correctly. cant set input, i')
        i = None
        sampsdict_use = None
        pca_use = None

    i = torch.unsqueeze(torch.unsqueeze(i, dim=0), dim=0)

    print('calculating pc_means')
    sampsall = []
    for k in sampsdict_use.keys():
        sampsall.extend(sampsdict_use[k])
    # find mean of higher order features
    proj_samps = [pca_use.transform(m) for m in sampsall]
    proj_samps_flat = proj_samps[0]
    for k in proj_samps[1:]:
        proj_samps_flat = np.concatenate((proj_samps_flat, k), axis=0)
    pc_means = np.nanmean(proj_samps_flat, axis=0)

    # decide if the x,y,z mesh needs to be calculated or not
    print('calculating mesh')
    pcranges = dyn.define_dynamics_bounds(pca_use, sampsall, resvec=None, bfrac=0.1, D=D)

    print('deciding on resolution of samples')
    for j in range(D):
        # first check if first dim is very small in range
        if len(pcranges[j]) < 30:
            pcranges[j] = np.linspace(pcranges[j][0], pcranges[j][-1], 30)

        # then coarse grain the sampling
        if KEops['nonuniform_samp'] and D > 2:
            print('diverting to non-uniform sampling for high dim')
            # scale by amount of variance explained
            tfun = lambda v_, i_, T: int((T * v_[i_] ** (len(v_)) / (np.prod(v_))) ** (1 / (len(v_))))
            v = pca_use.explained_variance_ratio_[:D]
            Nm = np.max([2, tfun(v, j, 1000)])
            pcranges[j] = np.linspace(pcranges[j][0], pcranges[j][-1], Nm)
        else:  # for lower dim
            print('Skipping nonuniform sampling')
            resj = int(len(pcranges[j]) / 30)
            pcranges[j] = pcranges[j][::resj]

    net = rd['net']

    pcmesh = np.meshgrid(*pcranges)
    crds_base = np.concatenate(tuple(k.reshape(-1, 1) for k in pcmesh), axis=1)
    nc = crds_base.shape[0]
    print(nc)

    if not usepcmeans:
        print('using visited locations in full-dim space, but sampled uniformly as best as possible in D-dim space')
        # for every point in pcranges, find the closest real activity point in full pc space and use that
        crds_temp = []
        proj_samps_flat_lowD = np.array([k[0:D] for k in proj_samps_flat])
        print(proj_samps_flat_lowD.shape)
        best_idxs = []
        for k in range(nc):
            best_point_idx = np.argmin(np.array(np.linalg.norm(crds_base[k, :] - proj_samps_flat_lowD, axis=1)))
            if best_point_idx not in best_idxs:  # avoid duplicates
                crds_temp.append(proj_samps_flat[best_point_idx])  # use entire PC space
                best_idxs.append(best_point_idx)
        crds_base = np.array(crds_temp)
        nc = crds_base.shape[0]  # after removing duplicates
        print(nc)
    else:
        print('using mean responses in dimensions above D for initial starting points to KE min')

    crds = []
    for k in range(nc):
        crds.append([crds_base[k, :], net, pca_use, i, pc_means, reg_idx, D, constrained, verbose, usepcmeans])

    del rd
    output = []

    print('checks 262')
    print(len(crds))
    print(nc)

    for k in crds:
        output.append(run_min(k))
    # with mp.Pool(num_proc) as pool:
    #    print('running')
    #    output = pool.map(run_min, crds)
    # output = []
    # for k in crds:
    #    output.append(run_min(k))

    return crds, output


def main():
    num = int(sys.argv[1])  # RNN number
    s_idx = int(sys.argv[2])  # full(0), nok_cl(1), nok_nocl (2)
    tphase = int(sys.argv[3])  # simple(0), int(1), pred(2), nocatch(3), catch(4), block(5)
    idx = int(sys.argv[4])  # index within that phase of learning. remember that int and pred can be variable amounts
    epoch = sys.argv[5]  # "wait", "start", "iti"
    num_proc = int(sys.argv[6])  # nuymber of processors
    constrained = bool(sys.argv[7])  # constrained or unconstrained minimizations
    reg_idx = int(sys.argv[8])  # which region to do dynamics for
    blocktype = sys.argv[9]  # must be mixed, low, high, inf, iind
    legacy = ast.literal_eval(sys.argv[10])
    dbase = sys.argv[11]
    savedir = sys.argv[12]

    # the full taxonomy of names
    subdirlist = ['full_cl_redo/', 'nok_cl/', 'nok_nocl/', 'pkind_mem/', 'pkind_pred/', 'pkind_int/', 'full_cl/']
    # dbase = '/scratch/dh148/dynamics/results/rnn/ac/20230206_clstudy/'
    datadir_dat = lambda s_idx_: dbase + subdirlist[s_idx_]

    # savedir = dbase+subdirlist[s_idx]+'dynamics/KEmin_constrained/wdim/'

    fgen = lambda num_, idx_, base, sess, s_idx_: (datadir_dat(s_idx_) + str(num_) + '/' + base + str(num_) +
                                                   '_' + sess + '_' + str(idx_))

    fname_funs = [
        lambda num_, idx_, s_idx_: datadir_dat(s_idx_) + str(num_) + '/' + 'rnn_kindergarten_' + str(num_) + '_simple',
        lambda num_, idx_, s_idx_: datadir_dat(s_idx_) + str(num_) + '/' + 'rnn_kindergarten_' + str(
            num_) + '_int_0_' + str(idx_),
        lambda num_, idx_, s_idx_: datadir_dat(s_idx_) + str(num_) + '/' + 'rnn_pred_' + str(num_) + '_' + str(idx_),
        lambda num_, idx_, s_idx_: fgen(num_, idx_, 'rnn_curric_', 'nocatch', s_idx_),
        lambda num_, idx_, s_idx_: fgen(num_, idx_, 'rnn_curric_', 'catch', s_idx_),
        lambda num_, idx_, s_idx_: fgen(num_, idx_, 'rnn_curric_', 'block', s_idx_),
        lambda num_, idx_, s_idx_: fgen(num_, idx_, 'rnn_curric_', 'block', s_idx_) + '_freeze'
        ]

    # load network and previous data
    modelname = fname_funs[tphase](num, idx, s_idx) + '.model'
    if legacy:
        dataname = (dbase + subdirlist[s_idx] + 'dynamics/KEmin_constrained/' +
                    fname_funs[tphase](num, idx, s_idx).split('/')[-1] + '_1k_reparse.dat')
    else:
        dataname = fname_funs[tphase](num, idx, s_idx) + '_1k.json'
    print(modelname)
    print(dataname)

    KEops = {'reg_idx': reg_idx, 'tphase': tphase, 'epoch': epoch, 'blocktype': blocktype, 'num_proc': num_proc,
             'constrained': constrained, 'verbose': False, 'nonuniform_samp': True}

    output, crds = KEmin(modelname, dataname, KEops)

    # run the code

    if reg_idx == 0:
        basename = modelname.split('/')[-1].split('.')[0]
    else:
        basename = modelname.split('/')[-1].split('.')[0] + '_' + blocktype

    pickle.dump({'crds': crds, 'output': output},
                open(savedir + 'kemin_' + basename + 'reg_' + str(reg_idx) + '_' + epoch + '.dat',
                     'wb'))


if __name__ == '__main__':
    main()

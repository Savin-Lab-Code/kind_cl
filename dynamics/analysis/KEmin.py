# find fixed points, core code

import pickle
import json
import copy
from os.path import exists
from sklearn.cluster import DBSCAN as dbscan
from sklearn.metrics import pairwise_distances
from scipy.io import savemat
from scipy.linalg import schur

import torch
import numpy as np
from scipy.optimize import minimize
from dynamics.analysis import dynamics_analysis as dyn
from dynamics.process.rnn import wt_protocols, parse


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

    if verbose:
        print(s0, end='\r')
    
    if constrained:  # do the PCA-constrained minimization
        if reg_idx == 0:
            KEfun = lambda xx: dyn.KEcalc_pcconstrain(xx, s0, net, pca_use, i, pc_means, reg_idx=reg_idx, D=D)[0]
            KEgradfun = lambda xx: dyn.KEcalc_pcconstrain(xx, s0, net, pca_use, i, pc_means, reg_idx=reg_idx, D=D)[1]
            KEjacfun = lambda xx: dyn.KEcalc_pcconstrain(xx, s0, net, pca_use, i, pc_means, reg_idx=reg_idx, D=D)[2]
        elif reg_idx == 1:
            KEfun = lambda xx: dyn.KEcalc_pcconstrain(s0, xx, net, pca_use, i, pc_means, reg_idx=reg_idx, D=D)[0]
            KEgradfun = lambda xx: dyn.KEcalc_pcconstrain(s0, xx, net, pca_use, i, pc_means, reg_idx=reg_idx, D=D)[1]
            KEjacfun = lambda xx: dyn.KEcalc_pcconstrain(s0, xx, net, pca_use, i, pc_means, reg_idx=reg_idx, D=D)[2]
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


def KEmin(modelname, dataname, KEops, D=None):
    """
    The kinetic energy minimization suite. laods 1k.json data, preprocesses to get pca and samples, then runs KE min
    @param modelname: (str) model file to load network
    @param dataname: (str) 1_k.json file of previously simulated data
    @param KEops: (dict) of options for the minimization. mostly subselecting epoch and trial types. see below
    @param D: (int) PC dimension where KE is done. if None, set by PCA variance explained
    @return: list of coordinates, kinetic energy, jacobians, and eigenvalues
    """

    reg_idx = KEops['reg_idx']  # 0 = OFC, 1 = STR
    tphase = KEops['tphase']  # see taskdict, below
    epoch = KEops['epoch']  # 'start','wait','reward'
    blocktype = KEops['blocktype']  # 'low','mixed','high. used only for STR
    constrained = KEops['constrained']  # do full minimizaiton
    verbose = KEops['verbose']  # print optimizaiton results as it goes?
    bd = KEops['basedirectory']  # where config files are

    # each phase of training might have different options. these config files have them hard coded
    # TODO set bd more generally
    # bd = '/home/dh148/projects/dynamics/dynamics/analysis/'
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

    if D is None:
        D = rd['D'][reg_idx]  # the dimensionality of the PC space

    block_idx = dynops['block_idx']
    block = rd['behdict']['blocks']
    offers = rd['behdict']['offers']
    outcomes = rd['behdict']['outcomes']
    ns = len(block)

    print('choosing inputs')
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
                print('epoch not speicified correctly. cant assign t_idx')
                t_idx = None

        elif reg_idx == 1:
            # choose if ofc inputs should be based on a trial that was an opt-out or a rewarded
            if tphase > 3:
                octype = 0  # use an opt-out trial's input to visualize that trial trajecotry in flowfields
            else:
                octype = 1  # opt-out might not exist in nocatch phase
            if blocktype == 'mixed':
                block_idx = np.argwhere((block[:ns] == 0) & (outcomes[:ns] == octype)
                                        & (np.exp(offers[:ns]).astype(int) == 20))[0, 0]
            elif blocktype == 'high':
                block_idx = np.argwhere((block[:ns] == 1) & (outcomes[:ns] == octype)
                                        & (np.exp(offers[:ns]).astype(int) == 20))[0, 0]
            elif blocktype == 'low':
                block_idx = np.argwhere((block[:ns] == 2) & (outcomes[:ns] == octype)
                                        & (np.exp(offers[:ns]).astype(int) == 20))[0, 0]
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
                print('epoch not speicified correctly. cant assign t_idx')
                t_idx = None
        else:
            print('reg_idx not defined correctly')
            t_idx = None
            block_idx = None
    else:
        t_idx = rd['tdict']['iti'][block_idx] - 1

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
        # rnnuse = rd['net'].rnn2
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
            print('Skipping nonuniform?')
            resj = int(len(pcranges[j]) / 30)
            pcranges[j] = pcranges[j][::resj]

    net = rd['net']
    pcmesh = np.meshgrid(*pcranges)
    crds_base = np.concatenate(tuple(k.reshape(-1, 1) for k in pcmesh), axis=1)
    nc = crds_base.shape[0]
    crds = []
    for k in range(nc):
        crds.append([crds_base[k, :], net, pca_use, i, pc_means, reg_idx, D, constrained, verbose])

    del rd
    output = []
    for k in crds:
        output.append(run_min(k))

    return crds, output


def KEprocess(fname, dosave=True, eps=3, doadaptive=False, savedir=None):
    """
    extracts the fixed and slow points from KE minization file, clusters with DBSCAN,
    then converts to a matlab file
    :param fname: file name of kinetic energy minima located by minimization
    :param dosave: if true, saves .mat file. otherwise just returns dict
    :param eps: the DBscan parameter
    :param doadaptive: should the epsilon parameter be chosen adaptive for each network, based on PC1 range?
    :param savedir: if None, save in original data dir. this helps for trying different hyperparams
    :return: (dict) of coordinates, schur modes, eigenvalues, etc.
    """

    if not exists(fname):
        # print('data file does not exist')
        return None

    if fname.split('.')[-1] == 'dat':
        # print('loading legacy .dat data')
        with open(fname, 'rb') as f:
            dat = pickle.load(f)

    # format everything into a matlab-friendly thing
    ns = len(dat['output'])
    crds = np.array([dat['output'][k][0] for k in range(ns)])
    KE = np.array([dat['output'][k][1] for k in range(ns)])
    jac_pc = np.array([dat['output'][k][2][0] for k in range(ns)])
    evals_full = np.array([dat['output'][k][2][1] for k in range(ns)])
    D = crds.shape[1]

    # filter the data into slow and fast points, then cluster to get the effecgtive points
    # create cutoffs for fixed points and slow points. filter data
    mask_fixed = KE < 0.0001

    # define slow points that only change magnitude < 5% across duration of trial
    lam = 2.5
    eps_norm = 0.05
    sdiffnorm = 2 * np.sqrt(KE) * lam / (np.linalg.norm(crds, axis=1))
    mask_slow = sdiffnorm < eps_norm

    mask = (mask_fixed) | (mask_slow)
    # potentially get ride of this
    mask = np.ones(mask.shape).astype(bool)

    crds_masked = crds[mask, :]
    KE_masked = KE[mask]
    jac_pc_masked = jac_pc[mask, :, :]
    sdiffnorm_masked = sdiffnorm[mask]
    isfixed_masked = KE_masked < 0.0001
    ns_masked = sum(mask)

    # decide on hyperparameter
    if doadaptive:
        xvals = [k[0][0] for k in dat['crds']]
        xrange = max(xvals) - min(xvals)
        eps = np.abs(xrange * eps)
        if eps == 0:
            print('issue with range')
            return None

    # find number of dynamical systems features
    clustering = dbscan(eps=eps, min_samples=10).fit(crds_masked)
    labs = np.unique(clustering.labels_)
    labels = clustering.labels_

    # analyze the spread of each feature. relevant only for line attractors, but that analysis is later
    featurelen = []
    for k in labs:
        pts = crds_masked[labels == k]
        featurelen.append(np.max(pairwise_distances(pts)))

    fetdict = dict(zip(labs, featurelen))

    # determine if each labeled feature is in range. give it a bit of a buffer of 30%
    # if not, discard it as not meainginful
    min_pcvals = []
    max_pcvals = []
    for m in range(D):
        min_pcvals.append(1.3 * np.min(np.array([k[0][m] for k in dat['crds']])))
        max_pcvals.append(1.3 * np.max(np.array([k[0][m] for k in dat['crds']])))

    inrange = []  # is each masked fixed/slow point within bounds of PC_min and PC_max?
    for k in crds_masked:
        cond_all = True
        for m in range(D):
            cond_m = k[m] > min_pcvals[m] and k[m] < max_pcvals[m]  # if out of range for any PC
            cond_all = cond_all and cond_m
        inrange.append(cond_all)

    savedict = {'ns': ns_masked, 'crds': crds_masked, 'KE': KE_masked, 'jac_pc': jac_pc_masked,
                'evals_full_masked': evals_full, 'statediff': sdiffnorm_masked, 'labels': labels,
                'isfixed': isfixed_masked, 'fetdict': fetdict, 'D': D, 'inrange': inrange}
    if dosave:
        if savedir is None:
            savemat(fname.split('.')[0] + '.mat', savedict)
        else:
            savename = savedir + fname.split('/')[-1].split('.')[0] + '.mat'
            savemat(savename, savedict)
    return savedict


def classify_gen(eigs, D=2):
    """will classify eigenvalues in arbitraty dim as
      - attractor (stable) both < 1
      - source (unstable) both > 1
      - saddle (semistable eig_1 < 1, eig2 > 1)
      - line attractor (one eig is unity, within a tolerance)

      (handle these later. need to check textbook)
      - stablespiral (complex eigenvalues), norm < 1
      - unstable spiral (complex)
      - orbit (purely imaginary)"""

    tol1_low = 0.999
    tol1_high = 1.001

    ltmask = eigs < tol1_low  # attractors
    gtmask = eigs > tol1_high  # unstable
    linemask = (eigs > tol1_low) & (eigs < tol1_high)  # lines

    if np.sum(ltmask) == D:
        return 'attractor'
    elif np.sum(gtmask) == D:
        return 'source'
    elif np.sum(linemask) > 0 and np.sum(gtmask) > 0:
        return 'unstable_line'
    elif np.sum(linemask) > 0:
        return 'line'
    elif np.sum(ltmask) > 0 and np.sum(gtmask) > 0:
        return 'saddle'
    else:
        print('i missed something' + str(eigs))
        return None


def getnpts(dirnames, tphase, subtype='fixed', eps=3, doadaptive=False, savedir=None, restrictrange=False):
    """
    helper script to get number and type of fixed points
    if savedir is None: defautl to saving things in datadir. useful for different eps runs
    restrictrange is for excluding feature outside range of PC activity
    @param dirnames: (list) of filenames to open
    @param tphase: (list) of t_phase of training for each network
    @param subtype: (str) of which type of fixed points to count
    @param eps: (float) DBSCAN parameter if not using adaptive
    @param doadaptive: (bool) used adaptive eps parameter based on size of PC dim
    @param savedir: (str) directory to save matlab file. if None, dont save
    @param restrictrange: (bool) only keep features in PC range
    @return:
    """
    nfixed_all = []
    nslow_all = []
    ntot = []

    nattractor_all = []
    nsaddle_all = []
    nline_all = []
    nsource_all = []
    nusline_all = []

    dirnamecpy = copy.deepcopy(dirnames)
    tphasecpy = copy.deepcopy(tphase)

    fetlen_all = []
    fetlentype_all = []
    inrange_all = []
    Dall = []

    for m in dirnames:
        # try:
        if savedir is None:
            dosave = False
        else:
            dosave = True

        dat = KEprocess(m, dosave=dosave, eps=eps, doadaptive=doadaptive, savedir=savedir)

        if dat is None:
            nfixed = np.nan
            nslow = np.nan
            nattractor = np.nan
            nsaddle = np.nan
            nline = np.nan
            nsource = np.nan
            npts = []
            D = np.nan

            fetlen = []
            fetlentype = []

        else:
            D = dat['D']

            if restrictrange:
                mask = dat['inrange']
            else:
                mask = np.ones(len(dat['labels'])).astype(bool)

            npts = np.unique(dat['labels'][np.array(dat['labels'] > -1) & np.array(mask)])

            nfixed = 0
            nslow = 0
            nattractor = 0
            nsaddle = 0
            nline = 0
            nsource = 0
            nusline = 0

            # maximum length of feature, and its type
            fetlen = []
            fetlentype = []

            # also track if feature is is in the range of network activity
            inrange = []

            for k in npts:
                pt_k_idx = np.argwhere(dat['labels'] == k)[0, 0]

                fetlen.append(dat['fetdict'][k])
                inrange.append(dat['inrange'][k])

                if dat['isfixed'][pt_k_idx]:
                    nfixed += 1
                else:
                    nslow += 1

                test = schur(dat['jac_pc'][pt_k_idx], output='real')
                evals = np.diag(test[0])
                c = classify_gen(evals, D)

                # add points only if following condition is met
                cond1 = dat['isfixed'][pt_k_idx] and subtype == 'fixed'  # update only if fixed
                cond2 = not dat['isfixed'][pt_k_idx] and subtype == 'slow'  # update only if slow
                cond3 = subtype == 'all'  # update all of them

                # only update if fixed
                if cond1 or cond2 or cond3:
                    if c == 'attractor':
                        nattractor += 1
                        fetlentype.append('attractor')
                    elif c == 'saddle':
                        nsaddle += 1
                        fetlentype.append('saddle')
                    elif c == 'line':
                        nusline += 1
                        fetlentype.append('line')
                    elif c == 'unstable_line':
                        nline += 1
                        fetlentype.append('unstable_line')
                    elif c == 'source':
                        nsource += 1
                        fetlentype.append('source')
                    else:
                        print('none chosen')

        nfixed_all.append(nfixed)
        nslow_all.append(nslow)
        ntot.append(len(npts))
        nattractor_all.append(nattractor)
        nsaddle_all.append(nsaddle)
        nline_all.append(nline)
        nusline_all.append(nusline)
        nsource_all.append(nsource)
        fetlen_all.append(fetlen)
        fetlentype_all.append(fetlentype)
        Dall.append(D)
        inrange_all.append(inrange)

    typelist = [nattractor_all, nsaddle_all, nline_all, nsource_all, nusline_all]

    return nfixed_all, nslow_all, ntot, typelist, dirnamecpy, tphasecpy, fetlen_all, fetlentype_all, Dall, inrange_all

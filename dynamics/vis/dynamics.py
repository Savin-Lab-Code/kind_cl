# will plot dynamical systems stuff

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.linalg import schur
import cmath
import networkx as nx
import torch
import dynamics.analysis as dyn
from dynamics.process.rnn import parse
import pickle

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def plotspectra(evals, eps=0.2, txt='', goodvals=None):
    """
    plots spectra of a linearized dynamics. returns numerical non-zero eigenvalues
    :param evals:
    :param eps
    :param txt:
    :param goodvals: if supplied, everything else is grayed out. otherwise choose epsilon ball to gray
    :return:
    """
    f = plt.figure(figsize=(7, 7))
    plt.vlines(0, -1, 1, color='gray')
    plt.hlines(0, -1, 1, color='gray')
    enorms = np.array([np.abs(k) for k in evals])
    keep_idx = np.argwhere(enorms > eps)
    ekeep = evals[enorms > eps]
    elose = evals[enorms <= eps]

    if goodvals is not None:
        goodvals = goodvals[enorms[goodvals] > eps]

        badvals = ~np.isin(np.array(range(len(evals))), goodvals)
        plt.scatter(np.real(evals[badvals]), np.imag(evals[badvals]), alpha=0.05, color='gray')
        plt.scatter(np.real(evals[badvals]), -np.imag(evals[badvals]), alpha=0.05, color='gray')
        plt.scatter(np.real(evals[goodvals]), np.imag(evals[goodvals]), color='red')
        plt.scatter(np.real(evals[goodvals]), -np.imag(evals[goodvals]), color='red')
        keep_idx = goodvals
    else:
        plt.scatter(np.real(elose), np.imag(elose), alpha=0.05, color='gray')
        plt.scatter(np.real(ekeep), np.imag(ekeep), color='red')

    circle1 = plt.Circle((0, 0), 1, facecolor='None', edgecolor='gray')
    plt.gca().add_patch(circle1)
    plt.xlabel('real')
    plt.ylabel('imag')
    plt.title('eivenvalue spetra: ' + txt)
    emaxnorm = np.max(np.abs(evals))
    emaxnorm = np.max([1, emaxnorm])
    plt.xlim([-emaxnorm-0.5, emaxnorm+0.5])
    plt.ylim([-emaxnorm-0.5, emaxnorm+0.5])
    plt.show()

    return f, keep_idx


def plotSchur(dFdh, s, txt='', eps=0.2, eps_real=1e-4, eps_stable=0.05, makenetwork=False):
    """
    visualized Schur modes as a feed-forward network. attempt to make these results more digestible in graph form
    :param dFdh: Jacobian from dynamics linearization. just w.r.t dh though, not wrt state or inputs
    :param s: the state
    :param txt: plotting test
    :param eps: threhold for being considered a zero: norm < eps
    :param eps_real: threshold on imaginary part to be considered real
    :param eps_stable: threshold from 1 for being considered stable mode
    :paeram makenetwork: (bool) to decide if network plot should be made. hard for large networks
    :return:
    """
    # look at schur modes
    # isolate and classify main modes and transitions
    # either decyaing(=1)/unstable(>1)/stable(~1) and/or damped(complex) pure (real).
    [T, U] = schur(dFdh, output='complex')
    ns = len(T)

    # what to keep
    idx_good = np.squeeze(np.argwhere(np.abs(np.diag(T)) > eps))
    ngood = len(idx_good)
    T_keep = np.zeros((ngood, ngood), dtype='complex')
    U_keep = np.zeros((ns, ngood), dtype='complex')
    for k in range(ngood):
        T_keep[k, :] = T[idx_good[k], idx_good]
        U_keep[:, k] = U[:, idx_good[k]]

    mask = np.abs(np.imag(T_keep)) < eps_real
    T_keep[mask] = np.real(T_keep[mask])

    f0, ax = plt.subplots(1, 2)
    plt.tight_layout()
    im1 = ax[0].imshow(np.real(T_keep))
    ax[0].set_xlabel('mode')
    ax[0].set_title('reduced Schur components (Re[T])')
    f0.colorbar(im1, ax=ax[0])

    im2 = ax[1].imshow(np.imag(T_keep))
    ax[1].set_xlabel('mode')
    ax[1].set_title('(Im[T])')
    f0.colorbar(im2, ax=ax[1])

    # grab cc pairs
    idx_pair = lambda x, idx: np.argwhere([cmath.isclose(k.conj(), x[idx]) for k in x])[0][0]
    # iterate over the good indices, combining cc pairs and collecting info
    eval_list = []
    nkeep = len(T_keep)
    for k in range(nkeep):
        pairtest = idx_pair(np.diag(T_keep), k)
        if pairtest >= k:
            eval_list.append([k, pairtest])

    if makenetwork:
        # start building a graph
        G = nx.MultiDiGraph()
        for k in eval_list:
            lam = T_keep[k[0], k[0]]
            evec = U_keep[:, k[0]]

            # add the node and its eigenvectors
            G.add_nodes_from([k[0]])

            # decide on color
            if np.abs(np.real(lam)) > 1.0:
                color = 'red'  # unstable node
            elif 1.0 - np.real(lam) < eps_stable:
                color = 'green'
            elif np.abs(np.real(lam)) < 1.0:
                color = 'cyan'

            if np.abs(np.imag(lam)) < eps_real:
                mkr = 's'  # purely real
            else:
                mkr = 'o'

            G.nodes[k[0]]['lam'] = lam
            G.nodes[k[0]]['evec'] = evec
            G.nodes[k[0]]['color'] = color
            G.nodes[k[0]]['mkr'] = mkr

        # add edges directed
        for k in range(len(eval_list)):
            for m in range(k + 1, len(eval_list)):
                weight = np.max([np.abs(T_keep[eval_list[k][0], eval_list[m][0]]),
                                 np.abs(T_keep[eval_list[k][1], eval_list[m][0]]),
                                 np.abs(T_keep[eval_list[k][0], eval_list[m][1]]),
                                 np.abs(T_keep[eval_list[k][1], eval_list[m][1]])])

                ebatch = [(eval_list[k][0], eval_list[m][0], weight)]
                np.argmax(np.array(ebatch))

                G.add_edges_from(ebatch)

        print(eval_list)

        # try formatting the nodes
        f = plt.figure(figsize=(7, 7))
        # pos = nx.circular_layout(G)
        # pos = nx.spring_layout(G)
        # pos = nx.planar_layout(G)
        pos = nx.spiral_layout(G)
        smax = max([np.abs(G.nodes[k]['lam']) for k in G.nodes])
        # smin = min([np.abs(G.nodes[k]['lam']) for k in G.nodes])

        sizelist = []
        for k in G.nodes:
            color = G.nodes[k]['color']
            mkr = G.nodes[k]['mkr']
            size = np.abs(G.nodes[k]['lam']) / (smax) * 3000
            sizelist.append(size)
            nx.draw_networkx_nodes(G, pos, nodelist=[k], node_size=size, node_color=color, node_shape=mkr, alpha=0.7)

        # node labes
        labels = [(k, "%d:\n  %3.2f + \n %3.2f j" % (k, np.real(G.nodes[k]['lam']),
                                                     np.imag(G.nodes[k]['lam']))) for k in G.nodes]
        labels = dict(labels)
        nx.draw_networkx_labels(G, pos, labels)

        # edes
        weights = np.array([k[2] for k in G.edges])
        weight_max = max(weights)
        weights = weights / weight_max * 3
        nx.draw_networkx_edges(G, pos, width=weights, arrows=True, node_size=sizelist, arrowsize=20)

        # edge labels
        labels = [((k[0], k[1]), "%3.2f" % k[2]) for k in G.edges]
        labels = dict(labels)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

        # plt.xlim([xmin-1, xmax + 1])
        # plt.ylim([ymin - 1, ymax + 1])
        plt.title('Schur network: ' + txt)
        plt.show()
    else:
        f = None

    # mode overlap and eigenvalues
    overlaps = []
    hnorm = np.linalg.norm(np.dot(s, s))
    for k in eval_list:
        overlaps.append([np.linalg.norm(np.dot(s, U_keep[:, k[0]])) / hnorm,
                         np.linalg.norm(np.dot(s, U_keep[:, k[1]])) / hnorm])

    # overlap with all modes, for comparison
    overlaps_all = []
    for k in range(ns):
        overlaps_all.append(np.linalg.norm(np.dot(s, U[:, k])) / hnorm)

    # plot all
    g, ax = plt.subplots(1, 2)
    plt.tight_layout()
    ax[1].plot(np.sort(overlaps_all)[::-1])
    ax[1].set_title('state overlap, all Schur modes')
    ax[1].set_xlabel('modes')

    ind = 0
    for k in overlaps:
        ax[0].bar(ind - 0.2, k[0], label=str(eval_list[ind]), color='blue', width=0.4)
        ax[0].bar(ind + 0.2, k[1], color='blue', width=0.5)
        ind += 1

    labels = [str(k) for k in G.nodes]

    ax[0].set_xticks(range(len(overlaps)), labels=labels)
    ax[0].set_xlabel('mode')
    ax[0].set_ylabel('normed overlap')
    ax[0].set_title('state overlap, non-zero schur modes')
    plt.show()

    return f, g


def sampleactivity(trial_idx, sflat, inp, tdict, txt='', nstate=30, nreg=2, celltype='hidden'):
    """
    will plot activity from each region (if there are two) alongside trial timing information
    :param trial_idx: trial index
    :param sflat: (tensor) of dimensions [nt][nreg][h,c][1,1,nn]
    :param inp: (ndarray) of dim [nt,din]. the general input before parse_inputs splits it for each region
    :param tdict: (dict) of timing indices for each trial event
    :param txt: (str) plotting text
    :param nstate: (int) how many traces to plot
    :param nreg: (int) number of regions in network. usually 2: OFC,STR
    :param celltype: (str) 'hidden' or 'cell' activity
    :return:
    """
    # trial offer
    # offer = np.exp(inpflat[tdict['start'][trial_idx]][0, 0, 0].detach().numpy())  # offer on that trial
    offer = np.exp(inp[tdict['start'][trial_idx], 0])
    print(offer)

    if celltype == 'hidden':
        cidx = 0
    elif celltype == 'cell':
        cidx = 1
    else:
        cidx = 0


    # data for that trial
    if nreg > 1:  # assume 2 for now
        s_trial_ofc = [k[0] for k in sflat[tdict['start'][trial_idx]:tdict['end'][trial_idx]]]
        s_trial_str = [k[1] for k in sflat[tdict['start'][trial_idx]:tdict['end'][trial_idx]]]
    else:  # doa  redundant plot
        s_trial_ofc = sflat[tdict['start'][trial_idx]:tdict['end'][trial_idx]]
        s_trial_str = sflat[tdict['start'][trial_idx]:tdict['end'][trial_idx]]

    # inp_trial = inpflat[tdict['start'][trial_idx]:tdict['end'][trial_idx]]
    inp_trial = inp[tdict['start'][trial_idx]:tdict['end'][trial_idx], :]
    print('here 258')
    print(inp_trial.shape)
    start_trial = tdict['start'][trial_idx]
    iti_trial = tdict['iti'][trial_idx]

    f = plt.figure()
    for k in range(nstate):
        sj_ofc = [j[cidx][0, 0, k].detach().numpy() for j in s_trial_ofc]
        plt.plot(sj_ofc)

    plt.vlines(iti_trial - start_trial, -1, 1, color='gray')
    titletext = "OFC sample {:s} activity, trial {:d}, offer val= {:d}: {:s}".format(celltype,trial_idx,
                                                                                  int(offer), txt)
    plt.title(titletext)
    plt.show()

    h = plt.figure()
    for k in range(nstate):
        sj_str = [j[cidx][0, 0, k].detach().numpy() for j in s_trial_str]
        plt.plot(sj_str)

    plt.vlines(iti_trial - start_trial, -1, 1, color='gray')
    titletext = "STR sample {:s} activity, trial {:d}, offer val= {:d}: {:s}".format(celltype,trial_idx,
                                                                                  int(offer), txt)
    plt.title(titletext)
    plt.show()

    # inputs
    g, ax = plt.subplots(2, 1)
    ltext = ['log(offer)', 'trial start', 'prev. reward (raw value)', 'prev action (0=wait,1=optout, 2=iti)']
    for k in [0, 1, 3]:
        # ax[0].plot([m[0, 0, k] for m in inp_trial], label=ltext[k])
        ax[0].plot(inp_trial[:, k], label=ltext[k])

    ax[0].vlines(iti_trial - start_trial, -1, 1, color='gray')
    ax[0].set_xlabel('timesteps')
    ax[0].set_ylabel('inputs')
    ax[0].set_title('inputs')
    ax[0].legend()

    for k in [2]:
        #ax[1].plot([m[0, 0, k] for m in inp_trial], label=ltext[k], color='k')
        ax[1].plot(inp_trial[:, k], label=ltext[k])

    ax[1].vlines(iti_trial - start_trial, -1, 1, color='gray')
    ax[1].set_xlabel('timesteps')
    ax[1].set_ylabel('inputs')
    ax[1].set_title('inputs')
    ax[1].legend()
    plt.show()

    return f, g


def dominant_schur(dFdh, state, thresh=0.1):
    """
    will calculate overlap of state with schur modes, and give indices of ones with substantial overlap
    :param dFdh: linearized dynamics
    :param state: input state
    :param thresh: threshold on relative overlap
    :return:
    """
    [_, U] = schur(dFdh, output='complex')
    nn_state = len(state)
    overlap = np.zeros((nn_state,))
    snorm = np.linalg.norm(state)
    for k in range(nn_state):
        overlap[k] = abs(np.dot(state, U[k, :])) / snorm

    good_idx = np.argwhere(overlap > thresh)
    return overlap, good_idx


def genflowfields(num, reg_idx, s_idx, tphase, idx, blocktype, epoch, D, verbose=True):
    # tphase = 5 #simple(0), int(1), pred(2), nocatch(3), catch(4), block(5)
    # idx = 3 #index within that phase of learning. remember that int and pred can be variable amounts
    # num = 5  # RNN number
    # s_idx = 0  #full(0), nok_cl(1), nok_nocl (2)
    # epoch = "wait" #"wait", "start", "iti"
    # D = 2 # dimensionality being investigated. choose 2 or 3 please
    # reg_idx = 0  # which region to do dynamics for
    # blocktype = 'mixed'  # mixed, low, high

    num_proc = 1  # nuymber of processors
    constrained = True  # constrained or unconstrained minimizations

    tpdict = {3: 'nocatch', 4: 'catch', 5: 'block'}

    # load path information. TODO. make modular at some point

    # the full taxonomy of names
    # TODO: make this modular soon
    subdirlist = ['full_cl_redo/', 'nok_cl/', 'nok_nocl/', 'pkind_mem/', 'pkind_pred/', 'pkind_int/', 'full_cl/']
    dbase = '/scratch/dh148/dynamics/results/rnn/ac/20230206_clstudy/'
    datadir_dat = lambda s_idx: dbase + subdirlist[s_idx]  # where models live
    savedir = dbase + subdirlist[s_idx] + 'dynamics/KEmin_constrained/'  # where dat file of simulated data lives
    savedir_flows = savedir + 'flows/'  # where flowfields will be saved

    fgen = lambda num, idx, base, sess, s_idx: datadir_dat(s_idx) + str(num) + '/' + base + str(
        num) + '_' + sess + '_' + str(idx)

    fname_funs = [
        lambda num, idx, s_idx: datadir_dat(s_idx) + str(num) + '/' + 'rnn_kindergarten_' + str(num) + '_simple',
        lambda num, idx, s_idx: datadir_dat(s_idx) + str(num) + '/' + 'rnn_kindergarten_' + str(num) + '_int_0_' + idx,
        lambda num, idx, s_idx: datadir_dat(s_idx) + str(num) + '/' + 'rnn_pred_' + str(num) + '_pred_' + idx,
        lambda num, idx, s_idx: fgen(num, idx, 'rnn_curric_', 'nocatch', s_idx),
        lambda num, idx, s_idx: fgen(num, idx, 'rnn_curric_', 'catch', s_idx),
        lambda num, idx, s_idx: fgen(num, idx, 'rnn_curric_', 'block', s_idx),
        lambda num, idx, s_idx: fgen(num, idx, 'rnn_curric_', 'block', s_idx) + '_freeze'
        ]

    # load network and previous data
    modelname = fname_funs[tphase](num, idx, s_idx) + '.model'
    kemin_name = datadir_dat(s_idx) + 'dynamics/KEmin_constrained/' + 'kemin_rnn_curric_' + str(num) + '_' + tpdict[
        tphase] + '_' + str(idx) + 'reg_' + str(reg_idx) + '_D2_' + epoch + '.dat'

    if verbose: print(modelname)
    if verbose: print([reg_idx, blocktype, epoch])
    if verbose: print(kemin_name)

    if verbose: print('running pre-processing')

    # each phase of training might have different options. these config files have them hard coded
    tphase_dict = {0: 'block.cfg', 1: 'block.cfg', 2: 'block.cfg', 3: 'nocatch.cfg', 4: 'catch.cfg', 5: 'block.cfg'}
    taskdict = {0: 'kind', 1: 'kind', 2: 'pred', 3: 'task', 4: 'task', 5: 'task'}
    dynops = {'numhidden': [256, 256], 'resim': False, 'configname': tphase_dict[tphase], 'epoch': epoch,
              'zeromean': False}
    rd = dyn.dynamics_preprocess_suite(savedir, modelname, simtype=taskdict[tphase], dynops=dynops)

    # setup for phase portaits

    # try to define dynamic bounds, set up for pca dynamics and minimization
    # TODO: this will eventually need to be changed for each epoch, and made to work with each task
    if blocktype == 'mixed':
        block_idx = 13
    elif blocktype == 'high':
        block_idx = 130
    elif blocktype == 'low':
        block_idx = 50

    # t_idx = rd['tdict']['start'][block_idx]+1
    t_idx = rd['tdict']['iti'][block_idx] - 1

    rnn_input = rd['inp'][t_idx, :]
    i = torch.tensor(np.expand_dims(np.expand_dims(rnn_input, axis=0), axis=0)).to(torch.float32)
    dt = 0.05  # if more behdat needed, i'll make it an input. otherwise hardcode
    n_neur = 256 * 2  # feature space

    if verbose: print('parsing regions-specific stuff')
    if reg_idx == 0:
        pca_use = rd['pca_ofc']
        rnnuse = rd['net'].rnn1
        sampsdict_use = rd['sampsdict_ofc']
        rnn_input = torch.squeeze(torch.squeeze(i))

    elif reg_idx == 1:
        pca_use = rd['pca_str']
        rnnuse = rd['net'].rnn2
        sampsdict_use = rd['sampsdict_str']
        i_ofc = torch.tensor(np.expand_dims(rd['behdict']['preds'][t_idx], axis=0)).to(torch.float32)
        rnn_input = torch.squeeze(torch.squeeze(torch.cat((i[:, :, :2], i_ofc), axis=2)))

    if verbose: print('calculating pc_means')
    sampsall = []
    for k in sampsdict_use.keys():
        sampsall.extend(sampsdict_use[k])

    # find mean of higher order features. TODO: do i need this?
    proj_samps = [pca_use.transform(m) for m in sampsall]
    proj_samps_flat = proj_samps[0]
    for k in proj_samps[1:]:
        proj_samps_flat = np.concatenate((proj_samps_flat, k), axis=0)
    pc_means = np.nanmean(proj_samps_flat, axis=0)

    # load the the KE min results and find the full range, then redecide on it. call is 'mesh'
    if verbose: print('using KE min to find good bounds for flowfields')

    kedat1 = pickle.load(open(kemin_name, 'rb'))
    kedat = parse.KEconvert2matlab(kemin_name, dosave=False)

    labs = np.unique(kedat['labels'][kedat['labels'] > 0])
    labs_idx = [np.argwhere(kedat['labels'] == k)[0, 0] for k in labs]
    xvals_fp = [kedat['crds'][k][0] for k in labs_idx]
    yvals_fp = [kedat['crds'][k][1] for k in labs_idx]

    xvals_base = [k[0][0] for k in kedat1['crds']]
    yvals_base = [k[0][1] for k in kedat1['crds']]

    xmin = np.min([np.min(xvals_base), np.min(xvals_fp)])
    ymin = np.min([np.min(yvals_base), np.min(yvals_fp)])

    xmax = np.max([np.max(xvals_base), np.max(xvals_fp)])
    ymax = np.max([np.max(yvals_base), np.max(yvals_fp)])

    nx = 100
    ny = 100
    xvals = np.linspace(xmin * 1.2, xmax * 1.2, nx)
    yvals = np.linspace(ymin * 1.2, ymax * 1.2, nx)

    mesh = [xvals, yvals]

    # try running the PCA dynamics.
    if verbose: print('running dynamics, dim=' + str(D))
    resvec = [50, 50,
              20]  # even if using 2D, giving 3 args helps prevent an error in finding dynamics bounds on PC activity

    net = rd['net']
    sgrads, outputs, mesh = dyn.pcadynamics(rnnuse, pca_use, sampsdict_use, rnn_input, net, reg_idx=reg_idx,
                                            D=D, resvec=resvec, mesh=mesh)

    # TODO: once block and reward type are coded into STR, this will need to be made modular
    if reg_idx == 0:
        basename_dyn = modelname.split('/')[-1].split('.')[0] + '_D' + str(D) + '_' + epoch + '_boutique'

    else:
        # basename_dyn = modelname.split('/')[-1].split('.')[0] + '_D'+str(D)+'_'+epoch+'_'+blocktype+'_boutique'
        basename_dyn = modelname.split('/')[-1].split('.')[0] + '_D' + str(
            D) + '_' + epoch + '_' + blocktype + '_boutique_end'

    if verbose: print('saving: ' + basename_dyn)
    dyn.savedynamics(sgrads, outputs, mesh, rd['samps_pcdict_ofc'], savedir_flows, basename_dyn, reg_idx, block=None)

    return None

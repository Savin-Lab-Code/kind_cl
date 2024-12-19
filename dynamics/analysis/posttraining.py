# analysis suite to take in a model file, .json data file, and do all the things to it for analysis
import pickle
import json
from dynamics.process.rnn import wt_protocols
from dynamics.process.rnn import parse
from dynamics.analysis import KEmin, flowfields


def run_suite(modelname, dataname, configname, simops, do_dict=None):
    """
    runs all the stuff
    @param modelname: (str) name of .model file
    @param dataname: (str) name of  1k or 10k .json file from training
    @param configname: (str) name of  1k config file. usually at dynamics/dynamics/anlaysis/1k_task.cfg
    @param simops: (dict) describing what type of sim it is. mostly epoch, tphase, RNN num information
    @param do_dict: (dict) for each type of simulation. overwrite existing result. but checks on per-anlaysis basis
    @return:
    """

    # idx = simops['idx']
    num = simops['num']
    s_idx = simops['s_idx']  # cltype
    t_idx = simops['t_idx']
    epoch = simops['epoch']
    # blocktype = simops['blocktype']  # mixed, high, low, kind, inf. kind and inf just choose first trial for pblock
    savedir_KE = simops['savedir_KE']  # same, for KE min data. probably $results/dynamics/KEmin_constrained
    savedir_allbeh = simops['savedir_allbeh']  # savme, for allbeh.json. probalby $results/
    savedir_flows = simops['savedir_flows']  # where flow data will be saved
    basedirectory = simops['basedirectory']  # where config files live
    simtype = simops['simtype']  # 'task','kind','inf'
    choicetype = simops['choicetype']  # how trial idx and t_idx is chosen for deciding on inputs for KE min, flowfields
    usepcmeans = simops['usepcmeans']  # will mean PC responses be used in KEmin for initial conditions for > D dim ?
    if 'verbose' in simops:
        verbose = simops['verbose']  #
    else:
        verbose = False

    if do_dict is None:
        do_dict = {'sim': False, 'KE': False, 'doKE': True, 'flow': False, 'beh2mat': False, 'allbeh': False}

    # the base file name
    basename = modelname.split('/')[-1].split('.')[0]

    if do_dict['sim']:
        print('simulating 1k trials')
        print(dataname)
        print('using config:')
        print(basedirectory + configname)
        dat = wt_protocols.sim_task(modelname, configname=basedirectory + configname, simtype=simtype, task_seed=num)
        if t_idx > 2:
            dat_json = parse.dat2json(dat)
        else:
            dat_json = parse.dict2json(dat)
        if t_idx > 2:
            print('saving')
            with open(dataname, 'w') as f:
                json.dump(dat_json, f)
        else:
            print('bypassing save. early stage of kind.')
    else:
        print('skipping simulation')

    for j in [0, 1]:
        reg_idx = j
        print('running KE minimization: ' + str(reg_idx))
        if t_idx < 3:
            typelist = ['kind']
        elif j == 1 and t_idx > 4:
            # typelist = ['mixed','high','low']
            print('running KE for all block types')
            typelist = ['mixed', 'high', 'low']
        else:  # catch and nocatch
            typelist = ['mixed']

        for k in typelist:
            KEops = {'reg_idx': reg_idx, 'tphase': t_idx, 'epoch': epoch, 'blocktype': k,
                     'num_proc': 1, 'constrained': True, 'block_idx': None, 'nonuniform_samp': True, 'verbose': verbose,
                     'choicetype': choicetype, 'usepcmeans': usepcmeans}
            if 'D' in simops.keys():
                Duse = simops['D']
            else:
                Duse = None  # will use data dimesionality. might fail
            savename_kemin0 = savedir_KE + 'kemin_' + basename + 'reg_' + str(reg_idx) + '_' + k + '_' + epoch + '.dat'

            if do_dict['KE']:
                crds, outputs = KEmin.KEmin(modelname, dataname, KEops, D_overwrite=Duse, bd=basedirectory)
                pickle.dump({'crds': crds, 'output': outputs}, open(savename_kemin0, 'wb'))
                parse.KEconvert2matlab(savename_kemin0, dosave=True)
            else:
                print('skipping KE calc')

    if t_idx > 4 and do_dict['flow']:
        print('running dynamical flow fields for final task stage')
        for j in [0, 1]:
            reg_idx = j

            if reg_idx == 1:  # all block types for STr
                typelist = ['mixed', 'high', 'low']
            else:  # mixed only for OFC
                typelist = ['mixed']

            for k in typelist:
                kemin_name = savedir_KE + 'kemin_' + basename + 'reg_' + str(reg_idx) + '_' + k + '_' + epoch + '.dat'
                # kemin_name = None  # for debugging purposes

                fo = {'reg_idx': reg_idx, 's_idx': s_idx, 'tphase': t_idx, 'epoch': epoch, 'blocktype': k,
                      'verbose': True, 'useKEbounds': False, 'configname': basedirectory + configname,
                      'savedir': savedir_flows, 'choicetype': choicetype}  # flowops dictionary

                # alows direct control over aspects of flowfields:
                # dimensionality, data used for defining bounds, rnn inputs, and trial idx
                overwrite = {'D': 2, 'dat2fit': None, 'rnn_input': None, 'block_idx': None}

                flowname = (savedir_flows + 'flowfields_' + basename + 'reg_' +
                            str(reg_idx)+'_' + k + '_' + epoch + '.mat')

                flowfields.genflowfields(modelname, dataname, kemin_name, flowname, fo, overwrite=overwrite)

    else:
        print('skipping flow')

    # parse the behavior data to matlab format. go ahead and do for all task files

    if t_idx > 2:
        savedir = modelname.split('rnn_')[0]
        savename = savedir + 'ratTrial_' + basename + '.mat'
        if s_idx == 2:
            loadname = modelname.split('.')[0] + '_1k.json'  # load 1k data. only thing saved now...
        else:
            loadname = modelname.split('.')[0] + '.json'  # NOT the 1k data, but 10k data

        if do_dict['beh2mat']:
            print('converting behavioral data to .mat file')
            print(savename)
            _ = parse.generate_rattrial_rnn(loadname=loadname, savename=savename)
        else:
            print('skipping conversion of behavioral data to .mat file.')

        for m in range(2):
            if m == 0:
                print('appending training data to allbeh.json')
                dataname_train = modelname.split('.')[0]+'.json'
                fname_global = savedir_allbeh + 'rnn_' + str(num) + '_allbeh.json'
                print(fname_global)
                print(dataname_train)
            else:  # the frozen data
                print('appending training data to allbeh_1k.json')
                dataname_train = modelname.split('.')[0] + '_1k.json'
                fname_global = savedir_allbeh + 'rnn_' + str(num) + '_allbeh_1k.json'
                print(fname_global)
                print(dataname_train)
            try:
                parse.joinbeh(dataname_train, fname_global, overwrite=do_dict['allbeh'])
            except FileNotFoundError:
                print('source file might be missing. probably dat2json error for corrupted file')

    return None

# analysis suite to take in a model file, .json data file, and do all the things to it for analysis
import pickle
import json
from dynamics.process.rnn import wt_protocols
from dynamics.process.rnn import parse
from dynamics.analysis import KEmin, flowfields
from os.path import exists


def run_suite(modelname, dataname, configname, simops, redo=None):
    """
    runs all the stuff
    @param modelname: (str) name of .model file
    @param dataname: (str) name of  10k .json file from training
    @param configname: (str) name of  1k config file. usually at dynamics/dynamics/anlaysis/1k_task.cfg
    @param simops: (dict) describing what type of sim it is. mostly epoch, tphase, RNN num information
    @param redo: (dict) for each type of simulation. overwrite existing result. but checks on per-anlaysis basis
    @return:
    """

    num = simops['num']
    s_idx = simops['s_idx']  # cltype
    t_idx = simops['t_idx']
    epoch = simops['epoch']
    savedir_KE = simops['savedir_KE']  # same, for KE min data. probably $results/dynamics/KEmin_constrained
    savedir_allbeh = simops['savedir_allbeh']  # savme, for allbeh.json. probalby $results/
    savedir_flows = simops['savedir_flows']  # where flow data will be saved
    basedirectory = simops['basedirectory']  # where config files live
    simtype = simops['simtype']  # 'task','kind','inf'

    if redo is None:
        redo = {'sim': False, 'KE': False, 'doKE': True, 'flow': False, 'beh2mat': False, 'allbeh': False}

    # the base file name
    basename = modelname.split('/')[-1].split('.')[0]

    redomask = (exists(dataname) and redo['sim']) or not exists(dataname)
    if redomask:
        print('simulating 1k trials')
        dat = wt_protocols.sim_task(modelname, configname=configname, simtype=simtype, task_seed=num)
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
        print('skipping simulation. already peformed')

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
                     'num_proc': 1, 'constrained': True, 'block_idx': None, 'nonuniform_samp': True,
                     'verbose': False, 'basedirectory':basedirectory}
            if 'D' in simops.keys():
                Duse = simops['D']  # either None to use proper dim, or 2
                if Duse == 2:
                    print('using 2D')
            else:
                Duse = None
            savename_kemin0 = savedir_KE + 'kemin_' + basename + 'reg_' + str(reg_idx) + '_' + k + '_' + epoch + '.dat'

            redomask = (exists(savename_kemin0) and redo['KE']) or not exists(savename_kemin0)
            redomask = redomask and redo['doKE']
            if redomask:
                crds, outputs = KEmin.KEmin(modelname, dataname, KEops, D=Duse)
                pickle.dump({'crds': crds, 'output': outputs}, open(savename_kemin0, 'wb'))
                parse.KEconvert2matlab(savename_kemin0, dosave=True)
            else:
                print('skipping file already exists: ' + savename_kemin0)

    if t_idx > 4 and redo['flow']:  # make the flow fields fully depndent on redo flag. don't really want them
        print('running dynamical flow fields for final task stage')
        for j in [0, 1]:
            reg_idx = j

            if reg_idx == 1:  # all block types for STr
                typelist = ['mixed', 'high', 'low']
            else:  # mixed only for OFC
                typelist = ['mixed']

            for k in typelist:
                kemin_name = savedir_KE + 'kemin_' + basename + 'reg_' + str(reg_idx) + '_' + k + '_' + epoch + '.dat'

                fo = {'reg_idx': reg_idx, 's_idx': s_idx, 'tphase': t_idx, 'epoch': epoch, 'blocktype': k,
                      'verbose': True, 'useKEbounds': False, 'basedirectory': basedirectory,
                      'savedir': savedir_flows}  # flowops dictionary

                flowname = (savedir_flows + 'flowfields_' + basename + 'reg_' +
                            str(reg_idx)+'_' + k + '_' + epoch + '.mat')
                redomask = (exists(flowname) and redo['flow']) or not exists(flowname)

                if redomask:
                    flowfields.genflowfields(modelname, dataname, kemin_name, flowname, fo)
                else:
                    print('skipping file. already exists: '+flowname)
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

        redomask = redo['beh2mat']
        if redomask:
            print('converting behavioral data to .mat file')
            print(savename)
            _ = parse.generate_rattrial_rnn(loadname=loadname, savename=savename)
        else:
            print('skipping conversion of behavioral data to .mat file. exists')

        for m in range(2):
            if m == 0:
                print('appending training data to allbeh.json')
                dataname_train = modelname.split('.')[0]+'.json'
                fname_global = savedir_allbeh + 'rnn_' + str(num) + '_allbeh.json'
            else:  # the frozen data
                print('appending training data to allbeh_1k.json')
                dataname_train = modelname.split('.')[0] + '_1k.json'
                fname_global = savedir_allbeh + 'rnn_' + str(num) + '_allbeh_1k.json'
            try:
                parse.joinbeh(dataname_train, fname_global, overwrite=redo['allbeh'])
            except FileNotFoundError:
                print('source file might be missing. probably dat2json error for corrupted file')

    return None

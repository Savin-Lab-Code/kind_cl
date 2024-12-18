# run a defined curriculum using the new format of CL_task and CL_stage classes

import torch
from dynamics.utils import utils
from dynamics.process.rnn import wt_protocols, wt_nets, wt_costs_generic
import sys


def fulltrain_run(fname):

    ops, uniqueops = utils.parse_configs(fname)

    # the unique ops. assign for easier reading
    savedir = uniqueops['savedir']
    modelname = uniqueops['modelname']
    numhidden = uniqueops['nn']
    netseed = uniqueops['netseed']
    bias = uniqueops['bias']
    nrounds = [uniqueops['nrounds1'], uniqueops['nrounds2'], uniqueops['nrounds3']]
    nrounds_start = [uniqueops['nrounds1_s'], uniqueops['nrounds2_s'], uniqueops['nrounds3_s']]
    optim_fname = uniqueops['adam_fname']

    # set up network
    device = torch.device(ops['device'])

    print('ops')
    print(ops)

    print('uniqueops')
    print(uniqueops)

    # allow for a restart as a standard
    print('choosing model type')
    nd = wt_nets.netdict[ops['modeltype']]  # dictionary of models and size params
    netfun = nd['net']
    net = netfun(din=nd['din'], dout=nd['dout'], num_hiddens=numhidden, seed=netseed, rnntype=ops['rnntype'],
                 bias_init=bias, snr=ops['snr_hidden'], dropout=ops['dropout'],
                 rank=ops['rank'], inputcase=nd['inputcase'])

    # a previously saved model?
    print('moving model to device')
    net.to(device)
    if modelname != '':
        print('loading previous model')
        net.load_state_dict(torch.load(modelname, map_location=device.type))
    else:
        print('saving initial network')
        savename = savedir + 'rnn_init_' + str(netseed)
        torch.save(net.state_dict(), savename + '.model')  # save model

    # do kindergarten training
    print('doing pretraining curiculum')
    stagelist_names = ops['stagelist_cl']
    stagelist = []
    for k in stagelist_names:
        stagelist.append(utils.parse_config_CLstage(k))  # returns list CL stage objects

    wt_protocols.training_curric_general(net, stagelist, ops=ops, savedir=savedir,
                                         device=device, optim_fname=optim_fname)


    # do curriculum training
    print('beginning curriculum training')
    savename = savedir + 'rnn_curric_' + str(netseed)
    print(savename)
    _, _ = wt_protocols.curriculumtraining(net, ops=ops, savename=savename, device=device, savetmp=True,
                                           nrounds=nrounds, nrounds_start=nrounds_start, optim_fname=optim_fname)

    print('done!')


if __name__ == '__main__':
    print('beginning')
    fulltrain_run(sys.argv[1])
    print('done')

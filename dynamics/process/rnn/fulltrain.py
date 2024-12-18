<<<<<<< HEAD
# main run code. takes in a config file, then decides which curriculum stages to use
=======
# a redo of the RNN training, from start to finish
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)

import torch
from dynamics.utils import utils
from dynamics.process.rnn import wt_protocols, wt_nets
import sys
<<<<<<< HEAD
=======
from datetime import datetime
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)


def fulltrain_run(fname):

    ops, uniqueops = utils.parse_configs(fname)

    # the unique ops. assign for easier reading
    savedir = uniqueops['savedir']
    modelname = uniqueops['modelname']
    numhidden = uniqueops['nn']
    netseed = uniqueops['netseed']
    usekindergarten = uniqueops['usekindergarten']
    bias = uniqueops['bias']
    nrounds = [uniqueops['nrounds1'], uniqueops['nrounds2'], uniqueops['nrounds3']]
    nrounds_start = [uniqueops['nrounds1_s'], uniqueops['nrounds2_s'], uniqueops['nrounds3_s']]
    roundstart_pred = uniqueops['roundstart_pred']
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
    print('deciding on pre-processing steps')
<<<<<<< HEAD
    if usekindergarten:
        print('beginning kindergarten')
=======

    if usekindergarten:
        print('beginning kindergarten')
        print(datetime.now())
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
        savename = savedir + 'rnn_kindergarten_' + str(netseed)
        print(savename)
        net, _, _, optim_fname = wt_protocols.training_kindergarten(net, ops=ops, savename=savename, device=device,
                                                                    savelog=True, optim_fname=optim_fname)
    if ops['kindergarten_prediction']:
        print('kindergarten + prediction')
<<<<<<< HEAD
=======
        print(datetime.now())
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
        savename = savedir + 'rnn_pred_' + str(netseed)
        print(savename)
        net, _, _, optim_fname = wt_protocols.training_prediction(net, ops, savename=savename, device=device,
                                                                  savelog=False, roundstart=roundstart_pred,
                                                                  optim_fname=optim_fname)
    if ops['kindergarten_d2m']:
        print('sham + kind + pred')
        savename = savedir + 'rnn_d2m_' + str(netseed)
        print(savename)

        net, _, _, optim_fname = wt_protocols.training_sham(net, ops, savename=savename, device=device, savelog=False,
                                                            roundstart=roundstart_pred, optim_fname=optim_fname)
    # do curriculum training
    print('beginning curriculum training')
<<<<<<< HEAD
=======
    print(datetime.now())
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
    savename = savedir + 'rnn_curric_' + str(netseed)
    print(savename)
    _, _ = wt_protocols.curriculumtraining(net, ops=ops, savename=savename, device=device, savetmp=True,
                                           nrounds=nrounds, nrounds_start=nrounds_start, optim_fname=optim_fname)

    print('done!')


if __name__ == '__main__':
    print('beginning')
    fulltrain_run(sys.argv[1])
    print('done')

# networks for wait time task
import torch
from torch import nn
import numpy as np

<<<<<<< HEAD
=======
def flat2gates(s, rnn):
    """
    turns a flattened state into something read for the RNN of choice. add custom network layers to this
    Also adds batch and time dims

    @param s: (tensor) state, as flattened object
    @param rnn: (torch.nn.module) object. could be custom
    @return:
    """

    if type(s) is np.ndarray:
        func = np.expand_dims
    elif type(s) is torch.Tensor:
        func = torch.unsqueeze
    else:
        print('base type of flattened array undedermined. asumming numpy compatible')
        func = np.expand_dims

    if type(rnn) is torch.nn.modules.rnn.GRU or type(rnn) is torch.nn.modules.rnn.RNN:
        sf = func(func(s, dim=0), dim=0)
    elif type(rnn) is torch.nn.modules.rnn.LSTM:
        nh = int(len(s)/2)
        sf = (func(func(s[:nh], dim=0), dim=0),
              func(func(s[nh:], dim=0), dim=0))
    else:
        print('that gate type not yet supported: ' + str(type(rnn)))
        sf = None

    return sf


def gates2flat(s, rnn):
    """
    turns a dim-padded state for the network into unpadded and flattened object

    @param s: (tensor, tuple) state, as flattened object
    @param rnn: (torch.nn.module) object. could be custom
    @return:
    """

    if type(rnn) is torch.nn.modules.rnn.GRU or type(rnn) is torch.nn.modules.rnn.RNN:
        if type(s) is torch.Tensor:
            sf = torch.squeeze(torch.squeeze(s, dim=0), dim=0)
        else:
            sf = np.squeeze(np.squeeze(s, axis=0), axis=0)

    elif type(rnn) is torch.nn.modules.rnn.LSTM:
        ngate = len(s)
        if type(s[0]) is torch.Tensor:
            sf = torch.squeeze(torch.squeeze(s[0], dim=0), dim=0)
            for j in range(1, ngate):
                sf = torch.cat((sf, torch.squeeze(torch.squeeze(s[j], dim=0), dim=0)))
        else:
            sf = np.squeeze(np.squeeze(s[0], axis=0), axis=0)
            for j in range(1, ngate):
                sf = torch.cat((sf, np.squeeze(np.squeeze(s[j], axis=0), axis=0)))


    else:
        print('that gate type not yet supported: '+str(type(rnn)))
        sf = None

    return sf

>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)

class RNNModel(nn.Module):
    # The RNN model that only outputs policy

<<<<<<< HEAD
    def __init__(self, din, dout, num_hiddens=32, seed=None, initbias=5.0, rnntype='RNN', _=None,
=======
    def __init__(self, din, dout, num_hiddens=32, seed=None, initbias=5.0, rnntype='RNN', dropout=None,
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
                 inputcase=11, **kwargs):
        """

        :param din: (int) size of input  (number inputs)
        :param dout (int) size of output (number actions):
        :param num_hiddens: (int) number of hidden unites
        :param seed: seed to torch random number generator. for reproducibilty
        :param initbias: initial bias for waiting. 5.0 works for nonRNN models
        :param rnntype: 'RNN', 'GRU', or 'LSTM'
<<<<<<< HEAD
        :param _: normally dropout. not used
        :param inputcase: (int) for type of input expected. see utils.py
=======
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
        :param kwargs:
        """

        self.inputcase = inputcase

        if 'rank' in kwargs:
            kwargs.pop('rank')  # back compatibility. needed for low-rank RNN
        super(RNNModel, self).__init__(**kwargs)

        # set random seed
        if seed is not None:
            self.seed = seed
            torch.manual_seed(self.seed)

        if rnntype == 'RNN':
            rnn_layer = nn.RNN(din, num_hiddens, batch_first=True)
        elif rnntype == 'GRU':
            rnn_layer = nn.GRU(din, num_hiddens, batch_first=True)
        elif rnntype == 'LSTM':
            rnn_layer = nn.LSTM(din, num_hiddens, batch_first=True)
        else:
            print('wrong RNN type: RNN,GRU,LSTM options. default to RNN')
            rnn_layer = nn.RNN(din, num_hiddens, batch_first=True)

        self.rnn = rnn_layer  # the vanilla RNN, LSTM, GRU, etc
        self.num_hiddens = self.rnn.hidden_size  # number of hidden states
        self.dout = dout  # output size. number of possible actions
        self.din = din  # input size. number inputs
        self.rnntype = rnntype

        # output layer. not bidirectional
        self.num_directions = 1

        # try just a linear network
        # self.linear = nn.Linear(self.din, self.dout)  # for linear policy
        self.linear = nn.Linear(self.num_hiddens, self.dout)

        # introduce a bias
        bias = self.linear.bias.clone()
        bias[0] = initbias
        self.linear.bias = torch.nn.Parameter(bias)

    def forward(self, inputs, state, isITI=False, device=torch.device('cpu')):
        # assume inputs is already a tensor of correct shape.
        # preprocessing should happen out of this class method
        Y, state = self.rnn(inputs, state)

        # reshape action
        # The fully connected layer will first change the shape of `Y` to
        # (`num_steps` * `batch_size`, `num_hiddens`). Its output shape is
        # (`num_steps` * `batch_size`, `dim`).
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
<<<<<<< HEAD
=======

        # output = self.linear(inputs.reshape((-1, inputs.shape[-1])))  # for linear policy

>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
        output = nn.Softmax(dim=1)(output)

        returndict = {'policy': output, 'value': None, 'state': state,
                      'activations': None, 'pred_supervised': None, 'output_sham': None}
        return returndict

    def begin_state(self, device, batchsize=1):
        # how network is intialized
        sigma = 1.0
<<<<<<< HEAD
=======
        # if you want zeros
        # return torch.zeros((self.num_directions * self.rnn.num_layers,
        #                batchsize, self.num_hiddens), device=device)

        # return torch.tensor(1.0)  # to use a linear policy, not RNN
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
        if isinstance(self.rnn, nn.LSTM):
            return (sigma * torch.rand((self.num_directions * self.rnn.num_layers,
                                        batchsize, self.num_hiddens),
                                       device=device),
                    sigma * torch.rand((self.num_directions * self.rnn.num_layers,
                                        batchsize, self.num_hiddens),
                                       device=device))
        else:
            return sigma * torch.rand((self.num_directions * self.rnn.num_layers,
                                       batchsize, self.num_hiddens), device=device)

    def parseiputs(self, inp, inputcase=None):
        """
        restructures inputs for network specific needs. not used until multiregion networks.
        But need in all for consistency
        :param inp:
        :param inputcase:
        :return:
        """

        if inputcase is None:
            inputcase = self.inputcase

        if inputcase > 0:
            return inp
        else:
            print('something is up with inputcase in the network')
            return inp


class RNNModelSupervisable(RNNModel):
<<<<<<< HEAD
    # The RNN model that can also undergo supervised learning beforehand, with kindergarten tasks

    # keep for setting standard options
    def __init__(self, din=4, dout=3, num_hiddens=32, seed=None, bias_init=6.0, rnntype='RNN',
                 _=None, **kwargs):
        # dropout not used, set to _ in inputs
=======
    # The RNN model that can also undergo supervised learning beforehand, in kindergarten()

    # keep for setting standard options
    def __init__(self, din=4, dout=3, num_hiddens=32, seed=None, bias_init=6.0, rnntype='RNN',
                 dropout=None, **kwargs):
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
        super(RNNModelSupervisable, self).__init__(din, dout, num_hiddens, seed, bias_init, rnntype, **kwargs)

    def forward(self, inputs, state, isITI=False, device=torch.device('cpu')):
        # assume inputs is already a tensor of correct shape.
        # preprocessing should happen out of this class method
        Y, state = self.rnn(inputs, state)

        # reshape action
        # The fully connected layer will first change the shape of `Y` to
        # (`num_steps` * `batch_size`, `num_hiddens`). Its output shape is
        # (`num_steps` * `batch_size`, `dim`).
        Y = Y.reshape((-1, Y.shape[-1]))

        # split output and softmax policy bit
        output1 = self.linear(Y)
        output_supervised = output1[:, 2:]
        Y2 = output1[:, :2]
        policy = nn.Softmax(dim=1)(Y2)

        returndict = {'policy': policy, 'value': None, 'state': state,
                      'activations': None, 'pred_supervised': output_supervised}
        return returndict


class RNNModelActorCritic(RNNModel):
    # outputs policy, value, and kindergarten training ouputs

    def __init__(self, din=4, dout=6, num_hiddens=32, seed=None, bias_init=6.0, rnntype='RNN',
<<<<<<< HEAD
                 _=None, **kwargs):
=======
                 dropout=None, **kwargs):
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
        """

        :param din: (int) size of input  (number inputs)
        :param dout (int) size of output (number actions):
        :param num_hiddens: (int) number of hidden unites
        :param seed: seed to torch random number generator. for reproducibilty
<<<<<<< HEAD
        :param bias_init: (float) for bias to add to outputs weight for waiting. not used
        :param rnntype: (str) for type of RNN used.
        :param _: dropout normally here. not used
=======
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
        :param kwargs:
        """

        super(RNNModelActorCritic, self).__init__(din, dout, num_hiddens, seed, bias_init, rnntype, **kwargs)

    def forward(self, inputs, state, isITI=False, device=torch.device('cpu')):
        # assume inputs is already a tensor of correct shape.
        # preprocessing should happen out of this class method
        Y, state = self.rnn(inputs, state)

        # reshape action
        # The fully connected layer will first change the shape of `Y` to
        # (`num_steps` * `batch_size`, `num_hiddens`). Its output shape is
        # (`num_steps` * `batch_size`, `dim`).
        Y = Y.reshape((-1, Y.shape[-1]))

        output1 = self.linear(Y)

        # only do softmax on first two entries, which is pwait, and poo
        Y2 = output1[:, :2]
        output_policy = nn.Softmax(dim=1)(Y2)
        output_value = output1[:, 2]

        returndict = {'policy': output_policy, 'value': output_value, 'state': state,
                      'activations': None, 'pred_supervised': None}
        return returndict


class RNNModelActorCriticSupervisable(RNNModel):
    # outputs policy, value, and kindergarten training ouputs

    def __init__(self, din=4, dout=6, num_hiddens=32, seed=None, bias_init=6.0, rnntype='RNN',
<<<<<<< HEAD
                 _=None, **kwargs):
        # _ for dropout in call. not used here. for backwards compabilitity

=======
                 dropout=None, **kwargs):
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
        super(RNNModelActorCriticSupervisable, self).__init__(din, dout, num_hiddens, seed, bias_init,
                                                              rnntype, **kwargs)

    def forward(self, inputs, state, isITI=False, device=torch.device('cpu')):
        # assume inputs is already a tensor of correct shape.
        # preprocessing should happen out of this class method
        Y, state = self.rnn(inputs, state)

        # reshape action
        # The fully connected layer will first change the shape of `Y` to
        # (`num_steps` * `batch_size`, `num_hiddens`). Its output shape is
        # (`num_steps` * `batch_size`, `dim`).
        Y = Y.reshape((-1, Y.shape[-1]))

        output1 = self.linear(Y)
        output_supervised = output1[:, 3:]

        # only do softmax on first two entries, which is pwait, and poo
        Y2 = output1[:, :2]
        output_policy = nn.Softmax(dim=1)(Y2)
        output_value = output1[:, 2]

        returndict = {'policy': output_policy, 'value': output_value, 'state': state,
                      'activations': None, 'pred_supervised': output_supervised}
        return returndict


class RNNModel_wITI(RNNModel):
    # an actor-critic model with auxiliary cost ability, but also task-engaged- and and ITI-specific actions

    def __init__(self, din=4, dout=7, num_hiddens=32, seed=None, bias_init=6.0, rnntype='RNN',
<<<<<<< HEAD
                 _=None, **kwargs):
        # _ for dropout in init, for backwards compatibility
=======
                 dropout=None, **kwargs):
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
        super(RNNModel_wITI, self).__init__(din, dout, num_hiddens, seed, bias_init, rnntype, **kwargs)

    def forward(self, inputs, state, isITI=False, device=torch.device('cpu')):
        # assume inputs is already a tensor of correct shape.
        # preprocessing should happen out of this class method
        Y, state = self.rnn(inputs, state)

        # reshape action
        # The fully connected layer will first change the shape of `Y` to
        # (`num_steps` * `batch_size`, `num_hiddens`). Its output shape is
        # (`num_steps` * `batch_size`, `dim`).
        Y = Y.reshape((-1, Y.shape[-1]))

        output1 = self.linear(Y)
        output_supervised = output1[:, 4:]

        # actions are now pwait poptout, pwait_ITI
        Y2 = output1[:, :3]
        if isITI:  # when in the ITI, set prob of wait and optout to zero. will this throw an in-place error?
            Y2[:, :2] = -1000.0
        else:  # in a trial. force the no "ITI" action
            Y2[:, 2] = -1000.0
        output_policy = nn.Softmax(dim=1)(Y2)

        output_value = output1[:, 3]

        returndict = {'policy': output_policy, 'value': output_value, 'state': state,
                      'activations': None, 'pred_supervised': output_supervised}
        return returndict


class RNNModelTrialProbPrediction(RNNModel_wITI):
    # instead of analog prediction of error, predicts probability of trial offers as aux. output
    # includes the ITI action space as well
    # outputs: actions ([0,1,2]), value ([3]), kindergarten ([4,5,6]), ptrial([7,8,9,10,11,12])

    def __init__(self, din=4, dout=12, num_hiddens=32, seed=None, bias_init=6.0, snr=0.0, rnntype='RNN',
<<<<<<< HEAD
                 _=None, **kwargs):
=======
                 dropout=None, **kwargs):
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)

        super(RNNModelTrialProbPrediction, self).__init__(din, dout, num_hiddens, seed, bias_init,
                                                          rnntype, **kwargs)

    def forward(self, inputs, state, isITI=False, device=torch.device('cpu')):
        # assume inputs is already a tensor of correct shape.
        # preprocessing should happen out of this class method
        Y, state = self.rnn(inputs, state)

        # reshape action
        # The fully connected layer will first change the shape of `Y` to
        # (`num_steps` * `batch_size`, `num_hiddens`). Its output shape is
        # (`num_steps` * `batch_size`, `dim`).
        Y = Y.reshape((-1, Y.shape[-1]))

        # supervised ouputs for kindergarten
        output1 = self.linear(Y)
        output_supervised = output1[:, 4:7]

        # only do softmax on first two entries, which is pwait, and poo
        Y2 = output1[:, :3]
        if isITI:  # when in the ITI, set prob of wait and optout to zero. will this throw an in-place error?
            Y2[:, :2] = -1000.0
        else:  # in a trial. force the no "ITI" action
            Y2[:, 2] = -1000.0
        output_policy = nn.Softmax(dim=1)(Y2)

        # probability of next trial
        Y3 = output1[:, 7:]  # activations for probability of each volume on next trial
        # output_probvolumepred = nn.Softmax(dim=1)(Y3)
        trialpred_activations = Y3

        output_value = output1[:, 3]

        returndict = {'policy': output_policy, 'value': output_value, 'state': state,
                      'activations': trialpred_activations, 'pred_supervised': output_supervised}
        return returndict


class RNNModel_multiregion(nn.Module):
    #
    # a multi-region RNN with an OFC-like and DMS-like compartment. OFC--> DMS
    # this will be explicitly creaed as a 2-unit network, becuase I don't know how to feed the optimizer
    # the parameters otherwise

    def __init__(self, din, dout,  num_hiddens, seed=None, rnntype='RNN', snr=0.0, bias_init=None, dropout=0.0,
                 inputcase=11, **kwargs):
        """
        this
        :param din: (list: int) size of input of each region
        :param dout (list: int) size of outputs for each region:
        :param num_hiddens: (list:int) number of hidden units for each region
        :param seed: seed to torch random number generator. for reproducibilty
        :param rnntype: 'RNN', 'GRU', or 'LSTM'
        :param snr: (float) variance of noise on hidden units. default is no noise
        :param bias_init: deprecated. kept for historical purposes
        :param dropout: (float) proportion of RNN-output layer  connections dropped at each forward call
        :param kwargs:
        """
        super(RNNModel_multiregion, self).__init__(**kwargs)

<<<<<<< HEAD
        self.inputcase = inputcase
=======
        self.inputcase=inputcase
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)

        # set random seed
        if seed is not None:
            self.seed = seed
            torch.manual_seed(self.seed)

        self.nregions = len(num_hiddens)  # had better be two...

        rnn = []

        if rnntype == 'RNN':
            for k in range(self.nregions):
                if k == 0:
                    nh_k = din[k]
                else:
                    nh_k = np.cumsum(num_hiddens[:k]) + din[k]
                rnn.append(nn.RNN(nh_k, num_hiddens[k], batch_first=True))
        elif rnntype == 'GRU':
            for k in range(self.nregions):
                if k == 0:
                    nh_k = din[k]
                else:
                    nh_k = np.cumsum(num_hiddens[:k]) + din[k]
                rnn.append(nn.GRU(nh_k, num_hiddens[k], batch_first=True))
        elif rnntype == 'LSTM':
            for k in range(self.nregions):
                if k == 0:
                    nh_k = din[k]
                else:
                    nh_k = int(np.cumsum(num_hiddens[:k]) + din[k])
                rnn.append(nn.LSTM(nh_k, num_hiddens[k], batch_first=True))
        else:
            print('wrong RNN type: RNN,GRU,LSTM options. default to RNN')
            for k in range(self.nregions):
                if k == 0:
                    nh_k = din[k]
                else:
                    nh_k = np.cumsum(num_hiddens[:k]) + din[k]
                rnn.append(nn.RNN(nh_k, num_hiddens[k], batch_first=True))

        self.rnntype = rnntype  # easier to remember type this way
        self.rnn1 = rnn[0]  # the list of  vanilla RNN, LSTM, GRU, etc
        self.rnn2 = rnn[1]  # the list of  vanilla RNN, LSTM, GRU, etc
        self.num_hiddens = num_hiddens  # list of hidden state numbers
        self.dout = dout  # same, for output size
        self.din = din  # same, for input size
        self.snr = snr

        # output layer. not bidirectional
        self.num_directions = 1

        # build the output heads
        linear = []
        for k in range(self.nregions):
            linear.append(nn.Linear(num_hiddens[k], dout[k]).requires_grad_(True))
        self.linear1 = linear[0]
        self.linear2 = linear[1]

    def forward(self, inputs, state, isITI=False, device=torch.device('cpu')):
        """
        propagates network forward, produces output. first unit is ofc (prediction loss), seonc unit is DMS
        (action, value, entropy losses)
        :param inputs: (list) of inputs for each subunit. each a 3-d tensor of form (nbatch=1,nt=1, num_hiddenk)
        :param state: (list) of states for each subunit. for LSTM, this is a 2-tuple of cell and hidden
                        which are themselves 3-d tensors for size (nbatch, nt, num_hidden1), where nbatch
                        is size of minibatch, nt number of time points (=1), and number of units
        :param isITI: (bool) for if this timestep is an ITI step, which restricts action space
<<<<<<< HEAD
        :param device: (torch.device) object for specifying where computation will happen. 'cpu' or 'gpu' normally
=======
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
        :return:
        """
        # assume inputs is already a tensor of correct shape.
        # preprocessing should happen out of this class method

        # unfortunately this does not give state at all times, which I want for my architecture
        # have to loop over sequence length
        batchsize, L, din = inputs[0].shape
        nhidden = state[0][0].shape[-1]

        dout1 = self.dout[0]
        state1_hidden = torch.zeros((batchsize, L, nhidden), device=device)
        state1_cell = torch.zeros((batchsize, L, nhidden), device=device)
        Y1 = torch.zeros((batchsize, L, dout1), device=device)
        state1k = state[0]
        for k in range(L):
            ik = torch.unsqueeze(inputs[0][:, k, :], dim=1)
            Y1k, state1k = self.rnn1(ik, state1k)
            # add noise
            Y1k = self.linear1(nn.Dropout(p=self.dropout)(Y1k))  # apply dropout to OFC output
            state1_hidden[:, k, :] = state1k[0][0, :, :] + self.snr*torch.randn(state1k[0][0, :, :].shape,
                                                                                device=device)
            state1_cell[:, k, :] = state1k[1][0, :, :]
            Y1[:, k, :] = torch.squeeze(Y1k, dim=1)

        # repackage state 1
        state1 = (torch.unsqueeze(state1_hidden[:, -1, :], dim=1),
                  torch.unsqueeze(state1_cell[:, -1, :], dim=1))
<<<<<<< HEAD
=======
        # Y1, state1 = self.rnn1(inputs[0], state[0])
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)

        if self.rnntype == 'LSTM':
            inp2 = torch.cat((inputs[1], state1_hidden), dim=2)  # hidden state of OFC is now input along w other inputs
        else:  # TODO: not functional. this will be hard to write modular for gru and vanilla RNN
            inp2 = torch.cat((inputs[1], state1_hidden), dim=2)  # just entire state for GRU and vanilla RNN
        Y2, state2 = self.rnn2(inp2, state[1])

        # add noise to state2
        state2 = list(state2)
        state2[0] = state2[0] + self.snr*torch.randn(state2[1].shape, device=device)
        state2 = tuple(state2)

        # reshape action
        # The fully connected layer will first change the shape of `Y` to
        # (`num_steps` * `batch_size`, `num_hiddens`). Its output shape is
        # (`num_steps` * `batch_size`, `dim`).
        Y1 = Y1.reshape((-1, Y1.shape[-1]))
        Y2 = Y2.reshape((-1, Y2.shape[-1]))

        # OFC supervised ouputs for kindergarten on the DMS units: trial offer and counting
        pred_activations = Y1[:, 0:5]  # activations for predicting block, or trial... whatever you want
        output_supervised_ofc = torch.unsqueeze(Y1[:, 5], dim=1)  # reward rate prediction

        # the rest of the outputs from DMS
        output2 = self.linear2(nn.Dropout(p=self.dropout)(Y2))  # apply dropout on STR output
        output_supervised_dms = output2[:, 4:]  # reward offer memory and tria length integration

        output_supervised = torch.cat((output_supervised_dms, output_supervised_ofc), dim=1)

        # only do softmax on first two entries, which is pwait, and poo
        pi_activation = output2[:, :3]
        if isITI:  # when in the ITI, set prob of wait and optout to zero. will this throw an in-place error?
            pi_activation[:, :2] = -1000.0
        else:  # in a trial. force the no "ITI" action
            pi_activation[:, 2] = -1000.0
        output_policy = nn.Softmax(dim=1)(pi_activation)

        output_value = output2[:, 3]

        # concat state
        state = [state1, state2]

        returndict = {'policy': output_policy, 'value': output_value, 'state': state,
                      'activations': pred_activations, 'pred_supervised': output_supervised}

        return returndict

    def begin_state(self, device, batchsize=1):
        """
        initilizes a network
        :param device:
        :param batchsize:
        :return:
        """
        # how network is intialized
        sigma = 1.0

        # return torch.tensor(1.0)  # to use a linear policy, not RNN
        statelist = []
        if isinstance(self.rnn1, nn.LSTM):
            statelist.append((sigma * torch.rand((self.num_directions * self.rnn1.num_layers,
                                                  batchsize, self.num_hiddens[0]), device=device),
                              sigma * torch.rand((self.num_directions * self.rnn1.num_layers,
                                                  batchsize, self.num_hiddens[0]), device=device)))

            statelist.append((sigma * torch.rand((self.num_directions * self.rnn2.num_layers,
                                                  batchsize, self.num_hiddens[1]), device=device),
                              sigma * torch.rand((self.num_directions * self.rnn2.num_layers,
                                                  batchsize, self.num_hiddens[1]), device=device)))

        else:
            statelist.append(sigma * torch.rand((self.num_directions * self.rnn1.num_layers,
                                                 batchsize, self.num_hiddens[0]), device=device))
            statelist.append(sigma * torch.rand((self.num_directions * self.rnn2.num_layers,
                                                 batchsize, self.num_hiddens[1]), device=device))

        return statelist

    def parseiputs(self, inp, inputcase=None):
        """
        chops up input into region-specific inputs.
        will look more like a hardcoded lookup table than anything else, but keeps simulation code cleaner
        :param inp:
        :param inputcase:
        :return:
        """

        if inputcase is None:
            inputcase = self.inputcase

        if inputcase == 11 and self.nregions == 2:
<<<<<<< HEAD
=======
            #inp1 = inputs to RNN 1
            #inp2 = inputs to RNN2
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
            inp1 = inp
            inp2 = inp[:, :, 0:2]
            inputs_all = [inp1, inp2]
        elif inputcase == 15 and self.nregions == 2:
            inp1 = inp[:, :, 0:4]
            i2 = np.array([0, 1, 4, 5])  # include d2m inputs to STR
            inp2 = inp[:, :, i2]
            inputs_all = [inp1, inp2]
        elif inputcase == 16:  # the photo stim
            inp1 = inp[:, :, 0:5]
            inp2 = inp[:, :, 0:2]
            inputs_all = [inp1, inp2]
        else:  # for kindergarten tasks
<<<<<<< HEAD
            print('incorrect inputcase specified. defaulting to inputcase 11 style')
=======
            print('ended up here in other inputcase. wt_nets 518')
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
            inp1 = inp
            inp2 = inp[:, :, 0:2]
            inputs_all = [inp1, inp2]

        return inputs_all


class RNNModel_multiregion_2(RNNModel_multiregion):
    #
    # a multi-region RNN with an OFC-like and STR-like compartment. OFC--> STr
    # sends output of OFC network to STR component, not activity of hidden units

<<<<<<< HEAD
    # TODO: do I have to include this, or can it just be assumed from class inheritance?
=======
    # TODO: do I have to include this, or can it just be assumed from class inheritance
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
    def __init__(self, din, dout, num_hiddens, seed=None, rnntype='RNN', snr=0.0, bias_init=0, dropout=0.0, rank=None,
                 inputcase=11, **kwargs):
        """
        this
        :param din: (list: int) size of input of each region
        :param dout (list: int) size of outputs for each region:
        :param num_hiddens: (list:int) number of hidden units for each region
        :param seed: seed to torch random number generator. for reproducibilty
        :param rnntype: 'RNN', 'GRU', or 'LSTM'
        :param snr: (float) variance of noise on hidden units. default is no noise
        :param bias_init: deprecated. kept for historical purposes
        :param dropout: proportion of neurons to be dropped at every forward call
        :param rank
        :param kwargs:
        """
        super(RNNModel_multiregion, self).__init__(**kwargs)

        self.inputcase = inputcase

        # set random seed
        if seed is not None:
            self.seed = seed
            torch.manual_seed(self.seed)

        self.nregions = len(num_hiddens)  # had better be two...

        rnn = []
        self.rank = rank

        if rnntype == 'RNN':
            for k in range(self.nregions):
                if k == 0:
                    nh_k = din[k]
                else:
                    nh_k = dout[k-1] + din[k] - 1  # -1 accounts for also having it track reward rate as out
                rnn.append(nn.RNN(nh_k, num_hiddens[k], batch_first=True))
        elif rnntype == 'GRU':
            for k in range(self.nregions):
                if k == 0:
                    nh_k = din[k]
                else:
                    nh_k = dout[k-1] + din[k] - 1
                rnn.append(nn.GRU(nh_k, num_hiddens[k], batch_first=True))
        elif rnntype == 'LSTM':
            for k in range(self.nregions):
                if k == 0:
                    nh_k = din[k]
                else:
                    nh_k = dout[k-1] + din[k] - 1

                rnn.append(nn.LSTM(nh_k, num_hiddens[k], batch_first=True))
        else:
            print('wrong RNN type: RNN,GRU,LSTM options. default to RNN')
            for k in range(self.nregions):
                if k == 0:
                    nh_k = din[k]
                else:
                    nh_k = dout[k-1] + din[k] - 1
                rnn.append(nn.RNN(nh_k, num_hiddens[k], batch_first=True))

        self.rnntype = rnntype  # easier to remember type this way
        self.rnn1 = rnn[0]  # the list of  vanilla RNN, LSTM, GRU, etc
        self.rnn2 = rnn[1]  # the list of  vanilla RNN, LSTM, GRU, etc
        self.num_hiddens = num_hiddens  # list of hidden state numbers
        self.dout = dout  # same, for output size
        self.din = din  # same, for input size
        self.snr = snr
        self.dropout = dropout

        # output layer. not bidirectional
        self.num_directions = 1

        # build the output heads
        linear = []
        for k in range(self.nregions):
            linear.append(nn.Linear(num_hiddens[k], dout[k]).requires_grad_(True))
        self.linear1 = linear[0]
        self.linear2 = linear[1]

    def forward(self, inputs, state, isITI=False, device=torch.device('cpu')):
        """
        propagates network forward, produces output. first unit is ofc (prediction loss), second unit is STR
        will send output (block prop estimates) to STR
        (action, value, entropy losses)
        :param inputs: (list) of inputs for each subunit. each a 3-d tensor of form (nbatch=1,nt=1, num_hiddenk)
        :param state: (list) of states for each subunit. for LSTM, this is a 2-tuple of cell and hidden
                        which are themselves 3-d tensors for size (nbatch, nt, num_hidden1), where nbatch
                        is size of minibatch, nt number of time points (=1), and number of units
        :param isITI: (bool) for if this timestep is an ITI step, which restricts action space
<<<<<<< HEAD
        :param device: (torch.device) object for specifying where computation will happen. 'cpu' or 'gpu' normally
=======
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
        :return:
        """
        # assume inputs is already a tensor of correct shape.
        # preprocessing should happen out of this class method

        # unfortunately this does not give state at all times, which I want for my architecture
        # have to loop over sequence length
        batchsize, L, din = inputs[0].shape
        if self.rnntype == 'LSTM':
            nhidden = state[0][0].shape[-1]
        else:  # and RNN
            nhidden = state[0].shape[-1]

        dout1 = self.dout[0]
<<<<<<< HEAD
        # TODO: make this agnostic to LSTM vs. RNN
        state1_hidden = torch.zeros((L, batchsize, nhidden), device=device)
        if self.rnntype == 'LSTM':
            state1_cell = torch.zeros((L, batchsize, nhidden), device=device)
        else:
            state1_cell = None
=======
        state1_hidden = torch.zeros((L, batchsize, nhidden), device=device)
        if self.rnntype == 'LSTM':
            state1_cell = torch.zeros((L, batchsize, nhidden), device=device)
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
        Y1 = torch.zeros((batchsize, L, dout1), device=device)
        state1k = state[0]
        for k in range(L):
            ik = torch.unsqueeze(inputs[0][:, k, :], dim=1)
            Y1k, state1k = self.rnn1(ik, state1k)
            # add noise
            Y1k = self.linear1(Y1k)

            if self.rnntype == 'LSTM':
                state1_hidden[k, :, :] = state1k[0][0, :, :] + self.snr * torch.randn(state1k[0][0, :, :].shape,
                                                                                      device=device)
                state1_cell[k, :, :] = state1k[1][0, :, :]
            else:
                state1_hidden[k, :, :] = state1k[0, :, :] + self.snr * torch.randn(state1k[0, :, :].shape,
                                                                                   device=device)
            Y1[:, k, :] = torch.squeeze(Y1k, dim=1)

        # repackage state 1
        if self.rnntype == 'LSTM':
            state1 = (torch.unsqueeze(state1_hidden[-1, :, :], dim=0),
                      torch.unsqueeze(state1_cell[-1, :, :], dim=0))
        else:
            state1 = torch.unsqueeze(state1_hidden[-1, :, :], dim=0)
<<<<<<< HEAD
        # Y1, state1 = self.rnn1(inputs[0], state[0])
=======
        # Y1, state1 = self.rnn1(inputs[0], state[0])  #THIS IS SLOPPY
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)

        # OFC supervised ouputs for kindergarten on the DMS units: trial offer and counting
        Y1 = Y1.reshape((-1, Y1.shape[-1]))
        pred_activations = Y1[:, 0:dout1-1]  # activations for predicting block, or trial... whatever you want
        output_supervised_ofc = torch.unsqueeze(Y1[:, dout1-1], dim=1)  # reward rate prediction

        ofc_output = torch.reshape(pred_activations, (batchsize, L, -1))

        if self.rnntype == 'LSTM':
            inp2 = torch.cat((inputs[1], ofc_output),
                             dim=2)  # hidden state of OFC is now input along w other inputs
        else:  # TODO: functional? this will be hard to write modular for gru and vanilla RNN
            inp2 = torch.cat((inputs[1], ofc_output),
                             dim=2)  # just entire state for GRU and vanilla RNN
        Y2, state2 = self.rnn2(inp2, state[1])

        # add noise to state2
        if self.rnntype == 'LSTM':
            state2 = list(state2)
            state2[0] = state2[0] + self.snr*torch.randn(state2[1].shape, device=device)
            state2 = tuple(state2)
        else:
            state2 = state2 + self.snr * torch.randn(state2.shape, device=device)

        # reshape action
        # The fully connected layer will first change the shape of `Y` to
        # (`num_steps` * `batch_size`, `num_hiddens`). Its output shape is
        # (`num_steps` * `batch_size`, `dim`).
        Y2 = Y2.reshape((-1, Y2.shape[-1]))

        # the rest of the outputs from DMS
        output2 = self.linear2(Y2)
        output_supervised_dms = output2[:, 4:6]  # reward offer memory and tria length integration
        output_supervised = torch.cat((output_supervised_dms, output_supervised_ofc), dim=1)

        # include ouputs that are related to sham tasks. would be extra ouputs in STr
        if self.dout[1] > 6:
            output_sham = output2[:, 6:]  # output of sham tasks, such as delay 2 match
        else:
<<<<<<< HEAD
=======
            #output_sham = np.nan
            output_sham = torch.tensor(np.nan, device=device)

        # only do softmax on first two entries, which is pwait, and poo
        pi_activation = output2[:, :3]
        if isITI:  # when in the ITI, set prob of wait and optout to zero. will this throw an in-place error?
            pi_activation[:, :2] = -1000.0
        else:  # in a trial. force the no "ITI" action
            pi_activation[:, 2] = -1000.0
        output_policy = nn.Softmax(dim=1)(pi_activation)

        output_value = output2[:, 3]

        # concat state
        state = [state1, state2]

        returndict = {'policy': output_policy, 'value': output_value, 'state': state,
                      'activations': pred_activations, 'pred_supervised': output_supervised,
                      'output_sham': output_sham}

        return returndict

class RNNModel_multiregion_2_abstract(RNNModel_multiregion_2):
    #
    # a multiregion network that has a separate output head from OFC for prediction. NOT the connection

    # TODO: do I have to include this, or can it just be assumed from class inheritance?

    def __init__(self, din, dout, num_hiddens, seed=None, rnntype='RNN', snr=0.0, bias_init=0, dropout=0.0, rank=None,
                 inputcase=11, **kwargs):
        """
        this
        :param din: (list: int) size of input of each region
        :param dout (list: int) size of outputs for each region:
        :param num_hiddens: (list:int) number of hidden units for each region
        :param seed: seed to torch random number generator. for reproducibilty
        :param rnntype: 'RNN', 'GRU', or 'LSTM'
        :param snr: (float) variance of noise on hidden units. default is no noise
        :param bias_init: deprecated. kept for historical purposes
        :param dropout: proportion of neurons to be dropped at every forward call
        :param rank: NOT used
        :param inputcase: (int) which type of inputes will you be reciving. defined in utils.py set_inputs()
        :param kwargs:
        """

        # must include full init becasuse the way that din for striatum is handled is different



        #super(RNNModel_multiregion_2, self).__init__(din, dout, num_hiddens,**kwargs)
        super(RNNModel_multiregion, self).__init__(**kwargs)

        self.inputcase = inputcase

        # set random seed
        if seed is not None:
            self.seed = seed
            torch.manual_seed(self.seed)

        self.nregions = len(num_hiddens)  # had better be two...

        rnn = []
        self.rank = rank

        if rnntype == 'RNN':
            for k in range(self.nregions):
                if k == 0:
                    nh_k = din[k]
                else:
                    nh_k = din[k] + 3  # accounts for 3 inputs
                rnn.append(nn.RNN(nh_k, num_hiddens[k], batch_first=True))
        elif rnntype == 'GRU':
            for k in range(self.nregions):
                if k == 0:
                    nh_k = din[k]
                else:
                    nh_k = din[k] + 3  # accounts for 3 inputs
                rnn.append(nn.GRU(nh_k, num_hiddens[k], batch_first=True))
        elif rnntype == 'LSTM':
            for k in range(self.nregions):
                if k == 0:
                    nh_k = din[k]
                else:
                    nh_k = din[k] + 3  # accounts for 3 inputs

                rnn.append(nn.LSTM(nh_k, num_hiddens[k], batch_first=True))
        else:
            print('wrong RNN type: RNN,GRU,LSTM options. default to RNN')
            for k in range(self.nregions):
                if k == 0:
                    nh_k = din[k]
                else:
                    nh_k = dout[k - 1] + din[k] + 3  # accounts for 3 inputs
                rnn.append(nn.RNN(nh_k, num_hiddens[k], batch_first=True))

        self.rnntype = rnntype  # easier to remember type this way
        self.rnn1 = rnn[0]  # the list of  vanilla RNN, LSTM, GRU, etc
        self.rnn2 = rnn[1]  # the list of  vanilla RNN, LSTM, GRU, etc
        self.num_hiddens = num_hiddens  # list of hidden state numbers
        self.dout = dout  # same, for output size
        self.din = din  # same, for input size
        self.snr = snr
        self.dropout = dropout

        # output layer. not bidirectional
        self.num_directions = 1

        # build the output heads
        linear = []
        for k in range(self.nregions):
            linear.append(nn.Linear(num_hiddens[k], dout[k]).requires_grad_(True))
        self.linear1 = linear[0]
        self.linear2 = linear[1]


    def forward(self, inputs, state, isITI=False, device=torch.device('cpu')):
        """
        propagates network forward, produces output. first unit is ofc (prediction loss), second unit is STR
        will send output (block prop estimates) to STR
        (action, value, entropy losses)
        :param inputs: (list) of inputs for each subunit. each a 3-d tensor of form (nbatch=1,nt=1, num_hiddenk)
        :param state: (list) of states for each subunit. for LSTM, this is a 2-tuple of cell and hidden
                        which are themselves 3-d tensors for size (nbatch, nt, num_hidden1), where nbatch
                        is size of minibatch, nt number of time points (=1), and number of units
        :param isITI: (bool) for if this timestep is an ITI step, which restricts action space
        :return:
        """
        # assume inputs is already a tensor of correct shape.
        # preprocessing should happen out of this class method

        # unfortunately this does not give state at all times, which I want for my architecture
        # have to loop over sequence length
        batchsize, L, din = inputs[0].shape
        if self.rnntype == 'LSTM':
            nhidden = state[0][0].shape[-1]
        else:  # and RNN
            nhidden = state[0].shape[-1]

        dout1 = self.dout[0]
        state1_hidden = torch.zeros((L, batchsize, nhidden), device=device)
        if self.rnntype == 'LSTM':
            state1_cell = torch.zeros((L, batchsize, nhidden), device=device)
        Y1 = torch.zeros((batchsize, L, dout1), device=device)
        state1k = state[0]
        for k in range(L):
            ik = torch.unsqueeze(inputs[0][:, k, :], dim=1)
            Y1k, state1k = self.rnn1(ik, state1k)
            # add noise
            Y1k = self.linear1(Y1k)

            if self.rnntype == 'LSTM':
                state1_hidden[k, :, :] = state1k[0][0, :, :] + self.snr * torch.randn(state1k[0][0, :, :].shape,
                                                                                      device=device)
                state1_cell[k, :, :] = state1k[1][0, :, :]
            else:
                state1_hidden[k, :, :] = state1k[0, :, :] + self.snr * torch.randn(state1k[0, :, :].shape,
                                                                                   device=device)
            Y1[:, k, :] = torch.squeeze(Y1k, dim=1)

        # repackage state 1
        if self.rnntype == 'LSTM':
            state1 = (torch.unsqueeze(state1_hidden[-1, :, :], dim=0),
                      torch.unsqueeze(state1_cell[-1, :, :], dim=0))
        else:
            state1 = torch.unsqueeze(state1_hidden[-1, :, :], dim=0)
        # Y1, state1 = self.rnn1(inputs[0], state[0])  #THIS IS SLOPPY

        # OFC supervised ouputs for kindergarten on the DMS units: trial offer and counting
        Y1 = Y1.reshape((-1, Y1.shape[-1]))

        # convention is first 3 are connection, next is integration output head, remaining are states output head
        pred_activations = Y1[:, 4:dout1]  # activations for predicting block in separate output head
        output_supervised_ofc = torch.unsqueeze(Y1[:, 3], dim=1)  # reward rate prediction
        ofc_output2str = Y1[:, :3]  # the connections between ofc and str

        ofc_output = torch.reshape(ofc_output2str, (batchsize, L, -1))

        if self.rnntype == 'LSTM':
            inp2 = torch.cat((inputs[1], ofc_output),
                             dim=2)  # hidden state of OFC is now input along w other inputs
        else:  # TODO: functional? this will be hard to write modular for gru and vanilla RNN
            inp2 = torch.cat((inputs[1], ofc_output),
                             dim=2)  # just entire state for GRU and vanilla RNN
        Y2, state2 = self.rnn2(inp2, state[1])

        # add noise to state2
        if self.rnntype == 'LSTM':
            state2 = list(state2)
            state2[0] = state2[0] + self.snr*torch.randn(state2[1].shape, device=device)
            state2 = tuple(state2)
        else:
            state2 = state2 + self.snr * torch.randn(state2.shape, device=device)

        # reshape action
        # The fully connected layer will first change the shape of `Y` to
        # (`num_steps` * `batch_size`, `num_hiddens`). Its output shape is
        # (`num_steps` * `batch_size`, `dim`).
        Y2 = Y2.reshape((-1, Y2.shape[-1]))

        # the rest of the outputs from DMS
        output2 = self.linear2(Y2)
        output_supervised_dms = output2[:, 4:6]  # reward offer memory and tria length integration
        output_supervised = torch.cat((output_supervised_dms, output_supervised_ofc), dim=1)

        # include ouputs that are related to sham tasks. would be extra ouputs in STr
        if self.dout[1] > 6:
            output_sham = output2[:, 6:]  # output of sham tasks, such as delay 2 match
        else:
            #output_sham = np.nan
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
            output_sham = torch.tensor(np.nan, device=device)

        # only do softmax on first two entries, which is pwait, and poo
        pi_activation = output2[:, :3]
        if isITI:  # when in the ITI, set prob of wait and optout to zero. will this throw an in-place error?
            pi_activation[:, :2] = -1000.0
        else:  # in a trial. force the no "ITI" action
            pi_activation[:, 2] = -1000.0
        output_policy = nn.Softmax(dim=1)(pi_activation)

        output_value = output2[:, 3]

        # concat state
        state = [state1, state2]

        returndict = {'policy': output_policy, 'value': output_value, 'state': state,
                      'activations': pred_activations, 'pred_supervised': output_supervised,
                      'output_sham': output_sham}

        return returndict


class lowrank_LSTM(nn.LSTM):

    def __init__(self, rank=1, **kwargs):
        super(lowrank_LSTM, self).__init__(**kwargs)

        # then override the weight matrices to have low rank
        hidden_size = self.hidden_size

        mi = nn.Parameter(torch.Tensor(hidden_size, rank))
        ni = nn.Parameter(torch.Tensor(hidden_size, rank))
        mf = nn.Parameter(torch.Tensor(hidden_size, rank))
        nf = nn.Parameter(torch.Tensor(hidden_size, rank))
        mg = nn.Parameter(torch.Tensor(hidden_size, rank))
        ng = nn.Parameter(torch.Tensor(hidden_size, rank))
        mo = nn.Parameter(torch.Tensor(hidden_size, rank))
        no = nn.Parameter(torch.Tensor(hidden_size, rank))

        with torch.no_grad():
            mi.normal_(std=1/hidden_size)
            ni.normal_(std=1/hidden_size)
            mf.normal_(std=1/hidden_size)
            nf.normal_(std=1/hidden_size)
            mg.normal_(std=1/hidden_size)
            ng.normal_(std=1/hidden_size)
            mo.normal_(std=1/hidden_size)
            no.normal_(std=1/hidden_size)

        # combine into the concatenate recurrent weights
        hh_i = torch.matmul(mi, ni.T)
        hh_f = torch.matmul(mf, nf.T)
        hh_g = torch.matmul(mg, ng.T)
        hh_o = torch.matmul(mo, no.T)

        hh = torch.cat((hh_i, hh_f, hh_g, hh_o), 0)

        # replace with new one, and assigne weights
        del self._parameters['weight_hh_l0']
        self.weight_hh_l0 = hh

        self.register_parameter('mi', mi)
        self.register_parameter('mf', mf)
        self.register_parameter('mg', mg)
        self.register_parameter('mo', mo)
        self.register_parameter('ni', ni)
        self.register_parameter('nf', nf)
        self.register_parameter('ng', ng)
        self.register_parameter('no', no)
        self.flatten_parameters()


class lowrank_RNN(nn.RNN):

    def __init__(self, rank=1, **kwargs):
        super(lowrank_RNN, self).__init__(**kwargs)

        # then override the weight matrices to have low rank
        hidden_size = self.hidden_size

        m = nn.Parameter(torch.Tensor(hidden_size, rank))
<<<<<<< HEAD
        # n = nn.Parameter(torch.Tensor(hidden_size, rank))

        with torch.no_grad():
            m.normal_(std=1/hidden_size)
=======
        n = nn.Parameter(torch.Tensor(hidden_size, rank))

        # TODO: make m a n somewhat overlap to bias the spectrum?
        with torch.no_grad():
            m.normal_(std=1/hidden_size)
            # n.normal_(std=1/hidden_size)
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
            n = m + 0.01/hidden_size * torch.rand((hidden_size, rank))

        # combine into the concatenate recurrent weights
        hh = torch.matmul(m, n.T)

        # replace with new one, and assigne weights
        del self._parameters['weight_hh_l0']
        self.weight_hh_l0 = hh
<<<<<<< HEAD
=======

>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
        self.register_parameter('m', m)
        self.register_parameter('n', m)
        self.flatten_parameters()


class RNNModel_lowrank(RNNModel_multiregion_2):
<<<<<<< HEAD
    def __init__(self, rank=None, **kwargs):
        # device = kwargs.pop('device')
        super(RNNModel_lowrank, self).__init__(**kwargs)
        if self.rank is None:
            self.rank = [None, None]
=======
    def __init__(self, rank=[None, None], **kwargs):
        # device = kwargs.pop('device')
        super(RNNModel_lowrank, self).__init__(**kwargs)
        self.rank = rank
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
        rnn = []
        # replace the rnn liat
        for k in range(self.nregions):
            if k == 0:
                nh_k = self.din[k]
            else:
                nh_k = self.dout[k - 1] + self.din[k] - 1
            if self.rnntype == 'LSTM':
                if rank[k] is not None:
                    rnn.append(lowrank_LSTM(rank=self.rank[k], input_size=nh_k,
                                            hidden_size=self.num_hiddens[k], batch_first=True))
                else:
                    rnn.append(nn.LSTM(nh_k, self.num_hiddens[k], batch_first=True))
            elif self.rnntype == 'RNN':
                if rank[k] is not None:
                    rnn.append(lowrank_RNN(rank=self.rank[k], input_size=nh_k,
                                           hidden_size=self.num_hiddens[k], batch_first=True))
                else:
                    rnn.append(nn.RNN(nh_k, self.num_hiddens[k], batch_first=True))
            elif self.rnntype == 'GRU':
                rnn.append(nn.GRU(nh_k, self.num_hiddens[k], batch_first=True))
            else:
                print('rnn type not correct. must be LSTM, RNN, GRU. default to RNN')
                rnn.append(nn.RNN(nh_k, self.num_hiddens[k], batch_first=True))

        self.rnn1 = rnn[0]
        self.rnn2 = rnn[1]
        # self.updateweights(device=device)

    def updateweights_LSTM(self, device=torch.device('cpu')):
        # recreate connectivity
        hh_i1 = torch.matmul(self.rnn1.mi, self.rnn1.ni.T)
        hh_f1 = torch.matmul(self.rnn1.mf, self.rnn1.nf.T)
        hh_g1 = torch.matmul(self.rnn1.mg, self.rnn1.ng.T)
        hh_o1 = torch.matmul(self.rnn1.mo, self.rnn1.no.T)
        hh1 = torch.cat((hh_i1, hh_f1, hh_g1, hh_o1), 0)
        hh1 = hh1.to(device)

        hh_i2 = torch.matmul(self.rnn2.mi, self.rnn2.ni.T)
        hh_f2 = torch.matmul(self.rnn2.mf, self.rnn2.nf.T)
        hh_g2 = torch.matmul(self.rnn2.mg, self.rnn2.ng.T)
        hh_o2 = torch.matmul(self.rnn2.mo, self.rnn2.no.T)
        hh2 = torch.cat((hh_i2, hh_f2, hh_g2, hh_o2), 0)
        hh2 = hh2.to(device)

        self.rnn1.weight_hh_l0 = hh1
        self.rnn2.weight_hh_l0 = hh2

        self.rnn1.flatten_parameters()
        self.rnn2.flatten_parameters()

        return None

    def updateweights_RNN(self, device=torch.device('cpu')):
        # recreate connectivity
        hh1 = torch.matmul(self.rnn1.m, self.rnn1.n.T)
        hh1 = hh1.to(device)

        hh2 = torch.matmul(self.rnn2.m, self.rnn2.n.T)
        hh2 = hh2.to(device)

        self.rnn1.weight_hh_l0 = hh1
        self.rnn2.weight_hh_l0 = hh2

        self.rnn1.flatten_parameters()
        self.rnn2.flatten_parameters()

        return None

    def updateweights(self, device=torch.device('cpu')):
        if self.rnntype == 'LSTM':
            self.updateweights_LSTM(device)
        elif self.rnntype == 'RNN':
            self.updateweights_RNN(device)

        return None

    def forward(self, inputs, state, isITI=False, device=torch.device('cpu'), train=True):

        if train:
            self.updateweights(device=device)

        returndict = super(RNNModel_lowrank, self).forward(inputs, state, isITI=isITI, device=device)
        return returndict

    def begin_state(self, device, batchsize=1):
        """
        initilizes a network, but to all zeros. seems to be a requirement for low-rank
        networks to stay on the low-rank manifold
        :param device:
        :param batchsize:
        :return:
        """

        statelist = super(RNNModel_lowrank, self).begin_state(device, batchsize=batchsize)
        # assume this has the list-->tuple--> tensor form of LSTMS
        if self.rnntype == 'LSTM':
            statelist = [tuple(0.0*m for m in k) for k in statelist]
        else:  # and RNN
            statelist = [0.0 * k for k in statelist]
        return statelist


class RNNModel_multiregion_2_softmax(RNNModel_multiregion_2):

    def forward(self, inputs, state, isITI=False, device=torch.device('cpu')):
        """
        propagates network forward, produces output. first unit is ofc (prediction loss), second unit is STR
        will send output (block prop estimates) to STR
        (action, value, entropy losses)
        :param inputs: (list) of inputs for each subunit. each a 3-d tensor of form (nbatch=1,nt=1, num_hiddenk)
        :param state: (list) of states for each subunit. for LSTM, this is a 2-tuple of cell and hidden
                        which are themselves 3-d tensors for size (nbatch, nt, num_hidden1), where nbatch
                        is size of minibatch, nt number of time points (=1), and number of units
        :param isITI: (bool) for if this timestep is an ITI step, which restricts action space
<<<<<<< HEAD
        :param device: (torch.device) object for specifying where computation will happen. 'cpu' or 'gpu' normally
=======
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
        :return:
        """
        # assume inputs is already a tensor of correct shape.
        # preprocessing should happen out of this class method

        # unfortunately this does not give state at all times, which I want for my architecture
        # have to loop over sequence length
        batchsize, L, din = inputs[0].shape
        nhidden = state[0][0].shape[-1]

        dout1 = self.dout[0]
        state1_hidden = torch.zeros((batchsize, L, nhidden), device=device)
        state1_cell = torch.zeros((batchsize, L, nhidden), device=device)
        Y1 = torch.zeros((batchsize, L, dout1), device=device)
        state1k = state[0]
        for k in range(L):
            ik = torch.unsqueeze(inputs[0][:, k, :], dim=1)
            Y1k, state1k = self.rnn1(ik, state1k)
            # add noise
            Y1k = self.linear1(nn.Dropout(p=self.dropout)(Y1k))
            state1_hidden[:, k, :] = state1k[0][:, k, :] + self.snr*torch.randn(state1k[0][:, k, :].shape,
                                                                                device=device)
            state1_cell[:, k, :] = state1k[1][:, k, :]
            Y1[:, k, :] = torch.squeeze(Y1k, dim=1)

        # repackage state 1
        state1 = (torch.unsqueeze(state1_hidden[:, -1, :], dim=1),
                  torch.unsqueeze(state1_cell[:, -1, :], dim=1))
<<<<<<< HEAD
=======
        # Y1, state1 = self.rnn1(inputs[0], state[0])
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)

        # OFC supervised ouputs for kindergarten on the DMS units: trial offer and counting
        Y1 = Y1.reshape((-1, Y1.shape[-1]))
        pred_activations = Y1[:, 0:dout1-1]  # activations for predicting block, or trial... whatever you want
        output_supervised_ofc = torch.unsqueeze(Y1[:, dout1-1], dim=1)  # reward rate prediction

        ofc_output = torch.reshape(torch.nn.Softmax(dim=1)(pred_activations), (batchsize, L, -1))

        if self.rnntype == 'LSTM':
            inp2 = torch.cat((inputs[1], ofc_output),
                             dim=2)  # hidden state of OFC is now input along w other inputs
        else:  # TODO: not functional. this will be hard to write modular for gru and vanilla RNN
            inp2 = torch.cat((inputs[1], ofc_output),
                             dim=2)  # just entire state for GRU and vanilla RNN
        Y2, state2 = self.rnn2(inp2, state[1])

        # add noise to state2
        state2 = list(state2)
        state2[0] = state2[0] + self.snr*torch.randn(state2[1].shape, device=device)
        state2 = tuple(state2)

        # reshape action
        # The fully connected layer will first change the shape of `Y` to
        # (`num_steps` * `batch_size`, `num_hiddens`). Its output shape is
        # (`num_steps` * `batch_size`, `dim`).
        Y2 = Y2.reshape((-1, Y2.shape[-1]))

        # the rest of the outputs from DMS
        output2 = self.linear2(nn.Dropout(p=self.dropout)(Y2))
        output_supervised_dms = output2[:, 4:]  # reward offer memory and tria length integration

        output_supervised = torch.cat((output_supervised_dms, output_supervised_ofc), dim=1)

        # only do softmax on first two entries, which is pwait, and poo
        pi_activation = output2[:, :3]
        if isITI:  # when in the ITI, set prob of wait and optout to zero. will this throw an in-place error?
            pi_activation[:, :2] = -1000.0
        else:  # in a trial. force the no "ITI" action
            pi_activation[:, 2] = -1000.0
        output_policy = nn.Softmax(dim=1)(pi_activation)

        output_value = output2[:, 3]

        # concat state
        state = [state1, state2]

        returndict = {'policy': output_policy, 'value': output_value, 'state': state,
                      'activations': pred_activations, 'pred_supervised': output_supervised}

        return returndict


<<<<<<< HEAD
=======

>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
netdict = {'RNNModel': {'net': RNNModel, 'din': 4, 'dout': 5,
                        'inputcase': 11},
           'RNNModelActorCritic': {'net': RNNModelActorCritic, 'din': 4, 'dout': 3,
                                   'inputcase': 11},
           'RNNModelSupervisable': {'net': RNNModelSupervisable, 'din': 4, 'dout': 3,
                                    'inputcase': 11},
           'RNNModelActorCriticSupervisable': {'net': RNNModelActorCriticSupervisable, 'din': 4, 'dout': 6,
                                               'inputcase': 11},
           'RNNModelActorCriticPredictive': {'net': RNNModelActorCriticSupervisable, 'din': 4, 'dout': 10,
                                             'inputcase': 11},
           'RNNModelActorCriticCued': {'net': RNNModelActorCriticSupervisable, 'din': 5, 'dout': 6,
                                       'inputcase': 11},
           'RNNModel_wITI': {'net': RNNModel_wITI, 'din': 4, 'dout': 7,
                             'inputcase': 11},
           'RNNModel_wITI_block': {'net': RNNModel_wITI, 'din': 5, 'dout': 7,
                                   'inputcase': 11},
           'RNNModelTrialProbPrediction': {'net': RNNModelTrialProbPrediction, 'din': 4, 'dout': 12,
                                           'inputcase': 11},
           'RNNModelTrialProbPrediction_allt': {'net': RNNModelTrialProbPrediction, 'din': 4, 'dout': 10,
                                                'inputcase': 11},
           'RNNModelTrialProbPrediction_block': {'net': RNNModelTrialProbPrediction, 'din': 5, 'dout': 12,
                                                 'inputcase': 11},
           'RNNModel_multiregion': {'net': RNNModel_multiregion, 'din': [4, 2], 'dout': [6, 6], 'inputcase': 11},
           'RNNModel_multiregion_2': {'net': RNNModel_multiregion_2, 'din': [4, 2], 'dout': [6, 6], 'inputcase': 11},
           'RNNModel_multiregion_2allt': {'net': RNNModel_multiregion_2, 'din': [4, 2], 'dout': [4, 6],
                                          'inputcase': 11},
           'RNNModel_multiregion_2_softmax': {'net': RNNModel_multiregion_2_softmax, 'din': [4, 2], 'dout': [4, 6],
                                              'inputcase': 11},
           'RNNModel_lowrank': {'net': RNNModel_lowrank, 'din': [4, 2], 'dout': [4, 6], 'inputcase': 11},
           'RNNModel_multiregion_wsham': {'net': RNNModel_multiregion_2, 'din': [4, 4], 'dout': [4, 9],
                                          'inputcase': 15},
<<<<<<< HEAD
           'RNNModel_photostim_ofc': {'net': RNNModel_multiregion_2, 'din': [5, 2], 'dout': [4, 6],
                                      'inputcase': 16}
=======
           'RNNModel_multiregion_2allt_5statepred': {'net': RNNModel_multiregion_2, 'din': [4, 2], 'dout': [6, 6],
                                                     'inputcase': 11},
           'RNNModel_multiregion_2allt_2statepred': {'net': RNNModel_multiregion_2, 'din': [4, 2], 'dout': [3, 6],
                                                     'inputcase': 11},
           'RNNModel_multiregion_2_abstract': {'net': RNNModel_multiregion_2_abstract, 'din':[4,2], 'dout':[9, 6],
                                               'inputcase': 11}
>>>>>>> 5ef8d19 (updating process methods for codeocean resubmission)
           }

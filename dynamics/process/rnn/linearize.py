# code to linearize dynamics of an RNN
import torch
from torch import nn
import numpy as np


class LSTM_fullcopy(nn.Module):
    # class to copy an LSTM and produce all gates

    def __init__(self, net, nreg=1, reg=1, rnntype='LSTM', **kwargs):
        super(LSTM_fullcopy, self).__init__(**kwargs)

        self.reg = reg
        self.nreg = nreg
        self.dout = net.dout  # output size. number of possible actions
        self.din = net.din  # input size. number inputs
        self.rnntype = rnntype

        if nreg == 1:
            rnn = net.rnn
        else:
            if reg == 0:
                rnn = net.rnn1
            elif reg == 1:
                rnn = net.rnn2
                self.rnn1 = net.rnn1
                self.linear1 = net.linear1
            else:
                print('wrong RNN region specification. defaulting to first region ')
                rnn = net.rnn1

        self.num_hiddens = rnn.hidden_size  # number of hidden states
        nh = self.num_hiddens
        self.num_directions = 1  # output layer. not bidirectional

        # copy the params from the (sub)-network
        i = 0*nh
        self.Wii = rnn.weight_ih_l0[i:i+nh, :]
        i = 1*nh
        self.Wif = rnn.weight_ih_l0[i:i+nh, :]
        i = 2*nh
        self.Wig = rnn.weight_ih_l0[i:i+nh, :]
        i = 2 * nh
        self.Wio = rnn.weight_ih_l0[i:i+nh, :]

        i = 0*nh
        self.Whi = rnn.weight_hh_l0[i:i+nh, :]
        i = 1*nh
        self.Whf = rnn.weight_hh_l0[i:i+nh, :]
        i = 2*nh
        self.Whg = rnn.weight_hh_l0[i:i+nh, :]
        i = 3*nh
        self.Who = rnn.weight_hh_l0[i:i+nh, :]

        i = 0*nh
        self.bii = rnn.bias_ih_l0[i:i+nh]
        i = 1*nh
        self.bif = rnn.bias_ih_l0[i:i+nh]
        i = 2*nh
        self.big = rnn.bias_ih_l0[i:i+nh]
        i = 3*nh
        self.bio = rnn.bias_ih_l0[i:i+nh]

        i = 0*nh
        self.bhi = rnn.bias_hh_l0[i:i+nh]
        i = 1*nh
        self.bhf = rnn.bias_hh_l0[i:i+nh]
        i = 2*nh
        self.bhg = rnn.bias_hh_l0[i:i+nh]
        i = 3*nh
        self.bho = rnn.bias_hh_l0[i:i+nh]

    def parse_inputs(self, x, h, c, device=torch.device('cpu')):
        """
        will parse up input to the RNN of interest. if first RNN, do nothing. otherwise, decide which network to
        follow
        :param x: input
        :param h: (tuple) hidden state. dim self.nreg
        :param c: (tuple) cell state. dim = self.nreg
        :param device: torch.device where comp is performed
        :return: parsed_inp, parsed_h, parsed_c
        """
        # if not linearizing region 1 (OFC), then send through regular OFC rnn.
        # adopting RNN_multiregion2 architecture
        if self.nreg == 1:
            parsed_inp = x
            parsed_h = h
            parsed_c = c
        else:
            if self.reg == 0:
                parsed_inp = x
                parsed_h = h[0]
                parsed_c = c[0]
            else:  # regularizing a network besides first subunit (OFC)
                # assume only two units. assume x is the full in type to OFC, not filtered by region yet
                x_str = x[:, :, 0:2]  # striatum inputs
                parsed_h = h[1]
                parsed_c = c[1]

                batchsize, L, din = x.shape
                Y1 = torch.zeros((batchsize, L, self.dout[0]), device=device)

                # repackage h and c of first unit
                state1k = (h[0], c[0])
                for k in range(L):  # loop over steps. should be 1 step at a time
                    ik = torch.unsqueeze(x[:, k, :], dim=1)
                    Y1k, state1k = self.rnn1(ik, state1k)
                    # add noise
                    Y1k = self.linear1(Y1k)
                    # state1_hidden[:, k, :] = state1k[0][0, :, :] + self.snr * torch.randn(state1k[0][0, :, :].shape,
                    #                                                                      device=device)
                    # state1_cell[:, k, :] = state1k[1][0, :, :]
                    Y1[:, k, :] = torch.squeeze(Y1k, dim=1)

                # OFC supervised ouputs for kindergarten on the DMS units: trial offer and counting
                Y1 = Y1.reshape((-1, Y1.shape[-1]))
                pred_activations = Y1[:, 0:self.dout[0] - 1]  # activations for predicting block, or trial
                ofc_output = torch.reshape(pred_activations, (batchsize, L, -1))

                if self.rnntype == 'LSTM':
                    parsed_inp = torch.cat((x_str, ofc_output), dim=2)  # add ofc_output as input
                else:
                    print('only LSTM supported for this code. check self.rnntype')
                    parsed_inp = None

        return parsed_inp, parsed_h, parsed_c

    def forward(self, x, h, c):
        """
        goal is to get the state variables of the RNN of interest. that's it. follows multiregion2 model
        :param x: inputs
        :param h: hidden state. all regions as a list
        :param c: cell state. all regions as a list
        :return: i,h,f,t states
        """

        if self.nreg > 1:
            h = h[self.reg]
            c = c[self.reg]

        # squeeze the input down to remove batch and layer
        x = torch.squeeze(torch.squeeze(x))
        if len(h.shape) > 1:
            h = torch.squeeze(torch.squeeze(h))
        if len(c.shape) > 1:
            c = torch.squeeze(torch.squeeze(c))

        s = torch.sigmoid

        it = s(torch.matmul(self.Wii, x) + self.bii + torch.matmul(self.Whi, h) + self.bhi)
        ft = s(torch.matmul(self.Wif, x) + self.bif + torch.matmul(self.Whf, h) + self.bhf)
        gt = torch.tanh(torch.matmul(self.Wig, x) + self.big + torch.matmul(self.Whg, h) + self.bhg)
        ot = s(torch.matmul(self.Wio, x) + self.bio + torch.matmul(self.Who, h) + self.bho)
        ct = ft*c + it*gt
        ht = ot*torch.tanh(ct)

        return it, ft, gt, ot, ct, ht

    def begin_state(self, device, batchsize=1):
        # how network is intialized
        sigma = 1.0
        # return torch.tensor(1.0)  # to use a linear policy, not RNN

        return (sigma * torch.rand((self.num_directions * self.rnn.num_layers,
                                    batchsize, self.num_hiddens),
                                   device=device),
                sigma * torch.rand((self.num_directions * self.rnn.num_layers,
                                    batchsize, self.num_hiddens),
                                   device=device))

    def linearize_full(self, x, h, c):
        """
        does the larger Jocobian, with state  = [h,c]. I think we can treat earlier gates (g,i,f,o) as not
        independent, since they are are multiplied into c. So c should be sufficient. Plus individual t-1 terms
        from them do not appear in expressions.

        Using wikipedia dimension order
        J = [h_h, h_c
             c_h, c_c]
        :param x: input
        :param h: hidden state. all regions as list
        :param c: cell state. all regions as list
        :return:
        """

        it, ft, gt, ot, ct, ht = self.forward(x, h, c)

        # make sure that correct region specified for weights, and correct h,c, regions
        if self.nreg > 1:
            h = h[self.reg]
            c = c[self.reg]

        # detach t-1 states
        s = torch.sigmoid
        x = torch.squeeze(torch.squeeze(x))
        if len(h.shape) > 1:
            h0 = torch.squeeze(torch.squeeze(h))
        else:
            print('shape of hidden state is weird. check shape')
            print(h.shape)
            h0 = None

        o = s(torch.matmul(self.Wio, x) + self.bio + torch.matmul(self.Who, h0) + self.bho)
        c = np.expand_dims(c[0, 0, :].detach().numpy(), axis=1)
        o = np.expand_dims(o.detach().numpy(), axis=1)  # higher dim

        nh = self.num_hiddens

        # detach t states
        it = np.expand_dims(it.detach().numpy(), axis=1)
        ft = np.expand_dims(ft.detach().numpy(), axis=1)
        gt = np.expand_dims(gt.detach().numpy(), axis=1)
        ot = np.expand_dims(ot.detach().numpy(), axis=1)
        ct = np.expand_dims(ct.detach().numpy(), axis=1)

        ones = np.ones((1, nh))

        kr = lambda x_kr: np.kron(x_kr, ones)

        # helpful derivatives: da = da_t/ dh_tm1
        di = kr(it * (1 - it)) * self.Whi.detach().numpy()
        df = kr(ft * (1 - ft)) * self.Whf.detach().numpy()
        dg = kr(1 - gt ** 2) * self.Whg.detach().numpy()
        do = kr(ot * (1 - ot)) * self.Who.detach().numpy()

        # more useful derivatives at t-1: dh_{t-1}/ dc_{t-1}, and similar for other states
        dhtm1_dctm1 = o*(1-np.tanh(c)**2)

        # the derivatives, in earnest
        # 1. dh_t / dh_t-1
        dc = df * c + di * gt + it * dg
        dht_dhtm1 = do * kr(np.tanh(ct)) + kr(ot) * (kr((1 - np.tanh(ct) ** 2)) * dc)

        # 2. dh_t / dc_t-1
        # helpers
        df_c = kr(ft*(1-ft))*dhtm1_dctm1*self.Whf.detach().numpy()
        di_c = kr(it*(1-it))*dhtm1_dctm1*self.Whf.detach().numpy()
        dg_c = kr((1-gt**2))*dhtm1_dctm1*self.Whg.detach().numpy()

        dht_dctm1_1 = kr(ot*(1-np.tanh(ct)**2))*(df_c*kr(c) + ft + di_c*kr(gt) + kr(it)*dg_c)
        dht_dctm1_2 = kr(ot*(1-ot)*np.tanh(ct))*dhtm1_dctm1*self.Who.detach().numpy()
        dht_dctm1 = np.diag(np.diag(dht_dctm1_1 + dht_dctm1_2))

        # 3. dc_t / dh_t-1
        dct_dhtm1 = df * kr(c) + dg * kr(it) + kr(gt) * di

        # 4. dc_t / dc_t-1. correct as diagonal, but why
        dct_dctm1_0 = ft
        dct_dctm1_1 = kr(it*(1-it) * gt) * dhtm1_dctm1*self.Whi.detach().numpy()
        dct_dctm1_2 = kr(it * (1 - gt**2)) * dhtm1_dctm1 * self.Whg.detach().numpy()
        dct_dctm1_3 = kr(ft * (1 - ft) * c) * dhtm1_dctm1 * self.Whf.detach().numpy()
        dct_dctm1 = dct_dctm1_0 + dct_dctm1_1 + dct_dctm1_2 + dct_dctm1_3
        dct_dctm1 = np.diag(np.diag(dct_dctm1))

        J1 = np.concatenate((dht_dhtm1, dht_dctm1), axis=1)
        J2 = np.concatenate((dct_dhtm1, dct_dctm1), axis=1)
        J = np.concatenate((J1, J2), axis=0)

        return J


def linprop(A, si, T=0.5, dt=0.05):
    """
    will propagate a linearized operator. do it discrete time as an AR1 process:
    x_t+1 = A x_t
    :param A:
    :param si:
    :param T:
    :param dt:
    :return:
    """

    tvec = np.linspace(0, T, int(T / dt))
    nt = len(tvec)
    ns = len(si)
    sall = np.zeros((ns, nt))

    sall[:, 0] = si

    for k in range(1, nt):
        st = A @ sall[:, k - 1]
        sall[:, k] = st

    return sall

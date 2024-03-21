# simulation environments for the wait time task
import numpy as np
import random
import dynamics.process.rnn.wt_delay2match as wt_d2m


class wt_env:
    # the wait time environment that will keep track of trials and output rewards
    def __init__(self, ops):

        # generate new trial info
        vdict = {0: [5, 10, 20, 40, 80],
                 1: [20, 40, 80],
                 2: [5, 10, 20]}

        self.ops = ops
        self.nstep = ops['ctmax']  # maximum trial duration before timeout
        self.seed = ops['seed']  # random seed for trials
        self.ns = ops['ns']  # how many trials to complete
        nstot = ops['ns'] + np.max([ops['ctmax'], 500]) + 1  # a padded total trials for pre-generating trial stuff
        self.ntmax = int(ops['T'] / ops['dt']) + 1
        self.tmesh = np.linspace(0, ops['T'], self.ntmax)  # time bins
        np.random.seed(self.seed)

        # set the master lists about optouts, rewards, delays, etc. add some padded extra trials for last batch
        self.vols = [np.random.choice(vdict[0], size=(nstot,)),
                     np.random.choice(vdict[1], size=(nstot,)),
                     np.random.choice(vdict[2], size=(nstot,))]  # volume of water reward

        self.delays = np.random.exponential(scale=ops['lam'], size=(nstot,))  # how long until reward arrives
        self.iscatch = np.random.binomial(1, ops['pcatch'], size=(nstot,)).astype(bool)  # is a catch trial

        # things to keep track of
        self.rewards_pertrial = []  # total reward from each trial. just curious
        self.trials = []  # running list of all trials

        # start a new trial. create a new one
        self.n = 0  # ho many trials completed
        self.trial = {'rewprev': 0, 'aprev': 0, 't': 0, 'w': self.ops['w'],
                      'block': 0, 'prevblock': 1, 'ct': 1, 'rewards': 0,
                      'v': self.vols[0][0], 'rewardDelay': self.delays[0], 'iscatch': self.iscatch[0],
                      'rt_idx': np.digitize(self.delays[0], bins=self.tmesh) - 1,
                      'auxinps': None}

    def step(self, action):
        """
        take a step forward in time
        @param action: (int) for deciding action taken. 0 = wait, 1 = opt-out, 2 = waitITI
        @return:
        """
        # step forward
        makenewtrial = False

        if action == 0:  # a wait action.

            if self.trial['t'] == self.trial['rt_idx'] and not self.trial['iscatch']:  # the rewarded time bin
                self.trial['outcome'] = 1
                rew_t = self.trial['w'] + self.trial['v']
                makenewtrial = True

            else:  # wait, but no reward
                rew_t = self.trial['w']

                if self.trial['t'] == self.ntmax:
                    self.trial['outcome'] = -1
                    makenewtrial = True

        else:  # an opt-out action
            rew_t = self.trial['w'] + self.ops['Ract']
            self.trial['outcome'] = 0
            makenewtrial = True

        # update previous rewards and actions for next simulation step
        self.trial['rewards'] += rew_t
        self.trial['rewprev'] = rew_t
        self.trial['aprev'] = action
        self.trial['t'] += 1

        if makenewtrial:
            self.newtrial()

        return rew_t

    def newtrial(self):
        """
            updates (dict) that keeps track of location within a trial, and a session. generates new trials
            :return:
            """

        prevtrial = self.trial
        self.trials.append(prevtrial)
        self.rewards_pertrial.append(prevtrial['rewards'])
        np.random.seed(self.n)  # the random block code wasnt random

        # make a new trial with some placeholder spots
        trial = {'rewards': 0, 'rewprev': prevtrial['rewprev'], 'aprev': prevtrial['aprev']}

        blocklen = self.ops['blocklen']
        self.n += 1

        # decide whether or not a block switch should occur
        if self.ops['useblocks']:
            # what block were you previously
            block = prevtrial['block']
            prevblock = prevtrial['prevblock']
            if prevtrial['ct'] < blocklen:
                trial['ct'] = prevtrial['ct'] + 1
                trial['block'] = block
                trial['prevblock'] = prevblock
            elif np.random.binomial(1, self.ops['pblock']) == 0:  # a failed, probabalistic block transition
                trial['ct'] = prevtrial['ct'] + 1
                trial['block'] = block
                trial['prevblock'] = prevblock
            else:  # block transition
                trial['ct'] = 0
                if block == 2 or block == 1:

                    trial['block'] = 0
                    trial['prevblock'] = block

                elif block == 0 and prevblock == 2:
                    trial['block'] = 1
                    trial['prevblock'] = block

                elif block == 0 and prevblock == 1:
                    trial['block'] = 2
                    trial['prevblock'] = block
        else:
            trial['block'] = 0
            trial['preblock'] = 0

        trial['v'] = self.vols[trial['block']][self.n]  # volume of water reward
        trial['rewardDelay'] = self.delays[self.n]  # how long until reward arrives
        trial['iscatch'] = self.iscatch[self.n]  # is a trial a catch trial

        trial['rt_idx'] = np.digitize(trial['rewardDelay'], bins=self.tmesh) - 1  # time index of reward
        trial['t'] = 0  # current index within a trial
        trial['w'] = self.ops['w']  # time penalty. track in case ever want to make adaptive
        trial['rewards'] = 0  # reset accrued reward on the trial
        trial['auxinps'] = None  # for sham tasks outside of the wt task. used in more advanced envs

        self.trial = trial

        return None


class wt_env_wITI(wt_env):
    # the wait time environment, but includes a variable ITI
    def __init__(self, ops):
        super().__init__(ops)
        nstot = ops['ns'] + ops['ctmax'] + 1  # a padded total trials for pre-generating trial stuff
        if ops['iti_random'] > 0:
            if ops['iti_type'] == 'guassian':  # sample from normal
                self.iti = np.floor(np.random.normal(ops['iti_random'], scale=3, size=nstot)).astype(int)
            else:  # uniform sampling from 1 to iti_rand
                self.iti = np.array([random.randint(1, ops['iti_random']) for _ in range(nstot)])
        else:
            self.iti = ops['iti_determ']*np.ones((nstot,)).astype(int)
        self.beginITI = False  # follows if trial was rewarded

    def step(self, action):

        # step forward
        makenewtrial = False

        # handle the inter-trial interval
        if self.beginITI:
            # rew_t = self.trial['w']
            rew_t = 0.0
            self.trial['t'] -= 1  # counteracts the counter later. easeir coding
            if self.trial['t_iti'] == self.iti[self.n]:
                makenewtrial = True
            else:
                self.trial['t_iti'] += 1

        else:
            if action == 0:  # a wait action.

                if self.trial['t'] == self.trial['rt_idx'] and not self.trial['iscatch']:  # the rewarded time bin
                    self.trial['outcome'] = 1
                    rew_t = self.trial['w'] + self.trial['v']
                    self.beginITI = True
                    self.trial['t_iti'] = 1  # iti counter

                else:  # wait, but no reward
                    rew_t = self.trial['w']

                    if self.trial['t'] == self.ntmax:
                        self.trial['outcome'] = -1
                        self.beginITI = True
                        self.trial['t_iti'] = 1

            else:  # an opt-out action
                rew_t = self.trial['w'] + self.ops['Ract']
                self.trial['outcome'] = 0
                self.beginITI = True
                self.trial['t_iti'] = 1

        # update previous rewards and actions for next simulation step
        self.trial['rewards'] += rew_t
        self.trial['rewprev'] = rew_t
        self.trial['aprev'] = action
        self.trial['t'] += 1

        if makenewtrial:
            self.newtrial()

        return rew_t

    def newtrial(self):
        super().newtrial()
        self.beginITI = False


# a class where the reward is given in pieces
class wt_env_wITI_batchedreward(wt_env_wITI):
    """
    an environment with a variable ITI, but doles reward out in even increments over the ITI
    """

    def __init__(self, ops):
        super().__init__(ops)

    def step(self, action):

        # step forward
        makenewtrial = False

        # handle the inter-trial interval
        if self.beginITI:

            self.trial['t'] -= 1  # counteracts the counter later. easeir coding
            if self.trial['t_iti'] == self.iti[self.n]:
                makenewtrial = True
                rew_t = 0.0  # dont reward last ITI to ovoid bleedover from previous trial to next
            else:
                if self.trial['outcome'] == 1:  # reward
                    rew_t = self.trial['v'] / self.iti[self.n]  # give proportion of reward at timestep
                elif self.trial['outcome'] == -1:  # an timeout
                    rew_t = 0.0
                elif self.trial['outcome'] == 0:  # an opt-out
                    rew_t = self.ops['Ract'] / self.iti[self.n]
                else:  # somethign went wront
                    rew_t = 0

            self.trial['t_iti'] += 1

        else:
            if action == 0:  # a wait action.

                if self.trial['t'] == self.trial['rt_idx'] and not self.trial['iscatch']:  # the rewarded time bin
                    self.trial['outcome'] = 1
                    rew_t = self.trial['w'] + self.trial['v']/self.iti[self.n]
                    self.beginITI = True
                    self.trial['t_iti'] = 1  # iti counter

                else:  # wait, but no reward
                    rew_t = self.trial['w']

                    if self.trial['t'] == self.ntmax:
                        self.trial['outcome'] = -1
                        self.beginITI = True
                        self.trial['t_iti'] = 1

            else:  # an opt-out action
                rew_t = self.trial['w'] + self.ops['Ract']/self.iti[self.n]
                self.trial['outcome'] = 0
                self.beginITI = True
                self.trial['t_iti'] = 1

        # update previous rewards and actions for next simulation step
        self.trial['rewards'] += rew_t
        self.trial['rewprev'] = rew_t
        self.trial['aprev'] = action
        self.trial['t'] += 1

        if makenewtrial:
            self.newtrial()

        return rew_t

    def newtrial(self):
        super().newtrial()
        self.beginITI = False


class wt_wshamtasks(wt_env_wITI_batchedreward):
    """
    a class that also includes step-by-step stimulus inputs about sham tasks.
    in particular, the delay to match/non-match task
    """

    def __init__(self, ops):
        super().__init__(ops)

        # create the data long enough to fit an entire minibatch
        self.shamidx = 0
        self.ntall_d2m = ops['ctmax']+1
        nt = int(ops['tmax_d2m'] / ops['dt'])  # how many timesteps per minibatch of d2m data
        self.ntd2m = nt
        nbatch = int(np.ceil(self.ntall_d2m/self.ntd2m))
        # initialize
        inp = None
        targ = None
        # seed trials by current main task trial
        for j in range(nbatch):
            inpj, targj, _, _, _ = wt_d2m.trainingdata(self.ops, seed=self.n, T=self.ntd2m, batchsize=1)
            if j == 0:
                inp = inpj
                targ = targj
            else:
                inp = np.concatenate((inp, inpj), axis=1)
                targ = np.concatenate((targ, targj), axis=1)
        self.d2m_inp = inp
        self.d2m_targ = targ

        self.trial['auxinps'] = [self.d2m_inp[0, self.shamidx, -2], self.d2m_inp[0, self.shamidx, -1]]

    def step(self, action):
        self.trial['auxinps'] = [self.d2m_inp[0, self.shamidx, -2], self.d2m_inp[0, self.shamidx, -1]]

        rew_t = super().step(action)
        self.shamidx += 1

        # need to make more sham data? it's a lot to preload
        if self.shamidx == self.ntall_d2m:
            nbatch = int(np.ceil(self.ntall_d2m / self.ntd2m))
            # initialize
            inp = None
            targ = None
            for j in range(nbatch):
                inpj, targj, _, _, _ = wt_d2m.trainingdata(self.ops, seed=self.n, T=self.ntd2m, batchsize=1)
                if j == 0:
                    inp = inpj
                    targ = targj
                else:
                    inp = np.concatenate((inp, inpj), axis=1)
                    targ = np.concatenate((targ, targj), axis=1)

            self.d2m_inp = inp
            self.d2m_targ = targ
            self.shamidx = 0

        return rew_t


envdict = {'wt_env': wt_env,
           'wt_env_wITI': wt_env_wITI,
           'wt_env_wITI_batchedreward': wt_env_wITI_batchedreward,
           'wt_wshamtasks': wt_wshamtasks}

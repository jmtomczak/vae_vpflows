from __future__ import print_function

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils.distributions import log_Bernoulli, log_Normal_diag, log_Normal_standard
from utils.visual_evaluation import plot_histogram

import numpy as np

from scipy.misc import logsumexp

def xavier_init(m):
    s =  np.sqrt( 2. / (m.in_features + m.out_features) )
    m.weight.data.normal_(0, s)

class linIAF(nn.Module):
    def __init__(self,args):
        super(linIAF, self).__init__()
        self.args = args

    def forward(self, L, z):
        '''
        :param L: batch_size (B) x latent_size^2 (L^2)
        :param z: batch_size (B) x latent_size (L)
        :return: z_new = L*z
        '''
        # L->tril(L)
        L_matrix = L.view( -1, self.args.z1_size, self.args.z1_size ) # resize to get B x L x L
        LTmask = torch.tril( torch.ones(self.args.z1_size, self.args.z1_size), k=-1 ) # lower-triangular mask matrix (1s in lower triangular part)
        I = Variable( torch.eye(self.args.z1_size, self.args.z1_size).expand(L_matrix.size(0), self.args.z1_size, self.args.z1_size) )
        if self.args.cuda:
            LTmask = LTmask.cuda()
            I = I.cuda()
        LTmask = Variable(LTmask)
        LTmask = LTmask.unsqueeze(0).expand( L_matrix.size(0), self.args.z1_size, self.args.z1_size ) # 1 x L x L -> B x L x L
        LT = torch.mul( L_matrix, LTmask ) + I # here we get a batch of lower-triangular matrices with ones on diagonal

        # z_new = L * z
        z_new = torch.bmm( LT , z.unsqueeze(2) ).squeeze(2) # B x L x L * B x L x 1 -> B x L

        return z_new

class combination_L(nn.Module):
    def __init__(self,args):
        super(combination_L, self).__init__()
        self.args = args

    def forward(self, L, y):
        '''
        :param L: batch_size (B) x latent_size^2 * number_combination (L^2 * C)
        :param y: batch_size (B) x number_combination (C)
        :return: L_combination = y * L
        '''
        # calculate combination of Ls
        L_tensor = L.view( -1, self.args.z1_size**2, self.args.number_combination ) # resize to get B x L^2 x C
        y = y.unsqueeze(1).expand(y.size(0), self.args.z1_size**2, y.size(1)) # expand to get B x L^2 x C
        L_combination = torch.sum( L_tensor * y, 2 ).squeeze()
        return L_combination

class Gate(nn.Module):
    def __init__(self):
        super(Gate, self).__init__()

    def forward(self, h, g):
        return h * g

#=======================================================================================================================
class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()

        self.args = args

        # encoder: q(z | x)
        self.q_z_layers_pre = nn.ModuleList()
        self.q_z_layers_gate = nn.ModuleList()

        self.q_z_layers_pre.append( nn.Linear(np.prod(self.args.input_size), 300) )
        self.q_z_layers_gate.append( nn.Linear(np.prod(self.args.input_size), 300) )

        self.q_z_layers_pre.append( nn.Linear(300, 300) )
        self.q_z_layers_gate.append( nn.Linear(300, 300) )

        self.q_z_mean = nn.Linear(300, self.args.z1_size)
        self.q_z_logvar = nn.Linear(300, self.args.z1_size)

        # decoder: p(x | z)
        self.p_x_layers_pre = nn.ModuleList()
        self.p_x_layers_gate = nn.ModuleList()

        self.p_x_layers_pre.append( nn.Linear(self.args.z1_size, 300) )
        self.p_x_layers_gate.append( nn.Linear(self.args.z1_size, 300) )

        self.p_x_layers_pre.append( nn.Linear(300, 300) )
        self.p_x_layers_gate.append( nn.Linear(300, 300) )

        self.p_x_mean = nn.Linear(300, np.prod(self.args.input_size))

        # convex combination linear Inverse Autoregressive flow
        self.encoder_y = nn.Linear( 300, self.args.number_combination )
        self.encoder_L = nn.Linear( 300, (self.args.z1_size**2) * self.args.number_combination )

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.linIAF = linIAF(self.args)
        self.combination_L = combination_L(self.args)

        self.Gate = Gate()

        # Xavier initialization (normal)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                xavier_init(m)

    # AUXILIARY METHODS
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if self.args.cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def calculate_likelihood(self, X, dir, mode='test', S=5000):
        # set auxiliary variables for number of training and test sets
        N_test = X.size(0)

        # init list
        likelihood_test = []

        MB = 500

        if S <= MB:
            R = 1
        else:
            R = S / MB
            S = MB

        for j in range(N_test):
            if j % 100 == 0:
                print('{:.2f}%'.format(j / (1. * N_test) * 100))
            # Take x*
            x_single = X[j].unsqueeze(0)

            a = []
            for r in range(0, R):
                # Repeat it for all training points
                x = x_single.expand(S, x_single.size(1))

                # pass through VAE
                x_mean, x_logvar, z_0, z_T, z_q_mean, z_q_logvar = self.forward(x)

                # RE
                RE = log_Bernoulli(x, x_mean, dim=1)

                # KL
                log_p_z = log_Normal_standard(z_T, dim=1)
                log_q_z = log_Normal_diag(z_0, z_q_mean, z_q_logvar, dim=1)
                KL = -(log_p_z - log_q_z)

                a_tmp = (RE - KL)

                a.append( a_tmp.cpu().data.numpy() )

            # calculate max
            a = np.asarray(a)
            a = np.reshape(a, (a.shape[0] * a.shape[1], 1))
            likelihood_x = logsumexp( a )
            likelihood_test.append(likelihood_x - np.log(len(a)))

        likelihood_test = np.array(likelihood_test)

        plot_histogram(-likelihood_test, dir, mode)

        return -np.mean(likelihood_test)

    def calculate_lower_bound(self, X_full):
        # CALCULATE LOWER BOUND:
        lower_bound = 0.
        RE_all = 0.
        KL_all = 0.

        MB = 500

        for i in range(X_full.size(0) / MB):
            x = X_full[i * MB: (i + 1) * MB].view(-1, np.prod(self.args.input_size))

            # pass through VAE
            x_mean, x_logvar, z_0, z_T, z_q_mean, z_q_logvar = self.forward(x)

            # RE
            RE = log_Bernoulli( x, x_mean )

            # KL
            log_p_z = log_Normal_standard( z_T, dim=1)
            log_q_z = log_Normal_diag( z_0, z_q_mean, z_q_logvar, dim=1 )
            KL = - torch.sum( log_p_z - log_q_z )

            RE_all += RE.cpu().data[0]
            KL_all += KL.cpu().data[0]

            # CALCULATE LOWER-BOUND: RE + KL - ln(N)
            lower_bound += (-RE + KL).cpu().data[0]

        lower_bound = lower_bound / X_full.size(0)

        return lower_bound

    # THE MODEL: VARIATIONAL POSTERIOR
    def q_z(self, x):
        h0_pre = self.q_z_layers_pre[0](x)
        h0_gate = self.sigmoid( self.q_z_layers_gate[0](x) )
        h0 = self.Gate( h0_pre, h0_gate )

        h1_pre = self.q_z_layers_pre[1](h0)
        h1_gate = self.sigmoid( self.q_z_layers_gate[1](h0) )
        h1 = self.Gate( h1_pre, h1_gate )

        z_q_mean = self.q_z_mean(h1)
        z_q_logvar = self.q_z_logvar(h1)
        return z_q_mean, z_q_logvar, h1

    # THE MODEL: HOUSEHOLDER FLOW
    def q_z_Flow(self, z, h_last):
        # ccLinIAF
        L = self.encoder_L(h_last)
        y = self.softmax(self.encoder_y(h_last))
        L_combination = self.combination_L(L, y)
        z['1'] = self.linIAF(L_combination, z['0'])
        return z

    # THE MODEL: GENERATIVE DISTRIBUTION
    def p_x(self, z):
        h0_pre = self.p_x_layers_pre[0](z)
        h0_gate = self.sigmoid( self.p_x_layers_gate[0](z) )
        h0 = self.Gate( h0_pre, h0_gate )

        h1_pre = self.p_x_layers_pre[1](h0)
        h1_gate = self.sigmoid( self.p_x_layers_gate[1](h0) )
        h1 = self.Gate( h1_pre, h1_gate )

        x_mean = self.sigmoid( self.p_x_mean(h1) )
        x_logvar = 0.
        return x_mean, x_logvar

    # THE MODEL: FORWARD PASS
    def forward(self, x):
        z = {}
        # z ~ q(z | x)
        z_q_mean, z_q_logvar, h_last = self.q_z(x)
        z['0'] = self.reparameterize(z_q_mean, z_q_logvar)
        # Householder Flow:
        z = self.q_z_Flow(z, h_last)

        # x_mean = p(x|z)
        x_mean, x_logvar = self.p_x(z['1'] )

        return x_mean, x_logvar, z['0'], z['1'], z_q_mean, z_q_logvar

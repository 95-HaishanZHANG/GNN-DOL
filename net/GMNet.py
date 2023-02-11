import torch
import torch.nn as nn
import torch.nn.functional as F
import cvxpy as cp
import numpy as np
from net.mpn import MPNet
# from net.cvxpylayer import CvxpyLayer
from cvxpylayers.torch import CvxpyLayer

class GMNet(nn.Module):
    def __init__(self, model_params, mode, batch_size=1, prob_size=30):
        super(GMNet, self).__init__()
        encoder_params = model_params['encoder_model_params']
        self.mpn_layer = MPNet(model_params, mode)
        self.soft = nn.Softmax(dim=1)
        self.b_s = batch_size
        self.prob_size = prob_size
        self.mode = mode
        if self.mode in ['train', 'val']:
            self._get_QP_problem()
            self.qp_layer = CvxpyLayer(self.prob, parameters=[self.Q_cp, self.A1_cp, self.A2_cp, self.G_cp, self.h_cp], variables=[self.x])

    def _generate_graph_struct(self, n_emb, e_emb, e_idx):
        node_num = n_emb.shape[1]
        edge_num = e_emb.shape[1]
        # A = np.ones((node_num, node_num)) - np.eye(node_num)
        G = np.zeros((self.b_s*node_num, edge_num), dtype=np.float32)
        H = np.zeros((self.b_s*node_num, edge_num), dtype=np.float32)
        row, col = e_idx
        # print('e_idx: ', G.shape, H.shape, e_idx.shape, edge_num)
        # print(row, col)
        edge_idx = np.tile(range(edge_num), self.b_s)
        G[row, edge_idx] = 1
        H[col, edge_idx] = 1
        # print(G[0])
        return G, H
    
    def _kronecker(self, A, B):
        # print('A shape: ', A.shape, B.shape)
        AB = torch.einsum("iab, icd->iacbd", A, B)
        # print('AB shape: ', AB.shape)
        AB = AB.view(self.b_s, A.size(1)*B.size(1), A.size(2)*B.size(2))
        return AB
    
    def _get_Q(self, n_emb_cur, e_emb_cur, n_emb_nxt, e_emb_nxt, num_x1, num_x2):
        m = num_x1 # self.prob_size  
        n = num_x2 # self.prob_size
        n_emb_cur = n_emb_cur.unsqueeze(dim=0).reshape(self.b_s, m, -1)
        e_emb_cur = e_emb_cur.unsqueeze(dim=0).reshape(self.b_s, m, -1)
        n_emb_nxt = n_emb_nxt.unsqueeze(dim=0).reshape(self.b_s, n, -1)
        e_emb_nxt = e_emb_nxt.unsqueeze(dim=0).reshape(self.b_s, n, -1)

        Kpp_g = torch.bmm(n_emb_cur, n_emb_nxt.transpose(1, 2)).cuda() # m * n
        Kqq_g = torch.bmm(e_emb_cur, e_emb_nxt.transpose(1, 2)).cuda() # m * n
        Kpp_xd1 = torch.diagonal(torch.bmm(n_emb_cur, n_emb_cur.transpose(1, 2)), dim1=-2, dim2=-1)
        Kpp_xd2 = torch.diagonal(torch.bmm(n_emb_nxt, n_emb_nxt.transpose(1, 2)), dim1=-2, dim2=-1)
        Kpp_x1 = Kpp_xd1.unsqueeze(-1).repeat(1, 1, Kpp_g.shape[-1])
        Kpp_x2 = Kpp_xd2.unsqueeze(-1).repeat(1, 1, Kpp_g.shape[-2]).transpose(1,2)
        Kp = Kpp_x1 + Kpp_x2 - 2 * Kpp_g
        Kp = Kp.transpose(1, 2).reshape(self.b_s, -1, 1).squeeze()
        Kqq_xd1 = torch.diagonal(torch.bmm(e_emb_cur, e_emb_cur.transpose(1, 2)), dim1=-2, dim2=-1)
        Kqq_xd2 = torch.diagonal(torch.bmm(e_emb_nxt, e_emb_nxt.transpose(1, 2)), dim1=-2, dim2=-1)
        Kqq_x1 = Kqq_xd1.unsqueeze(-1).repeat(1, 1, Kqq_g.shape[-1])
        Kqq_x2 = Kqq_xd2.unsqueeze(-1).repeat(1, 1, Kqq_g.shape[-2]).transpose(1,2)
        Kq = Kqq_x1 + Kqq_x2 - 2 * Kqq_g
        Kq = Kq.transpose(1, 2).reshape(self.b_s, -1, 1).squeeze()
        
        Mn = Kp.unsqueeze(dim=1)
        Me = Kq.unsqueeze(dim=1)
        
        return (Me+Mn)/2.
    
    def _get_QP_problem(self):
        m = self.prob_size
        n = self.prob_size
        E = np.identity(m*n)
        one = np.ones((m*n, 1))
        zero = np.zeros((m*n, 1))

        self.A1 = np.zeros((n, m*n), dtype = 'float32')
        self.A2 = np.zeros((m, m*n), dtype = 'float32')
        self.G = np.vstack((-E, E)).astype(np.float32)
        self.h = np.vstack((zero, one)).astype(np.float32)
        for i in range(n):
            self.A1[i, (i*m):((i+1)*m)] = 1
        for j in range(m):
            for i in range(n):
                self.A2[j, (i*m+j)] = 1

        tmp_A1 = np.zeros((self.b_s, n, m*n)).astype(np.float32)
        tmp_A2 = np.zeros((self.b_s, m, m*n)).astype(np.float32)
        tmp_G = np.zeros((self.b_s, (2*m*n), m*n)).astype(np.float32)
        tmp_h = np.zeros((self.b_s, (2*m*n), 1)).astype(np.float32)
        for i in range(self.b_s):
            tmp_A1[i] = self.A1
            tmp_A2[i] = self.A2
            tmp_G[i] = self.G
            tmp_h[i] = self.h
        self.A1, self.A2, self.G, self.h = tmp_A1, tmp_A2, tmp_G, tmp_h
        
        self.x = cp.Variable((m*n, 1), nonneg=True)
        self.Q_cp = cp.Parameter((1, m*n))
        self.A1_cp = cp.Parameter((n, m*n))
        self.A2_cp = cp.Parameter((m, m*n))
        self.G_cp = cp.Parameter((2*m*n, m*n))
        self.h_cp = cp.Parameter((2*m*n, 1))
        
        # Note: to apply SCS solver on GPU, we use the sum_squares to approximate x^TQx, since it is affine.
        # If you want to train the model on CPU, there is no need to use the affine approximation.  
        obj = cp.Minimize(cp.sum_squares(self.Q_cp @ self.x)) 

        cons = [self.A1_cp @ self.x == 1,
                self.A2_cp @ self.x <= 2,
                self.G_cp @ self.x <= self.h_cp]
        self.prob = cp.Problem(obj, cons)

    def _get_QP_problem_(self, Q_t, num_x1, num_x2):
        m = num_x1 
        n = num_x2 
        E = np.identity(m*n)
        one = np.ones((m*n, 1))
        zero = np.zeros((m*n, 1))

        self.A1 = np.zeros((n, m*n), dtype = 'float32')
        self.A2 = np.zeros((m, m*n), dtype = 'float32')
        self.G = np.vstack((-E, E)).astype(np.float32)
        self.h = np.vstack((zero, one)).astype(np.float32)
        for i in range(n):
            self.A1[i, (i*m):((i+1)*m)] = 1
        for j in range(m):
            for i in range(n):
                self.A2[j, (i*m+j)] = 1
        
        self.x = cp.Variable((m*n, 1), nonneg=True)
        
        obj = cp.Minimize(cp.sum_squares(Q_t.cpu().T @ self.x)) 
        cons = [self.A1 @ self.x == 1,
                self.A2 @ self.x <= 2,
                self.G @ self.x <= self.h]
        self.prob = cp.Problem(obj, cons)

    def forward(self, g_data):
        n_emb_cur, e_emb_cur = g_data.x1, g_data.e1
        n_emb_nxt, e_emb_nxt = g_data.x2, g_data.e2
        matching_idx = g_data.matching_idx
        num_x1, num_x2 = g_data.num_x1, g_data.num_x2
        n_emb_cur, n_emb_nxt, e_emb_cur, e_emb_nxt = self.mpn_layer(n_emb_cur, n_emb_nxt, e_emb_cur, e_emb_nxt, matching_idx, num_x1, num_x2)
        
        if self.mode in ['train', 'val']:
            Q_t = self._get_Q(n_emb_cur, e_emb_cur, n_emb_nxt, e_emb_nxt, self.prob_size, self.prob_size)
            A1_t, A2_t, G_t, h_t = map(torch.from_numpy, [self.A1, self.A2, self.G, self.h])
            Q_t = Q_t.requires_grad_()
            A1_t = A1_t.cuda() 
            A2_t = A2_t.cuda() 
            G_t = G_t.cuda() 
            h_t = h_t.cuda() 
            x_out = self.qp_layer(Q_t, A1_t, A2_t, G_t, h_t)
            x_star = x_out[0].squeeze()
        else:
            Q_t = self._get_Q(n_emb_cur, e_emb_cur, n_emb_nxt, e_emb_nxt, num_x1, num_x2)
            self._get_QP_problem_(Q_t, num_x1, num_x2)
            self.prob.solve(solver=cp.SCS, gpu= True, use_indirect=True)
            x_star = self.x.value

        return x_star

class GMNet_(nn.Module):
    def __init__(self, model_params):
        super(GMNet_, self).__init__()
        encoder_params = model_params['encoder_model_params']
        self.encoder = MLPEncoder(**encoder_params)
        self.mpn_layer = MPNet(model_params)
        self.soft = nn.Softmax(dim=1)
        self.b_s = 1
        
    
    def _get_Q(self, n_emb_cur, e_emb_cur, n_emb_nxt, e_emb_nxt, num_x1, num_x2):
        m = num_x1 #self.prob_size 
        n = num_x2 # self.prob_size
        # print('e_idx_cur: ', e_idx_cur.shape)
        # print('shape: ', n_emb_cur.shape, n_emb_nxt.shape, e_emb_cur.shape, e_emb_nxt.shape)
        n_emb_cur = F.normalize(n_emb_cur, dim=1)
        n_emb_nxt = F.normalize(n_emb_nxt, dim=1)
        e_emb_cur = F.normalize(e_emb_cur, dim=1)  
        e_emb_nxt = F.normalize(e_emb_nxt, dim=1)
        n_emb_cur = n_emb_cur.unsqueeze(dim=0)
        e_emb_cur = e_emb_cur.unsqueeze(dim=0)
        n_emb_nxt = n_emb_nxt.unsqueeze(dim=0)
        e_emb_nxt = e_emb_nxt.unsqueeze(dim=0)
        # print('shape: ', n_emb_cur.shape, n_emb_nxt.shape, e_emb_cur.shape, e_emb_nxt.shape)

        Kpp_g = torch.bmm(n_emb_cur, n_emb_nxt.transpose(1, 2)).cuda() # m * n
        Kqq_g = torch.bmm(e_emb_cur, e_emb_nxt.transpose(1, 2)).cuda() # m * n
        # print('shape: ', Kpp_g.shape, Kqq_g.shape)
        Kpp_xd1 = torch.diagonal(torch.bmm(n_emb_cur, n_emb_cur.transpose(1, 2)), dim1=-2, dim2=-1)
        Kpp_xd2 = torch.diagonal(torch.bmm(n_emb_nxt, n_emb_nxt.transpose(1, 2)), dim1=-2, dim2=-1)
        Kpp_x1 = Kpp_xd1.unsqueeze(-1).repeat(1, 1, Kpp_g.shape[-1])
        Kpp_x2 = Kpp_xd2.unsqueeze(-1).repeat(1, 1, Kpp_g.shape[-2]).transpose(1,2)
        Kp = Kpp_x1 + Kpp_x2 - 2 * Kpp_g
        Kp = Kp.transpose(1, 2).reshape(self.b_s, -1, 1).squeeze()
        Kqq_xd1 = torch.diagonal(torch.bmm(e_emb_cur, e_emb_cur.transpose(1, 2)), dim1=-2, dim2=-1)
        Kqq_xd2 = torch.diagonal(torch.bmm(e_emb_nxt, e_emb_nxt.transpose(1, 2)), dim1=-2, dim2=-1)
        Kqq_x1 = Kqq_xd1.unsqueeze(-1).repeat(1, 1, Kqq_g.shape[-1])
        Kqq_x2 = Kqq_xd2.unsqueeze(-1).repeat(1, 1, Kqq_g.shape[-2]).transpose(1,2)
        Kq = Kqq_x1 + Kqq_x2 - 2 * Kqq_g
        Kq = Kq.transpose(1, 2).reshape(self.b_s, -1, 1).squeeze()
        
        Mn = Kp.unsqueeze(dim=1)
        Me = Kq.unsqueeze(dim=1)
        # print(Mn.shape, Me.shape)
        
        
        return (Me+Mn)/2., Me, Kpp_g.squeeze(), Kqq_g.squeeze(), e_emb_cur, e_emb_nxt
    
    def _get_QP_problem(self, Q1_t, num_x1, num_x2):
        m = num_x1 #self.prob_size
        n = num_x2 # self.prob_size
        I = np.identity(m+n)
        E = np.identity(m*n)
        one = np.ones((m*n, 1))
        zero = np.zeros((m*n, 1))

        # b = np.zeros((m, 1), dtype = 'float32')
        self.A = np.zeros(((m+n), m*n), dtype = 'float32')
        self.A1 = np.zeros((n, m*n), dtype = 'float32')
        self.A2 = np.zeros((m, m*n), dtype = 'float32')
        self.G = np.vstack((-E, E)).astype(np.float32)
        self.h = np.vstack((zero, one)).astype(np.float32)
        for i in range(m):
            self.A2[i, (i*n):((i+1)*n)] = 1
            self.A[i, (i*n):((i+1)*n)] = 1
        for j in range(n):
            for i in range(m):
                self.A1[j, (i*n+j)] = 1
                self.A[(m+j), (i*n+j)] = 1
        for i in range(n):
            I[m+i] = -I[m+i]

        # tmp_A = np.zeros((self.b_s, m+n, m*n)).astype(np.float32)
        # tmp_A1 = np.zeros((self.b_s, m, m*n)).astype(np.float32)
        # tmp_A2 = np.zeros((self.b_s, n, m*n)).astype(np.float32)
        # tmp_G = np.zeros((self.b_s, (2*m*n), m*n)).astype(np.float32)
        # tmp_h = np.zeros((self.b_s, (2*m*n), 1)).astype(np.float32)
        # for i in range(self.b_s):
        #     tmp_A[i] = self.A
        #     tmp_A1[i] = self.A1
        #     tmp_A2[i] = self.A2
        #     tmp_G[i] = self.G
        #     tmp_h[i] = self.h
        # self.A, self.A1, self.A2, self.G, self.h = tmp_A[0], tmp_A1[0], tmp_A2[0], tmp_G[0], tmp_h[0]
        # Q1_t, A_t, A1_t, A2_t, G_t, h_t = map(torch.from_numpy, [Q1_t, self.A, self.A1, self.A2, self.G, self.h])
        
        self.x = cp.Variable((m*n, 1), nonneg=True)
        # self.x = torch.Tensor(self.x)
        # # cp.Variable((m*n, 1))
        # self.Q1_cp = cp.Parameter((1, m*n))
        # self.Q2_cp = cp.Parameter((m*n, m*n))
        # self.A_cp = cp.Parameter((m+n, m*n))
        # self.A1_cp = cp.Parameter((m, m*n))
        # self.A2_cp = cp.Parameter((n, m*n))
        # self.G_cp = cp.Parameter((2*m*n, m*n))
        # self.h_cp = cp.Parameter((2*m*n, 1))
        
        obj = cp.Minimize(cp.sum_squares(Q1_t.cpu().T @ self.x)) # + cp.norm(self.b_cp, 1))
        cons = [self.A1 @ self.x <= 2,
                self.A2 @ self.x == 1]
                #self.G @ self.x <= self.h] # self.G_cp @ self.x <= self.h_cp, self.A2_cp @ self.x <= 1, b_cp >= 0, b_cp <= 1,
        self.prob = cp.Problem(obj, cons)
        print('is_dcp: ', self.prob.is_dcp(dpp=True))

    def forward(self, g_data):
        n_emb_cur, e_emb_cur = g_data.x1, g_data.e1
        n_emb_nxt, e_emb_nxt = g_data.x2, g_data.e2
        matching_idx = g_data.matching_idx
        num_x1, num_x2 = g_data.num_x1, g_data.num_x2
        print('shape: ', n_emb_cur.shape, n_emb_nxt.shape, e_emb_cur.shape, e_emb_nxt.shape)
        n_emb_cur, n_emb_nxt, e_emb_cur, e_emb_nxt = self.mpn_layer(n_emb_cur, n_emb_nxt, e_emb_cur, e_emb_nxt, matching_idx, num_x1, num_x2)
        Q1_t, _, Kp, Kq, e_emb_cur, e_emb_nxt = self._get_Q(n_emb_cur, e_emb_cur, n_emb_nxt, e_emb_nxt, num_x1, num_x2) # 换一种方式计算
        # self.qp_layer = CvxpyLayer(self.prob, parameters=[self.Q1_cp, self.A1_cp, self.A2_cp, self.G_cp, self.h_cp], variables=[self.x]) # , self.G_cp, self.h_cp
        # A_t, A1_t, A2_t, G_t, h_t = map(torch.from_numpy, [self.A, self.A1, self.A2, self.G, self.h])
        self._get_QP_problem(Q1_t, num_x1, num_x2)
        # self.prob.solve()
        # self.prob.solve(solver=cp.SCS, gpu= True, use_indirect=True)
        # print(self.x.value)
        x_star = Q1_t# torch.from_numpy(Q1_t)
        # print('shape: ', Kq.shape)
        # Q1_t = Q1_t.requires_grad_()
        # A1_t = A1_t.cuda() 
        # A2_t = A2_t.cuda() 
        # G_t = G_t.cuda() 
        # h_t = h_t.cuda() 
        # x_star = self.qp_layer(Q1_t, A2_t, A1_t, G_t, h_t) # , G_t, h_t
        return x_star, Q1_t, Kp, Kq # x_star[0].squeeze(), Q_t, Kp, Kq



if __name__ == "__main__":
    GMNet()
    print('111')
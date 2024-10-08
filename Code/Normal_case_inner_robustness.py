# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 10:14:21 2022

@author: marko
"""

import numpy as np
from scipy.linalg import block_diag
import cvxpy as cp
from quadprog import solve_qp
import os
from datetime import datetime 
import cvxopt
from cvxopt import matrix, solvers
    
import matplotlib.pyplot as plt
from matplotlib import cm
import tikzplotlib

def leder_obj(A_s, Ag, n, xi, pi):
    
    return 0.5*xi @ A_s.T @ Ag @ A_s @ xi - xi @ A_s.T @ Ag @ n + 0.5*n @ Ag @ n 

def prepare_follower_pseudo_grad(N, P, Q, S1, S2, S3, r1, r2, r3, pi, Ni_list):
    
    for i in range(N):  
            for j in range(N):
                if i == j:
                    current = P[i]   
                else:
                    current = Q[i]*Ni_list[j]  
                if j == 0:   
                    row = current 
                else:
                    row = np.concatenate((row, current),axis=1) 
            if i == 0:
                mat = row
            else: 
                mat = np.concatenate((mat, row), axis=0)
            
    vec = np.concatenate((np.concatenate((r1+S1 @ pi, r2+S2 @ pi), axis=0), r3+S3 @ pi),axis=0)
    
    F1 = mat
    F2 = vec
    
    return F1, F2

def calculate_NE(K, gamma, N, P, Q, S1, S2, S3, r1, r2, r3, A, b, G, h, pi, x_init, Ni_list):
    
    F1, F2 = prepare_follower_pseudo_grad(N, P, Q, S1, S2, S3, r1, r2, r3, pi, Ni_list)
    xi = x_init

    K_l = np.concatenate((np.concatenate((G, A), axis=0), -A), axis=0)
    k_r = np.concatenate((np.concatenate((h,b)), -b))
           
    # qp_C = np.transpose(-K_l_load)
    # qp_b = np.squeeze(-k_r_load)

    qp_C = np.transpose(-K_l)
    qp_b = np.squeeze(-k_r)
    
    list_of_xi = []
    
    for it in range(K):
        
        grad = F1 @ xi + F2
        
        x_hat = xi - gamma*grad
        
        qp_G = np.eye(len(xi))
        qp_a = x_hat
        
        try:
            
            sol,_,_,_,_,_ = solve_qp(qp_G, qp_a, qp_C, qp_b)
            xi = 0.5*xi + 0.5*np.array(sol)
            
        except:

            P2 = np.eye(len(x_hat))
            x = cp.Variable(len(P2))
            q = x_hat
            
            prob = cp.Problem(cp.Minimize(cp.quad_form(x, P2) - 2*q.T @ x),
                          [G @ x <= h,
                          A @ x == b])

            prob.solve()

            xi = 0.5*xi + 0.5*np.squeeze(np.array(x.value))
            
        list_of_xi.append(xi)
        
        if it == 0:
            previous_xi = xi
        else:
            norm_conv = np.linalg.norm(xi - previous_xi)
            previous_xi = xi
        
    
    return xi, np.array(list_of_xi), norm_conv

def calculate_best_response(P, q, G, h, A, b):
    
    K_l = np.concatenate((np.concatenate((G, A), axis=0), -A), axis=0)
    k_r = np.concatenate((np.concatenate((h,b)), -b))
    
    qp_C = np.transpose(-K_l)
    qp_b = np.squeeze(-k_r)
    qp_G = P
    qp_a = -q
    
    sol,_,_,_,_,_ = solve_qp(qp_G, qp_a, qp_C, qp_b)
    xi = np.array(sol)
    
    return xi
    
def active_constraints(xi, G, h):
    
    delta = G @ xi - h

    list_of_active_left = []
    list_of_active_right = []
    
    list_of_inactive_left = []
    list_of_inactive_right = []
    
    for i in range(len(delta)):
        
        if np.abs(delta[i])<0.00000000000001:
                
            list_of_active_left.append(np.squeeze(G[i,:]))
            list_of_active_right.append(h[i])
                
        else:
                
            list_of_inactive_left.append(np.squeeze(G[i,:]))
            list_of_inactive_right.append(h[i])
    
    G_A_comp = np.array(list_of_inactive_left)
    h_A_comp = np.array(list_of_inactive_right)
    
    G_A = np.array(list_of_active_left)
    h_A = np.array(list_of_active_right)
    
    return G_A_comp, h_A_comp, G_A, h_A

def adjust_constraints(G, h, A, b, x):
    
    G_A_comp, h_A_comp, G_A, h_A = active_constraints(x, G, h)
    
    A_new = A
    G_new = G_A_comp
    
    b_new = b
    h_new = h_A_comp
    
    if len(G_A)>0:
        
        A_new = np.concatenate((A_new, G_A), axis=0)
        b_new = np.concatenate((b_new, h_A), axis=0)
        
    if len(A_new)>len(x):
        
        A_new = A_new[:len(x), :]
        b_new = b_new[:len(x)]
        
    return A_new, b_new, G_new, h_new

def calculate_dual_var(P, q, G, h, A, b):

    x = cp.Variable(len(P))
    prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, P) + q.T @ x),
                     [G @ x <= h,
                      A @ x == b])
    
    prob.solve()
    xopt = np.squeeze(np.array(x.value))
    lamb = np.squeeze(np.array(prob.constraints[0].dual_value))
    
    return xopt, lamb 
        
def compute_jacobian(P, S, A_new, b_new, G_new, h_new, xi, lambi):
    
    K = np.concatenate((P, G_new.T), axis=1)
    K = np.concatenate((K, A_new.T), axis=1)
    
    K_row2 = np.concatenate((np.diag(lambi) @ G_new, np.diag(np.squeeze(G_new @ xi - h_new))), axis=1)
    K_row2 = np.concatenate((K_row2, np.zeros((len(G_new), len(A_new)))), axis=1)
    
    K_row3 = np.concatenate((A_new, np.zeros((len(A_new), len(A_new)+len(G_new)))), axis=1)
    
    K = np.concatenate((K, K_row2), axis=0)
    K = np.concatenate((K, K_row3), axis=0)
    
    Right = np.concatenate((-S, np.zeros((len(A_new)+len(G_new), S.shape[1]))), axis=0)
    dx_lam_nu = np.linalg.inv(K) @ Right
    
    jacobian = dx_lam_nu[:len(P),:]

    
    return jacobian 
    
def grid_search(K, gamma, N, P, Q, S1, S2, S3, r1, r2, r3, A, b, G, h, x_init, A_s, Ag, N_des, low_p, up_p, N_p, Ni_list):

    p0 = np.linspace(low_p, up_p, N_p)
    p1 = np.linspace(low_p, up_p, N_p)
    p2 = np.linspace(low_p, up_p, N_p)
    p3 = np.linspace(low_p, up_p, N_p)
    
    iter2 = 0
    
    Cost_matrix = np.zeros((len(p0), len(p1), len(p2), len(p3)))
    
    for i in range(len(p0)):
        
        pi0 = p0[i]
        
        for j in range(len(p1)):
            
            pi1 = p1[j]
            
            for k in range(len(p2)):
                
                pi2 = p2[k]
                
                for l in range(len(p3)):
                    
                    pi3 = p3[l]
                    
                    iter2 += 1
                    
                    if iter2 % 1000 == 0:
                        print('It_g: ', iter2)
                
                    pi = np.array([pi0, pi1, pi2, pi3])
                    
                    xi, _ = calculate_NE(K, gamma, N, P, Q, S1, S2, S3, r1, r2, r3, A, b, G, h, pi, x0, Ni_list)
                
                    cost = leder_obj(A_s, Ag, N_des, xi, pi)
                    
                    Cost_matrix[i,j,k,l] = cost
                
    return Cost_matrix 

def test_for_different_prices(list_of_prices, K, gamma, N, P, Q, S1, S2, S3, r1, r2, r3, A, b, G, h, x0, Ni_list):
    
    list_of_losses = []
    it = 0
    
    for price in list_of_prices:
        
        print('Iteration: '+str(it))
        it += 1
        
        xi, _ = calculate_NE(K, gamma, N, P, Q, S1, S2, S3, r1, r2, r3, A, b, G, h, price, x0, Ni_list)
        
        cost = leder_obj(A_s, Ag, N_des, xi, price)
        list_of_losses.append(cost)
        
    return np.array(list_of_losses)

def test_for_epsilon_surr(price, K, gamma, N, P, Q, S1, S2, S3, r1, r2, r3, A, b, G, h, x0, Ni_list, epsilon_max=0.5):
    
    N_samples_per_eps = 50
    
    N_eps= 10
    
    epsilons = np.linspace(0.0, epsilon_max, N_eps)
    
    list_of_results_for_eps = []
    
    total_it = 0.0
    
    for eps in epsilons:
        
        list_of_results = []
        
        for i in range(N_samples_per_eps):
            
            print('Test it: '+str(total_it))
            total_it += 1
            
            distr = np.random.rand(len(price))
            distr /= np.linalg.norm(distr)
            
            price_eps = price + eps*distr
            
            xi, _ = calculate_NE(K, gamma, N, P, Q, S1, S2, S3, r1, r2, r3, A, b, G, h, price_eps, x0, Ni_list)
            cost = leder_obj(A_s, Ag, N_des, xi, price_eps)
            
            list_of_results.append(cost)
            
        list_of_results_for_eps.append(list_of_results)
        
    list_of_results_for_eps = np.array(list_of_results_for_eps)
    
    array_of_min = np.min(list_of_results_for_eps, axis=1)
    array_of_max = np.max(list_of_results_for_eps, axis=1)
    
    fig = plt.figure(dpi=180)
    ax  = fig.add_subplot()
    
    ax.plot(epsilons, array_of_min, label='min')
    #ax.plot(epsilons, array_of_max, label='max')
    
    ax.set_xlim((epsilons[0], epsilons[-1]))
    ax.legend()
    ax.set_xlabel('epsilon')
    ax.set_ylabel(r'$J_G\left(\pi\right) $')
    ax.grid('on')  
    
    return list_of_results_for_eps

def plot_graphs(N_des, name='11_04_2022_16_14_50_projected_False_armijo_True[3. 2. 3. 2.]'):
    
    folder = os.getcwd()
    folder = folder + '/Results_dummy/' + name
    
    list_of_losses = np.load(folder + '/list_of_losses.npy')
    list_of_xi     = np.load(folder + '/list_of_xi.npy')
    list_of_pi     = np.load(folder + '/list_of_pi.npy')
    
    load_folder = os.getcwd() + '/Parameters_ACC'
    
    N1 = np.load(load_folder + '/N1.npy')
    N2 = np.load(load_folder + '/N2.npy')
    N3 = np.load(load_folder + '/N3.npy')    
    
    k = np.arange(len(list_of_losses))
    
    list_of_colors = ['tab:red', 'tab:orange', 'tab:gray', 'tab:cyan', 'tab:brown', 'tab:pink', 'tab:green', 'tab:olive', 'tab:purple']
    
    fig1, ax1 = plt.subplots(dpi=180)
    
    for i in range(list_of_pi.shape[1]):
        
        desired_value = N_des[i]
        color_ = list_of_colors[i]
        
        ax1.plot(k, N1*np.squeeze(list_of_xi[:, 0+i])+N2*np.squeeze(list_of_xi[:, 4+i])+N3*np.squeeze(list_of_xi[:, 8+i]), color=color_, label='Station '+str(i+1))
        ax1.plot(k, len(k)*[desired_value],  '--', color=color_,)   
        
    ax1.grid('on')
    ax1.legend()
    ax1.set_xlabel("Iteration [k]")
    ax1.set_ylabel("Vehicle accumulation")
    ax1.set_xlim((0, len(k)))
    
    fig1.savefig(folder + "/Accumulation.jpg", dpi=180)
    tikzplotlib.save(folder + "/Accumulation.tex")  
    
    fig2, ax2 = plt.subplots(dpi=180)
    ax2.plot(k, list_of_losses)
    ax2.grid('on')
    ax2.legend()
    ax2.set_xlabel("Iteration [k]")
    ax2.set_ylabel("J_g")
    ax2.set_xlim((0, len(k))) 

    fig2.savefig(folder + "/loss.jpg", dpi=180)
    tikzplotlib.save(folder + "/loss.tex")  
    
    fig3, ax3 = plt.subplots(dpi=180)
    
    for i in range(list_of_pi.shape[1]):
        
        color_ = list_of_colors[i]
        
        ax3.plot(k, list_of_pi[:,i], color=color_, label='Station '+str(i+1))     

    ax3.grid('on')
    ax3.legend()
    ax3.set_xlabel("Iteration [k]")
    ax3.set_ylabel("Price")
    ax3.set_xlim((0, len(k)))    

    fig3.savefig(folder + "/Price.jpg", dpi=180)
    tikzplotlib.save(folder + "/Price.tex") 
    
def prepare_compute_initial_p(N, M, list_of_Ai, list_of_bi, list_of_Gi, list_of_hi, list_of_Pi, list_of_Qi, list_of_ri, list_of_Si, param_rho):
    
    #----- Prepare procedure -----
    
    #----- Make Sigma matrix -----
    
    Suma_mat = np.eye(M)
    for i in range(N-2):  
        Suma_mat = np.hstack((Suma_mat, np.eye(M)))
        
    list_of_theta_delta_i = []
    list_of_M_1_i = []
    list_of_M_2_i = []
    list_of_v_1_i = []
    list_of_v_2_i = []
    list_of_theta_z_i = []
    
    for i in range(N): 
        
        #----- read the values -----
        
        Ai = list_of_Ai[i]
        hi = list_of_hi[i]
        Si = list_of_Si[i]
        ri = list_of_ri[i]
        Pi = list_of_Pi[i]
        Qi = list_of_Qi[i]
        bi = list_of_bi[i]
        Gi = list_of_Gi[i]
        
        #----- define the values -----
        
        len_xi = N*M
        len_pi = M
        len_nii = Ai.shape[0]
        len_delta_i = len(hi)
        len_li = len_xi + len_pi + len_nii + len_delta_i 
        
        Wi = np.hstack((Pi, Qi @ Suma_mat))
        
        Delta_zero_mat = np.zeros((Wi.shape[0], len_delta_i))
        Ki = np.hstack((Wi, Si.T, Ai.T, Delta_zero_mat))
        
        Theta_delta_i = np.hstack((np.zeros((len_delta_i, len_li-len_delta_i)), np.eye(len_delta_i)))

        #----- Make Theta_xi^i matrix -----
        
        for j in range(N):   
            if j == 0:
                if i == 0:
                    Theta_xii = np.eye(M)
                else:
                    Theta_xii = np.zeros((M,M))  
            else:
                if i == j:
                    current = np.eye(M)
                else:
                    current = np.zeros((M,M))    
                Theta_xii = np.hstack((Theta_xii, current))
                
        Theta_xii = np.hstack((Theta_xii, np.zeros((Theta_xii.shape[0], len_li-len_xi))))
        
        p_min = 0.0
        p_max = 5.0
        
        Theta_pi = np.hstack((np.zeros((len_pi, len_xi)), np.eye(len_pi), np.zeros((len_pi, len_li-len_xi-len_pi))))
        A_pi     = np.vstack((np.eye(len_pi), -np.eye(len_pi)))
        b_pi     = np.hstack((np.zeros(len_pi)+p_max, np.zeros(len_pi)-p_min))
        
        M1_i = np.vstack((Ki, Ai @ Theta_xii))
        v1_i = np.hstack((-ri, bi))
        
        M2_i = np.vstack((Gi @ Theta_xii + Theta_delta_i, -Theta_delta_i, A_pi @ Theta_pi))
        eps_const = 0.00000001
        v2_i = np.hstack((hi, np.zeros(len_delta_i)-eps_const, b_pi))
        
        Theta_z_i = np.hstack((np.eye(len_xi+len_pi), np.zeros((len_xi+len_pi, len_nii+len_delta_i))))
        
        #----- Store it in the list for the use -----
        
        list_of_theta_delta_i.append(Theta_delta_i)
        list_of_M_1_i.append(M1_i)
        list_of_M_2_i.append(M2_i) 
        list_of_v_1_i.append(v1_i)
        list_of_v_2_i.append(v2_i)
        list_of_theta_z_i.append(Theta_z_i)
        
    return list_of_theta_delta_i, list_of_M_1_i, list_of_M_2_i, list_of_v_1_i, list_of_v_2_i, list_of_theta_z_i

def compute_initial_p(N, M, list_of_Ai, list_of_bi, list_of_Gi, list_of_hi, list_of_Pi, list_of_Qi, list_of_ri, list_of_Si, param_rho):
    
    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
    list_of_theta_delta_i, list_of_M_1_i, list_of_M_2_i, list_of_v_1_i, list_of_v_2_i, list_of_theta_z_i = prepare_compute_initial_p(N, M, list_of_Ai, list_of_bi, list_of_Gi, list_of_hi, list_of_Pi, list_of_Qi, list_of_ri, list_of_Si, param_rho)
    
    #----- Store values -----
    
    list_of_iterate_xi     = []
    list_of_iterate_pi     = []
    list_of_iterate_ni     = []
    list_of_iterate_deltai = []
    list_of_iterate_yi = []
    list_of_iterate_z = []
    
    for i in range(N):
        
        list_of_iterate_xi.append([])
        list_of_iterate_pi.append([])
        list_of_iterate_ni.append([])
        list_of_iterate_deltai.append([])
        list_of_iterate_yi.append([])
        
    #------------------------
    
    #----- Global variable -----
    
    z = np.array(N*M*[1.0/M] + M*[1.0])
    
    list_of_l_i = []
    
    list_of_y_i = N*[np.zeros(N*M + M)]
    
    N_iterations = 500
    
    for iteration in range(N_iterations):
        
        print("Iteration ", iteration)
        
        for i in range(N):

            Theta_delta_i = list_of_theta_delta_i[i]
            M1_i = list_of_M_1_i[i]
            M2_i = list_of_M_2_i[i]
            v1_i = list_of_v_1_i[i]
            v2_i = list_of_v_2_i[i]
            Theta_z_i = list_of_theta_z_i[i]
            
            yi = list_of_y_i[i]
            
            #----- l_i update step -----
            
            BP_i = param_rho * Theta_z_i.T @ Theta_z_i + 0.000001*np.eye(Theta_z_i.shape[1])
            Bq_i = Theta_z_i.T @ yi - Theta_delta_i.T @ np.ones(Theta_delta_i.shape[0]) - param_rho * Theta_z_i.T @ z
            
            x = cp.Variable(M1_i.shape[1])
    
            prob = cp.Problem(0.5*cp.Minimize(cp.quad_form(x, BP_i) + Bq_i.T @ x),
                              [M2_i @ x <= v2_i,
                               M1_i @ x == v1_i])
            
            MAX_atempts = 5
            success = False
            
            N_attempt = 0
            
            while (not success):
                
                prob.solve()
                if (not (x.value is None)):
                    success = True
                else:
                    N_attempt += 1
                    
                if N_attempt == MAX_atempts:
                    break
                
            len_li = len(np.squeeze(np.array(x.value)))
            
            #----- extract optimized values -----
            
            xi_opt      = np.hstack((np.eye(N*M), np.zeros((N*M, len_li-N*M)))) @ np.squeeze(np.array(x.value))
            pi_opt      = np.hstack((np.zeros((M, N*M)), np.eye(M), np.zeros((M, len_li-N*M-M)))) @ np.squeeze(np.array(x.value))
            ni_opt      = np.hstack((np.zeros((list_of_Ai[i].shape[0], N*M+M)), np.eye(list_of_Ai[i].shape[0]), np.zeros((list_of_Ai[i].shape[0], len_li-list_of_Ai[i].shape[0]-N*M-M)))) @ np.squeeze(np.array(x.value))
            delta_i_opt = Theta_delta_i @ np.squeeze(np.array(x.value))
            
            #----- Log data -----
            
            list_of_iterate_xi[i].append(xi_opt)
            list_of_iterate_pi[i].append(pi_opt)
            list_of_iterate_ni[i].append(ni_opt)
            list_of_iterate_deltai[i].append(delta_i_opt)
            
            #--------------------
            
            if iteration == 0:           
                list_of_l_i.append(np.squeeze(np.array(x.value))) 
                
            else:   
                list_of_l_i[i] = np.squeeze(np.array(x.value))  
            
        #----- z update step -----
        
        z_sum = np.zeros(len(z))
        
        for i in range(N):
            
            z_sum = z_sum + list_of_y_i[i]/param_rho + list_of_theta_z_i[i] @ list_of_l_i[i]
        
        z = z_sum/N
        list_of_iterate_z.append(z)
        
        #----- yi update step -----
        
        for i in range(N):
            
            list_of_y_i[i] = list_of_y_i[i] + param_rho * (list_of_theta_z_i[i] @ list_of_l_i[i] - z)
            list_of_iterate_yi[i].append(list_of_y_i[i])
            
    return list_of_iterate_xi, list_of_iterate_pi, list_of_iterate_ni, list_of_iterate_deltai, list_of_iterate_yi, list_of_iterate_z

def plot_ADMM_convergence(name):
    
    #folder = os.getcwd() + '/Results_dummy/' + name
    folder = name
    
    list_of_iterate_pi = np.load(folder + '/list_of_iterate_pi.npy')
    
    it_p1 = list_of_iterate_pi[0]
    it_p2 = list_of_iterate_pi[1]
    it_p3 = list_of_iterate_pi[2]
    
    M_k = it_p1.shape[0]
    M   = it_p1.shape[1]
    
    k   = np.arange(M_k)
    
    fig1, ax1 = plt.subplots(dpi=180)
    fig2, ax2 = plt.subplots(dpi=180)
    fig3, ax3 = plt.subplots(dpi=180)
    fig4, ax4 = plt.subplots(dpi=180)
    
    p1_12 = np.abs(it_p1[:,0] - it_p2[:,0])
    p1_13 = np.abs(it_p1[:,0] - it_p3[:,0])
    p1_23 = np.abs(it_p2[:,0] - it_p3[:,0])
    
    p2_12 = np.abs(it_p1[:,1] - it_p2[:,1])
    p2_13 = np.abs(it_p1[:,1] - it_p3[:,1])
    p2_23 = np.abs(it_p2[:,1] - it_p3[:,1]) 
    
    p3_12 = np.abs(it_p1[:,2] - it_p2[:,2])
    p3_13 = np.abs(it_p1[:,2] - it_p3[:,2])
    p3_23 = np.abs(it_p2[:,2] - it_p3[:,2])
    
    p4_12 = np.abs(it_p1[:,3] - it_p2[:,3])
    p4_13 = np.abs(it_p1[:,3] - it_p3[:,3])
    p4_23 = np.abs(it_p2[:,3] - it_p3[:,3])
    
    Points = M_k
    
    ax1.plot(k[:Points], p1_12[:Points])
    ax1.plot(k[:Points], p1_13[:Points])
    ax1.plot(k[:Points], p1_23[:Points])
    tikzplotlib.save(folder + "/PriceADMM1.tex") 

    ax2.plot(k[:Points], p2_12[:Points])
    ax2.plot(k[:Points], p2_13[:Points])
    ax2.plot(k[:Points], p2_23[:Points])
    tikzplotlib.save(folder + "/PriceADMM2.tex") 
    
    ax3.plot(k[:Points], p3_12[:Points])
    ax3.plot(k[:Points], p3_13[:Points])
    ax3.plot(k[:Points], p3_23[:Points])
    tikzplotlib.save(folder + "/PriceADMM3.tex")

    ax4.plot(k[:Points], p4_12[:Points])
    ax4.plot(k[:Points], p4_13[:Points])
    ax4.plot(k[:Points], p4_23[:Points])
    tikzplotlib.save(folder + "/PriceADMM4.tex")
    
    plt.show()

if __name__ == '__main__':
    
    load_folder = os.getcwd() + '/Parameters_ACC'
    
    N1 = np.load(load_folder + '/N1.npy')
    N2 = np.load(load_folder + '/N2.npy')
    N3 = np.load(load_folder + '/N3.npy')    
    
    Ni_list = [N1, N2, N3]
    
    G1 = np.load(load_folder + '/G1.npy')
    A1 = np.load(load_folder + '/A1.npy')
    b1 = np.array(np.load(load_folder + '/b1.npy'))
    h1 = np.squeeze(np.load(load_folder + '/h1.npy'))
    K_l1 = np.load(load_folder + '/K_l1.npy')
    k_r1 = np.load(load_folder + '/k_r1.npy')
    
    G2 = np.load(load_folder + '/G2.npy')
    A2 = np.load(load_folder + '/A2.npy')
    b2 = np.array(np.load(load_folder + '/b2.npy'))
    h2 = np.squeeze(np.load(load_folder + '/h2.npy'))
    K_l2 = np.load(load_folder + '/K_l2.npy')
    k_r2 = np.load(load_folder + '/k_r2.npy')
    
    G3 = np.load(load_folder + '/G3.npy')
    A3 = np.load(load_folder + '/A3.npy')
    b3 = np.array(np.load(load_folder + '/b3.npy'))
    h3 = np.squeeze(np.load(load_folder + '/h3.npy'))   
    K_l3 = np.load(load_folder + '/K_l3.npy')
    k_r3 = np.load(load_folder + '/k_r3.npy')
    
    r1 = np.squeeze(np.load(load_folder + '/r1.npy'))
    r2 = np.squeeze(np.load(load_folder + '/r2.npy'))
    r3 = np.squeeze(np.load(load_folder + '/r3.npy'))
    
    S1 = np.load(load_folder + '/S1.npy')
    S2 = np.load(load_folder + '/S2.npy')
    S3 = np.load(load_folder + '/S3.npy')

    G = np.array(block_diag(G1, G2, G3))
    h = np.concatenate((np.concatenate((h1, h2)), h3))
    
    A = np.array(block_diag(A1, A2, A3))
    b = np.concatenate((np.concatenate((b1, b2)), b3))
    
    K_l = np.array(block_diag(K_l1, K_l2, K_l3))
    k_r = np.concatenate((np.concatenate((k_r1, k_r2)), k_r3))
    
    N     = 3
    alpha = 0.000001
    gamma = 0.00001/1000
    
    P1  = np.load(load_folder + '/P1.npy')
    P2  = np.load(load_folder + '/P2.npy')
    P3  = np.load(load_folder + '/P3.npy')                                # They are all the same
    P   = [P1, P2, P3]
    
    Q1  = np.load(load_folder + '/Q1.npy')
    Q2  = np.load(load_folder + '/Q2.npy')
    Q3  = np.load(load_folder + '/Q3.npy')
    Q   = [Q1, Q2, Q3]
    
    A_s = 1.0*np.concatenate((np.concatenate((N1*np.eye(4), N2*np.eye(4)), axis=1), N3*np.eye(4)),axis=1)    
    Ag = 2.5*0.1*np.array([[4.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 3.0, 0], [0.0, 0.0, 0.0, 2.0]])
    
    N_des = np.load(load_folder + '/N_des.npy')
    
    #----- Initial conditions -----
    
    pi = np.array([4.0, 2.0, 3.0, 1.0])  
    
    x0 = np.array(len(pi)*[1.0] + len(pi)*[1.0] + len(pi)*[1.0])/len(pi)
    
    #----- Prepare save -----
    
    current_folder = os.getcwd() + '/Results_dummy'
    
    if not os.path.isdir(current_folder):
        os.makedirs(current_folder)
        
    now = datetime.now()
    date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
    
    name = current_folder + "/" + date_time
    
    #----- Projected p -----
    
    projected = False
    
    p_max = 5.0
    p_min = 0.0
    
    G_pi = np.array([[1.0, 0, 0, 0], [-1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, -1.0]])
    h_pi = np.array([p_max, -p_min, p_max, -p_min, p_max, -p_min, p_max, -p_min])
           
    #----- Projected with armijo -----
    
    projected_with_armijo = True
    
    beta = 0.25
    s_line = alpha
    nu = 0.00001
    
    list_of_counter_fail = []
    list_of_steps        = []
    
    #----- Prepare folder -----
    
    name = name + '_projected_' + str(projected)
    name = name + '_armijo_' + str(projected_with_armijo)
    
    name = name + str(pi)
    
    name_subfolder = date_time + '_projected_' + str(projected) + '_armijo_' + str(projected_with_armijo) + str(pi)

    if not os.path.isdir(name):
        os.makedirs(name)

    #----- List of active constratints -----

    list_A1_len = []
    list_A2_len = []
    list_A3_len = [] 
    
    #----- Complete procedure -----

    K_pi  = 350
    K_list     = [10, 50, 100, 250, 500, 1000, 2500, 5000]
    
    list_of_loss_evolutions = []
    list_of_conv_norm       = []
    
    for K in K_list:
        
        print('K parameter is: ', K)
        
        pi = np.array([4.0, 2.0, 3.0, 1.0]) 
        x0 = np.array(len(pi)*[1.0] + len(pi)*[1.0] + len(pi)*[1.0])/len(pi)
        
        list_of_losses = []
        list_of_xi     = []
        list_of_pi     = []
        list_of_sigma  = []
        list_of_grad   = []
        
        list_of_norm   = []
        
        for iteration in range(K_pi):
            
            print("It: ", iteration)
            
            list_of_pi.append(pi)
            #F1, F2 = prepare_follower_pseudo_grad(N, P, Q, S1, S2, S3, r1, r2, r3, pi, Ni_list)
            
            if not projected_with_armijo or iteration == 0:
                xi, xi_evol, norm_conv = calculate_NE(K, gamma, N, P, Q, S1, S2, S3, r1, r2, r3, A, b, G, h, pi, x0, Ni_list)
                
            list_of_norm.append(norm_conv)
                
            list_of_xi.append(xi)
            cost = leder_obj(A_s, Ag, N_des, xi, pi)
            #list_of_losses.append(cost/1000.0) 
            list_of_losses.append(cost)
            
            list_of_sigma.append(A_s @ xi)
        
            #----- Prepare best response -----
        
            q_br1 = r1 + S1 @ pi + Q1 @ (N2*xi[4:8] + N3*xi[8:])
            q_br2 = r2 + S2 @ pi + Q2 @ (N1*xi[:4]  + N3*xi[8:])
            q_br3 = r3 + S3 @ pi + Q3 @ (N2*xi[4:8] + N1*xi[:4])    
            
            A_new1, b_new1, G_new1, h_new1 = adjust_constraints(G1, h1, A1, b1, xi[:4])
            A_new2, b_new2, G_new2, h_new2 = adjust_constraints(G2, h2, A2, b2, xi[4:8])
            A_new3, b_new3, G_new3, h_new3 = adjust_constraints(G3, h3, A3, b3, xi[8:])
            
            list_A1_len.append(A_new1)
            list_A2_len.append(A_new2)
            list_A3_len.append(A_new3)
            
            xopt1, lamb1 = calculate_dual_var(P1, q_br1, G_new1, h_new1, A_new1, b_new1)
            x_br1 = calculate_best_response(P1, q_br1, G_new1, h_new1, A_new1, b_new1)
            jacob1= compute_jacobian(P1, S1, A_new1, b_new1, G_new1, h_new1, xi[:4], lamb1)
            
            #print(0.5*x_br1 @ P1 @ x_br1 + x_br1 @ q_br1)
            #print(0.5*xi[:4] @ P1 @ xi[:4] + xi[:4] @ q_br1)
        
            xopt2, lamb2 = calculate_dual_var(P2, q_br2, G_new2, h_new2, A_new2, b_new2)
            jacob2 = compute_jacobian(P2, S2, A_new2, b_new2, G_new2, h_new2, xi[4:8], lamb2)    
        
            xopt3, lamb3 = calculate_dual_var(P3, q_br3, G_new3, h_new3, A_new3, b_new3)
            jacob3 = compute_jacobian(P3, S3, A_new3, b_new3, G_new3, h_new3, xi[8:], lamb3) 
            
            #----- Update the leader decision -----
            
            jacobian_full = np.concatenate((np.concatenate((jacob1.T, jacob2.T), axis=1), jacob3.T), axis=1)
            
            dJ_l_dx = A_s.T @ Ag @ A_s @ xi - A_s.T @ Ag @ N_des
            
            dJ_l_dpi = jacobian_full @ dJ_l_dx 
            
            if projected:
                
                pi_hat = pi - alpha*dJ_l_dpi
                P_pi = np.eye(len(pi_hat))
                x_pi = cp.Variable(len(P_pi))
                q_pi = pi_hat
                
                prob = cp.Problem(cp.Minimize(cp.quad_form(x_pi, P_pi) - 2*q_pi.T @ x_pi),
                              [G_pi @ x_pi <= h_pi])
    
                prob.solve()
    
                pi = np.squeeze(np.array(x_pi.value))
                
            elif projected_with_armijo:
                
                found_step = False
                l = 0.0
                
                while not found_step:
                    
                    step = beta**l * s_line 
                    
                    pi_hat = pi - step*dJ_l_dpi
                    P_pi = np.eye(len(pi_hat))
                    x_pi = cp.Variable(len(P_pi))
                    q_pi = pi_hat
                
                    prob = cp.Problem(cp.Minimize(cp.quad_form(x_pi, P_pi) - 2*q_pi.T @ x_pi),
                              [G_pi @ x_pi <= h_pi])
    
                    prob.solve()
    
                    pi_plus = np.squeeze(np.array(x_pi.value)) 
                    xi_plus, _ , norm_conv = calculate_NE(K, gamma, N, P, Q, S1, S2, S3, r1, r2, r3, A, b, G, h, pi_plus, x0, Ni_list)
                    cost_plus = leder_obj(A_s, Ag, N_des, xi_plus, pi_plus)
                    
                    if cost - cost_plus >= nu * dJ_l_dpi @ (pi - pi_plus):
                        
                        found_step = True
                        
                    else:
                        
                        l += 1
                        
                list_of_grad.append(step*dJ_l_dpi)        
                pi = pi_plus
                xi = xi_plus
                
                list_of_steps.append(step)
                list_of_counter_fail.append(l)
                
            else:
                
                pi = pi - alpha*dJ_l_dpi
                                 
        np.save(name+'/list_of_losses_'+str(K)+'.npy', np.array(list_of_losses))
        np.save(name+'/list_of_pi_'+str(K)+'.npy'    , np.array(list_of_pi))
        np.save(name+'/list_of_xi_'+str(K)+'.npy'    , np.array(list_of_xi))
        
        list_of_loss_evolutions.append(list_of_losses)
        list_of_conv_norm.append(list_of_norm)
        
    fig_comp, ax_comp = plt.subplots(dpi=180)
    
    # fig_comp2, ax_comp2 = plt.subplots(dpi=180)
    
    norm_ = plt.Normalize(min(K_list), max(K_list))
    colormap = cm.viridis
    k_ = np.arange(K_pi)
    
    for ik, loss_ik in enumerate(list_of_loss_evolutions):
        
        color = colormap(norm_(K_list[ik]))
        ax_comp.plot(k_, list_of_loss_evolutions[ik], color=color)
        # ax_comp2.plot(k_, list_of_conv_norm[ik], color=color)
        
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm_)
    sm.set_array([])
    fig_comp.colorbar(sm, label='K Value')
    
    ax_comp.set_xlabel('Kpi')
    ax_comp.set_ylabel('Jg')     
    ax_comp.set_xlim((0, len(k_))) 
    
    ax_comp.grid('on')

    fig_comp.savefig(name + "/Compare1.jpg", dpi=180)
    tikzplotlib.save(name + "/Compare1.tex") 

    # ax_comp2.set_xlabel('Kpi')
    # ax_comp2.set_ylabel('norm')     
    # ax_comp2.set_xlim((0, len(k_)))  
    
    # ax_comp2.grid('on')

    # fig_comp2.savefig(name + "/Compare2.jpg", dpi=180)
    # tikzplotlib.save(name + "/Compare2.tex") 
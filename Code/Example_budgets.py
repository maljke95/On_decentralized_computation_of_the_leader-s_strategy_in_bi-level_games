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
    
    for iter in range(K):
        
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
    
    return xi, np.array(list_of_xi)

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
    
    budget_active = False
    
    delta = G @ xi - h

    list_of_active_left = []
    list_of_active_right = []
    
    list_of_inactive_left = []
    list_of_inactive_right = []
    
    for i in range(len(delta)):
        
        if np.abs(delta[i])<0.00000000000001:
                
            list_of_active_left.append(np.squeeze(G[i,:]))
            list_of_active_right.append(h[i])
            
            if i == len(delta)-1:
                budget_active = True
            
        else:
                
            list_of_inactive_left.append(np.squeeze(G[i,:]))
            list_of_inactive_right.append(h[i])
    
    G_A_comp = np.array(list_of_inactive_left)
    h_A_comp = np.array(list_of_inactive_right)
    
    G_A = np.array(list_of_active_left)
    h_A = np.array(list_of_active_right)
    
    return G_A_comp, h_A_comp, G_A, h_A, budget_active

def adjust_constraints(G, h, A, b, x):
    
    G_A_comp, h_A_comp, G_A, h_A, budget_active = active_constraints(x, G, h)
    
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
        
    return A_new, b_new, G_new, h_new, budget_active

def calculate_dual_var(P, q, G, h, A, b):

    x = cp.Variable(len(P))
    prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, P) + q.T @ x),
                     [G @ x <= h,
                      A @ x == b])
    
    prob.solve()
    xopt = np.squeeze(np.array(x.value))
    lamb = np.squeeze(np.array(prob.constraints[0].dual_value))
    
    return xopt, lamb 
        
def compute_jacobian(P, S, A_new, b_new, G_new, h_new, xi, lambi, budget_active):
    
    K = np.concatenate((P, G_new.T), axis=1)
    K = np.concatenate((K, A_new.T), axis=1)
    
    K_row2 = np.concatenate((np.diag(lambi) @ G_new, np.diag(np.squeeze(G_new @ xi - h_new))), axis=1)
    K_row2 = np.concatenate((K_row2, np.zeros((len(G_new), len(A_new)))), axis=1)
    
    K_row3 = np.concatenate((A_new, np.zeros((len(A_new), len(A_new)+len(G_new)))), axis=1)
    
    K = np.concatenate((K, K_row2), axis=0)
    K = np.concatenate((K, K_row3), axis=0)
    
    if budget_active:
        
        Right = np.vstack((-S, np.zeros((len(A_new)+len(G_new)-1, S.shape[1])), xi.T @ S.T))
        
    else:
    
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
    print(len(k))
    
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
    
def plot_budgets(lb1, lb2, lb3, lb_max1, lb_max2, lb_max3, folder):
    
    k = np.arange(len(lb1))
    
    fig1, ax1 = plt.subplots(dpi=180)
    
    ax1.plot(k, lb1)
    ax1.plot(k, len(k)*[lb_max1],  '--')
    
    ax1.grid('on')
    ax1.legend()
    ax1.set_xlabel("Iteration [k]")
    ax1.set_ylabel("Utilized discount budget B1")
    ax1.set_xlim((0, len(k)))
    
    fig1.savefig(folder + "/b1.jpg", dpi=180)
    tikzplotlib.save(folder + "/b1.tex")

    fig2, ax2 = plt.subplots(dpi=180)
    
    ax2.plot(k, lb2)
    ax2.plot(k, len(k)*[lb_max2],  '--')
    
    ax2.grid('on')
    ax2.legend()
    ax2.set_xlabel("Iteration [k]")
    ax2.set_ylabel("Utilized discount budget B2")
    ax2.set_xlim((0, len(k)))
    
    fig2.savefig(folder + "/b2.jpg", dpi=180)
    tikzplotlib.save(folder + "/b2.tex")


    fig3, ax3 = plt.subplots(dpi=180)
    
    ax3.plot(k, lb3)
    ax3.plot(k, len(k)*[lb_max3],  '--')
    
    ax3.grid('on')
    ax3.legend()
    ax3.set_xlabel("Iteration [k]")
    ax3.set_ylabel("Utilized discount budget B3")
    ax3.set_xlim((0, len(k)))
    
    fig3.savefig(folder + "/b3.jpg", dpi=180)
    tikzplotlib.save(folder + "/b3.tex")

    
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
    
    print(np.diag(S1))
    print(np.diag(S2))
    print(np.diag(S3))

    G = np.array(block_diag(G1, G2, G3))
    h = np.concatenate((np.concatenate((h1, h2)), h3))
    
    A = np.array(block_diag(A1, A2, A3))
    b = np.concatenate((np.concatenate((b1, b2)), b3))
    
    K_l = np.array(block_diag(K_l1, K_l2, K_l3))
    k_r = np.concatenate((np.concatenate((k_r1, k_r2)), k_r3))
    
    N     = 3
    K_pi  = 350
    K     = 5000
    alpha = 0.000001
    gamma = 0.00001
    
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
    
    list_of_losses = []
    list_of_xi     = []
    list_of_pi     = []
    list_of_sigma  = []
    list_of_grad   = []
    
    #----- Initial conditions -----

    #pi = np.array([3, 2, 3, 2])
    
    pi = np.array([4.0, 2.0, 3.0, 1.0])
    #pi = np.random.uniform(1.0,5.0, 4)
    
    x0 = np.array(len(pi)*[1.0] + len(pi)*[1.0] + len(pi)*[1.0])/len(pi)
    
    #qp_G = np.eye(len(x0))
    #qp_a = x0
    #qp_C = np.transpose(-K_l)
    #qp_b = np.squeeze(-k_r)
    
    #sol,_,_,_,_,_ = solve_qp(qp_G, qp_a, qp_C, qp_b)
    #x0 = np.array(sol)
    
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
    
    #----- List of active constratints -----
    
    list_A1_len = []
    list_A2_len = []
    list_A3_len = []
    
    list_of_budgets1 = []
    list_of_budgets2 = []
    list_of_budgets3 = []
    
    #----- Base price + budget constraints -----
    
    p_base = np.array([5.0, 3.0, 5.0, 3.0])
    
    B1 = 14000.0
    B2 = 13000.0
    B3 = 12000.0

    if not os.path.isdir(name):
        os.makedirs(name) 
        
    for iteration in range(K_pi):
        
        print("It: ", iteration)
        
        list_of_pi.append(pi)
        #F1, F2 = prepare_follower_pseudo_grad(N, P, Q, S1, S2, S3, r1, r2, r3, pi, Ni_list)
        
        add_c1 = S1 @ (p_base - pi)
        G1_ = np.concatenate((G1, add_c1.reshape(1,-1)),axis=0)
        
        add_c2 = S2 @ (p_base - pi)
        G2_ = np.concatenate((G2, add_c2.reshape(1,-1)),axis=0)
        
        add_c3 = S3 @ (p_base - pi)
        G3_ = np.concatenate((G3, add_c3.reshape(1,-1)),axis=0)
        
        h1_ = np.array(list(h1)+[B1])
        h2_ = np.array(list(h2)+[B2])
        h3_ = np.array(list(h3)+[B3])
        
        G_ = np.array(block_diag(G1_, G2_, G3_))
        h_ = np.concatenate((np.concatenate((h1_, h2_)), h3_))
        
        if not projected_with_armijo or iteration == 0:
            xi, xi_evol = calculate_NE(K, gamma, N, P, Q, S1, S2, S3, r1, r2, r3, A, b, G_, h_, pi, x0, Ni_list)
        
        list_of_xi.append(xi)
        cost = leder_obj(A_s, Ag, N_des, xi, pi)
        list_of_losses.append(cost/1000.0) 
        
        list_of_sigma.append(A_s @ xi)
        
        list_of_budgets1.append(xi[:4]  @ S1 @ (p_base-pi))
        list_of_budgets2.append(xi[4:8] @ S2 @ (p_base-pi))
        list_of_budgets3.append(xi[8:]  @ S3 @ (p_base-pi))
        
        #----- Prepare best response -----
    
        q_br1 = r1 + S1 @ pi + Q1 @ (N2*xi[4:8] + N3*xi[8:])
        q_br2 = r2 + S2 @ pi + Q2 @ (N1*xi[:4]  + N3*xi[8:])
        q_br3 = r3 + S3 @ pi + Q3 @ (N2*xi[4:8] + N1*xi[:4])    
        
        A_new1, b_new1, G_new1, h_new1, budget_active1 = adjust_constraints(G1_, h1_, A1, b1, xi[:4])
        A_new2, b_new2, G_new2, h_new2, budget_active2 = adjust_constraints(G2_, h2_, A2, b2, xi[4:8])
        A_new3, b_new3, G_new3, h_new3, budget_active3 = adjust_constraints(G3_, h3_, A3, b3, xi[8:])
        
        list_A1_len.append(A_new1)
        list_A2_len.append(A_new2)
        list_A3_len.append(A_new3)
        
        xopt1, lamb1 = calculate_dual_var(P1, q_br1, G_new1, h_new1, A_new1, b_new1)
        x_br1 = calculate_best_response(P1, q_br1, G_new1, h_new1, A_new1, b_new1)
        jacob1= compute_jacobian(P1, S1, A_new1, b_new1, G_new1, h_new1, xi[:4], lamb1, budget_active1)
        
        #print(0.5*x_br1 @ P1 @ x_br1 + x_br1 @ q_br1)
        #print(0.5*xi[:4] @ P1 @ xi[:4] + xi[:4] @ q_br1)
    
        xopt2, lamb2 = calculate_dual_var(P2, q_br2, G_new2, h_new2, A_new2, b_new2)
        jacob2 = compute_jacobian(P2, S2, A_new2, b_new2, G_new2, h_new2, xi[4:8], lamb2, budget_active2)    
    
        xopt3, lamb3 = calculate_dual_var(P3, q_br3, G_new3, h_new3, A_new3, b_new3)
        jacob3 = compute_jacobian(P3, S3, A_new3, b_new3, G_new3, h_new3, xi[8:], lamb3, budget_active3) 
        
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
                xi_plus, _ = calculate_NE(K, gamma, N, P, Q, S1, S2, S3, r1, r2, r3, A, b, G, h, pi_plus, x0, Ni_list)
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
                             
    np.save(name+'/list_of_losses.npy', np.array(list_of_losses))
    np.save(name+'/list_of_pi.npy'    , np.array(list_of_pi))
    np.save(name+'/list_of_xi.npy'    , np.array(list_of_xi))
    
    
    plot_graphs(N_des, name_subfolder)
    plot_budgets(list_of_budgets1, list_of_budgets2, list_of_budgets3, B1, B2, B3, name)
    
    #----- Grid serach -----
    
    # N_p = 20
    # cost_m = grid_search(K, gamma, N, P, Q, S1, S2, S3, r1, r2, r3, A, b, G, h, x0, A_s, Ag, N_des, p_min, p_max, N_p, Ni_list)
    
    # np.save(name+'/cost_m.npy', np.array(cost_m))
    
    #----- Different prices -----
    
    # list_of_prices = []
    
    # price = np.array([3.0, 2.0, 2.0, 2.0])
    
    # idx_station = 3
    
    # add_price = np.zeros(len(price))
    # add_price[idx_station] += 1.0
    
    # for i in range(-100,100):
        
    #     list_of_prices.append(price + i*add_price)
        
    # list_of_loss_diff_price=test_for_different_prices(list_of_prices, K, gamma, N, P, Q, S1, S2, S3, r1, r2, r3, A, b, G, h, x0, Ni_list)
    
    #----- test if it is the local min -----
    
    test_local = False
    
    if test_local:
        
        price_test = pi
    
        list_of_results_for_eps = test_for_epsilon_surr(price_test, K, gamma, N, P, Q, S1, S2, S3, r1, r2, r3, A, b, G, h, x0, Ni_list, epsilon_max=0.1)
    
    #plot_graphs(N_des, name='11_04_2022_16_14_50_projected_False_armijo_True[3. 2. 3. 2.]')
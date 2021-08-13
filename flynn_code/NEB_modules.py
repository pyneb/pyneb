import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy import interpolate
from scipy.ndimage import filters, morphology #For minimum finding
import time
from matplotlib.pyplot import cm
import pandas as pd
import h5py
def find_local_minimum(arr):
    """
    Returns the indices corresponding to the local minimum values. Taken 
    directly from https://stackoverflow.com/a/3986876
    
    Parameters
    ----------
    arr : Numpy array
        A D-dimensional array.

    Returns
    -------
    minIndsOut : Tuple of numpy arrays
        D arrays of length k, for k minima found

    """
    neighborhood = morphology.generate_binary_structure(len(arr.shape),1)
    local_min = (filters.minimum_filter(arr, footprint=neighborhood,\
                                        mode="nearest")==arr)
    
    background = (arr==0)
    #Not sure this is necessary - it doesn't seem to do much on the test
        #data I defined.
    eroded_background = morphology.binary_erosion(background,\
                                                  structure=neighborhood,\
                                                  border_value=1)
        
    detected_minima = local_min ^ eroded_background
    allMinInds = np.vstack(local_min.nonzero())
    minIndsOut = tuple([allMinInds[coordIter,:] for \
                        coordIter in range(allMinInds.shape[0])])
    return minIndsOut

def action(path,pot,minimum):
    x_coords = path[:,0]
    y_coords = path[:,1]
    a = 0
    for i in np.arange(0,len(x_coords),1):
        if i== 0:
            pass
        else:
            a += .5*(np.sqrt(2.0*(pot(x_coords[i],y_coords[i]) - minimum)) \
                           + np.sqrt(2.0*(pot(x_coords[i-1],y_coords[i-1])- minimum)))\
                            *np.sqrt(((x_coords[i] - x_coords[i-1]) ** 2 + \
                            (y_coords[i-1] - y_coords[i-1]) ** 2))
    return a

def energy(V,path,minimum):
    total = 0
    for point in path:
        total += V(point[0],point[1]) - minimum
    return(total)
def grad_2d(func,x,y):
    h = 10**(-8)
    ### assumes a 2-dim function f(x,y)
    df_dx = (func(x+h/2,y) - func(x-h/2,y))/h
    df_dy = (func(x,y+h/2) - func(x,y-h/2))/h
    return(df_dx,df_dy)

def eps(V,x,y,m,E_gs):
    result = np.sqrt(2*m*(V(x,y) - E_gs))
    return(result)

### Analytic surfaces
def V_sin(x,y,ax,ay):
    result = ax*np.cos(2.0*np.pi*x) + ay*np.cos(2.0*np.pi*y) 
    return(result)
def Q(r,d,alpha,r0):
    result = 0.5*d*(1.5*np.exp(-2.0*alpha*(r-r0)) - np.exp(-alpha*(r-r0)))
    return(result)
def J(r,d,alpha,r0):
    result = 0.25*d*(np.exp(-2.0*alpha*(r-r0)) - 6.0*np.exp(-alpha*(r-r0)) )
    return(result)
def V_LEPS(rAB,rBC,rAC,a,b,c,dab,dbc,dac,alpha,r0):
    V1 =  Q(rAB,dab,alpha,r0)/(1 + a) + Q(rBC,dbc,alpha,r0)/(1+b) + Q(rAC,dac,alpha,r0)/(1+c)
    V2 = np.sqrt((J(rAB,dab,alpha,r0)/(1+a))**2 + (J(rBC,dbc,alpha,r0)/(1+b))**2 + 
                 (J(rAC,dac,alpha,r0)/(1+c))**2 - J(rAB,dab,alpha,r0)*J(rBC,dbc,alpha,r0)/((1+a)*(1+b)) - 
                 J(rAC,dac,alpha,r0)*J(rBC,dbc,alpha,r0)/((1+b)*(1+c)) 
                 - J(rAB,dab,alpha,r0)*J(rAC,dac,alpha,r0)/((1+a)*(1+c)) )
    V = V1 - V2
    return(V)
def V_HO_LEPS(rAB,x):
    ### Parameters are from Bruce J. Berne, Giovanni Ciccotti,David F. Coker, Classical and Quantum Dynamics in Condensed Phase Simulations Proceedings of the International School of Physics (1998) Chapter 16
    rAC = 3.742
    a = .05
    b = .80
    c = .05
    dab = 4.746
    dbc = 4.746
    dac = 3.445
    r0 = 0.742
    alpha = 1.942
    kc = .2025
    c2 = 1.154
    V1 = V_LEPS(rAB,rAC-rAB,rAC,a,b,c,dab,dbc,dac,alpha,r0)
    V2 = 2.0*kc*(rAB - (0.5*rAC - x/c2))**2
    result = V1 + V2 
    return(result)
def make_time_plot(iters,times,N,M,k,dt,savefig=False):
        plt.plot(iters,times,'.-')
        plt.xlabel('Iterations')
        plt.ylabel('Time (seconds)')
        plt.title('N= '+str(N)+' k= '+str(k)+' dt= '+str(dt)+' Walltime')
        if savefig is not False:
            plt.savefig('Walltime_'+str(M)+'_N_'+str(N)+'_k_'+str(k)+'_dt_'+str(dt)+'.pdf')
        else:pass
        plt.show()
        plt.clf()    
        
class NEB():
    def __init__(self,f,M,N,x_lims,y_lims,grid_size,R0,RN,glb_min):
        self.f = f # potential function (can be analytic or an interpolated functions) CURRENTLY ASSUMES A 2D FUNCTION
        self.N = N # number of images
        self.M = M # max number of iterations
        self.x_lims = x_lims # x-axis upper and lower bounds 
        self.y_lims = y_lims # y-axis upper and lower bounds 
        self.grid_size = grid_size # size of the grid
        self.x = np.linspace(x_lims[0], x_lims[1], grid_size[0]) # define x-axis grid 
        self.y = np.linspace(y_lims[0], y_lims[1], grid_size[1]) # define y-axis grid
        self.xx0,self.yy0 = R0[0],R0[1] #define beginning (x,y) coords
        self.xx1,self.yy1 = RN[0],RN[1] # define ending (x,y) coords
        self.glb_min = glb_min
        self.E_gs = self.shift_V(self.xx0,self.yy0)
    def V(self,x,y):
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            assert len(x) == len(y)
            result = np.zeros(len(x))
            for i in range(len(x)):    
                result[i] = self.f(x[i],y[i]).item()
        else:
            result = self.f(x,y).item()
            
        return(result)
    def shift_V(self,x,y):
        result = self.V(x,y) - self.glb_min 
        return(result)
    def get_end_points(self):
        return(self.shift_V(self.xx0,self.yy0),self.shift_V(self.xx1,self.yy1))
    def get_init_path(self):
        ## returns the initial positions of every point on the chain.
        x_coords = np.linspace(self.xx0,self.xx1,self.N)
        y_coords = np.linspace(self.yy0,self.yy1,self.N)
        path = np.stack((x_coords,y_coords),axis=1)
        return(path)
    def get_tang_vect(self,path):
        #returns array of tangen vectors for each point on the chain. The position of each is given by chain coords. 
        #each row is the ith tangent vector directional information. 
        tan_vects = np.zeros(path.shape)
        for i in np.arange(0,len(path),1):
            if i==0:
                tan = (path[i+1] - path[i])
            else:
                if i==len(path)-1:
                    tan = (path[i] - path[i-1])
                else:
                    Vip1 = self.shift_V(path[i+1][0],path[i+1][1])
                    Vi = self.shift_V(path[i][0],path[i][1])
                    Vim1 =self.shift_V(path[i-1][0],path[i-1][1])
                    if (Vip1 > Vi) and (Vi > Vim1): 
                        tan = path[i+1] - path[i]
                    elif (Vip1 < Vi) and (Vi < Vim1): 
                        tan = path[i] - path[i-1]
                    elif (Vip1 < Vi) and (Vi > Vim1) or (Vip1 > Vi) and (Vi < Vim1): 
                        delta_V_min = min(abs(Vip1 - Vi),abs(Vim1 - Vi))
                        delta_V_max = max(abs(Vip1 - Vi),abs(Vim1 - Vi))
                        if Vip1 > Vim1:
                            tan = (path[i+1] - path[i])*delta_V_max + (path[i] - path[i-1])*delta_V_min
                        else: 
                            tan = (path[i+1] - path[i])*delta_V_min + (path[i] - path[i-1])*delta_V_max
                    else:pass
            norm = np.linalg.norm(tan)
            tan = tan/norm
            tan_vects[i] = tan
        tan_vects = np.array(tan_vects)
        return(tan_vects)
    def F_s(self,k,R,tan_vects):
        #returns 2d-array calculating force at each image.
        # R is an array of the position vectors on the chain. each ith row is assumed to be R_{i}
        force = np.zeros((R.shape[0],R.shape[1]))
        for i in np.arange(0,len(R),1):
            if i==0:
                force[i] = np.zeros((1,2))[0]
            elif i==len(R)-1:
                force[i] = np.zeros((1,2))[0]
            else:
                result = k*(np.linalg.norm(R[i+1] - R[i]) - np.linalg.norm(R[i]  - R[i-1]))*tan_vects[i]
                force[i] = result
        return(force)
    def F_r_finite(self,R,tan,params):
        ## gives the "real" force on each image
        force = np.zeros((R.shape[0],R.shape[1]))
        for i in np.arange(0,len(R),1):
            if i==0:
                force[i] = np.zeros((1,2))[0]
            elif i==len(R)-1:
                force[i] = np.zeros((1,2))[0]
            else:
                grad_Vx,grad_Vy = grad_2d(self.shift_V,R[i][0],R[i][1])
                grad_V = np.array((grad_Vx,grad_Vy))
                result = -grad_V + np.dot(grad_V,tan[i])*tan[i]
                force[i] = result
        return(force)
    def g_perp(self,path,tau,params):
        ## Taken from a talk Calculations of Tunneling Rates using the Line Integral NEB and Acceleration of Path Optimization using
        ## Gaussian Process Regression by Vilhjálmur Ásgeirsson 
        m = params['m']
        E_const = params['E_const']
        k = params['k']
        kappa = params['kappa']
        fix_r0 = params['fix_r0']
        fix_rn = params['fix_rn']
        N_idx = np.arange(0,len(path),1)
        g_perp= np.zeros((len(N_idx),2))
        E = eps(self.shift_V,path[:,0],path[:,1],m,0)
        for i in N_idx:
            if i==0:
                if fix_r0 is not False:
                    g_perp[i] = np.zeros((1,2))[0]
                else:
                    g_spr_0 = 0 #-1.0*np.linalg.norm(path[i+1]  - path[i])*tau[i]
                    f = -1.0*np.array(grad_2d(self.shift_V,path[i][0],path[i][1]))
                    f_norm = np.linalg.norm(f)
                    f_unit = f/f_norm
                    g_perp[i] = (g_spr_0 - (np.dot(g_spr_0,f_unit) - kappa*(self.shift_V(path[i][0],path[i][1]) - E_const))*f_unit)
            elif i==len(N_idx)-1:
                if fix_rn is not False:
                    g_perp[i] = np.zeros((1,2))[0]
                else:
                    g_spr_0 = 0 #-1.0*np.linalg.norm(path[i]  - path[i-1])*tau[i]
                    f = -1.0*np.array(grad_2d(self.shift_V,path[i][0],path[i][1]))
                    f_norm = np.linalg.norm(f)
                    f_unit = f/f_norm
                    g_perp[i] = (g_spr_0 - (np.dot(g_spr_0,f_unit) - kappa*(self.shift_V(path[i][0],path[i][1]) - E_const))*f_unit)
            else:
                f = -1*np.array(grad_2d(self.shift_V,path[i][0],path[i][1]))
                d_i = np.linalg.norm(path[i] - path[i-1])
                d_ip1 = np.linalg.norm(path[i+1] - path[i])
                d_ivec = (path[i] - path[i-1])/d_i
                d_ip1vec = (path[i+1] - path[i])/d_ip1
                g_i =.5*((m/E[i])*(d_i + d_ip1)*f - (E[i] + E[i-1])*d_ivec + (E[i+1] + E[i])*d_ip1vec) 
                g_perp[i] = g_i - np.dot(g_i,tau[i])*tau[i]
        return(g_perp) 
    
    def get_forces(self):
        functions = {
            'MEP': self.F_r_finite,
            'LAP': self.g_perp
            }
        return(functions)
    
    def FIRE(self,init_path,dt,eta,force_params,target='LAP'):
        ### minimize target function using FIRE algo
        ### Initialize the initial path. R0 is the starting point on V and RN is the end point
        action_array = np.zeros((self.M))
        energies = np.zeros((self.M))
        ### Initialize the path array
        path = np.full((self.M,self.N,2),init_path)
        ### Initialize the velocities, masses, and shift arrays for the FIRE Algorithm 
        v = np.full((self.M,self.N,2),np.zeros(init_path.shape))
        vp = np.full((self.M,self.N,2),np.zeros(init_path.shape))
        a = np.full((self.M,self.N,2),np.zeros(init_path.shape))
        mass = np.full(init_path.shape[0],1)
        shift = np.full((self.M,self.N,2),np.zeros(init_path.shape))
        start = time.time()
        ### define force function
        force = self.get_forces()[target]
        k = force_params['k']
        ### FIRE parameters, should maybe be passed in?
        min_fire=10
        dtmax=10.0
        dtmin=0.1
        finc=1.1
        fdec=0.5
        fadec=0.99
        alpha_st=0.1
        alpha=alpha_st
        maxmove=0.2
        fire_steps=0
        #### MAIN KERNEL (FIRE)
        for i in np.arange(0,self.M,1):
            ## calculate the new tangent vectors and forces after each shift.
            tau = self.get_tang_vect(path[i])
            F_spring = self.F_s(k,path[i],tau)
            g = force(path[i],tau,force_params)
            ## note the g for boundary images can contain a spring force. By default F_spring = 0 for boundary images.
            F =  F_spring + g
            for j in np.arange(0,self.N,1):
                if i==0:
                    vp[i][j]= np.zeros(v[i][j].shape)
                elif i==self.M-1:
                        pass
                else:
                    prod = np.dot(F[j],v[i-1][j])
                    if prod > 0:
                        vp[i][j]= (1.0 - alpha)*v[i-1][j]+alpha*np.linalg.norm(v[i-1][j])*F[j]/np.linalg.norm(F[j])
                        if(fire_steps > min_fire):
                            dt = min(dt*finc,dtmax)
                            alpha=alpha*fadec

                        fire_steps+=1
                    else:
                        vp[i][j] = np.zeros(v[i][j].shape)
                        alpha=alpha_st
                        dt=max(dt*fdec,dtmin)
                        fire_steps=0
                    v[i][j] = vp[i][j] + dt*F[j]
                    shift[i][j] = v[i][j]*dt + 0.5*F[j]/mass[j]*dt**2
                    if(np.linalg.norm(shift[i][j])>maxmove):
                        shift[i][j] = maxmove*shift[i][j]/np.linalg.norm(shift[i][j])
                    path[i+1][j] = path[i][j] + shift[i][j]
            action_array[i] = action(path[i],self.V,self.glb_min)
            energies[i] = energy(self.V,path[i],self.glb_min)
        end = time.time()
        total_time = end - start
        return(path[-1],action_array,energies,total_time)

    def QMV(self,init_path,dt,eta,force_params,target='LAP'):
        ### minimize target function using Quick min Verlet algo
        ### This algo seems much more stable than BFGS.
        ### Initialize the initial path. R0 is the starting point on V and RN is the end point
        action_array = np.zeros((self.M))
        energies = np.zeros((self.M))
        ### Initialize the path array
        path = np.full((self.M,self.N,2),init_path)
        ### Initialize the velocities, masses, and shift arrays for the QM Verlet Algorithm 
        v = np.full((self.M,self.N,2),np.zeros(init_path.shape))
        vp = np.full((self.M,self.N,2),np.zeros(init_path.shape))
        a = np.full((self.M,self.N,2),np.zeros(init_path.shape))
        mass = np.full(init_path.shape[0],1)
        shift = np.full((self.M,self.N,2),np.zeros(init_path.shape))
        start = time.time()
        ### define force function
        force = self.get_forces()[target]
        k = force_params['k']
        #### MAIN KERNEL (QM Verlet)
        for i in np.arange(0,self.M,1):
            ## calculate the new tangent vectors and forces after each shift.
            tau = self.get_tang_vect(path[i])
            F_spring = self.F_s(k,path[i],tau)
            g = force(path[i],tau,force_params)
            ## note the g for boundary images can contain a spring force. By default F_spring = 0 for boundary images.
            F =  F_spring + g
            for j in np.arange(0,self.N,1):
                if i==0:
                    vp[i][j]= np.zeros(v[i][j].shape)
                elif i==self.M-1:
                    pass
                else:
                    prod = np.dot(v[i-1][j],F[j])
                    if prod > 0:
                        vp[i][j]= prod*F[j]/np.linalg.norm(F[j])**2
                    else:
                        vp[i][j] = np.zeros(v[i][j].shape)
                    a[i][j] = F[j]/mass[j] - v[i][j]*eta/mass[j]
                    v[i][j] = vp[i][j] + dt*a[i][j] 
                    shift[i][j] = v[i][j]*dt + .5*a[i][j]*dt**2
                    x_img = path[i][j][0] + shift[i][j][0]
                    y_img = path[i][j][1] + shift[i][j][1]
                    
                    ## Add boundary check to make sure images don't get kicked out of bounds
                    if x_img < self.x_lims[0]:
                        path[i+1][j] = np.array([self.x_lims[0],path[i][j][1]]) 
                    elif x_img > self.x_lims[1]:
                        path[i+1][j] = np.array([self.x_lims[1],path[i][j][1]]) 
                    elif y_img < self.y_lims[0]:
                        path[i+1][j] = np.array([path[i][j][0],self.y_lims[0]]) 
                    elif y_img > self.y_lims[1]:
                        path[i+1][j] = np.array([path[i][j][1],self.y_lims[1]])
                    else:
                        path[i+1][j] = path[i][j] + shift[i][j]
            action_array[i] = action(path[i],self.V,self.glb_min)
            energies[i] = energy(self.V,path[i],self.glb_min)
        end = time.time()
        total_time = end - start
        return(path[-1],action_array,energies,total_time)
    
    def backtrack(self,Fi,Fim,alpha_0,gamma,N_0):
        eps = 10**(-3)
        FiRMS = 0.0
        FimRMS = 0.0
        alpha = alpha_0
        for row in Fi:
            FiRMS += np.sqrt(np.mean(row**2))
        for row in Fim:
            FimRMS += np.sqrt(np.mean(row**2))
        chk = (FiRMS - FimRMS)/(abs(FiRMS + FimRMS))
        skip = False
        N_back = N_0
        if chk > eps:
            alpha = alpha_0*gamma
            skip = True
            N_back = N_0
        else:
            N_back = N_back - 1
            if N_back < 0:
                N_back = N_0
                if alpha < alpha_0:
                    alpha = alpha_0
                    skip=True
                else:
                    alpha = alpha/gamma
            else: pass
        return(alpha,skip)
    
    def BFGS(self,init_path,alpha,beta,gamma,s_max,force_params,target='LAP'):
        ### Initialize arrays
        action_array = np.zeros((self.M))
        energies = np.zeros((self.M))
        path = np.full((self.M,self.N,2),init_path)
        F = np.full((self.M,self.N,2),np.zeros(init_path.shape))
        sigma = np.full((self.M,self.N,2),np.zeros(init_path.shape))
        shift = np.full((self.M,self.N,2),np.zeros(init_path.shape))
        y =  np.full((self.M,self.N,2),np.zeros(init_path.shape))
        H = np.full((self.M,self.N,2,2),np.identity(2))
        rho = np.zeros(self.M)
        start = time.time()
        k = force_params['k']
        force = self.get_forces()[target]
        
        #### MAIN KERNEL (BFGS)
        for i in np.arange(0,self.M-1,1):
            tau = self.get_tang_vect(path[i])
            F_spring = self.F_s(k,path[i],tau)
            F_real = self.F_r_finite(path[i],tau)
            g = force(path[i],tau,force_params)

            F[i] =  F_spring + g
            for j in np.arange(0,self.N,1):
                shift[i][j] = np.matmul(H[i][j],F[i][j])
            shift = s_max*shift
            path[i+1] = path[i] + alpha*shift[i]
            path_hold = path[i].copy()
            F_hold = F[i].copy()
            
            tau = self.get_tang_vect(path[i+1])
            F_spring = self.F_s(k,path[i+1],tau)
            g = force(path[i+1],1.0,tau,force_params)
            F[i+1] =  F_spring + g
            
            alpha,skip = self.backtrack(F[i+1],F[i],alpha,gamma,20)
            if skip == True:
                for j in np.arange(0,self.N,1):
                    H[i][j] = np.identity(2)
                path[i] = path_hold
                F[i] = F_hold
                i = i - 1
            else:
                sigma[i] = path[i+1] - path[i]
                y[i] = -F[i+1] + F[i]
                factor = 0
                for j in np.arange(0,self.N,1):
                    factor += np.dot(y[i][j],sigma[i][j])
                rho[i] = factor 
                for j in np.arange(0,self.N,1):
                    if (np.array_equal(H[i][j],np.identity(2))==True):
                        norm = np.dot(y[i][j],y[i][j])
                        if norm != 0:
                            H[i][j] = 1.0/(rho[i]*norm)*np.identity(2)
                        else: pass
                    else:pass
                for j in np.arange(0,self.N,1):
                    A = np.identity(2) - np.outer(sigma[i][j],y[i][j])*rho[i]
                    B = np.identity(2) - np.outer(y[i][j],sigma[i][j])*rho[i]
                    mat_mul1 = np.matmul(H[i][j],B)
                    H[i+1][j] = np.matmul(A,mat_mul1) + np.outer(sigma[i][j],sigma[i][j])*rho[i]
            
            action_array[i] = action(path[i],self.V,self.glb_min)
            energies[i] = energy(self.V,path[i],self.glb_min)
        end = time.time()
        total_time = end - start
        return(path[-1],action_array,energies,total_time)
    def make_convergence_plot(self,diffs,images,iter_steps,k,dt,savefig=False):
        fig, ax = plt.subplots(2,figsize = (12, 10))
        for i,diff in enumerate(diffs):
            ax[0].plot(images,diff[:,0],'.-',label='x '+str(iter_steps[i+1])+'-'+str(iter_steps[i]))
        ax[0].set_title('x',fontsize=14)
        ax[0].legend(bbox_to_anchor=(1, 1), loc='upper left')
        #ax[0].set_xlabel('Image Label',fontsize=13)
        ax[0].set_ylabel(r'$|R_{i}- R_{i-1}|$',fontsize=13)
        ax[0].set_xticks(images)
        for i,diff in enumerate(diffs):
            ax[1].plot(images,diff[:,1],'.-',label='y '+str(iter_steps[i+1])+'-'+str(iter_steps[i]))
        ax[1].set_title('y',fontsize=14)
        ax[1].legend(bbox_to_anchor=(1, 1), loc='upper left')
        ax[1].set_xlabel('Image Label',fontsize=13)
        ax[1].set_ylabel(r'$|R_{i}- R_{i-1}|$',fontsize=13)
        ax[1].set_xticks(images)
        fig.suptitle('N= '+str(self.N)+' k= '+str(k)+' dt= '+str(dt)+' Convergence', fontsize=16)
        if savefig is not False:
            plt.savefig('x-ycon_M_'+str(self.M)+'_N_'+str(self.N)+'_k_'+str(k)+'_dt_'+str(dt)+'.pdf')
        else: pass
        plt.show()
    
def make_cplot(init_paths,paths,grid,zz,params,savefig=False):
    ##plotting function. takes in multiply pathes and init pathes. assumes the init paths and pathes have the same order.
    ## params is a dictionary that should at least contain 'M', 'N', and 'k'.
    color=iter(cm.rainbow(np.linspace(0,1,len(paths))))
    fig, ax = plt.subplots(1,1,figsize = (12, 10))
    im = ax.contourf(grid[0],grid[1],zz,cmap='Spectral_r',levels=MaxNLocator(nbins = 200).tick_values(-2,20))
    ax.contour(grid[0],grid[1],zz,colors=['black'],levels=[params['E_gs']])              
    for init_path in init_paths:
        ax.plot(init_path[:, 0], init_path[:, 1], '.-', color = 'orange',ms=10)
    for path in paths:
        c=next(color)
        ax.plot(path[:, 0], path[:, 1], '.-', color = c,ms=10)
        
    ax.set_ylabel('$Q_{30}$',size=20)
    ax.set_xlabel('$Q_{20}$',size=20)
    ax.set_title('M = '+str(params['M'])+' N = '+str(params['N'])+' k='+str(params['k']))
    cbar = fig.colorbar(im)
    if savefig is not False:
        plt.savefig('Finalpath_M_'+str(params['M'])+'_N_'+str(params['N'])+'_k_'+str(params['k'])+'.pdf')
    else:pass
    plt.show()  
 

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

def extract_gs_inds(allMinInds,coordMeshTuple,zz,pesPerc=0.5):
    #Uses existing indices, in case there's some additional filtering I need to
    #do after calling "find_local_minimum"
    if not isinstance(pesPerc,np.ndarray):
        pesPerc = np.array(len(coordMeshTuple)*[pesPerc])
        
    nPts = zz.shape
    maxInd = np.array(nPts)*pesPerc
    
    allowedIndsOfIndices = np.ones(len(allMinInds[0]),dtype=bool)
    for cIter in range(len(coordMeshTuple)):
        allowedIndsOfIndices = np.logical_and(allowedIndsOfIndices,allMinInds[cIter]<maxInd[cIter])
        
    allowedMinInds = tuple([inds[allowedIndsOfIndices] for inds in allMinInds])
    actualMinIndOfInds = np.argmin(zz[allowedMinInds])
    
    gsInds = tuple([inds[actualMinIndOfInds] for inds in allMinInds])
    
    return gsInds
def find_approximate_contours(coordMeshTuple,zz,eneg=0,show=False):
    nDims = len(coordMeshTuple)
    
    fig, ax = plt.subplots()
    
    if nDims == 1:
        sys.exit("Err: weird edge case I haven't handled. Why are you looking at D=1?")
    elif nDims == 2:
        allContours = np.zeros(1,dtype=object)
        if show:
            cf = ax.contourf(*coordMeshTuple,zz,cmap="Spectral_r")
            plt.colorbar(cf,ax=ax)
        #Select allsegs[0] b/c I'm only finding one level; ccp.allsegs is a
            #list of lists, whose first index is over the levels requested
        allContours[0] = ax.contour(*coordMeshTuple,zz,levels=[eneg]).allsegs[0]
    else:
        allContours = np.zeros(zz.shape[2:],dtype=object)
        possibleInds = np.indices(zz.shape[2:]).reshape((nDims-2,-1)).T
        for ind in possibleInds:
            meshInds = 2*(slice(None),) + tuple(ind)
            localMesh = (coordMeshTuple[0][meshInds],coordMeshTuple[1][meshInds])
            # print(localMesh)
            allContours[tuple(ind)] = \
                ax.contour(*localMesh,zz[meshInds],levels=[eneg]).allsegs[0]
            
    if not show:
        plt.close(fig)
    return allContours

def subspace_2d(data_dict,const_names,const_comps,plane_names):
    # returns a 2d slice of parameter space given fixed coordinates
    ### first convert data into a pandas dataframe. it is easier to work with 
    df = pd.DataFrame(data_dict)
    
    subspace = df.loc[df[const_names[0]]==const_comps[0]]
    for i in np.arange(1,len(const_comps),1):
        subspace = df.loc[df[const_names[i]]==const_comps[i]]
    
    x = subspace[plane_names[0]]
    y = subspace[plane_names[1]]
    V = subspace[plane_names[2]]
    
    df2 = pd.DataFrame({'x':x,'y':y,'z':V})
    x1 = np.sort(df2.x.unique())
    x2 = np.sort(df2.y.unique())
    xx,yy = np.meshgrid(x1,x2)
    zz = pd.DataFrame(None, index = x1, columns = x2, dtype = float)
    
    for i, r in df2.iterrows():
        zz.loc[r.x, r.y] = np.around(r.z,3)
    zz = zz.to_numpy()
    zz = zz.T
    return(xx,yy,zz)
def make_nd_grid(data_dict,coord_keys,energy_key,return_grid=True):
    shape = np.zeros(len(coord_keys),dtype=int)
    if return_grid==True:
        coord_arrays = []
        grids = []
        for i,key in enumerate(coord_keys):
            coord_arrays.append(np.sort(np.unique(data_dict[key])))
            shape[i] = len(coord_arrays[i])     
        for key in coord_keys:
            grids.append(np.array(data_dict[key]).reshape(*shape))
        zz = np.array(data_dict[energy_key]).reshape(*shape)
        return(grids,zz)
    else:
        for i,key in enumerate(coord_keys):
            shape[i]=len(np.unique(data_dict[key]))
        zz = np.array(data_dict[energy_key]).reshape(*shape)
        return(zz)
    
def grad(f,coords,h=10**(-8)):
    if len(coords.shape) == 1:
        coords = coords.reshape((1,-1))
    nPoints,dim = coords.shape
    gradOut = np.zeros((nPoints,dim))
    for i in range(dim):
        ds = np.zeros(coords.shape)
        for j in range(nPoints):
            ds[j][i] =  h*.5
        forward = coords + ds
        backward = coords - ds
        gradOut[:,i] = (f(forward) - f(backward))/h
    ## if there is only 1 coord, return a row vector
    if gradOut.shape[0] == 1:
        gradOut = gradOut[0]
    else:
        pass
    return(gradOut)


def mass_tensor_wrapper(data_dict,nDims,coord_keys,mass_keys,mass_func=None):
    uniq_coords = []
    mass_mins = []
    func_list = []
    l_bnds = np.zeros(len(coord_keys))
    u_bnds = np.zeros(len(coord_keys))
    ## each component of this tensor should contain a function f(\vec{r}) = a where a is real.
    for i,key in enumerate(coord_keys):
        array = np.sort(np.unique(data_dict[key]))
        uniq_coords.append(array)
        l_bnds[i] = min(array)
        u_bnds[i] = max(array) 
    for key in mass_keys:
        mass_mins.append(min(data_dict[key]))
        M_grid = make_nd_grid(data_dict,coord_keys,key,return_grid=False)
        func_list.append(interp_wrapper(uniq_coords,M_grid))
    def mass_tensor(coords):
        if mass_func == None:
            M = np.identity(nDims)
        elif mass_func == True:
            if isinstance(coords,np.ndarray)==True:
                pass
            else:
                coords = np.array(coords)
            M = np.zeros((nDims,nDims)).flatten()
            if len(coords.shape) == 1:
                for i in range(len(coords)):
                    if coords[i] > u_bnds[i]:
                        coords[i] = u_bnds[i]
                    elif coords[i] < l_bnds[i]:
                        coords[i] = l_bnds[i]
                    else: pass
            else:pass
            for i,func in enumerate(func_list):
                M[i] = func(coords)
                M = M.reshape(nDims,nDims)
        return(M)
    return(mass_tensor)
def eps(V,mass_func,coords,E_gs):
    ## M is a nDim x nDim matrix
    ## coords is number of images x nDim matrix
    # V is the potential function
    # E_gs is the energy of the ground state ( aka the constrain energy)
    npts = coords.shape[0]
    nDims = coords.shape[1]
    dx = np.zeros(coords.shape)
    result = np.zeros(coords.shape[0])
    lin_eles = np.zeros(coords.shape[0])
    for i in np.arange(0,npts-1,1):
        pot = V(coords[i])
        dx[i] = coords[i+1] - coords[i]
        lin_ele = np.dot(dx[i],np.dot(mass_func(coords[i]),dx[i]))
        result[i] = np.sqrt(2*(pot - E_gs)*lin_ele)
        lin_eles[i] = lin_ele
    return(result,lin_eles)
   
def action(path,eps):
    a = 0
    ### loop over rows (coords of each image)
    for i in np.arange(0,eps.shape[0],1):
        if i== 0:
            pass
        else:
            a += .5*(eps[i] + eps[i-1])*np.linalg.norm(path[i] - path[i-1])
    return a

### inerpolators 
def coord_interp_wrapper(orig_data,orig_V,l_rb,u_rb):
    def coord_nd_interp(eval_point):
        try:
            result = interpolate.interpn(orig_data,orig_V,eval_point)
        except:
            #print("interpolator failed at: ",eval_point)
            rb = eval_point.copy()
            # check if it is a row vector
            if len(eval_point.shape) == 1:
                for i in range(len(eval_point)):
                    if eval_point[i] > u_rb[i]:
                        rb[i] = u_rb[i]
                    elif eval_point[i] < l_rb[i]:
                        rb[i] = l_rb[i]
                    else: pass
                E = interpolate.interpn(orig_data,orig_V,rb).item()
                result = E*np.exp(np.linalg.norm(eval_point-rb)**(1/2)) + np.linalg.norm(eval_point-rb)*5
            else: # case where we have a n x d matrix of coordinates
                # pick row
                index = []
                for i in range(eval_point.shape[0]):
                    # pick column
                    for j in range(eval_point.shape[1]):
                        if eval_point[i][j] > u_rb[j]:
                            rb[i][j] = u_rb[j]
                            index.append(i)
                        elif eval_point[i][j] < l_rb[j]:
                            rb[i][j] = l_rb[j]
                            index.append(i)
                        else: pass
                E = interpolate.interpn(orig_data,orig_V,rb)
                for k in index:
                    E[k] = E[k]*np.exp(np.linalg.norm(eval_point-rb)**(1/2))+ np.linalg.norm(eval_point-rb)*5
                result = E
        if isinstance(result,np.float64) == True:
            pass
        elif len(result) == 1:
            result = result.item()
        else:pass
        return(result)
    return(coord_nd_interp)
def interp_wrapper(orig_data,orig_V):
    def nd_interp(eval_point):
        result = interpolate.interpn(orig_data,orig_V,eval_point)
        if isinstance(result,np.float64) == True:
            pass
        elif len(result) == 1:
            result = result.item()
        else:pass
        return(result)
    return(nd_interp)

class NEB():
    def __init__(self,f,mass_func,M,N,R0,RN,glb_min,lower_bndy,upper_bndy):
        self.f = f # potential function (can be analytic or an interpolated functions)
        self.mass_func = mass_func ## function that computes the mass tensor at a given \vec{r}
        self.N = N # number of images
        self.M = M # max number of iterations
        self.R0 = R0 # initial band starting point
        self.RN = RN # intial band ending point 
        self.lower_bndy = lower_bndy ### row vector containing upper bounds of each coord
        self.upper_bndy = upper_bndy ### row vector containing lower bounds of each coord
        self.glb_min = glb_min
        self.E_gs_0 = self.shift_V(self.R0)
    def shift_V(self,r):
        result = self.f(r) - self.glb_min 
        return(result)
    def get_end_points(self):
        return(self.shift_V(self.R0),self.shift_V(self.RN))
    def get_init_path(self):
        ## returns the initial positions of every point on the chain.
        coords = []
        for i in range(len(self.R0)):
            xi = np.linspace(self.R0[i],self.RN[i],self.N)
            coords.append(xi)
        #y_coords = np.linspace(self.yy0,self.yy1,self.N)
        path = np.stack(coords,axis=1)
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
                    Vip1 = self.shift_V(path[i+1])
                    Vi = self.shift_V(path[i])
                    Vim1 =self.shift_V(path[i-1])
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
        nDim = R.shape[1]
        force = np.zeros((R.shape[0],R.shape[1]))
        for i in np.arange(0,len(R),1):
            if i==0:
                force[i] = np.zeros((1,nDim))[0]
            elif i==len(R)-1:
                force[i] = np.zeros((1,nDim))[0]
            else:
                result = k*(np.linalg.norm(R[i+1] - R[i]) - np.linalg.norm(R[i]  - R[i-1]))*tan_vects[i]
                force[i] = result
        return(force)
    def F_r_finite(self,R,tan,params):
        ## gives the "real" force on each image
        nDim = R.shape[1]
        force = np.zeros((R.shape[0],R.shape[1]))
        for i in np.arange(0,len(R),1):
            if i==0:
                force[i] = np.zeros((1,nDim))[0]
            elif i==len(R)-1:
                force[i] = np.zeros((1,nDim))[0]
            else:
                grad_V = grad(self.shift_V,R[i])
                grad_V = np.array((grad_V))
                result = -grad_V + np.dot(grad_V,tan[i])*tan[i]
                force[i] = result
        return(force)
    def g_perp(self,path,E,mu,tau,params):
        ## Taken from a talk Calculations of Tunneling Rates using the Line Integral NEB and Acceleration of Path Optimization using
        ## Gaussian Process Regression by Vilhjálmur Ásgeirsson 
        nDim = path.shape[1]
        E_const = params['E_const']
        k = params['k']
        kappa = params['kappa']
        fix_r0 = params['fix_r0']
        fix_rn = params['fix_rn']
        N_idx = np.arange(0,len(path),1)
        g_perp= np.zeros((len(N_idx),nDim))
        E_R0 = E[0]
        E_RN = E[-1]
        delta = .01 ## tolerance for defining when to apply spring force.
        for i in N_idx:
            if i==0:
                if fix_r0 is not False:
                    g_perp[i] = np.zeros((1,nDim))[0]
                else:
                    #g_spr_0 = -1.0*np.linalg.norm(path[i]  - path[i-1])*tau[i]
                    g_spr_0 = 0.0
                    f = -1.0*np.array(grad(self.shift_V,path[i]))
                    f_norm = np.linalg.norm(f)
                    f_unit = f/f_norm
                    g_perp[i] = (g_spr_0 - (np.dot(g_spr_0,f_unit) - kappa*(self.shift_V(path[i]) - E_const))*f_unit)
            elif i==len(N_idx)-1:
                if fix_rn is not False:
                    g_perp[i] = np.zeros((1,nDim))[0]
                else:
                    if E_RN <= E_const+delta:
                        g_spr_0 = -1.0*np.linalg.norm(path[i]  - path[i-1])*tau[i]
                    else:
                        g_spr_0 = 0.0
                    f = -1.0*np.array(grad(self.shift_V,path[i]))
                    f_norm = np.linalg.norm(f)
                    f_unit = f/f_norm
                    g_perp[i] = (g_spr_0 - (np.dot(g_spr_0,f_unit) - kappa*(self.shift_V(path[i]) - E_const))*f_unit)
            else:
                f = -1*np.array(grad(self.shift_V,path[i]))
                d_i = np.linalg.norm(path[i] - path[i-1])
                d_ip1 = np.linalg.norm(path[i+1] - path[i])
                d_ivec = (path[i] - path[i-1])/d_i
                d_ip1vec = (path[i+1] - path[i])/d_ip1
                g_i =.5*((mu[i]/E[i])*(d_i + d_ip1)*f - (E[i] + E[i-1])*d_ivec + (E[i+1] + E[i])*d_ip1vec) 
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
        nDim= init_path.shape[1]
        action_array = np.zeros((self.M))
        energies = np.zeros((self.M))
        ### Initialize the path array
        path = np.full((self.M,self.N,nDim),init_path)
        ### Initialize the velocities, masses, and shift arrays for the FIRE Algorithm 
        v = np.full((self.M,self.N,nDim),np.zeros(init_path.shape))
        vp = np.full((self.M,self.N,nDim),np.zeros(init_path.shape))
        a = np.full((self.M,self.N,nDim),np.zeros(init_path.shape))
        mass = np.full(init_path.shape[0],1)
        shift = np.full((self.M,self.N,nDim),np.zeros(init_path.shape))
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
        delta = 10**(-4) ### convergence threshold for the action defined by abs(action[i] - actions[i-1]) < delta 
        #### MAIN KERNEL (FIRE)
        for i in np.arange(0,self.M,1):
            # calculate the mass comp M_{ij}dx^{i}dx^{j} and eps = \sqrt(2 (V(\vect{r}) - E_gs) M_{ij}dx^{i}dx^{j}) of each image
            var_eps,mu = eps(self.shift_V,self.mass_func,path[i],0.0)
            ## calculate the new tangent vectors and forces after each shift.
            tau = self.get_tang_vect(path[i])
            # calculate the spring/harmonic force on each image
            F_spring = self.F_s(k,path[i],tau)
            # calculate the "real" force of the image
            g = force(path[i],var_eps,mu,tau,force_params)
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
                    '''
                    for k,coord in enumerate(path[i+1][j]):
                        if coord > self.upper_bndy[k]:
                            path[i+1][j][k] = path[i][j][k]
                        elif coord < self.lower_bndy[k]:
                            path[i+1][j][k] = path[i][j][k]
                        else:
                            pass
                    '''
            #print('finished iteration', i)
            action_array[i] = action(path[i],var_eps)
            ## action convergence test
            #if i > 10:
           #     if abs(action_array[i] - action_array[i-1]) < delta:
            #        n = i
            #        break
            #    else:
            #        n = i
            #        pass
        end = time.time()
        total_time = end - start
        return(path[-1],action_array,total_time)

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
        delta = 10**(-4) ### convergence threshold for the action defined by abs(action[i] - actions[i-1]) < delta 
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
                    path[i+1][j] = path[i][j] + shift[i][j]
                    
            action_array[i] = action(path[i],self.shift_V,self.glb_min)
            if i > 10:
                if abs(action_array[i] - action_array[i-1]) < delta:
                    n = i
                    break
                else:
                    n = i
                    pass
        end = time.time()
        total_time = end - start
        return(path[n],action_array[:n],total_time)
    
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
def make_cplot(init_paths,paths,grid,zz,params,savefig=False):
    ##plotting function. takes in multiply pathes and init pathes. assumes the init paths and pathes have the same order.
    ## params is a dictionary that should at least contain 'M', 'N', and 'k'.
    color=iter(cm.rainbow(np.linspace(0,1,len(paths))))
    fig, ax = plt.subplots(1,1,figsize = (12, 10))
    im = ax.contourf(grid[0],grid[1],zz,cmap='Spectral_r',levels=MaxNLocator(nbins = 200).tick_values(-5,9))
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

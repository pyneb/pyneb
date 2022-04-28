import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import time
import sys
### add pyneb
sys.path.insert(0, '../../src/pyneb')
import solvers
import utilities

def linear_path(R0,RN,NImgs):
    ## returns the initial positions of every point on the chain.
    path = np.zeros((NImgs,len(R0)))
    for i in range(len(R0)):
        xi = np.linspace(R0[i],RN[i],NImgs)
        path[:,i] = xi
    return(path)

class LepsPot():
    def __init__(self,params={}):
        defaultParams = {"a":0.05,"b":0.8,"c":0.05,"dab":4.746,"dbc":4.746,\
                         "dac":3.445,"r0":0.742,"alpha":1.942,"rac":3.742,"kc":0.2025,\
                         "c_ho":1.154}
        for key in defaultParams.keys():
            if key in params:
                setattr(self,key,params[key])
            else:
                setattr(self,key,defaultParams[key])
    
    def _q(self,r,d,alpha,r0):
        return d/2*(3/2*np.exp(-2*alpha*(r-r0)) - np.exp(-alpha*(r-r0)))
    
    def _j(self,r,d,alpha,r0):
        return d/4*(np.exp(-2*alpha*(r-r0)) - 6*np.exp(-alpha*(r-r0)))
    
    def leps_pot(self,rab,rbc):
        q = self._q
        j = self._j
                
        rac = rab + rbc
        
        vOut = q(rab,self.dab,self.alpha,self.r0)/(1+self.a) +\
            q(rbc,self.dbc,self.alpha,self.r0)/(1+self.b) +\
            q(rac,self.dac,self.alpha,self.r0)/(1+self.c)
        
        jab = j(rab,self.dab,self.alpha,self.r0)
        jbc = j(rbc,self.dbc,self.alpha,self.r0)
        jac = j(rac,self.dac,self.alpha,self.r0)
        
        jTerm = jab**2/(1+self.a)**2+jbc**2/(1+self.b)**2+jac**2/(1+self.c)**2
        jTerm = jTerm - jab*jbc/((1+self.a)*(1+self.b)) - jbc*jac/((1+self.b)*(1+self.c)) -\
            jab*jac/((1+self.a)*(1+self.c))
        
        vOut = vOut - np.sqrt(jTerm)
        
        return vOut
    
    def __call__(self,coords): #(rab, x)
        """
        
        LEPs potential plus harmonic oscillator.
        Taken from Parameters are from Bruce J. Berne, Giovanni Ciccotti,David F. Coker, \
        Classical and Quantum Dynamics in Condensed Phase Simulations \
        Proceedings of the International School of Physics (1998) Chapter 16 \
        
        Call this function with a numpy array of rab and x:
        
            xx, yy = np.meshgrid(np.arange(0,4,0.01),np.arange(-2,2,0.01))
            zz = leps_plus_ho(xx,yy),
        
        and plot it as
        
            fig, ax = plt.subplots()
            ax.contour(xx,yy,zz,np.arange(-10,70,1),colors="k")
    
        """
        if isinstance(coords,np.ndarray) == False:
            coords = np.array(coords)
        if len(coords.shape) == 1:
            coords = coords.reshape(1,-1) 
        else:pass
            
        if len(coords.shape) >= 3:
            rab = coords[0]
            x = coords[1]
        else:
            rab = coords[:,0]
            x = coords[:,1]
        

        
        vOut = self.leps_pot(rab,self.rac-rab)
        vOut += 2*self.kc*(rab-(self.rac/2-x/self.c_ho))**2
        
        return vOut



#Define potential function
V_func = LepsPot()
M_func = None # for LAP 
auxFunc = None # for MEP

x = np.linspace(.5,3.25,500)
y = np.linspace(-3,3.05,500)
uniq_coords = [x,y]

xx,yy = np.meshgrid(x,y)
grids = [xx,yy]
coords = np.array([xx,yy])
EE_0 = V_func(np.array([xx,yy]))

#Calculate minima of PES
minima_id =  utilities.SurfaceUtils.find_all_local_minimum(EE_0)
minima_coords = np.array((xx[minima_id],yy[minima_id])).T
V_min = V_func(minima_coords)
sorted_ascending_idx = np.argsort(V_min) #places global min first

# redefine minima_coords so that they are in ascending order
minima_coords = minima_coords[sorted_ascending_idx]
V_min = V_func([minima_coords[:,0],minima_coords[:,1]])

#########
gs_coord = minima_coords[0]
E_gs = V_func(gs_coord)
print(f'Ground State Location: {gs_coord}')
print(f'Ground State Energy: {E_gs}')
V_func_shift = utilities.shift_func(V_func,shift=E_gs)#shift by the ground state
EE = V_func_shift(np.array([xx,yy]))



#### Starting NEB

NImgs = 42 # number of images
k = 2.0 # spring constant for entire band
kappa = 1.0 # harmonic force strength (only used if force_RN or force_R0 is True)
E_const = E_gs # energy contour to constrain the end points to
nDims = len(uniq_coords)
force_R0 = False
force_RN = False
springR0 = False
springRN = False
endPointFix = (force_R0,force_RN)
springForceFix = (springR0,springRN)

### Optimization parameters 
## Velocity Verlet parameter set
dt = .01
NIterations = 500


### define initial path
#beginning point 
R0 = [0.74, 1.30]
# end point
RN = [ 3.00, -1.30]

#init_path_constructor = utils.init_NEB_path(R0,RN,NImgs)
#init_path = init_path_constructor.linear_path()
init_path = linear_path(R0,RN,NImgs)
### Define parameter dictionaries (mostly for book keeping)
neb_params ={'k':k,'kappa':kappa,'constraintEneg':E_const}

#### Compute LAP
# LAP function you want to minimize
target_func_LAP = utilities.TargetFunctions.action
# LAP specialized function that takes the gradient of target function
target_func_grad_LAP = utilities.GradientApproximations().discrete_action_grad_const

LAP_params = {'potential':V_func_shift,'nPts':NImgs,'nDims':nDims,'mass':M_func,'endpointSpringForce': springForceFix ,\
                 'endpointHarmonicForce':endPointFix,'target_func':target_func_LAP,\
                 'target_func_grad':target_func_grad_LAP,'nebParams':neb_params}

### define the least action object 
### This essentially defines the forces given the target and gradient functions 
lap = solvers.LeastActionPath(**LAP_params)

### Define the optimizer object to use. Note the initial band is passed
### here and the operations defined in LeastActionPath are applied to
### the band.
minObj_LAP = solvers.VerletMinimization(lap,initialPoints=init_path)

### Begining the optimization procedure. Results are all of the velocities
### band positions, and forces for each iteration of the optimization.
t0 = time.time()
tStepArr, alphaArr, stepsSinceReset = minObj_LAP.fire2(dt,NIterations,useLocal=False)
allPaths_LAP = minObj_LAP.allPts
final_path_LAP = allPaths_LAP[-1]


t1 = time.time()
total_time_LAP = t1 - t0
print(f'Total Time LAP: {total_time_LAP}')
action_array_LAP = np.zeros(NIterations+2)
for i,path in enumerate(allPaths_LAP):
    ## computer action over interpolated path using 500 points
    path_call = utilities.InterpolatedPath(path)
    action_array_LAP[i] = np.around(path_call.compute_along_path(utilities.TargetFunctions.action,500,tfArgs=[V_func_shift])[1][0],3)
min_action_LAP = np.around(action_array_LAP[-1],4)


#### Compute MEP
# MEP function you want to minimize
target_func_MEP = utilities.TargetFunctions.mep_default
# MEP specialized function that takes the gradient of target function
target_func_grad_MEP = utilities.potential_central_grad 
MEP_params = {'potential':V_func,'nPts':NImgs,'nDims':nDims,'auxFunc':auxFunc,'endpointSpringForce': springForceFix ,\
                 'endpointHarmonicForce':endPointFix,'target_func':target_func_MEP,\
                 'target_func_grad':target_func_grad_MEP,'nebParams':neb_params}
t0 = time.time()
mep = solvers.MinimumEnergyPath(**MEP_params)
minObj_MEP = solvers.VerletMinimization(mep,initialPoints=init_path)
tStepArr, alphaArr, stepsSinceReset = minObj_MEP.fire2(dt,NIterations,useLocal=False)
allPaths_MEP = minObj_MEP.allPts

t1 = time.time()
total_time_MEP = t1 - t0
final_path_MEP = allPaths_MEP[-1]
print(f'Total Time MEP: {total_time_MEP}')
### Compute the action of each path in allPts_MEP
action_array_MEP = np.zeros(NIterations+2)
for i,path in enumerate(allPaths_MEP):
    ## computer action over interpolated path using 500 points
    path_call = utilities.InterpolatedPath(path)
    action_array_MEP[i] = np.around(path_call.compute_along_path(utilities.TargetFunctions.action,500,tfArgs=[V_func_shift])[1][0],3)
min_action_MEP =  np.around(action_array_MEP[-1],4)


### Find stationary points on domain using the MEP
maxima,minima,saddle = utilities.get_crit_pnts(V_func, final_path_MEP,method='central')


# Plot the results.
### Make surface contour plot
fig, ax = plt.subplots(1,1)
im = ax.contourf(grids[0],grids[1],EE,cmap='Spectral_r',extend='both',levels=MaxNLocator(nbins = 200).tick_values(0,6))
ax.contour(grids[0],grids[1],EE,colors=['black'],levels=MaxNLocator(nbins = 15).tick_values(0,6))  
ax.plot(init_path[:, 0], init_path[:, 1], '.-', color = 'green',ms=10,label='Initial Path')
ax.plot(final_path_LAP[:, 0], final_path_LAP[:, 1], '.-',ms=10,label='LAP',color='purple')
ax.plot(final_path_MEP[:, 0], final_path_MEP[:, 1], '.-',ms=10,label='MEP',color='red')    
ax.plot(final_path_MEP[:,0][saddle],final_path_MEP[:,1][saddle],'*',color='black',markersize=12)
ax.plot(final_path_MEP[:,0][minima],final_path_MEP[:,1][minima],'X',color='yellow',markersize=12)
ax.set_ylabel('$y$',size=20)
ax.set_xlabel('$x$',size=20)
ax.set_title(f'N = {NImgs}, k={k}, kappa={kappa}')
ax.legend(frameon=True,fancybox=True)
cbar = fig.colorbar(im)
plt.show()  
plt.clf()

### Plot actions
plt.plot(range(NIterations+2),action_array_LAP,label='Final LAP '+str(min_action_LAP))
plt.plot(range(NIterations+2),action_array_MEP,label='Final MEP '+str(min_action_MEP))
plt.xlabel('Iterations')
plt.ylabel('Action')
plt.legend(frameon=True,fancybox=True)
plt.show()
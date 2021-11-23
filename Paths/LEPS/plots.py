import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import time
import sys
from datetime import date
### add pyneb
sys.path.insert(0, '../../py_neb/')
sys.path.insert(0, '../../../flynn_code/py_neb_demos')
import utilities
today = date.today()

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
        Taken from Parameters are from Bruce J. Berne, Giovanni Ciccotti,David F. Coker, 
        Classical and Quantum Dynamics in Condensed Phase Simulations 
        Proceedings of the International School of Physics (1998) Chapter 16
        
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


plt_title = 'LEPSHO_plot_v1.pdf'

### Paths to plot
# './PyNeb_6_camel_back_symm_LAP_path.txt'
LAP_path = np.loadtxt('./PyNeb_LEPSHO_LAP_path.txt',skiprows=1,delimiter=',')
MEP_path = np.loadtxt('./PyNeb_LEPSHO_MEP_path.txt',skiprows=1,delimiter=',')
DP_path = np.loadtxt('./LEPS_DPM_Path.txt',skiprows=1,delimiter=',')
paths = {'LAP': LAP_path,'MEP': MEP_path,'DP': DP_path}
path_names = paths.keys()
action_dict = {}
#Define potential function
V_func = LepsPot()

### Shift Surface by global minimum
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
print('E_gs: ',E_gs)
V_func_shift = utilities.shift_func(V_func,shift=E_gs)#shift by the ground state
EE = V_func_shift(np.array([xx,yy]))



## Interpolate the paths 
for name in path_names:
    path = paths[name]
    path_call = utilities.InterpolatedPath(path)
    action_dict[name] = np.around(path_call.compute_along_path(utilities.TargetFunctions.action,500,tfArgs=[V_func_shift])[1][0],3)

with open('MB_action_values.txt', 'w+') as f:
    for name in path_names:
        f.write(str(name)+': '+str(action_dict[name])+'\n')
for name in action_dict.keys():
    print(name+': ', action_dict[name])

### Plot the results

fig, ax = plt.subplots(1,1,figsize = (8, 6))
im = ax.contourf(grids[0],grids[1],EE,cmap='Spectral_r',extend='both',levels=MaxNLocator(nbins = 200).tick_values(0,6))
ax.contour(grids[0],grids[1],EE,colors=['black'],levels=MaxNLocator(nbins = 15).tick_values(0,6))  
ax.plot(DP_path[:, 0], DP_path[:, 1],label='DPM '+str(action_dict['DP']),linestyle='--',color='black',linewidth=2.0)
ax.plot(LAP_path[:, 0], LAP_path[:, 1],label='LAP '+str(action_dict['LAP']),linestyle='-',color='purple',linewidth=2.0)
ax.plot(MEP_path[:, 0], MEP_path[:, 1],label='MEP '+str(action_dict['MEP']),linestyle='-',color='red',linewidth=2.0)
ax.plot(LAP_path[0][0],LAP_path[0][1],marker='^',color='green',markersize=12)
ax.plot(LAP_path[-1][0],LAP_path[-1][1],marker='^',color='green',markersize=12)
cbar = fig.colorbar(im)
plt.legend(frameon=True,fancybox=True)
plt.xlabel(r'$x$',size=24)
plt.ylabel(r'$y$',size=24)
plt.savefig(plt_title)
plt.show()

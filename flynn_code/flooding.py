import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.ndimage import filters, morphology #For minimum finding
import itertools as it
import types
import time
from scipy.spatial import cKDTree
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
def V_sin(coords):
    if isinstance(coords,np.ndarray) == False:
        coords = np.array(coords)
    
    if len(coords.shape) == 1:
        #print('its a scalar')
        #print(coords)
        coords = coords.reshape(1,-1) 
    else:pass
        
    if len(coords.shape) >= 3:
        #print('its a grid')
        #print(coords)
        x = coords[0]
        y = coords[1]
    else:
        #print('its a vector')
        #print(coords)
        x = coords[:,0]
        y = coords[:,1]
    result = 1*np.cos(2.0*np.pi*x) + 1*np.cos(2.0*np.pi*y) 
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
def V_HO_LEPS(coords):
    ##coords is a list of [x,y]
    #V_HO_LEPS(rAB,x)
    ### Parameters are from Bruce J. Berne, Giovanni Ciccotti,David F. Coker, Classical and Quantum Dynamics in Condensed Phase Simulations Proceedings of the International School of Physics (1998) Chapter 16
    if isinstance(coords,np.ndarray) == False:
        coords = np.array(coords)
    
    if len(coords.shape) == 1:
        #print('its a scalar')
        #print(coords)
        coords = coords.reshape(1,-1) 
    else:pass
        
    if len(coords.shape) >= 3:
        #print('its a grid')
        #print(coords)
        rAB = coords[0]
        x = coords[1]
    else:
        #print('its a vector')
        #print(coords)
        rAB = coords[:,0]
        x = coords[:,1]
    
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
def V_1d(x):
    result = ((x**2 - 1.5)**2 - .5*x)* np.exp(-.18*x**2) 
    return(result)
def V_shift(pot,glb_minimum):
    def V(x):
        result = pot(x) - glb_minimum
        return(result)
    return(V)

'''
def flood(V,waterPnts,x_max,spatialStep,waterStep,upper_bndy,lower_bndy):
    # V should be either a grid or function.
    if isinstance(waterPnts,np.ndarray)==False:
        waterPnts = np.array(waterPnts)
    if len(waterPnts.shape) == 0:
        waterPnts = np.atleast_1d(waterPnts)
        waterPnts = waterPnts.reshape((1,-1))
    elif len(waterPnts.shape) == 1:
        waterPnts = waterPnts.reshape((1,-1))
    else:
        raise ValueError('Coordinates to place water must be np array')
    
    nPts, nDim = waterPnts.shape
    if isinstance(V, types.FunctionType):
        print('V is a function')
    elif isinstance(V, np.ndarray):
        print('V is a grid')
        gridDims = V.shape 
    else:
        raise ValueError('V must be a numpy ndarray or a python function')
    barriers = np.zeros(nPts)
    saddlePnts = np.zeros((nPts,nDim))
    def single_flood(V,x_gs,x_max,dx,dd,upper_bndy,lower_bndy):
        neighbor_shift = list(it.product([0,1,-1], repeat=nDim))[1:]
        neighbor_shift = np.array(neighbor_shift)
        print(neighbor_shift)
        x_neighbors = np.full(neighbor_shift.shape,x_gs)
        ## initialize all the points to the GS
        d = V(x_gs) ## initialize the water height to V(x_gs)
        V_neighbors = V(x_neighbors)  #initialize energy at each neighbor
        shift = 0
        print('x_max ',x_max)
        print('x_neighbors initial \n',x_neighbors)
        print('d initial',d)
        print('V_neighbor initial \n',V_neighbors)
        ### Look for closest local minima and chose that as x_max. 
        ### while any of the neighbor points haven't hit the x_max coord
        ### keep adding water and shifting the neighbors up.
        #print((x_neighbors > x_max).any())
        print(x_neighbors[:,0])
        print(x_max[0])
        print(x_neighbors[:,0] < x_max[0])
        while (x_neighbors[:,0] < x_max[0]).all() == True :
            print(x_neighbors[:,0] < x_max[0])
            print('x_max ',x_max)
            print('x_neighbors \n',x_neighbors)
            print('d',d)
            print('V_neighbors \n',V_neighbors)
            for i,point in enumerate(x_neighbor):
                if (V(point) < d) == True:
            # if there are wet points shift up by dx a bit to check neighbors
                    #shift += dx
                    point_shift = point + neighbor_shift[i]*shift
                # spawn more neighbors
                    p_iNeighbors_shift = np.array(list(it.product([0,1,-1], repeat=nDim))[1:])
                    p_iNeighbors = point + p_iNeighbors_shift*dx
                    np.concatenate(x_neighbor,p_iNeighbors,axis=0)
                for i,point in enumerate(x_shift):
                    for coord_indx,upper in enumerate(upper_bndy):
                        if (point[coord_indx] > upper):
                            pass
                        else:
                            x_neighbors[i] = point
                for i,point in enumerate(x_shift):
                    for coord_indx,lower in enumerate(lower_bndy):
                        if (point[coord_indx] < lower):
                            pass
                        else:
                            x_neighbors[i] = point
                V_neighbors = V(x_neighbors)
            else:
                check = np.isclose(V(x_neighbors),d,atol=.01)
                saddle_ind = check.nonzero()
                #print('saddle_ind ',saddle_ind)
                saddle_pnt = x_neighbors[saddle_ind]
                d += dd
            #time.sleep(.5)
        return(d,saddle_pnt)

    for i,minimum in enumerate(waterPnts):
       bar_height,saddle_loc = single_flood(V,minimum,x_max\
                                            ,spatialStep,waterStep\
                                            ,upper_bndy,lower_bndy)
       print('bar_height ',bar_height)
       print('saddle_loc ',saddle_loc)
       barriers[i] = bar_height
       saddlePnts[i] = saddle_loc
    return(barriers,saddlePnts)
'''
nDim = 2
#### LEPS EXAMPLE
rAB = np.linspace(-np.pi/3, np.pi/3,100) 
x = np.linspace(-np.pi/3, np.pi/3,100) 
#ubndy = np.array([3.5,3])
#lbndy = np.array([0.3,-3])
rrAB,xx = np.meshgrid(rAB,x)
coords = np.array([rrAB,xx])
zz = V_sin(np.array([rrAB,xx]))
upper_bndy = np.array([np.pi/3,np.pi/3])
lower_bndy = np.array([-np.pi/3,-np.pi/3])


minima_id = find_local_minimum(zz)
minima_coords = np.array((rrAB[minima_id],xx[minima_id])).T
V_min = V_sin(minima_coords)
sorted_ascending_idx = np.argsort(V_min) #places global min first
# redine minima_coords
minima_coords = minima_coords[sorted_ascending_idx]
V_min = V_sin([minima_coords[:,0],minima_coords[:,1]])
#########
x_gs = minima_coords[0]
x_max = minima_coords[1]
x_gs = x_gs.reshape(1,-1)
E_gs = V_sin(x_gs)

### initialize 
neighbor_shift = list(it.product([0,1,-1], repeat=nDim))[1:]
neighbor_shift = np.array(neighbor_shift)

x_neighbors = np.full(neighbor_shift.shape,x_gs) # positions of all the nodes
## initialize all the points to the GS
d = V_sin(x_gs) ## initialize the water height to V(x_gs)
V_neighbors = V_sin(x_neighbors)  #initialize energy at each neighbor
shift = .05

#while (x_neighbors[:,0] < x_max[0]).all() == True :
neighbor_shift = x_neighbors +  2*neighbor_shift*shift
#for node in neighbor_shift:
    # create a square around it 
for i,point in enumerate(neighbor_shift):
    for j in range(len(lower_bndy)):
        if point[j] < lower_bndy[j]:
            point[j] = lower_bndy[j] 
            x_neighbors[i][j] =  point[j]
    else:
        x_neighbors[i] = neighbor_shift[i]
for i,point in enumerate(neighbor_shift):
    for j in range(len(upper_bndy)):
        if point[j] > upper_bndy[j]:
            point[j] = upper_bndy[j] 
            x_neighbors[i][j] =  point[j]
    else:
        x_neighbors[i] = neighbor_shift[i]
print(x_neighbors)  
#print(local_min ^ eroded_background)
fig, ax = plt.subplots(1,1,figsize = (12, 10))

im = ax.contourf(rrAB,xx,zz,cmap='Spectral_r',extend='both',levels=MaxNLocator(nbins = 200).tick_values(-1,1))
ax.contour(rrAB,xx,zz,colors=['black'],levels=MaxNLocator(nbins = 10).tick_values(-1,1))  

plt.plot(x_gs[:,0],x_gs[:,1],'o',color='orange',ms='12')
plt.plot(x_max[0],x_max[1],'o',color='red',ms='12')
plt.plot(x_neighbors[:,0],x_neighbors[:,1],'o',color='black',ms='10')
#plt.plot(x_0,barrier,'x',color='black')
cbar = fig.colorbar(im)
plt.show()  


'''
spatialStep = .01
waterStep = .01
#V,waterPnts,glb_min,spatialStep,waterStep
barrier,x_0 = flood(V_HO_LEPS,x_gs,x_max,spatialStep,waterStep,ubndy,lbndy)

fig, ax = plt.subplots(1,1,figsize = (12, 10))

im = ax.contourf(rrAB,xx,zz,cmap='Spectral_r',extend='both',levels=MaxNLocator(nbins = 200).tick_values(-5,5))
ax.contour(rrAB,xx,zz,colors=['black'],levels=MaxNLocator(nbins = 50).tick_values(-2,15))  
for i in range(0,len(minima_coords),1):
    plt.plot(minima_coords[i][0],minima_coords[i][1],'o',color='black',ms='8')
plt.plot(x_0[0][0],x_0[0][1],'x',color='black',ms='12')
cbar = fig.colorbar(im)
plt.show()  
plt.clf()


#### 1 DIM EXAMPLE
x = np.linspace(-2,2,100)
ubndy = np.array([2])
lbndy = np.array([-2])
y = V_1d(x)

## first find minima
minima_id = find_local_minimum(y)
print(minima_id[0])
minima_coords = np.array((x[minima_id[0]],y[minima_id[0]])).T

V_min= V_1d(minima_coords[:,0])
sorted_ascending_idx = np.argsort(V_min)#places global min first
# redine minima_coords
minima_coords = minima_coords[sorted_ascending_idx]
V_min = V_1d(minima_coords[:,0])

#########
x_gs = minima_coords[0][0]
x_max = minima_coords[1][0]
E_gs = V_1d(x_gs)

spatialStep = .0001
waterStep = .0001
#V,waterPnts,glb_min,spatialStep,waterStep
barrier,x_0 = flood(V_1d,x_gs,x_max,spatialStep,waterStep,ubndy,lbndy)
#barrier,x_0 = flood(V,x_gs,x_max,glb_minimum,spatialStep = .0001,waterStep=.0001)


plt.plot(x,y)
for i in range(0,len(minima_coords),1):
    plt.plot(minima_coords[i][0],minima_coords[i][1],'x',color='black')
plt.plot(x_0,barrier,'x',color='black')
plt.hlines(barrier,-2,2,color='red')
plt.xlabel('x')
plt.ylabel('V')
plt.show()
'''
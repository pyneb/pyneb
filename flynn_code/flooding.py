import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import filters, morphology #For minimum finding
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

def V(x,c):
    result = ((x**2 - c)**2 - .5*x)* np.exp(-.18*x**2) 
    return(result)

def flood(V,c,x_gs,glb_minimum,x_max,dx,dd):
    d = V(x_gs,c) - glb_minimum
    shift = 0
    n = 0 
    x_r = x_gs+shift
    x_l = x_gs-shift
    V_r = V(x_r,c) -glb_minimum
    V_l  = V(x_l,c) - glb_minimum
    while x_r <= x_max and x_l <= x_max:
        # check if there are any "wet points"
        if d >= V_r or d >= V_l:
        # if there are wet points shift up by dx a bit to check neighbors
            n += 1
            shift += dx
        # recalculate the right and left "neighboring" points
            x_r = x_gs+shift
            x_l = x_gs-shift
            V_r = V(x_r,c) - glb_minimum
            V_l  = V(x_l,c)- glb_minimum
        else:
            saddle_pnt = x_r
            d += dd
    return(d,saddle_pnt)
def V_shift(V,x,c,glb_minimum):
    result = V(x,c) - glb_minimum
    return(result)

c = 1.5
x = np.linspace(-2,2,100)
y = V(x,c)
## first find minima
minima_id = find_local_minimum(y)
minima_coords = np.array((x[minima_id[0]],y[minima_id[0]])).T# Nx2 array with (x,y) coords in each row
glb_minima_idx = np.argmin(V(minima_coords[:,0],c))
glb_minimum = V(minima_coords[:,0][glb_minima_idx],c)

#shift PES by global minimum
y = V_shift(V,x,c,glb_minimum)
minima_id = find_local_minimum(y)
minima_coords = np.array((x[minima_id[0]],y[minima_id[0]])).T
x_gs = minima_coords[:,0][0]
E_gs = V_shift(V,x_gs,c,glb_minimum) 


x_max = minima_coords[:,0][1]

barrier,x_0 = flood(V,c,x_gs,glb_minimum,x_max,dx = .0001,dd=.0001)

print(barrier,x_0)

plt.plot(x,y)
for i in range(0,len(minima_coords),1):
    plt.plot(minima_coords[i][0],minima_coords[i][1],'x',color='black')
plt.plot(x_0,barrier,'o')
plt.hlines(barrier,-2,2,color='red')
plt.xlabel('x')
plt.ylabel('V')
plt.show()

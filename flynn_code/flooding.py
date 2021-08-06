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
    result = (x**2 - c)**2
    return(result)

def flood(V,c,gs_coord,dx,dd):
    d = 2.246865 #V(gs_coord,c)
    shift = 0
    n = 0 
    end = False
    while end == False:
        r = V(gs_coord + shift,c)
        l  = V(gs_coord - shift,c)
        if d >= r or d >= l:    
            n += 1
            shift += dx
        else:
            print('done')
            end = True
            coord_r = gs_coord + shift - dx
            coord_l = gs_coord - shift + dx
    return(d,coord_r,coord_l)
c = 1.5
x = np.linspace(-2,2,100)
y = V(x,c)
minima_id = find_local_minimum(y)
minima_coords = np.array((x[minima_id[0]],y[minima_id[0]])).T# Nx2 array with (x,y) coords in each row
d,shift_r,shift_l = flood(V,c,minima_coords[0][0],dx = .0001,dd=.01)

plt.plot(x,y)
for i in range(0,len(minima_coords),1):
    plt.plot(minima_coords[i][0],minima_coords[i][1],'x',color='black')
plt.plot(shift_r,d,'o')
plt.plot(shift_l,d,'o')
plt.hlines(d,-2,2,color='red')
plt.show()

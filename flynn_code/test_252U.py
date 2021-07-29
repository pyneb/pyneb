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
def get_endpoints(s):
    """
        returnd end points of surface: 
        end : minimum energy point 
        start: the other local minimum furthest from it
    """
    local_minima = np.array(detect_local_minima(s))
    minima_vals = s[local_minima[0, :], local_minima[1, :]]
    order = np.argsort(minima_vals)
    ordered_minima = local_minima[:, order]

#     print(local_minima)
#     print(minima_vals)

    m1 = tuple(ordered_minima[:, 0])
    _m1 = np.tile(ordered_minima[:, 0][:, None], ordered_minima.shape[1])
    distances = ((ordered_minima - _m1) ** 2).sum(axis = 0)
    m2 = tuple(ordered_minima[:, distances.argmax()])
    if m1[0] < m2[0]:
        start = m1
        end = m2
    else:
        start = m2
        end = m1
#     print(start, end)
    return start, end

data_path = '../252U_Test_Case/252U_PES.h5'   
data = h5py.File(data_path, 'r')
Q20_grid = np.array(data['Q20'])
Q30_grid = np.array(data['Q30'])
V_grid = np.array(data['PES'])


### interpolate the grid
print('starting interpolation')
#tck = interpolate.bisplrep(Q20_grid,Q30_grid,V_grid,kx=2, ky=2)
#zz_bspline = interpolate.bisplev(Q20_grid[:,1],Q30_grid[0],tck) 
f = interpolate.RectBivariateSpline(Q20_grid[:,1], Q30_grid[0], V_grid, kx=4, ky=4, s=0)
#f = interpolate.Rbf(Q20_grid, Q30_grid, V_grid,smoothing=0, kernel='linear') 
# define new grid

Q20 = np.linspace(Q20_grid[:,0][0],Q20_grid[:,0][-1],500)
Q30 = np.linspace(Q30_grid[0][0],Q30_grid[0][-1],500)
xx, yy = np.meshgrid(Q20,Q30)
zz = f(Q20,Q30)
minima = find_local_minimum(V_grid)
minima_interp = find_local_minimum(zz.T)
#print(f(xx,yy[minima]))


fig, ax = plt.subplots()
im = ax.contourf(xx, yy, zz.T, cmap='Spectral_r',levels=MaxNLocator(nbins = 200).tick_values(-2,20))
ccp = ax.contour(xx, yy, zz.T, levels = [0])
cbar = fig.colorbar(im)

ax.plot(Q20_grid[minima],Q30_grid[minima],'o',color='black')
ax.plot(xx[minima_interp],yy[minima_interp],'x',color='black')
plt.show()
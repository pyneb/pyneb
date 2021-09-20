import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy import interpolate
from scipy.ndimage import filters, morphology #For minimum finding
import pandas as pd
import itertools
import h5py
import sys
import warnings
#sys.path.insert(0, '../../py_neb')
import py_neb_temp

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
        
        gsInds = tuple([inds[actualMinIndOfInds] for inds in allowedMinInds])
        
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
            if show:
                plt.show(fig)
                
        if not show:
            plt.close(fig)
        
        return allContours  
class PES():
    '''
    Class imports hdf5 and starts PES instance. functions contained in this class 
    offer utilites for checking data shapes, gets the domain boundary values, unique
    coordinates, mesh grids, and subspace slices.
    '''
    def __init__(self,file_path):
        data = h5py.File(file_path,'r')
        ### section organizes keys 
        self.keys = list(data.keys())
        self.mass_keys = [i for i in self.keys if i.startswith('B')]
        self.multi_pole_keys = [i for i in self.keys if i.startswith('Q')]
        self.other_poss_coord_keys = ['pairing']
        self.other_coord_keys = [coord for key in self.keys for coord in self.other_poss_coord_keys if key==coord]# other coordinate keys
        self.possible_energy_key = ['PES','EHFB','E_HFB'] #possible energy keys
        self.energy_key = [energy for key in self.keys for energy in self.possible_energy_key if key==energy]
        self.coord_keys = self.multi_pole_keys + self.other_coord_keys
        wanted_keys = self.coord_keys + self.energy_key + self.mass_keys
        self.data_dict = {}
        for key in wanted_keys:
            self.data_dict[key] = np.array(data[key])
    def get_keys(self,choice='coords'):
        string_out = f'Coordinates: {self.coord_keys} \nMass Components: {self.mass_keys} \nEnergy Key: {self.energy_key}'
        print(string_out)
        if choice == 'coords':
            return(self.coord_keys)
        elif choice == 'mass':
            return(self.mass_keys)
        else:
            raise ValueError('choice must be mass or coords')
            return()
    def get_data_shapes(self):
        shape_dict = {}
        for key in self.data_dict.keys():
            print(self.data_dict[key].shape)
            shape_dict[key] = self.data_dict[key].shape
        return(shape_dict)
    def get_unique(self,return_type='array'):
        uniq_coords = {}
        if return_type =='array':
            uniq_coords = [np.sort(np.unique(self.data_dict[key])) for key in self.coord_keys]
        elif return_type =='dict':
            for key in self.coord_keys:
                uniq_coords[key] = np.sort(np.unique(self.data_dict[key]))
        else:
            raise ValueError('Return types can only be array and dict')
        return(uniq_coords)
    def get_grids(self,return_coord_grid=True,shift_GS=True,ignore_val=-1760):
        uniq_coords = self.get_unique(return_type='dict')
        if return_coord_grid==True:
            grids = []
            coord_arrays = [uniq_coords[key] for key in uniq_coords.keys()]
            shape = [len(coord_arrays[i]) for i in range(len(uniq_coords.keys()))]
            grids = [self.data_dict[key].reshape(*shape) for key in uniq_coords.keys()]
            zz = self.data_dict[self.energy_key[0]].reshape(*shape)
            if shift_GS==True:
                minima_ind = py_neb_temp.find_local_minimum(zz)
                ### ignore the fill values. Stolen from Dan's Code
                allowedInds = tuple(np.array([inds[zz[minima_ind]!=ignore_val] for inds in minima_ind]))
                gs_ind = extract_gs_inds(allowedInds,grids,zz,pesPerc=0.25)
                E_gs_shift = zz[gs_ind] 
                EE = zz - E_gs_shift
                E_gs = EE[gs_ind]
                gs_coord = np.array([grids[i][gs_ind] for i in range(len(grids))])
                return(grids,zz,E_gs,gs_coord)
            else:
                return(grids,zz)
        else:
            shape = [len(uniq_coords[key]) for key in uniq_coords.keys()]
            zz = self.data_dict[self.energy_key].reshape(*shape)
            if shift_GS==True:
                minima_ind = py_neb_temp.find_local_minimum(zz)
                ### ignore the fill values. Stolen from Dan's Code
                allowedInds = tuple(np.array([inds[zz[minima_ind]!=ignore_val] for inds in minima_ind]))
                gs_ind = py_neb_temp.extract_gs_inds(allowedInds,grids,zz,pesPerc=0.25)
                E_gs_shift = zz[gs_ind] 
                EE = zz - E_gs_shift
                gs_coord = np.array([grids[i][gs_ind] for i in range(len(grids))])
            else: pass
            return(zz)
    def get_mass_grids(self):
        # returns the grids for each comp. of the tensor as a flattend array
        # ex) tensor (B2020,B2030 \n B2030,B3030) will be represented as 
        # [B2020,B2030,B2030,B3030] where each index contains a grid.
        uniq_coords = self.get_unique(return_type='dict')
        grids = []
        coord_arrays = [uniq_coords[key] for key in uniq_coords.keys()]
        shape = [len(coord_arrays[i]) for i in range(len(uniq_coords.keys()))]
        grids = {key:self.data_dict[key].reshape(*shape) for key in self.mass_keys}
        return(grids)
    def get_2dsubspace(self,restrict_dict,plane):
        # returns a 2d slice of parameter space given fixed coordinates
        ### first convert data into a pandas dataframe. it is easier to work with 
        df = pd.DataFrame(self.data_dict)
        const_names = restrict_dict.keys()
        for key in const_names:
            subspace = df.loc[df[key]==restrict_dict[key]]
        x = subspace[plane[0]]
        y = subspace[plane[1]]
        V = subspace[self.energy_key[0]]
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
    def get_boundaries(self):
        uniq_coords = self.get_unique(return_type='dict')
        keys = uniq_coords.keys()
        nDims = len(keys)
        l_bndy = np.zeros(nDims)
        u_bndy = np.zeros(nDims)
        for i,key in enumerate(keys):
            l_bndy[i] = min(uniq_coords[key])
            u_bndy[i] = max(uniq_coords[key])
        return(l_bndy,u_bndy)





class init_NEB_path:
    def __init__(self,R0,RN,NImgs):
        self.R0 = R0
        self.RN = RN
        self.NImgs = NImgs
        if isinstance(R0,np.ndarray)==False:
            R0 = np.array(R0)
        if isinstance(RN,np.ndarray)==False:
            RN = np.array(RN)
        if len(R0.shape) != 1 or len(R0.shape) != 1 :
            raise ValueError('R0 or RN are not 1-d row vectors')
        
    def linear_path(self):
            ## returns the initial positions of every point on the chain.
            path = np.zeros((self.NImgs,len(self.R0)))
            for i in range(len(self.R0)):
                xi = np.linspace(self.R0[i],self.RN[i],self.NImgs)
                path[:,i] = xi
            return(path)
        
class PES_plot:
    def __int__(self):
        return
    def make_single(self):
        return
    def make_tile_plot(self):
        return
    
    
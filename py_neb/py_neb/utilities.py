

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
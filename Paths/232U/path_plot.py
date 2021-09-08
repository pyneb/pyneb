import sys
sys.path.append("..//..//Daniels_Code/")

from NEB_Class import *

pesFile = "..//..//PES/232U.h5"
dsets, attrs = FileIO.new_read_from_h5(pesFile)

coords = ["Q20","Q30"]

gridShape = [len(np.unique(dsets[c])) for c in coords]

(xx, yy) = [dsets[c].reshape(gridShape) for c in coords]
zz = dsets["PES"].reshape(gridShape)

fig, ax = Utilities.standard_pes(xx,yy,zz,clipRange=(-10,30))
ax.contour(xx,yy,zz,levels=[0],colors=["black"])

pathFiles = ['232U_Const_Mass.txt', 'PathPabloU232.csv', '232U_Var_Mass.txt']

for (pIter,p) in enumerate(pathFiles):
    pathIn = FileIO.read_path(p)
    if pathIn.shape[0] < 50:
        marker = "."
    else:
        marker = None
    label = pathFiles[pIter].replace(".txt","").replace(".csv","")
    ax.plot(pathIn[:,0],pathIn[:,1],marker=marker,label=label)
    
ax.legend(loc="upper left")
ax.set(title="232U PES")
fig.savefig("232U.pdf")
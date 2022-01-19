import sys, os
from scipy import interpolate

pyNebDir = os.path.expanduser("~/Research/ActionMinimization/py_neb/")

if pyNebDir not in sys.path:
    sys.path.insert(0,pyNebDir)
    
from py_neb import *

fName = "..//232U_original.dat"
with open(fName) as fIn:
    headerLine = fIn.readline()
header = headerLine.lstrip("#").split()

df = pd.read_csv(fName,delimiter="\s+",names=header,skiprows=1)
df = df.rename(columns={"BE":"PES","M22":"B2020","M32":"B2030","M33":"B3030",\
                        "E0":"E_ZPE"})

q20UnVals = np.unique(df["Q20"])
q30UnVals = np.unique(df["Q30"])

expectedMesh = np.meshgrid(q20UnVals,q30UnVals)
expectedFlat = np.array([[q2,q3] for q2 in q20UnVals for q3 in q30UnVals])

newDf = pd.DataFrame(data=expectedFlat,columns=["Q20","Q30"])

newDf = newDf.merge(df,on=["Q20","Q30"],how="outer")
newDf["is_interp"] = newDf["PES"].isna()

idxToInterp = newDf[newDf["is_interp"]==True].index
ptsToInterp = np.array(newDf[["Q20","Q30"]].iloc[idxToInterp])

interpCols = ["PES","B2020","B2030","B3030"]
for head in interpCols:    
    interp_func = interpolate.RBFInterpolator(np.array(df[["Q20","Q30"]]),df[head])
    newDf[head].iloc[idxToInterp] = interp_func(ptsToInterp)
    
zz = np.array(newDf["PES"]).reshape(expectedMesh[0].shape,order="F")
gsLoc = SurfaceUtils.find_local_minimum(zz)
eGS = zz[gsLoc]

newDf["PES"] -= eGS

h5File = h5py.File("..//232U_original.h5","w")
h5File.attrs.create("DFT","SKMs")
h5File.attrs.create("Ground_State",gsLoc)
h5File.attrs.create("Original_File","232U_original.dat")

h5File.create_group("interpolation")
h5File["interpolation"].attrs.create("method","scipy.interpolate.RBFInterpolator")

for col in newDf.columns:
    resArr = np.array(newDf[col]).reshape(expectedMesh[0].shape,order="F")
    h5File.create_dataset(col,data=resArr.flatten())

h5File.close()

# fig, ax = plt.subplots()
# ax.contourf(*expectedMesh,(zz-eGS).clip(-5,30),cmap="Spectral_r",levels=75)

# ax.scatter(*[q[gsLoc] for q in expectedMesh],marker="*",s=200,color="red")


# # h5File = h5py.File("..//PES/232U.h5","w")

# # for (hIter, head) in enumerate(datDict.keys()):
# #     h5File.create_dataset(head,data=datDict[head])
    
# # h5File.attrs.create("Original_File","232U_PES_SKMs.dat")
# # h5File.attrs.create("DFT","SKMs")

# # h5File.attrs.create("Ground_State",gsArr)

# # h5File.close()

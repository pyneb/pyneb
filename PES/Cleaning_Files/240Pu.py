from NEB_Class import *

with open("240Pu.dat") as fIn:
    allDat = fIn.readlines()

heads = allDat[2].replace("\n","").split(" ")
while "" in heads:
    heads.remove("")

allDat = allDat[3:]

isDummyArr = np.zeros(len(allDat),dtype=bool)
fillArr = np.array([-1760.,1,1,1,1,1,1,1,1,1])

formattedDat = np.zeros((len(allDat),len(heads)))
for (lIter,line) in enumerate(allDat):
    locLine = line.replace("\n","").split(" ")
    while "" in locLine:
        locLine.remove("")
    formattedDat[lIter] = np.array(locLine).astype(float)
    if np.array_equal(formattedDat[lIter,3:],fillArr):
        isDummyArr[lIter] = True
    
h5File = h5py.File("240Pu.h5","w")

for (hIter,head) in enumerate(heads):
    h5File.create_dataset(head,data=formattedDat[:,hIter])
    
h5File.create_dataset("Is_Dummy",data=isDummyArr)
    
h5File.close()
import numpy as np

######
# Note: Since in practice all the datasets had only one stratum, we implemented the code for one stratum
#####

def createIs(file_path,num_classes):

    #Matrix with indices for positive literals
    Iplus = []
    #Matrix with indeces for negative literals
    Iminus = []
    with open(file_path, 'r') as f:
        for line in f:
            split_line = line.split()
            assert split_line[2] == ':-'
            iplus = np.zeros(num_classes)
            iminus = np.zeros(num_classes)
            for item in split_line[3:]:
                if 'n' in item:
                    index = int(item[1:])
                    iminus[index] = 1
                else:
                    index = int(item)
                    iplus[index] = 1
            Iplus.append(iplus)
            Iminus.append(iminus)
    Iplus = np.array(Iplus)
    Iminus = np.array(Iminus)
    return Iplus, Iminus

def createM(file_path,num_classes):
    M = []
    with open(file_path, 'r') as f:
        for line in f:
            split_line = line.split()
            assert split_line[2] == ':-'
            m = np.zeros(num_classes)
            index = int(split_line[1])
            m[index] = 1
            M.append(m)
    M = np.array(M).transpose()
    return M

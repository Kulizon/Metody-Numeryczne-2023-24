import numpy as np
import matplotlib as plt

N = 5
x = [0 for i in range(N)]
mat = [[0 for i in range(N)] for j in range(N)] 

for i in range(N):
    x[i] = i+1

    mat[i][i] = 1.2
    if (i+1 < N):
        mat[i+1][i] = 0.2 # set 0.2 under diag
        mat[i][i+1] = (0.1)/(i+1) # set 0.1/(i+1) under diag
    if (i+2 < N):
        mat[i][i+2] = (0.15)/(i+1)**2 # set 0.15/(i+1)**2 under diag



print(np.matrix(mat))
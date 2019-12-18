
import numpy as np
import matplotlib.pyplot as plt


def moving_average(data, window_size=100): #used this approach https://stackoverflow.com/questions/11352047/finding-moving-average-from-data-points-in-python
    cumsum_vec = np.cumsum(np.insert(data, 0, 0)) 
    ma_vec = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size
    return ma_vec

    
file1 = np.load("lava2_0.npz")
file2 = np.load("lava2_1.npz")
file3 = np.load("lava2_2.npz")


plt.figure()
#plt.plot(file1[file1.files[0]], file1[file1.files[1]])
plt.plot(file1[file1.files[0]][99:]/2, moving_average(file1[file1.files[1]]))
plt.plot(file2[file2.files[0]][99:]/2, moving_average(file2[file2.files[1]]))
plt.plot(file3[file3.files[0]][99:]/2, moving_average(file3[file3.files[1]])) #/2 rectifies a bug
plt.xlabel("Steps")
plt.ylabel("Return")
plt.savefig('outA_.png')

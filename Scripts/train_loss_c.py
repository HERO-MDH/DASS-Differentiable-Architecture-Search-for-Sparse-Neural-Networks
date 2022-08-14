import numpy as np 
from numpy import genfromtxt
import matplotlib.pyplot as plt
my_data = genfromtxt('train_loss.csv', delimiter=',')
mydata0 =my_data[:,0]
# # print(mydata0)
# mydata1 = my_data[:,-2]
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth
for i in range(200):
    print(smooth(mydata0,20)[i])    
plt.plot(range(200), smooth(mydata0,20))
plt.show()
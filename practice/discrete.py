import numpy as np
import matplotlib.pyplot as plt

def x(n):
    y=np.where((n>=0) & (n<=3),2.0**n,0)
    return y

n_arr=np.arange(-10,10)
y=x(n_arr)
# y_new=x(-(n_arr-2))

# fig,axs=plt.subplots(1,2,figsize=(14,10))

# ax1=axs[0]
# ax1.stem(n_arr,y,linefmt='b-')

# ax2=axs[1]
# ax2.stem(n_arr,y_new,linefmt='g-')

# plt.tight_layout()
# plt.show()


y_sample=y[y!=0]

sq=np.abs(y_sample*y_sample)
print(sq)
energy=np.sum(sq)
power=energy/len(y_sample)

print(energy)
print(power)







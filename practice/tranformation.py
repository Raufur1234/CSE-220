import matplotlib.pyplot as plt
import numpy as np


def y_signal(t_arr):
    conditions=[
        (t_arr>=0) & (t_arr<2),
        (t_arr>=2) & (t_arr<=4)
    ]
    choices=[
        t_arr,2
    ]
    y=np.select(conditions,choices,default=0)
    return y

t=np.linspace(-10,10,1000,endpoint=True)
y=y_signal(t)
y_rev=y_signal(-t)
y_e=0.5*(y+y_rev)
y_o=0.5*(y-y_rev)
#y_trans=y_signal(-3*t+2)

fig,axs=plt.subplots(1,3,figsize=(14,10))
ax1=axs[0]
ax1.plot(t,y,'b-',linewidth=5)
ax1.set_title('x(t)')
ax1.grid(True)



ax2=axs[1]
ax2.plot(t,y_e,'g-',linewidth=5)
ax2.set_title('x(t)(even)')
ax2.grid(True)


ax3=axs[2]
ax3.plot(t,y_o,'r-',linewidth=5)
ax3.set_title('x(t)(odd)')
ax3.grid(True)

# ax2=axs[1]
# ax2.plot(t,y_trans,'g-',linewidth=5)
# ax2.set_title('x(-3t+2)')
# ax2.grid(True)

plt.tight_layout()
plt.show()
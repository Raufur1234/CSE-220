import numpy as np
import matplotlib.pyplot as plt

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
north = np.array([120, 135, 128, 142, 150, 158, 162, 170, 168, 180, 185, 195])
south = np.array([110, 118, 125, 130, 138, 145, 150, 155, 160, 168, 172, 180])
central = np.array([100, 108, 115, 120, 130, 140, 148, 152, 158, 165, 170, 175])


fig,axs= plt.subplots(2,2,figsize=(14,10))

ax1=axs[0,0]
ax1.plot(months,north,color='b',linestyle='-',label='North')
ax1.plot(months,south,color='g',linestyle='--',label='South')
ax1.plot(months,central,color='r',linestyle='dotted',label='Central')
ax1.set_xlabel('I am gay')
ax1.set_ylabel('ibdshd')
ax1.legend()
ax1.grid(True)

ax2=axs[0,1]
branches=['North','South','Central']
profits=[north[-1],south[-1],central[-1]]
ax2.bar(branches,profits,color=['b','g','r'])
ax2.set_xlabel('Branch')
ax2.set_ylabel('Profit (in thousand dollars)')
ax2.set_title('December Profit Comparison')
ax2.grid(axis='y')


ax3=axs[1,0]
ax3.scatter(months,north,color='b',s=60)
ax3.set_title('wow')
ax3.set_xlabel('dsds')
ax3.set_ylabel('dsdsd')


quarters=['Q1','Q2','Q3','Q4']

north_q=north.reshape(4,3).sum(axis=1)
south_q=south.reshape(4,3).sum(axis=1)
central_q=central.reshape(4,3).sum(axis=1)


ax4=axs[1,1]
ax4.bar(quarters,north_q, color='blue',label='North')
ax4.bar(quarters,south_q, color='green',bottom=north_q,label='South')
ax4.bar(quarters,central_q, color='red',bottom=north_q+south_q,label='Central')


plt.tight_layout()
plt.show()
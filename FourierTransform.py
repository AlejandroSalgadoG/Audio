import math
import numpy as np
import matplotlib.pyplot as plt

from Auxiliar import read_data

signal = read_data( "Records/a.wav" )
signal_idx = np.arange( signal.size ) 

fs = 44032
cut = 80000
x = signal[cut:cut+fs]
x_idx = signal_idx[cut:cut+fs]

plt.plot(signal_idx, signal)
plt.plot(x_idx, x)
plt.show()

n = x.size
dt = 1/n # inter sample time
p = n*dt # period
df = 1/p # frequency resolution
#print(df)

t = np.arange(0,p,dt) # time
nyquist = math.floor( t.size/2 ) + 1
coef = np.fft.fft( x )[:nyquist] / n

fig, (ax1, ax2) = plt.subplots(2)

ax1.scatter( np.arange(coef.size), abs(coef)*2, s=7 )
ax2.scatter( np.arange(coef.size), np.angle(coef), s=7 )

ax1.set_ylabel("Amplitude")
ax2.set_ylabel("Phase")
ax2.set_xlabel("Frequency")
plt.show()

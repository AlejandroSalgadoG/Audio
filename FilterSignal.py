import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from Auxiliar import read_data, reproduce_data, cos_wave, difference, get_magnitude_phase

signal = read_data( "Records/b1.wav" )
signal_idx = np.arange( signal.size ) 

fs = 44032
cut = 44032
x = signal[cut:cut+fs]
x_idx = signal_idx[cut:cut+fs]

#plt.plot(signal_idx, signal)
#plt.plot(x_idx, x)
#plt.show()

n = x.size
dt = 1/n # inter sample time
p = n*dt # period
df = 1/p # frequency resolution
#print(df)

t = np.arange(0,p,dt) # time
nyquist = math.floor( t.size/2 ) + 1
coef = np.fft.fft( x )[:nyquist] / n

t_f = np.arange(nyquist)
amps, phas = get_magnitude_phase( coef )

fig, (ax1, ax2) = plt.subplots(2)
ax1.scatter( t_f, amps, s=7 )
ax2.scatter( t_f, phas, s=7 )
plt.show()

d_amps = difference( amps )
mean, std = d_amps.mean(), d_amps.std()
noise_idx = (d_amps >= mean-std) & (d_amps <= mean+std)
coef[ np.append(False, noise_idx) ] = 0

amps, phas = get_magnitude_phase( coef )
good_idx, _ = find_peaks( amps )
bad_idx = np.ones( nyquist, dtype=np.bool )
bad_idx[ good_idx ] = False
coef[ bad_idx ] = 0

amps, phas = get_magnitude_phase( coef )
fig, (ax1, ax2) = plt.subplots(2)
ax1.scatter( t_f, amps, s=7 )
ax2.scatter( t_f, phas, s=7 )
plt.show()

waves = np.array([ cos_wave( f, amps[f], phas[f], t ) for f, c in enumerate(coef) if c != 0 ])
waves[0] = waves[0] / 2
x_p = np.sum( waves, axis=0, dtype=np.int16 )

plt.plot(x)
plt.plot(x_p)
plt.show()

for i in range(24):
    reproduce_data( x )
    reproduce_data( x_p )

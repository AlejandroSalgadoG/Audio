import math
import numpy as np
import matplotlib.pyplot as plt

from Auxiliar import read_data, reproduce_data, get_magnitude_phase, reconstruct_signal

signal = read_data( "Records/a.wav" )
signal_idx = np.arange( signal.size ) 

fs = 44032
cut = 80000
x = signal[cut:cut+fs]
x_idx = signal_idx[cut:cut+fs]

n = x.size
dt = 1/n # inter sample time
p = n*dt # period
df = 1/p # frequency resolution
#print(df)

t = np.arange(0,p,dt) # time
nyquist = math.floor( t.size/2 ) + 1
fourier_coef = np.fft.fft( x )
amplitudes, angles = get_magnitude_phase( fourier_coef / n )

# filter signal
fourier_coef[ 2100: ] = 0

fig, (ax1, ax2) = plt.subplots(2)

w = np.arange(nyquist)
new_amplitudes, new_angles = get_magnitude_phase( fourier_coef[:nyquist] / n )
ax1.scatter( w, new_amplitudes, s=7 )
ax2.scatter( w, new_angles, s=7 )

ax1.set_ylabel("Amplitude")
ax2.set_ylabel("Phase")
ax2.set_xlabel("Frequency")
plt.show()

filtered_signal = reconstruct_signal( fourier_coef )

plt.plot( signal_idx, signal )
plt.plot( x_idx, filtered_signal )
plt.show()

reproduce_data( x )
reproduce_data( filtered_signal * 2 )

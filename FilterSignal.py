import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from Auxiliar import read_data, reproduce_data, cos_wave, difference, apply_fourier, get_magnitude_phase

signal = read_data( "Records/c1_down.wav" )
signal_idx = np.arange( signal.size ) 

fs = 44032
cut = 44032
x = signal[cut:cut+fs]
x_idx = signal_idx[cut:cut+fs]

#plt.plot(signal_idx, signal)
#plt.plot(x_idx, x)
#plt.show()

n, dt, p, df, nyquist, t, coef = apply_fourier( x )
coef = coef[:nyquist] / n
amps, phas = get_magnitude_phase( coef )
t_f = np.arange(nyquist)

d_amps = difference( amps )
mean, std = d_amps.mean(), d_amps.std()
noise_idx = (d_amps >= mean-std) & (d_amps <= mean+std)
coef[ np.append(False, noise_idx) ] = 0
new_amps, new_phas = get_magnitude_phase( coef )

good_idx, _ = find_peaks( new_amps )
bad_idx = np.ones( nyquist, dtype=np.bool )
bad_idx[ good_idx ] = False
coef[ bad_idx ] = 0
new_amps, new_phas = get_magnitude_phase( coef )

plt.plot( t_f, amps )
plt.plot( t_f, new_amps )
plt.show()

waves = np.array([ cos_wave( f, new_amps[f], new_phas[f], t ) for f, c in enumerate(coef) if c != 0 ])
waves[0] = waves[0] / 2
x_p = np.sum( waves, axis=0, dtype=np.int16 )

plt.plot(x)
plt.plot(x_p)
plt.show()

for i in range(5):
    reproduce_data( x, fs=fs )
    reproduce_data( x_p, fs=fs )

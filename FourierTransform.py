import math
import numpy as np
import matplotlib.pyplot as plt

from Auxiliar import read_data, apply_fourier, get_magnitude_phase

signal = read_data( "Records/c1.wav" )
signal_idx = np.arange( signal.size ) 

fs = 44032
cut = 44032
x = signal[cut:cut+fs]
x_idx = signal_idx[cut:cut+fs]

plt.plot(signal_idx, signal)
plt.plot(x_idx, x)
plt.show()

n, dt, p, df, nyquist, t, coef = apply_fourier( x )
amps, phas = get_magnitude_phase( coef[:nyquist] / n )

plt.plot( np.arange(nyquist), amps )
plt.show()

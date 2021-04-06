import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from Auxiliar import read_data, reproduce_data, cos_wave, apply_fourier, get_magnitude_phase

data_files = ["a_1.wav", "a_2.wav", "a_3.wav", "a_4.wav", "a_5.wav"]
#data_files = ["e_1.wav", "e_2.wav", "e_3.wav", "e_4.wav", "e_5.wav"]
#data_files = ["i_1.wav", "i_2.wav", "i_3.wav", "i_4.wav", "i_5.wav"]
#data_files = ["o_1.wav", "o_2.wav", "o_3.wav", "o_4.wav", "o_5.wav"]
#data_files = ["u_1.wav", "u_2.wav", "u_3.wav", "u_4.wav", "u_5.wav"]

#data_files = ["c1.wav", "c2.wav", "c3.wav", "c4.wav", "c5.wav"]
#data_files = ["cs1.wav", "cs2.wav", "cs3.wav", "cs4.wav"]
#data_files = ["d1.wav", "d2.wav", "d3.wav", "d4.wav"]
#data_files = ["ds1.wav", "ds2.wav", "ds3.wav", "ds4.wav"]
#data_files = ["e1.wav", "e2.wav", "e3.wav", "e4.wav"]
#data_files = ["f1.wav", "f2.wav", "f3.wav", "f4.wav"]
#data_files = ["fs1.wav", "fs2.wav", "fs3.wav", "fs4.wav"]
#data_files = ["g1.wav", "g2.wav", "g3.wav", "g4.wav"]
#data_files = ["gs1.wav", "gs2.wav", "gs3.wav", "gs4.wav"]
#data_files = ["a1.wav", "a2.wav", "a3.wav", "a4.wav"]
#data_files = ["as1.wav", "as2.wav", "as3.wav", "as4.wav"]
#data_files = ["b1.wav", "b2.wav", "b3.wav", "b4.wav"]

#data_files = ["c1.wav", "d1.wav", "e1.wav", "f1.wav", "g1.wav", "a1.wav", "b1.wav"]
#data_files = ["c2.wav", "d2.wav", "e2.wav", "f2.wav", "g2.wav", "a2.wav", "b2.wav"]
#data_files = ["c3.wav", "d3.wav", "e3.wav", "f3.wav", "g3.wav", "a3.wav", "b3.wav"]
#data_files = ["c4.wav", "d4.wav", "e4.wav", "f4.wav", "g4.wav", "a4.wav", "b4.wav"]
#data_files = ["c1.wav", "cs1.wav", "d1.wav", "ds1.wav", "e1.wav", "f1.wav", "fs1.wav", "g1.wav", "gs1.wav", "a1.wav", "as1.wav", "b1.wav"]
#data_files = ["c2.wav", "cs2.wav", "d2.wav", "ds2.wav", "e2.wav", "f2.wav", "fs2.wav", "g2.wav", "gs2.wav", "a2.wav", "as2.wav", "b2.wav"]
#data_files = ["c3.wav", "cs3.wav", "d3.wav", "ds3.wav", "e3.wav", "f3.wav", "fs3.wav", "g3.wav", "gs3.wav", "a3.wav", "as3.wav", "b3.wav"]
#data_files = ["c4.wav", "cs4.wav", "d4.wav", "ds4.wav", "e4.wav", "f4.wav", "fs4.wav", "g4.wav", "gs4.wav", "a4.wav", "as4.wav", "b4.wav"]

fs = 44032
cut = 44032
f_cut = 8000

fig, ax = plt.subplots( len(data_files) )

for idx, data_file in enumerate(data_files):
    signal = read_data( "Records/%s" % data_file )
    signal_idx = np.arange( signal.size ) 

    x = signal[cut:cut+fs]
    x_idx = signal_idx[cut:cut+fs]

    n, dt, p, df, nyquist, t, coef = apply_fourier( x )
    amps, phas = get_magnitude_phase( coef[:nyquist] / n )

    ax[idx].plot( np.arange(f_cut), amps[:f_cut] )
    ax[idx].legend( [data_file] )
    
plt.show()

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from Auxiliar import read_data, reproduce_data, cos_wave, get_magnitude_phase

data_files = ["c1.wav", "c2.wav", "c3.wav", "c4.wav", "c5.wav"]
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

    n = x.size
    dt = 1/n # inter sample time
    p = n*dt # period
    df = 1/p # frequency resolution
    t = np.arange(0,p,dt) # time
    nyquist = math.floor( t.size/2 ) + 1
    coef = np.fft.fft( x )[:nyquist] / n

    amps, phas = get_magnitude_phase( coef )
    ax[idx].scatter( np.arange(f_cut), amps[:f_cut], s=7 )
    ax[idx].legend( [data_file] )
    
plt.show()

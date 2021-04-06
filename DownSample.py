import math
import numpy as np
import matplotlib.pyplot as plt

from Auxiliar import read_data, apply_fourier, get_magnitude_phase, down_sample, reproduce_data, save_data

notes = ["c1", "cs1", "d1", "ds1", "e1", "f1", "fs1", "g1", "gs1", "a1", "as1", "b1",
         "c2", "cs2", "d2", "ds2", "e2", "f2", "fs2", "g2", "gs2", "a2", "as2", "b2",
         "c3", "cs3", "d3", "ds3", "e3", "f3", "fs3", "g3", "gs3", "a3", "as3", "b3",
         "c4", "cs4", "d4", "ds4", "e4", "f4", "fs4", "g4", "gs4", "a4", "as4", "b4", "c5"]

notes = ["a", "e", "i", "o", "u"]

for note in notes:
    original_data = "%s.wav" % note
    downsamp_data = "%s_down3.wav" % note
    
    signal = read_data( "Records/%s" % original_data )
    signal_idx = np.arange( signal.size ) 
    
    fs = 44032
    cut = 44032*2
    x = signal[cut:cut+fs]
    x_idx = signal_idx[cut:cut+fs]
    
    fs_down = 5504
    x_down = down_sample(x, step=8)
    x_down_idx = down_sample(x_idx, step=8)
    
    plt.plot(x_idx, x)
    plt.plot(x_down_idx, x_down)
    plt.show()
    
    n, dt, p, df, nyquist, t, coef = apply_fourier( x )
    amps, phas = get_magnitude_phase( coef[:nyquist] / n )
    
    n_down, dt_down, p_down, df_down, nyquist_down, t_down, coef_down = apply_fourier( x_down )
    amps_down, phas_down = get_magnitude_phase( coef_down[:nyquist_down] / n_down )
    
    plt.plot( np.arange(nyquist), amps )
    plt.plot( np.arange(nyquist_down), amps_down )
    plt.show()
    
    reproduce_data( x )
    reproduce_data( x_down, fs=fs_down )
    
    save_data( x_down, "Records/%s" % downsamp_data, fs=fs_down )

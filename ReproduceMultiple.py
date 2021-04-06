import numpy as np

from Auxiliar import read_data, reproduce_data

fs = 8192
cut = 8192

data_files = ["a_1_8k", "e_1_8k", "i_1_8k", "o_1_8k", "u_1_8k",
              "a_2_8k", "e_2_8k", "i_2_8k", "o_2_8k", "u_2_8k",
              "a_3_8k", "e_3_8k", "i_3_8k", "o_3_8k", "u_3_8k",
              "a_4_8k", "e_4_8k", "i_4_8k", "o_4_8k", "u_4_8k",
              "a_5_8k", "e_5_8k", "i_5_8k", "o_5_8k", "u_5_8k"]

data_files = ["a_1_8k", "a_2_8k", "a_3_8k", "a_4_8k", "a_5_8k",
              "e_1_8k", "e_2_8k", "e_3_8k", "e_4_8k", "e_5_8k",
              "i_1_8k", "i_2_8k", "i_3_8k", "i_4_8k", "i_5_8k",
              "o_1_8k", "o_2_8k", "o_3_8k", "o_4_8k", "o_5_8k",
              "u_1_8k", "u_2_8k", "u_3_8k", "u_4_8k", "u_5_8k"]

for data_file in data_files:
    signal = read_data( "Records/%s.wav" % data_file, fs=fs )
    x = signal[cut:cut+fs]
    reproduce_data( x, fs=fs )

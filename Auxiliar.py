import wave
import numpy as np
from scipy.io.wavfile import write

def read_data(filename, chunk=1024, fs=44032, seconds=3):
    data_size = int(fs / chunk * seconds) * chunk
    wf = wave.open(filename, 'rb')
    data = wf.readframes( data_size )
    return np.frombuffer(data, dtype=np.int16)

def save_data(signal, filename, fs=44032):
    data = signal.tobytes()
    wf = wave.open(filename, 'wb')
    wf.setnchannels( 1 )
    wf.setsampwidth( 2 ) # size of int of 16 bits
    wf.setframerate( fs )
    wf.writeframes(data)
    wf.close()

import math
import wave
import numpy as np
import pyaudio

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

def read_frames(signal, chunk):
    for start in range(0, len(signal), chunk):
        yield signal[start:start + chunk]

def reproduce_data(signal, chunk=1024, fs=44032):
    audio = pyaudio.PyAudio()
    stream = audio.open( format=pyaudio.paInt16, channels=1, rate=fs, output=True )
    for data in read_frames(signal.tobytes(), chunk*2): stream.write(data)
    stream.close()
    audio.terminate()

def get_magnitude_phase( fourier_coef ):
    return abs(fourier_coef)*2, np.angle(fourier_coef)

def reconstruct_signal( fourier_coef ):
    return np.real( np.fft.ifft( fourier_coef ) ).astype( np.int16 )

def cos_wave( freq, amp, phase, t ):
    wave = amp * np.cos( 2*np.pi*freq*t + phase )
    return wave.astype(np.int16)

def difference(data, order=1): 
    if order <= 0: return data
    return difference(data[1:] - data[:-1], order-1)

def down_sample( signal, step=2 ):
    return signal[::step]

def apply_fourier( signal ):
    n = signal.size
    dt = 1/n # inter sample time
    p = n*dt # period
    df = 1/p # frequency resolution
    t = np.arange(0,p,dt) # time
    nyquist = math.floor( t.size/2 ) + 1
    coef = np.fft.fft( signal )
    return n, dt, p, df, nyquist, t, coef

def invert_fourier( amps, phas, t ):
    waves = np.array([ cos_wave( f, a, p, t ) for f, (a, p) in enumerate( zip(amps, phas) ) if a != 0 ])
    waves[0] = waves[0] / 2
    return np.sum( waves, axis=0, dtype=np.int16 )

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

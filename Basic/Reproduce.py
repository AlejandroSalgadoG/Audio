import wave
import pyaudio

filename = 'output.wav'

chunk = 1024 
fs = 44032
seconds = 2
n_frames = int(fs / chunk) * seconds
sample_format = pyaudio.paInt16  # 16 bits per sample

audio = pyaudio.PyAudio()
stream = audio.open(format = sample_format, channels = 1, rate = fs, output = True)

wf = wave.open(filename, 'rb')
for i in range(n_frames):
    data = wf.readframes(chunk)
    stream.write(data)

stream.close()
audio.terminate()

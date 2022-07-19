import matplotlib.pyplot as plt
from Fourier import FourierHandler

from Wav.WavReader import WavReader

wav_reader = WavReader("Data/hola2.wav")
data = wav_reader.get_data()

plt.plot(data)

fourier = FourierHandler(data)
coeff = fourier.transform()
amps = fourier.get_amplitudes(coeff)

plt.plot(amps)

print("done")
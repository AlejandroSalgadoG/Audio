import matplotlib.pyplot as plt
from Fourier import FourierHandler

from Wav.WavReader import WavReader
from Wav.WavWriter import WavWriter

wav_reader = WavReader("Data/sin2.wav")
data = wav_reader.get_data()

plt.plot(data)

fourier = FourierHandler(data)
coeff = fourier.time_to_freq()
amps = fourier.get_amplitudes(coeff)

plt.plot(amps)

signal = fourier.freq_to_time(coeff)

plt.plot(signal)

wav_writer = WavWriter(signal)
wav_writer.write_data("Data/output.wav")
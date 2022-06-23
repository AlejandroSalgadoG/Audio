from struct import *

with open("square.wav", "rb") as sound_file:
    data_file = sound_file.read()

riff = unpack("4c", data_file[:4])
[size] = unpack("<i", data_file[4:8])
wave = unpack("4c", data_file[8:12])

print(riff)
print(size, "-", len(data_file) - 8)
print(wave)

fmt_ = unpack("4c", data_file[12:16])
[sub_size] = unpack("<i", data_file[16:20])
[audio_format] = unpack("<h", data_file[20:22])
[n_channels] = unpack("<h", data_file[22:24])
[sample_rate] = unpack("<i", data_file[24:28])
[byte_rate] = unpack("<i", data_file[28:32])
[block_align] = unpack("<h", data_file[32:34])
[bits_per_sample] = unpack("<h", data_file[34:36])

print(fmt_)
print(sub_size, "- 16 for PCM")
print(audio_format, "- 1 for PCM")
print(n_channels, "- 1 for Mono")
print(sample_rate)
print(byte_rate, "-", sample_rate * n_channels * bits_per_sample / 8)
print(block_align, "-", n_channels * bits_per_sample / 8)
print(bits_per_sample)

data_id = unpack("4c", data_file[36:40])
data_size = unpack("<i", data_file[40:44])

print(data_id)
print(data_size, "-", len(data_file[44:]))

data = unpack("<44100h", data_file[44:]) # sample_rate * time

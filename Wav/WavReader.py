from struct import unpack
from typing import List

class WavReader:
    def __init__(self, wav_file_name: str):
        with open(wav_file_name, "rb") as sound_file:
            self.raw_data = sound_file.read()

        self.riff = unpack("4c", self.raw_data[:4])
        if b"".join(self.riff) != b"RIFF":
            raise Exception("No RIFF start")

        [self.size] = unpack("<i", self.raw_data[4:8])

        self.wave = unpack("4c", self.raw_data[8:12])
        if b"".join(self.wave) != b"WAVE":
            raise Exception("No WAVE string")

        self.fmt_ = unpack("4c", self.raw_data[12:16])
        if b"".join(self.fmt_) != b"fmt ":
            raise Exception("No fmt string")

        [self.sub_size] = unpack("<i", self.raw_data[16:20])
        if self.sub_size != 16:
            raise Exception(f"Found subsize of {self.sub_size}, should be 16")

        [self.audio_format] = unpack("<h", self.raw_data[20:22])
        if self.audio_format != 1:
            raise Exception(f"Found audio format of {self.audio_format}, should be 1")

        [self.n_channels] = unpack("<h", self.raw_data[22:24])
        if self.n_channels != 1:
            raise Exception(f"Found n_channels of {self.n_channels}, should be 1")

        [self.sample_rate] = unpack("<i", self.raw_data[24:28])
        [self.byte_rate] = unpack("<i", self.raw_data[28:32])
        [self.block_align] = unpack("<h", self.raw_data[32:34])

        [self.bits_per_sample] = unpack("<h", self.raw_data[34:36])
        if self.bits_per_sample % 8 != 0:
            raise Exception("Bits per sample are not full bytes")

        self.bytes_per_sample = self.bits_per_sample // 8

        self.data_id = unpack("4c", self.raw_data[36:40])
        if b"".join(self.data_id) != b"data":
            raise Exception(f"No data string")

        [self.data_size] = unpack("<i", self.raw_data[40:44])
        self.n_data_samples = self.data_size // self.bytes_per_sample

    def get_data(self) -> List[int]:
        return unpack(f"<{self.n_data_samples}h", self.raw_data[44:])

wav_reader = WavReader("Data/hola.wav")
data = wav_reader.get_data()
print(data)
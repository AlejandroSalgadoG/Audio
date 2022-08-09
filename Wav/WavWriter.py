from io import BytesIO
from struct import pack

from Fourier import TimeData

class WavWriter:
    def __init__(self, t_data: TimeData):
        self.data = t_data.data
        self.n_samples = len(self.data)

        self.sub_size = 16 # PCM
        self.audio_format = 1 # PCM
        self.n_channels = 1 # Mono
        self.sample_rate = 44100
        self.bits_per_sample = 16
        self.bytes_per_sample = self.bits_per_sample // 8

    def get_riff_header(self) -> bytes:
        riff = b"RIFF"
        size = pack("<i", 36 + self.n_samples * self.bytes_per_sample)
        wave = b"WAVE"
        return riff + size + wave

    def get_fmt_header(self) -> bytes:
        fmt_ = b"fmt "
        sub_size = pack("<i", self.sub_size)
        audio_format = pack("<h", self.audio_format)
        n_channels = pack("<h", self.n_channels)
        sample_rate = pack("<i", self.sample_rate)
        byte_rate = pack("<i", self.sample_rate * self.n_channels * self.bytes_per_sample)
        block_align = pack("<h", self.n_channels * self.bytes_per_sample)
        bites_per_sample = pack("<h", self.bits_per_sample)
        return fmt_ + sub_size + audio_format + n_channels + sample_rate + byte_rate + block_align + bites_per_sample

    def get_data_header(self) -> bytes:
        data_id = b"data"
        data_size = pack("<i", self.n_samples * self.bytes_per_sample)
        return data_id + data_size

    def get_data(self) -> bytes:
        return pack(f"<{self.n_samples}h", *self.data)

    def get_wav_content(self) -> bytes:
        return self.get_riff_header() + self.get_fmt_header() + self.get_data_header() + self.get_data()

    def get_file_obj(self) -> BytesIO:
        data = self.get_wav_content()
        file_obj = BytesIO()
        file_obj.write(data)
        file_obj.seek(0)
        return file_obj

    def write_data(self, file_name: str):
        data = self.get_wav_content()
        with open(file_name, "wb") as data_file:
            data_file.write(data)
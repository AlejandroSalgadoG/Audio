import numpy as np

from typing import List


class FreqData:
    def __init__(self, pos_coeff: List[np.complex128], odd_n_samples: bool):
        self.pos_coeff = pos_coeff
        self.odd_n_samples = odd_n_samples
        self.nyquist = pos_coeff.size - 1

    def get_amplitudes(self) -> List[np.float64]:
        return abs(self.pos_coeff) * 2

    def get_phases(self) -> List[np.float64]:
        return np.angle(self.pos_coeff)

    def get_coeff(self) -> List[np.complex128]:
        neg_coeff = np.conjugate(self.pos_coeff[-1 if self.odd_n_samples else -2:0:-1])
        return np.concatenate((self.pos_coeff, neg_coeff))


class TimeData:
    def __init__(self, data: List[np.int]):
        self.data = data
        self.n_samples = data.size
        self.nyquist = self.get_nyquist()

    def is_odd_n_samples(self) -> bool:
        return self.n_samples % 2 != 0

    def get_nyquist(self) -> int:
        return (self.n_samples - 1) // 2 if self.is_odd_n_samples() else self.n_samples // 2


class Time2Freq:
    @classmethod
    def transform(self, time_data: TimeData) -> FreqData:
        freq_raw_data = np.fft.fft(time_data.data) / time_data.n_samples
        return FreqData(freq_raw_data[:time_data.nyquist + 1], time_data.is_odd_n_samples())


class Freq2Time:
    @classmethod
    def transform(self, freq_data: FreqData) -> TimeData:
        coeff = freq_data.get_coeff()
        time_raw_data = np.real( np.fft.ifft( coeff * coeff.size ) )
        return TimeData(time_raw_data.astype(int))
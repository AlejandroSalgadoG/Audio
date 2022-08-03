import numpy as np

from typing import List, Optional


class FreqData:
    def __init__(self, pos_coeff: List[np.complex128], odd_n_samples: bool):
        self.pos_coeff = pos_coeff
        self.odd_n_samples = odd_n_samples
        self.nyquist = pos_coeff.size - 1

    def get_amplitudes(self, min_freq: Optional[int] = None, max_freq: Optional[int] = None) -> List[np.float64]:
        amps = abs(self.pos_coeff) * 2
        min_freq = 0 if min_freq is None else min_freq
        max_freq = self.nyquist if max_freq is None else max_freq
        return amps[min_freq:max_freq]

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

    def extract_portion(self, start: Optional[int] = None, end: Optional[int] = None):
        start = 0 if start is None else start
        end = self.n_samples if end is None else end
        return TimeData(self.data[start:end])

    def batch(self, size: int, offset: int = 0):
        n_batch = (self.n_samples - offset) // size

        if n_batch == 0:
            return BatchData(self, self.data, 0, self.n_samples)

        for i in range(n_batch):
            start = max(i*size + offset, 0)
            end = max((i+1)*size + offset, 0)
            yield BatchData(self, self.data[start:end], start, end)

        if end < self.n_samples:
            yield BatchData(self, self.data[end:], end, self.n_samples)

    def repeat(self, n_times: int):
        return TimeData(np.concatenate([self.data] * n_times))

    def concat(self, time_data):
        return TimeData(np.concatenate([self.data, time_data.data]))


class BatchData(TimeData):
    def __init__(self, time_data: TimeData, data: List[np.int], start: int, end: int):
        self.time_data = time_data
        self.data = data
        self.start = start
        self.end = end


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
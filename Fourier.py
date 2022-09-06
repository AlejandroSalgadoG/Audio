import numpy as np

from typing import Dict, List, Optional


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

    def batch(self, size: int, offset: int = 0, increment: Optional[int] = None):
        if increment is None:
            increment = size

        start = offset
        end = size + offset

        while start < self.n_samples:
            yield BatchData(self.data[max(0, start):end], start, end)
            start += increment
            end += increment

    def repeat(self, n_times: int):
        return TimeData(np.concatenate([self.data] * n_times))

    def concat(self, time_data):
        return TimeData(np.concatenate([self.data, time_data.data]))


class BatchData(TimeData):
    def __init__(self, data: List[np.int], start: int, end: int):
        super().__init__(data)
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


class Sample:
    def __init__(self, label: str, time_data: TimeData, freq_data: FreqData):
        self.label: str = label
        self.time_data: TimeData = time_data
        self.freq_data: FreqData = freq_data


class SampleFromTime:
    @classmethod
    def create(self, label: str, time_data: TimeData) -> Sample:
        return Sample(label, time_data, Time2Freq.transform(time_data))


class SampleFromFreq:
    @classmethod
    def create(self, label: str, freq_data: FreqData) -> Sample:
        return Sample(label, Freq2Time.transform(freq_data), freq_data)


class MultiSample:
    def __init__(self):
        self.all_samples: List[Sample] = []
        self.samples: Dict[str, List[Sample]] = {}

    def add_sample(self, sample: Sample):
        self.all_samples.append(sample)

        if sample.label not in self.samples:
            self.samples[sample.label] = []

        self.samples[sample.label].append(sample)

    def add_from_time_data(self, label: str, time_data: TimeData):
        self.add_sample( SampleFromTime.create(label, time_data) )

    def add_from_freq_data(self, label: str, freq_data: FreqData):
        self.add_sample( SampleFromFreq.create(label, freq_data) )

    def get_amplitudes(self, label: str, min_freq: int, max_freq: int):
        return ArrayList2Matrix.transform([
            sample.freq_data.get_amplitudes(min_freq, max_freq)
            for sample in self.samples[label]
        ])


class ArrayList2Matrix:
    @classmethod
    def transform(self, data_list: List[np.array]):
        return np.concatenate([np.expand_dims(data, axis=0) for data in data_list])
import numpy as np

from typing import List

class FrequencyDomain:
    def __init__(self, signal: List[int]):
        self.signal = signal
        self.n_samples = len(signal)
        self.nyquist = self.get_nyquist()
        self.coeff = self.time_to_freq()
        self.pos_coeff = self.coeff[:self.nyquist]

    def get_nyquist(self) -> int:
        return self.n_samples // 2 if self.n_samples % 2 == 0 else (self.n_samples - 1) // 2

    def time_to_freq(self) -> List[np.complex128]:
        return np.fft.fft(self.signal) / self.n_samples

    def get_amplitudes(self) -> List[np.float64]:
        return abs(self.pos_coeff) * 2

    def get_phases(self) -> List[np.float64]:
        return np.angle(self.pos_coeff)


class TimeDomain:
    def __init__(self, freq_signal: List[np.complex128]):
        self.coeff = freq_signal
        self.n_samples = len(freq_signal)
        self.signal = self.freq_to_time().astype(int)

    def freq_to_time(self) -> List[np.float64]:
        return np.real( np.fft.ifft(self.coeff * self.n_samples) )
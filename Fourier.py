import numpy as np

from typing import List

class FourierHandler:
    def __init__(self, signal: List[int]):
        self.signal = signal
        self.n_samples = len(signal)
        self.nyquist = self.get_nyquist(self.n_samples)

    def get_nyquist(self, size: int) -> int:
        return size // 2 if size % 2 == 0 else (size - 1) // 2

    def transform(self) -> List[np.complex128]:
        return np.fft.fft(self.signal) / self.n_samples

    def get_amplitudes(self, coeff: List[np.complex128]) -> List[int]:
        positive_coeff = coeff[:self.nyquist]
        return abs(positive_coeff) * 2
from typing import Iterator, List, Tuple

def batch_data(data: List[int], size: int, offset: int = 0) -> Iterator[Tuple[List[int], int, int]]:
    n_data = len(data)
    n_batch = (n_data - offset) // size

    if n_batch == 0:
        return data

    for i in range(n_batch):
        start = max(i*size + offset, 0)
        end = max((i+1)*size + offset, 0)
        yield data[start:end], start, end

    if end < n_data:
        yield data[end:], end, n_data
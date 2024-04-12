from typing import List
import pandas as pd


def extract_channels(tensor, index=None, columns=None) -> List[pd.DataFrame]:
    channels = []
    shape = tensor.shape
    assert len(shape) == 3, "The matrix should be a 3D tensor"

    for i in range(shape[0]):
        channel = tensor[i, :, :]
        channel = pd.DataFrame(channel, index=index, columns=columns)
        channels.append(channel)

    return channels
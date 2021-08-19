import torch
import torchaudio


class ToMono(torch.nn.Module):
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return torch.mean(waveform, dim=0, keepdim=True)


class Normalize(torch.nn.Module):
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return (waveform-waveform.mean()) / waveform.std()


class Pad(torch.nn.Module):
    def __init__(self, value: float, size: int):
        super(Pad, self).__init__()
        self.value = value
        self.size = size
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.pad(waveform, (0, self.size-max(waveform.shape)), "constant", self.value)


audio_transform = torch.nn.Sequential(*[
    ToMono(), #converts audio channels to mono 
    torchaudio.transforms.Resample(orig_freq=441000, new_freq=8000), # downsamples audio signal to 8000 HZ
    Normalize(), # normalize audio signal to have mean=0 & std=1
    Pad(value=0, size=32000),
])
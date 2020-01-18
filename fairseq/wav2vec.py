import torch
from fairseq.models.wav2vec import Wav2VecModel
import torchaudio

WAV_SCP_PATH = "/projects/audio-pipelines/fairseq/dummy_wav.scp"
WAV2VEC_PATH = "/projects/audio-pipelines/fairseq/wav2vec_large.pt"
DEVICE = "cuda:0"


def convert_files(scp_path: str):
    model = load_model()
    with open(scp_path, "r") as f:
        for row in f:
            wav_path = row.split(" ")[1].split("\n")[0]
            out_path = wav_path.split(".")[0] + ".torch"
            convert_file(wav_path, out_path, model)


def convert_file(wav_path: str, output_path: str, model: torch.nn.Module):
    wave = read_file(wav_path)
    z = model.feature_extractor(wave)
    c = model.feature_aggregator(z)
    print(z.shape, c.shape)
    save_file(c, output_path)


def read_file(wav_path: str) -> torch.tensor:
    waveform, sample_rate = torchaudio.load(wav_path)
    return waveform.to(DEVICE)


def load_model() -> torch.nn.Module:
    cp = torch.load(WAV2VEC_PATH)
    model = Wav2VecModel.build_model(cp["args"], task=None)
    model.load_state_dict(cp["model"])
    model.eval()
    return model.to(DEVICE)


def save_file(tensor: torch.tensor, output_path: str):
    torch.save(tensor, output_path)

import torch
import numpy as np
import os
import librosa
import re
import json
import yaml
import argparse
from pathlib import Path
from string import punctuation
from g2p_en import G2p
from scipy.io.wavfile import write

from models.StyleSpeech import StyleSpeech
from text import text_to_sequence
import audio as Audio
import utils
from mel2wav.interface import MelVocoder
from mel2wav.modules import Generator

# Global variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Text processing functions
def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon

def preprocess_english(text, lexicon_path):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon(lexicon_path)

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(text_to_sequence(phones, ['english_cleaners']))

    return torch.from_numpy(sequence).to(device)

# Audio processing functions
def preprocess_audio(audio_file, _stft):
    wav, sample_rate = librosa.load(audio_file, sr=None)
    if sample_rate != 16000:
        wav = librosa.resample(wav, sample_rate, 16000)
    mel_spectrogram, _ = Audio.tools.get_mel_from_wav(wav, _stft)
    return torch.from_numpy(mel_spectrogram).to(device)

# Model initialization
def get_StyleSpeech(config, checkpoint_path):
    model = StyleSpeech(config).to(device)
    model.load_state_dict(torch.load(checkpoint_path)['model'])
    model.eval()
    return model

# Modified load_model function for MelVocoder
def load_model(mel2wav_path, device=device):
    root = Path(mel2wav_path)
    with open(root / "args.yml", "r") as f:
        args_dict = yaml.safe_load(f)
    
    args = argparse.Namespace(**args_dict)
    
    netG = Generator(args.n_mel_channels, args.ngf, args.n_residual_layers).to(device)
    netG.load_state_dict(torch.load(root / "best_netG.pt", map_location=device))
    return netG

def synthesize(args, model, _stft):   
    # Preprocess input
    ref_mel = preprocess_audio(args['ref_audio'], _stft).transpose(0,1).unsqueeze(0)
    src = preprocess_english(args['text'], args['lexicon_path']).unsqueeze(0)
    src_len = torch.from_numpy(np.array([src.shape[1]])).to(device)
    
    # Create save directory
    save_path = args['save_path']
    os.makedirs(save_path, exist_ok=True)

    # Generate mel spectrogram
    style_vector = model.get_style_vector(ref_mel)
    mel_output = model.inference(style_vector, src, src_len)[0]

    print("Mel output shape:", mel_output.shape)

    # Adjust mel_output dimensions if necessary
    if mel_output.dim() == 2:
        mel_output = mel_output.unsqueeze(0)  # (n_mel, time) -> (1, n_mel, time)
    elif mel_output.dim() == 3 and mel_output.size(1) != 80:
        mel_output = mel_output.transpose(1, 2)  # (batch, time, n_mel) -> (batch, n_mel, time)

    print("Adjusted mel output shape:", mel_output.shape)

    # Convert mel to audio
    audio = vocoder.inverse(mel_output)
    audio = audio.cpu().squeeze().numpy()
    
    # Convert to 16-bit PCM
    audio_16bit = (audio * 32767).astype(np.int16)

    # Save audio
    write(os.path.join(save_path, 'synthesized_audio.wav'), 16000, audio_16bit)

    # # Plot spectrograms
    # mel_ref = ref_mel.cpu().squeeze().transpose(0, 1).detach()
    # mel = mel_output.cpu().squeeze().transpose(0, 1).detach()
    # utils.plot_data([mel_ref.numpy(), mel.numpy()], 
    #     ['Ref Spectrogram', 'Synthesized Spectrogram'], filename=os.path.join(save_path, 'plot.png'))
    
    print('Generation complete!')

# Main execution
if __name__ == "__main__":
    # Define arguments
    args = {
        "checkpoint_path": "pretrained/meta_stylespeech.pth",
        "config": "configs/config.json",
        "save_path": "results/",
        "ref_audio": "samples/0001_000352.wav",
        "text": "What are you doing now! professor!",
        "lexicon_path": "lexicon/librispeech-lexicon.txt"
    }
    
    # Load config
    with open(args['config']) as f:
        config = utils.AttrDict(json.load(f))

    # Initialize model
    model = get_StyleSpeech(config, args['checkpoint_path'])
    vocoder = MelVocoder(path='./pretrained')
    print('Model prepared')

    # Initialize STFT
    _stft = Audio.stft.TacotronSTFT(
                config.filter_length,
                config.hop_length,
                config.win_length,
                config.n_mel_channels,
                config.sampling_rate,
                config.mel_fmin,
                config.mel_fmax)

    # Run synthesis
    synthesize(args, model, _stft)
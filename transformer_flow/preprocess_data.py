import librosa
import os
import numpy as np
import soundfile as sf
import torch

def process_audio(audio, mel_filter, sampling_rate, n_fft=2048, hop_length=512):
    D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    spec_mag = np.abs(D)
    spec_phase = np.angle(D)
    spec_x = spec_mag * np.cos(spec_phase)
    spec_y = spec_mag * np.sin(spec_phase)
    if mel_filter is not None:
        spec_x, spec_y = mel_filter @ spec_x, mel_filter @ spec_y
    return spec_x, spec_y

def regenerate_audio(spec_x, spec_y, mel_filter_inv, sampling_rate, n_fft=2048, hop_length=512):
    if mel_filter_inv is not None:
        spec_x = mel_filter_inv @ spec_x
        spec_y = mel_filter_inv @ spec_y
    D = spec_x + 1j * spec_y
    audio = librosa.istft(D, hop_length=hop_length)
    return audio

def generate_mel_filter(sampling_rate, n_fft=2048, n_mels=80, fmin=0, fmax=None):
    mel_filter = librosa.filters.mel(sr=sampling_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    mel_filter_inv = np.linalg.pinv(mel_filter)
    return mel_filter, mel_filter_inv

if __name__ == '__main__':
    sampling_rate = 44100 
    n_fft = 1000
    root_dir = '../processed_jamendo_data'
    output_dir = '../square_jamendo_data'

    mel_filter, mel_filter_inv = generate_mel_filter(sampling_rate, n_mels=500, fmin=0, fmax=None, n_fft=n_fft)
    mel_filter, mel_filter_inv = None, None

    audio = librosa.load(os.path.join(root_dir, '0.wav'), sr=sampling_rate)[0]

    spec_x, spec_y = process_audio(audio, mel_filter, sampling_rate, n_fft=n_fft, hop_length=n_fft-100)

    r_audio = regenerate_audio(spec_x, spec_y, mel_filter_inv, sampling_rate, n_fft=n_fft, hop_length=n_fft-100)
    sf.write('output.wav', r_audio, sampling_rate)

    first_audio_len = 0
    for file in os.listdir(root_dir):
        if not file.endswith('.wav'): continue

        audio = librosa.load(os.path.join(root_dir, file), sr=sampling_rate)[0]

        if first_audio_len == 0:
            first_audio_len = len(audio)
        else:
            audio = audio[:first_audio_len]

        spec_x, spec_y = process_audio(audio, mel_filter, sampling_rate, n_fft=n_fft, hop_length=n_fft-100)

        spec = np.concatenate([spec_x, spec_y], axis=0)

        torch.save(torch.from_numpy(spec), os.path.join(output_dir, file.replace('.wav', '.pt')))

        #r_audio = regenerate_audio(spec_x, spec_y, mel_filter_inv, sampling_rate, n_fft=n_fft, hop_length=n_fft-100)
        #sf.write('output.wav', r_audio, sampling_rate)



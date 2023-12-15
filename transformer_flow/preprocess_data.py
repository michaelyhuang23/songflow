import librosa
import os
import numpy as np
import soundfile as sf
import torch
import multiprocessing

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

sampling_rate = 44100 
n_fft = 1000
root_dir = '../processed_jamendo_data/11'
output_dir = '../square_jamendo_data'
mel_filter, mel_filter_inv = None, None

def process_file(file, first_audio_len):
    sampling_rate = 44100 
    n_fft = 1000
    root_dir = '../processed_jamendo_data/11'
    output_dir = '../square_jamendo_data'
    mel_filter, mel_filter_inv = None, None

    if not file.endswith('.wav'): return

    audio = librosa.load(os.path.join(root_dir, file), sr=sampling_rate)[0]

    audio = audio[:first_audio_len]

    spec_x, spec_y = process_audio(audio, mel_filter, sampling_rate, n_fft=n_fft, hop_length=n_fft-100)

    spec = np.concatenate([spec_x, spec_y], axis=0)

    torch.save(torch.from_numpy(spec), os.path.join(output_dir, file.replace('.wav', '.pt')))

def parallel_process_file(args):
    process_file(*args)

if __name__ == '__main__':
    audio = librosa.load(os.path.join(root_dir, '0.wav'), sr=sampling_rate)[0]
    first_audio_len = len(audio)

    file_names = [file for file in os.listdir(root_dir) if file.endswith('.wav')]

    # Number of processes
    num_processes = multiprocessing.cpu_count()

    # Create a pool of processes
    pool = multiprocessing.Pool(processes=num_processes)

    # Map the process_folder function to the folders
    pool.map(parallel_process_file, zip(file_names, [first_audio_len]*len(file_names)))

    # Close the pool and wait for the work to finish
    pool.close()
    pool.join()
    #r_audio = regenerate_audio(spec_x, spec_y, mel_filter_inv, sampling_rate, n_fft=n_fft, hop_length=n_fft-100)
    #sf.write('output.wav', r_audio, sampling_rate)



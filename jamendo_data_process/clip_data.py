import librosa
import os
import numpy as np
import math
import soundfile as sf

def load_audio(file_path):
    # Load the audio file
    audio, sr = librosa.load(file_path, sr=None)  # sr=None ensures original sampling rate is used
    return audio, sr

def find_sound_start_end(audio, sr, threshold_percentile=1):
    # Compute the energy of the audio
    energy = librosa.feature.rms(y=audio)[0]
    threshold = np.percentile(energy, threshold_percentile)

    # Find frames exceeding the threshold
    frames = np.where(energy > threshold)[0]

    if len(frames) == 0:  # In case the entire audio is silent
        return 0, len(audio)

    # Convert frames to samples
    start_frame, end_frame = frames[0], frames[-1]
    start_sample = librosa.frames_to_samples(start_frame)
    end_sample = librosa.frames_to_samples(end_frame)

    print(f'fraction of audio kept: {(end_sample - start_sample) / len(audio)}')

    return start_sample, end_sample

def split_into_clips(audio, sr, clip_length=10):
    # Calculate the number of samples per clip
    samples_per_clip = int(clip_length * sr)

    # Calculate total number of clips
    total_clips = len(audio) // samples_per_clip

    clips = []
    for i in range(total_clips):
        start = i * samples_per_clip
        end = start + samples_per_clip
        clip = audio[start:end]
        clips.append(clip)

    return clips

def save_clips(clips, sr, output_dir, idx):
    for clip in clips:
        filename = os.path.join(output_dir, f"{idx}.wav")
        idx += 1
        sf.write(filename, clip, sr)
    return idx

root_dir = '../jamendo-data'
output_dir = '../processed_jamendo_data'

idx = 0

for folder in os.listdir(root_dir):
    if 'DS' in folder: continue
    folder_path = os.path.join(root_dir, folder)
    for file in os.listdir(folder_path):
        if not file.endswith('.mp3') : continue
        file_path = os.path.join(folder_path, file)

        audio, sr = load_audio(file_path)
        # Find the start and end of the sound
        start, end = find_sound_start_end(audio, sr)
        # Extract the portion of the audio where there is sound
        audio_with_sound = audio[start:end]
        # Split the audio into 10-second clips
        clips = split_into_clips(audio_with_sound, sr, 10)

        idx = save_clips(clips, sr, output_dir, idx)

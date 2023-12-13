import torch
from torch.nn import functional as F
from model import build_model
from dataset import SpecDataset
from preprocess_data import regenerate_audio
import os
import soundfile as sf

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

model_folder = 'dataset=square_jamendo_data_epochs=10_batch_size=5_lr=0.0005_num_layers=10_dim_feedforward=128'
model_path = os.path.join('results', model_folder, '9.pth')

config = {
    'dataset': 'square_jamendo_data',
    'epochs': 10,
    'batch_size': 5,
    'lr': 0.0005,
    'num_layers': 10,
    'dim_feedforward': 128
}


dataset = SpecDataset(os.path.join('../', config['dataset']))
flow = build_model(T=dataset.T, D=dataset.D, num_layers=config['num_layers'], dim_feedforward=config['dim_feedforward']).to(device)

checkpoint = torch.load(model_path, map_location=device)
flow.load_state_dict(checkpoint['model_state_dict'])
flow.eval()

n_fft = 1000
sampling_rate = 44100

def generate_audio(z=None):
    if z is None:
        c_data = flow.sample(1)[0]
    else:
        c_data = flow._transform.inverse(z)[0]
    spec = dataset.revert_tensor(c_data).cpu().numpy()
    spec_x = spec[:len(spec)//2]
    spec_y = spec[len(spec)//2:]
    r_audio = regenerate_audio(spec_x, spec_y, None, sampling_rate, n_fft=n_fft, hop_length=n_fft-100)
    return r_audio

with torch.no_grad():
    data = dataset[0].to(device)[None, ...]
    z = flow.transform_to_noise(data)
    print('finish transform to noise')
    r_audio = generate_audio(z)
    print('finish regenerate audio')
    sf.write('output.wav', r_audio, sampling_rate)

    for i in range(10):
        print(f'running {i}')
        r_audio = generate_audio()
        sf.write(f'output_{i}.wav', r_audio, sampling_rate)


    
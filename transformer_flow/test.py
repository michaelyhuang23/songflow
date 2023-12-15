import torch
from torch.nn import functional as F
from model import build_model
from dataset import SpecDataset
from preprocess_data import regenerate_audio
import os
import soundfile as sf

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
# /home/gridsan/yhuang1/songflow/transformer_flow/results/dataset=square_jamendo_data_epochs=2_batch_size=5_lr=1e-05_num_layers=10_dim_feedforward=128/1_2680.pth
model_folder = 'dataset=square_jamendo_data_epochs=2_batch_size=5_lr=1e-05_num_layers=10_dim_feedforward=128'
model_path = os.path.join('results', model_folder, '1_2680.pth')

config = {
    'dataset': 'square_jamendo_data',
    'epochs': 2,
    'batch_size': 5,
    'lr': 1e-05,
    'num_layers': 10,
    'dim_feedforward': 128
}


dataset = SpecDataset(os.path.join('../', config['dataset']))
flow = build_model(T=dataset.T, D=dataset.D, num_layers=config['num_layers'], dim_feedforward=config['dim_feedforward']).to(device)

checkpoint = torch.load(model_path, map_location=device)
share_checkpoints = {k: v for k, v in checkpoint['model_state_dict'].items() if 'autoregressive_net.mask' not in k and '_permutation' not in k}
flow.load_state_dict(share_checkpoints, strict=False)

# do not use eval mode! Transformer module has a bug where eval mode fails

n_fft = 1000
sampling_rate = 44100

def generate_audio(data):
    spec = dataset.revert_tensor(data).cpu().numpy().astype('float64')
    spec_x = spec[:len(spec)//2]
    spec_y = spec[len(spec)//2:]
    r_audio = regenerate_audio(spec_x, spec_y, None, sampling_rate, n_fft=n_fft, hop_length=n_fft-100)
    return r_audio

with torch.no_grad():
    data = dataset[0].to(device)[None, :, :] 

    r_audio = generate_audio(data[0])
    sf.write('input.wav', r_audio, sampling_rate)

    print(data.shape)

    z = flow.transform_to_noise(data)
    print(torch.min(z), torch.std(z), torch.max(z))
    print('finish transform to noise')
    c_data, _ = flow._transform.inverse(z)
    print(torch.max(data - c_data))
    c_data = c_data[0]
    print(c_data.shape)
    r_audio = generate_audio(c_data)
    print('finish regenerate audio')
    sf.write('output.wav', r_audio, sampling_rate)

    for i in range(10):
        print(f'running {i}')
        c_data = flow.sample(1)[0]
        print(c_data.shape)
        r_audio = generate_audio(c_data)
        sf.write(f'output_{i}.wav', r_audio, sampling_rate)


    
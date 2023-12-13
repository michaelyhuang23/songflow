from torch.nn import functional as F
import torch
import os
from model import build_model
from dataset import SpecDataset
import wandb
import argparse

def save_model(epoch, model, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='square_jamendo_data')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_layers', type=int, default=10)
    parser.add_argument('--dim_feedforward', type=int, default=128)

    args = parser.parse_args()

    results_dir = os.path.join('results', '_'.join([f'{k}={v}' for k, v in vars(args).items()]))
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    os.environ["WANDB_API_KEY"] = 'd1b0878433065e9dcd09b271a9f48a69e1a8e543' # Michael's api key
    os.environ["WANDB_MODE"] = 'offline'
    os.environ["WANDB_DIR"] = 'wandb_log'

    wandb.init(
        project="songflow",
        entity="michaelyhuang23"
    )
    wandb.config.update(args)
    wandb.config.update({'architecture': 'transformer'})

    dataset = SpecDataset(os.path.join('../', args.dataset))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    flow = build_model(T=dataset.T, D=dataset.D, num_layers=args.num_layers, dim_feedforward=args.dim_feedforward).to(device)

    optimizer = torch.optim.Adam(flow.parameters(), lr=args.lr)

    EPOCHS = args.epochs
    for epoch in range(EPOCHS):
        for i, batch in enumerate(dataloader):
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = -flow.log_prob(batch).mean()
            loss.backward()
            optimizer.step()
            wandb.log({'epoch': epoch, 'iter': i , 'loss': loss.item()})
            if i % min(len(dataset)//10, 10) == 0:
                print(f'Epoch {epoch}, Iter: {i}, Loss: {loss.item()}')
        save_model(epoch, flow, os.path.join(results_dir, f'{epoch}.pth'))

wandb.finish()
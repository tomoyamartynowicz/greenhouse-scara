import argparse
import os
import random
import sys

import numpy as np
import torch
from tqdm import tqdm

import imitate_episodes as original
from constants import TASK_CONFIGS


def detach_cpu(values):
    return {key: value.detach().cpu() for key, value in values.items()}


def save_atomic(value, path):
    temporary_path = path + '.tmp'
    torch.save(value, temporary_path)
    os.replace(temporary_path, path)


def train_bc(train_dataloader, val_dataloader, config, resume):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    resume_path = os.path.join(ckpt_dir, 'training_state.ckpt')
    best_path = os.path.join(ckpt_dir, 'policy_best.ckpt')

    original.set_seed(seed)
    policy = original.make_policy(policy_class, policy_config).to(config["device"])
    optimizer = original.make_optimizer(policy_class, policy)

    start_epoch = 0
    min_val_loss = np.inf
    best_epoch = -1
    train_history = []
    validation_history = []

    if resume:
        state = torch.load(resume_path, map_location=config['device'], weights_only=False)
        saved_config = state['config']
        if (saved_config['task_name'] != config['task_name'] or saved_config['policy_class'] != policy_class
                or saved_config['policy_config'] != policy_config or saved_config['seed'] != seed):
            raise ValueError('The resume checkpoint belongs to a different task or ACT configuration.')
        policy.load_state_dict(state['policy'])
        optimizer.load_state_dict(state['optimizer'])
        start_epoch = state['epoch'] + 1
        min_val_loss = state['min_val_loss']
        best_epoch = state['best_epoch']
        train_history = [detach_cpu(item) for item in state['train_history']]
        validation_history = [detach_cpu(item) for item in state['validation_history']]
        torch.set_rng_state(state['torch_rng'].cpu())
        if torch.cuda.is_available(): torch.cuda.set_rng_state_all([rng.cpu() for rng in state['cuda_rng']])
        np.random.set_state(state['numpy_rng'])
        random.setstate(state['python_rng'])
        del state
        print(f'Resuming at epoch {start_epoch}; target is {num_epochs} epochs')
    else:
        save_atomic({
            'format_version': 1,
            'epoch': -1,
            'policy': policy.state_dict(),
            'optimizer': optimizer.state_dict(),
            'min_val_loss': min_val_loss,
            'best_epoch': best_epoch,
            'train_history': train_history,
            'validation_history': validation_history,
            'config': config,
            'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all(),
            'numpy_rng': np.random.get_state(),
            'python_rng': random.getstate(),
        }, resume_path)

    for epoch in tqdm(range(start_epoch, num_epochs)):
        print(f'\nEpoch {epoch}')

        with torch.inference_mode():
            policy.eval()
            epoch_dicts = [original.forward_pass(data, policy) for data in val_dataloader]
            validation_summary = original.compute_dict_mean(epoch_dicts)
            validation_history.append(detach_cpu(validation_summary))
            val_loss = validation_summary['loss'].item()
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                best_epoch = epoch
                save_atomic(policy.state_dict(), best_path)
        print(f'Val loss:   {val_loss:.5f}')
        print(''.join(f'{key}: {value.item():.3f} ' for key, value in validation_summary.items()))

        policy.train()
        optimizer.zero_grad()
        epoch_dicts = []
        for data in train_dataloader:
            forward_dict = original.forward_pass(data, policy)
            forward_dict['loss'].backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_cpu(forward_dict))
            epoch_dicts.append(original.detach_dict(forward_dict))
        train_summary = original.compute_dict_mean(epoch_dicts)
        print(f"Train loss: {train_summary['loss'].item():.5f}")
        print(''.join(f'{key}: {value.item():.3f} ' for key, value in train_summary.items()))

        save_atomic({
            'format_version': 1,
            'epoch': epoch,
            'policy': policy.state_dict(),
            'optimizer': optimizer.state_dict(),
            'min_val_loss': min_val_loss,
            'best_epoch': best_epoch,
            'train_history': train_history,
            'validation_history': validation_history,
            'config': config,
            'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all(),
            'numpy_rng': np.random.get_state(),
            'python_rng': random.getstate(),
        }, resume_path)

        if epoch % 100 == 0:
            torch.save(policy.state_dict(), os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt'))
            original.plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    torch.save(policy.state_dict(), os.path.join(ckpt_dir, 'policy_last.ckpt'))
    best_state_dict = torch.load(best_path, map_location='cpu', weights_only=True)
    torch.save(best_state_dict, os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt'))
    original.plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')
    return best_epoch, min_val_loss, best_state_dict


def main(args):
    resume = args.pop('resume')
    if '--resume' in sys.argv:
        sys.argv.remove('--resume')
    resume_path = os.path.join(args['ckpt_dir'], 'training_state.ckpt')
    if resume and not os.path.isfile(resume_path):
        raise FileNotFoundError(f'No resume checkpoint found: {resume_path}')
    original.train_bc = lambda train, val, config: train_bc(train, val, config, resume)
    original.main(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument('--policy_class', default='ACT')
    parser.add_argument('--task_name', default='scara_default', choices=TASK_CONFIGS)
    parser.add_argument('--inputs_config', default=os.path.join(os.path.dirname(__file__), 'inputs.yaml'))
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pretrained_backbone', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--allow_single_episode', action='store_true')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--kl_weight', type=int, default=10)
    parser.add_argument('--chunk_size', type=int, default=30)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--dim_feedforward', type=int, default=3200)
    main(vars(parser.parse_args()))

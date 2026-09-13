import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import yaml
from greenhouse_scara_common.data import image_specs
import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
import time
from copy import deepcopy
from tqdm import tqdm
from pathlib import Path

from constants import ACT_STATE_DIM, TASK_CONFIGS
from utils import load_data # data functions
from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from policy import ACTPolicy, CNNMLPPolicy

import IPython
e = IPython.embed

def main(args):
    set_seed(args["seed"])
    # command line parameters
    policy_class = args['policy_class']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']

    # get task parameters
    task_config = TASK_CONFIGS[task_name]
    dataset_dir = args["dataset_dir"]
    ckpt_dir = args["ckpt_dir"]
    with open(args.get('inputs_config', Path(__file__).with_name('inputs.yaml'))) as f:
        inputs_config = yaml.safe_load(f)
    if args.get('allow_single_episode'):
        inputs_config['allow_single_episode'] = True
    stream_specs = image_specs(inputs_config['shape_meta']) # camera info as dict from yaml
    camera_names = list(stream_specs) # list of camera names from input yaml
    if len({tuple(spec["shape"][1:]) for spec in stream_specs.values()}) != 1:
        raise ValueError("ACT streams must share the same input height and width")
    num_episodes = len(list(Path(dataset_dir).glob("episode_*.hdf5")))

    # fixed parameters
    state_dim = ACT_STATE_DIM
    lr_backbone = 1e-5
    backbone = 'resnet18'
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': args['lr'],
                         'num_queries': args['chunk_size'],
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'camera_names': camera_names,
                         'stream_specs': stream_specs,
                         'pretrained_backbone': args.get('pretrained_backbone', True),
                         'state_dim' : state_dim
                         }
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,
                         'camera_names': camera_names,}
    else:
        raise NotImplementedError

    config = {
        'dataset_dir': dataset_dir,
        'num_episodes': num_episodes,
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'state_dim': state_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        'device': args.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'),
        'inputs_config': inputs_config,
        'max_train_steps': args.get('max_train_steps'),
        'max_val_steps': args.get('max_val_steps'),
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'camera_names': camera_names,
    }

    # dataset chunk_size == ACT num_queries
    train_dataloader, val_dataloader, stats = load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val, policy_config["num_queries"], inputs_config, args.get("num_workers", 4))

    print("===================================")
    print(f"Train samples : {len(train_dataloader.dataset)}")
    print(f"Val samples   : {len(val_dataloader.dataset)}")
    print(f"Train batches : {len(train_dataloader)}")
    print(f"Val batches   : {len(val_dataloader)}")
    print(f"Batch size    : {batch_size_train}")
    print(f"Num workers   : {train_dataloader.num_workers}")
    print("===================================")

    # save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    with open(os.path.join(ckpt_dir, "config.pkl"),"wb") as f:
        pickle.dump(config, f)

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data

    ## recommended non_blocking=True when pin_memory=True ref Pytorch
    device = next(policy.parameters()).device
    image_data = {key: value.to(device, non_blocking=True) for key, value in image_data.items()}
    qpos_data = qpos_data.to(device, non_blocking=True)
    action_data = action_data.to(device, non_blocking=True)
    is_pad = is_pad.to(device, non_blocking=True)
    return policy(qpos_data, image_data, action_data, is_pad) # TODO remove None


def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    policy.to(config["device"])
    optimizer = make_optimizer(policy_class, policy)

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    for epoch in tqdm(range(num_epochs)):
        ## Monitor GPU-memory annd timing
        if next(policy.parameters()).is_cuda: torch.cuda.synchronize()
        epoch_start = time.perf_counter()

        print(f'\nEpoch {epoch}')
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
                if config["max_val_steps"] and batch_idx + 1 >= config["max_val_steps"]: break
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        print(f'Val loss:   {epoch_val_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)
        if next(policy.parameters()).is_cuda: torch.cuda.synchronize()
        val_end = time.perf_counter()

        # training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            # backward
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
            if config["max_train_steps"] and batch_idx + 1 >= config["max_train_steps"]: break
        epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)
        if next(policy.parameters()).is_cuda: torch.cuda.synchronize()
        epoch_end = time.perf_counter()

        ## Monitor GPU-memory annd timing
        val_time = val_end - epoch_start
        train_time = epoch_end - val_end
        epoch_time = epoch_end - epoch_start
        peak_mem_gb = torch.cuda.max_memory_allocated() / 1024**3 if next(policy.parameters()).is_cuda else 0

        print(
            f"Epoch {epoch} performance: "
            f"total={epoch_time:.2f}s, "
            f"train={train_time:.2f}s, "
            f"val={val_time:.2f}s, "
            f"peak_gpu_mem={peak_mem_gb:.2f} GB"
        )

        if epoch % 100 == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    # save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f'Saved plots to {ckpt_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument('--ckpt_dir', required=True)
    parser.add_argument('--policy_class', default='ACT', choices=['ACT'])
    parser.add_argument('--task_name', default='scara_default', choices=TASK_CONFIGS)
    parser.add_argument('--inputs_config', type=Path, default=Path(__file__).with_name('inputs.yaml'))
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pretrained_backbone', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--allow_single_episode', action='store_true')
    parser.add_argument('--max_train_steps', type=int)
    parser.add_argument('--max_val_steps', type=int)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=1e-5)

    # for ACT
    parser.add_argument('--kl_weight', type=int, default=10)
    parser.add_argument('--chunk_size', type=int, default=30)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--dim_feedforward', type=int, default=3200)
    
    main(vars(parser.parse_args()))

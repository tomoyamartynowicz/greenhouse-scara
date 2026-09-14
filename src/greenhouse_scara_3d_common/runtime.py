"""Small shared training/evaluation loop; policy objectives stay in their own modules."""
import argparse
import copy
import importlib
import json
from pathlib import Path
import random
import time
import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader
from .dataset import PointCloudDataset, split_paths
from .vendor.dp3.model.diffusion.ema_model import EMAModel

WORKSPACE = Path(__file__).resolve().parents[2]
MODELS = {'dp3': 'greenhouse_scara_3d_diffusion_policy', 'flow3d': 'greenhouse_scara_3d_flow_matching_policy',
          'act3d': 'greenhouse_scara_3d_act', 'maniflow': 'greenhouse_scara_maniflow'}


def build_policy(cfg):
    if cfg['model'] not in MODELS: raise ValueError('Unknown model')
    if not 1 <= cfg['n_obs_steps'] <= cfg['horizon'] or not 1 <= cfg['n_action_steps'] <= cfg['horizon'] - cfg['n_obs_steps'] + 1:
        raise ValueError('Require horizon >= n_obs_steps + n_action_steps - 1')
    meta = {'obs': {'point_cloud': {'shape': [cfg['pointcloud']['num_points'], 6 if cfg['pointcloud']['use_color'] else 3], 'type': 'point_cloud'},
                    'agent_pos': {'shape': [4], 'type': 'low_dim'}}, 'action': {'shape': [4]}}
    return importlib.import_module(MODELS[cfg['model']] + '.policy').build(cfg, OmegaConf.create(meta))


def load_config(default, argv=None):
    parser = argparse.ArgumentParser(description='SCARA 3D training; override YAML with key=value')
    parser.add_argument('--config', default=str(default))
    args, overrides = parser.parse_known_args(argv)
    cfg = OmegaConf.to_container(OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(overrides)), resolve=True)
    # Each model's own repository root; also preserves unmoved local 3D projects.
    path_root = Path(default).resolve().parents[2]
    for key in ('dataset_dir', 'output_dir', 'calibration_file'):
        p = Path(cfg[key]).expanduser()
        cfg[key] = str(p if p.is_absolute() else path_root / p)
    cfg['calibration'] = OmegaConf.to_container(OmegaConf.load(cfg['calibration_file']), resolve=True)
    return cfg


def _loss(policy, batch, model, ema):
    out = policy.compute_loss(batch, ema_model=ema) if model == 'maniflow' else policy.compute_loss(batch)
    return out if isinstance(out, tuple) else (out, {})


def to_device(batch, device):
    return {k: to_device(v, device) if isinstance(v, dict) else v.to(device) for k, v in batch.items()}


def load_checkpoint(path, device='cpu'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    policy = build_policy(checkpoint['config'])
    policy.load_state_dict(checkpoint['ema'] if checkpoint['ema'] is not None else checkpoint['policy'])
    return policy.to(device).eval(), checkpoint


def _loader(dataset, batch_size, workers, shuffle, generator, model):
    if model == 'maniflow':
        # Upstream 75/25 split requires both sub-batches, and uses integer batch fractions.
        if batch_size < 4 or batch_size % 4 or len(dataset) < 4:
            raise ValueError('ManiFlow needs batch_size divisible by 4 and >=4 samples, also for validation')
        batch_size = min(batch_size, len(dataset) // 4 * 4)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      drop_last=model == 'maniflow', generator=generator, pin_memory=torch.cuda.is_available())


def train(cfg):
    seed = cfg['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    output = Path(cfg['output_dir'])
    latest = output / 'latest.ckpt'
    if latest.exists() and not cfg.get('resume', False):
        raise FileExistsError(f'{latest} exists; use another output_dir or resume=true')
    if cfg.get('resume') and not latest.exists(): raise FileNotFoundError(latest)
    if cfg['model'] == 'maniflow':
        if not cfg['use_ema']: raise ValueError('ManiFlow consistency training requires use_ema=true')
        if (cfg['policy']['flow_batch_ratio'], cfg['policy']['consistency_batch_ratio']) != (.75, .25):
            raise ValueError('This trainer supports the upstream 75/25 flow/consistency split')
    train_paths, val_paths = split_paths(cfg)
    train_ds, val_ds = PointCloudDataset(train_paths, cfg), PointCloudDataset(val_paths, cfg)
    train_loader = _loader(train_ds, cfg['batch_size'], cfg['num_workers'], True, generator, cfg['model'])
    val_loader = _loader(val_ds, cfg['val_batch_size'], cfg['num_workers'], False, None, cfg['model'])
    policy = build_policy(cfg)
    policy.set_normalizer(train_ds.normalizer())
    policy.to(cfg['device'])
    ema_policy = copy.deepcopy(policy).eval().requires_grad_(False) if cfg['use_ema'] else None
    ema = EMAModel(ema_policy, power=.75) if ema_policy is not None else None
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad],
                                lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'],
                                betas=tuple(cfg.get('optimizer_betas', [0.9, 0.999])))
    from diffusers.optimization import get_scheduler
    steps_per_epoch = min(len(train_loader), cfg['max_train_steps'] or len(train_loader))
    scheduler = get_scheduler('cosine', optimizer=optimizer, num_warmup_steps=cfg['warmup_steps'],
                              num_training_steps=steps_per_epoch * cfg['epochs'])
    start, best, global_step = 0, float('inf'), 0
    if cfg.get('resume'):
        state = torch.load(latest, map_location=cfg['device'], weights_only=False)
        allowed = {'resume', 'epochs', 'max_train_steps', 'max_val_steps', 'device', 'num_workers'}
        if {k:v for k,v in cfg.items() if k not in allowed} != {k:v for k,v in state['config'].items() if k not in allowed}:
            raise ValueError('Resume requires the same data, calibration and model configuration')
        policy.load_state_dict(state['policy']); optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        if ema_policy is not None:
            ema_policy.load_state_dict(state['ema']); ema.optimization_step = state['global_step']
        torch.set_rng_state(state['torch_rng'].cpu()); generator.set_state(state['loader_rng'].cpu())
        if torch.cuda.is_available() and state.get('cuda_rng'): torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']])
        np.random.set_state(state['numpy_rng']); random.setstate(state['python_rng'])
        start, best, global_step = state['epoch'] + 1, state['best_val'], state['global_step']
    output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(cfg), output / 'config.yaml')
    print(f"{cfg['model']}: {len(train_paths)} train / {len(val_paths)} validation episodes; {len(train_ds)} / {len(val_ds)} windows", flush=True)
    for epoch in range(start, cfg['epochs']):
        begin = time.monotonic(); policy.train()
        losses, metrics = [], {}
        for step, batch in enumerate(train_loader):
            batch = to_device(batch, cfg['device'])
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = _loss(policy, batch, cfg['model'], ema_policy)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg['grad_clip'])
            optimizer.step(); scheduler.step()
            if ema: ema.step(policy)
            global_step += 1; losses.append(loss.item())
            if cfg['max_train_steps'] and step + 1 >= cfg['max_train_steps']: break
        evaluated = ema_policy if ema_policy is not None else policy
        evaluated.eval(); vals = []; squared_error, elements = 0., 0
        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                batch = to_device(batch, cfg['device'])
                loss, _ = _loss(evaluated, batch, cfg['model'], ema_policy)
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite validation loss')
                vals.append(loss.item())
                # Sampling is checked on one validation batch per epoch, in physical action units.
                if step == 0:
                    prediction = evaluated.predict_action(batch['obs'])['action']
                    offset = cfg['n_obs_steps'] - 1
                    target = batch['action'][:, offset:offset + cfg['n_action_steps']]
                    if prediction.shape != target.shape or not torch.isfinite(prediction).all():
                        raise ValueError('Invalid predicted action chunk')
                    squared_error += (prediction - target).square().sum().item(); elements += target.numel()
                if cfg['max_val_steps'] and step + 1 >= cfg['max_val_steps']: break
        val = float(np.mean(vals)); improved = val < best; best = min(val, best)
        record = dict(epoch=epoch, train_loss=float(np.mean(losses)), val_loss=val,
                      sampled_action_mse=squared_error / elements, seconds=time.monotonic() - begin,
                      train_metrics={k:float(v) for k,v in metrics.items()})
        with (output / 'metrics.jsonl').open('a') as f: f.write(json.dumps(record) + '\n')
        state = dict(config=cfg, policy=policy.state_dict(), ema=ema_policy.state_dict() if ema_policy else None,
                     optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), epoch=epoch,
                     global_step=global_step, best_val=best, train_episodes=[str(p) for p in train_paths],
                     val_episodes=[str(p) for p in val_paths], torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                     numpy_rng=np.random.get_state(), python_rng=random.getstate(), loader_rng=generator.get_state())
        temporary = output / 'latest.ckpt.tmp'; torch.save(state, temporary); temporary.replace(latest)
        if improved:
            import shutil
            shutil.copy2(latest, output / 'best.ckpt')
        print(json.dumps(record), flush=True)
    return latest


def main(default): train(load_config(default))


def evaluate_main():
    parser = argparse.ArgumentParser(description='Offline SCARA 3D evaluation; no robot connection')
    parser.add_argument('checkpoint')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--max-batches', type=int, default=10)
    args = parser.parse_args()
    policy, state = load_checkpoint(args.checkpoint, args.device)
    cfg = state['config']
    dataset = PointCloudDataset([Path(p) for p in state['val_episodes']], cfg)
    loader = DataLoader(dataset, batch_size=cfg['val_batch_size'])
    errors = np.zeros(4); count = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = to_device(batch, args.device)
            predicted = policy.predict_action(batch['obs'])['action']
            start = cfg['n_obs_steps'] - 1
            target = batch['action'][:, start:start + cfg['n_action_steps']]
            if predicted.shape != target.shape or not torch.isfinite(predicted).all(): raise ValueError('Invalid policy output')
            errors += (predicted - target).square().sum((0, 1)).cpu().numpy()
            count += predicted.shape[0] * predicted.shape[1]
            if i + 1 >= args.max_batches: break
    print(json.dumps({'evaluated_action_steps': count, 'mse_per_joint': (errors / count).tolist(),
                      'single_episode_test': state['train_episodes'] == state['val_episodes']}))

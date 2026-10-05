"""Read-only LIBERO metadata/partition/config check; never loads MoGe or writes outputs."""
import argparse
from pathlib import Path
import yaml

from moge3_depth.config import load_config, resolve_config_path
from precompute_libero_depth import discover_video_jobs, assign_jobs_greedily


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=root / 'configs/libero_next1.yaml')
    parser.add_argument('--train-config', type=Path, default=root.parents[1] / 'configs/train/libero.yaml')
    args = parser.parse_args()
    config = load_config(args.config)
    train = yaml.safe_load(args.train_config.read_text())['datasets']['vla_data']
    for key in ('python_bin', 'moge_source_dir', 'flex_gemm_source_dir', 'checkpoint', 'dataset_root'):
        path = resolve_config_path(config, config['paths'][key])
        if not path.exists():
            raise FileNotFoundError(f'{key}: {path}')
    expected = {
        'output_root': train['depth_cache_root'],
        'dataset_root': train['data_root_dir'],
    }
    for key, value in expected.items():
        if resolve_config_path(config, config['paths'][key]) != Path(value).expanduser().resolve():
            raise ValueError(f'Teacher {key} does not match training: {value}')
    if config['data']['camera'] != 'observation.images.image':
        raise ValueError('ForeAct main-camera training must use main-camera depth')
    if config['data']['horizon'] != 1 or train['depth_cache_horizon'] != 1 or train['depth_frame_offsets'] != [1]:
        raise ValueError('Both configs must use t+1 depth')
    if config['preprocess']['model_size'] != train['depth_model_size']:
        raise ValueError('Teacher/training depth sizes differ')
    if config['storage']['episodes_per_directory'] != train['depth_episodes_per_directory']:
        raise ValueError('Teacher/training cache chunk sizes differ')
    expected_suites = {f'libero_{name}_no_noops_lerobot' for name in ('10', 'goal', 'spatial', 'object')}
    if set(config['data']['suites']) != expected_suites:
        raise ValueError('Expected all four LIBERO suites')
    gpus = config['distributed']['gpu_ids']
    workers = config['distributed']['num_workers']
    if len(gpus) != workers or len(set(gpus)) != workers:
        raise ValueError('num_workers must match distinct gpu_ids')
    jobs = discover_video_jobs(config)
    episodes = [episode for job in jobs for episode in job.episodes]
    identities = {(episode.suite_name, episode.episode_index) for episode in episodes}
    if len(identities) != len(episodes):
        raise ValueError('Duplicate suite/episode assignment')
    assignments = assign_jobs_greedily(jobs, workers)
    for index, jobs_per_worker in enumerate(assignments):
        print(f'shard={index} gpu={gpus[index]} video_files={len(jobs_per_worker)} episodes={sum(len(job.episodes) for job in jobs_per_worker)} target_frames={sum(job.target_frames for job in jobs_per_worker)}')
    print(f'PASS: {len(episodes)} episodes / {len(jobs)} video files; t+1 and per-suite cache paths match training.')
    print('No GPU/model loaded, no outputs created. Cache existence/completeness and inference were NOT tested.')
    print('Note: inherited teacher uses bicubic for its first resize; ForeAct RGB uses bilinear; see README.')


if __name__ == '__main__':
    main()

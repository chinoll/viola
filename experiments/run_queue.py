"""Download -> mmap -> single-GPU sweeps -> multi-GPU sweeps. No service shutdowns."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent.parent
DATA = Path('/data/datasets/imagenet-256-sd-vae-ft-mse-latents')
OUT = Path('/data/experiments/viola-fp8-muon/dit-b-patch32-standalone')
OUT.mkdir(parents=True, exist_ok=True)


def status(stage, **fields):
    record = dict(stage=stage, updated_at=time.time(), **fields)
    temp = OUT / 'status.tmp'
    temp.write_text(json.dumps(record, indent=2))
    temp.replace(OUT / 'status.json')
    print(json.dumps(record), flush=True)


def free_gpus():
    r = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                        '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True)
    return [int(row[0]) for line in r.stdout.splitlines() if (row := line.split(','))
            and int(row[1]) < 500 and int(row[2]) < 5]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adopt-pid', type=int)
    parser.add_argument('--adopt-run')
    args = parser.parse_args()
    if args.adopt_pid:
        previous = json.loads((OUT / 'status.json').read_text())
        status('training', run=args.adopt_run, gpus=[3], epochs=10, token_budget=16384,
               adopted_pid=args.adopt_pid, next_run='ddp1-tokens16384-bf16',
               training_started_at=previous.get('training_started_at', previous['updated_at']))
        while not (OUT / args.adopt_run / 'complete.json').exists():
            if not Path(f'/proc/{args.adopt_pid}').exists():
                raise RuntimeError('Adopted training exited without complete.json; leaving its results untouched.')
            time.sleep(15)
    while not (DATA / 'download_complete.json').exists():
        pid = int((Path('/data/viola/experiments/download_imagenet.pid')).read_text())
        if not Path(f'/proc/{pid}').exists():
            raise RuntimeError('ImageNet download exited before completion; see download_imagenet.log')
        status('downloading', bytes_on_disk=sum(p.stat().st_size for p in list(DATA.glob('*.parquet')) +
               list(DATA.glob('.cache/huggingface/download/*.incomplete'))))
        time.sleep(30)
    if not (DATA / 'mmap/metadata.json').exists():
        status('preparing_mmap')
        with (OUT / 'prepare.log').open('w') as log:
            subprocess.run([sys.executable, str(ROOT / 'experiments/prepare_latents.py'), '--source', str(DATA)],
                           check=True, stdout=log, stderr=subprocess.STDOUT)
    jobs = [(1, 256, 'fp8'), (1, 256, 'bf16')]
    jobs += [(world, batch, 'fp8') for world in (1, 2) for batch in (256, 128, 64)
             if (world, batch) != (1, 256)]
    for world, batch, momentum_dtype in jobs:
        name = f'ddp{world}-tokens{batch * 64}-' + ('bf16' if momentum_dtype == 'bf16' else 'balanced')
        output = OUT / name
        if (output / 'complete.json').exists():
            continue
        while True:
            available = free_gpus()
            # The single-card phase uses GPU 3, which was free when scheduled.
            if 3 in available and len(available) >= world:
                gpus = [3] + [g for g in available if g != 3][:world - 1]
                break
            status('waiting_for_free_gpus', next_run=name, required_gpus=world, free_gpus=available)
            time.sleep(30)
        status('training', run=name, gpus=gpus, epochs=10, token_budget=batch * 64,
               momentum_dtype=momentum_dtype, training_started_at=time.time())
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str, gpus)), PYTHONPATH=str(ROOT),
                   OMP_NUM_THREADS='4', TRITON_DEFAULT_FP_FUSION='1', TORCHINDUCTOR_COMPILE_THREADS='4')
        argv = [sys.executable, '-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={world}',
                str(ROOT / 'experiments/train_dit.py'), '--data', str(DATA / 'mmap'),
                '--output', str(output), '--epochs', '10', '--batch', str(batch), '--micro-batch', '64',
                '--momentum-dtype', momentum_dtype]
        with (OUT / (name + '.log')).open('a') as log:
            subprocess.run(argv, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        if momentum_dtype == 'bf16':
            status('comparing_fp8_bf16', steps=50040)
            with (OUT / 'comparison.log').open('w') as log:
                subprocess.run([sys.executable, str(ROOT / 'experiments/compare_runs.py'),
                    '--fp8', str(OUT / 'ddp1-tokens16384-balanced'), '--bf16', str(output),
                    '--output', str(OUT / 'fp8-vs-bf16')], cwd=ROOT, env=env,
                    stdout=log, stderr=subprocess.STDOUT, check=True)
    status('complete')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status('failed', error=repr(error))
        raise

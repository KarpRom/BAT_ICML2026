__author__ = "Houtan Ghaffari"
__email__ = "houtan.ghaffari@gmail.com"
__version__ = "1.0.0"

import os
import gc
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple, Optional
import argparse
import contextlib
from data.bat_audio_datasets import SSLAudioSet, SSLDataTransforms
from models import MLR_Student, MLR_Teacher
from .optim import EMA_Scheduler, RiseRunDecay, ema_update
from .utils import infinite_batch_iterator


def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def ssl_train_step(student: torch.nn.Module,
                   teacher: torch.nn.Module,
                   x: torch.Tensor,
                   optimizer: torch.optim.Optimizer,
                   data_transforms: torch.nn.Module,
                   scheduler: RiseRunDecay,
                   ema_scheduler: EMA_Scheduler,
                   clip_norm: Optional[float] = None,
                   mask_ratio: float = 0.8,
                   num_views: int = 16,
                   device: int = 0,
                   accumulation_steps: int = 1,
                   is_accumulating: bool = False) -> Tuple[float, float]:
    """
    Executes a single step of Self-Supervised Learning (SSL) pretraining.
    Computes global and local representation losses between the student and teacher networks,
    performs backpropagation, and updates the teacher via Exponential Moving Average (EMA).
    """
    with torch.no_grad():
        # x_views.shape = (V=batch_size*views, 1, T, F)
        x_views, mask_info = data_transforms(x, num_views, mask_ratio=mask_ratio)
        x_target, _ = data_transforms(x, num_views=1, mask_ratio=0)  # Get clean target

    # we don't need to synchronize devices during gradient accumulation
    sync_context = student.no_sync() if is_accumulating else contextlib.nullcontext()
    with sync_context:
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            student_cls_tokens, student_patch_tokens = student(x_views, mask_info)  # (V, D), (V, N, D)
            with torch.no_grad():
                _, teacher_patch_tokens = teacher(x_target)  # _, (B, N, D)

        # Convert back to float32 for loss computation
        student_cls_tokens = student_cls_tokens.float()
        student_patch_tokens = student_patch_tokens.float()
        mask = mask_info['mask'].float().to(device, non_blocking=True)

        teacher_patch_tokens = teacher_patch_tokens.repeat_interleave(num_views, dim=0).float()
        teacher_cls_tokens = teacher_patch_tokens.mean(dim=1)

        global_loss = (student_cls_tokens - teacher_cls_tokens).pow(2.).mean()
        local_loss = ((student_patch_tokens - teacher_patch_tokens).pow(2.).mean(dim=2) * mask).sum() / mask.sum()

        loss = global_loss + local_loss
        normalized_loss = loss / accumulation_steps
        normalized_loss.backward()

    if not is_accumulating:
        if clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(student.parameters(), clip_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        decay = ema_scheduler.step()
        ema_update(teacher.encoder, student.module.encoder, decay=decay)

    return global_loss.item(), local_loss.item()


def run_ssl_experiment(args: argparse.Namespace) -> None:
    local_rank = setup_ddp()
    is_rank_zero = (local_rank == 0)

    time_stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    log_dir = Path('logs/SSL')

    if is_rank_zero:
        log_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[SSL PRETRAINING] Starting distributed training. World size: {dist.get_world_size()}")
        history_save_path = log_dir / f'BAT_history_{time_stamp}.csv'
        state_save_path = f'BAT_state_{time_stamp}.pt'

    # Data
    # Matches both extensions: the Cuts_Database smoke-test corpus is .wav, the real
    # bee corpus built by detection/build_ssl_dataset.py is .flac. librosa.load()
    # (used in SSLAudioSet, via its soundfile backend) is format-agnostic either way.
    dataset_dir = Path(args.dataset_dir)
    ssl_data = np.array(
        [f.as_posix() for ext in ("*.wav", "*.flac") for f in dataset_dir.glob(f"**/{ext}")]
    ).astype(np.bytes_)

    train_dataset = SSLAudioSet(ssl_data, sr=args.sr)

    sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(train_dataset, sampler=sampler, batch_size=args.batch_size, num_workers=args.num_workers,
                              pin_memory=False, collate_fn=train_dataset.collate_fn, persistent_workers=True,
                              drop_last=True)

    infinite_loader = infinite_batch_iterator(train_loader)

    data_transforms = SSLDataTransforms(sr=args.sr, n_fft=args.n_fft, hop_length=args.hop_length,
                                        n_mels=args.n_mels, time_frame_size=args.time_frame_size,
                                        num_time_patches=args.time_frame_size // args.patch_size,
                                        num_freq_patches=args.n_mels // args.patch_size).to(local_rank)
    # Model
    student = MLR_Student(input_shape=(args.time_frame_size, args.n_mels),
                          patch_size=(args.patch_size, args.patch_size),
                          dim=args.dim, depth=args.depth, num_heads=args.num_heads, mlp_ratio=args.mlp_ratio,
                          pos_trainable=False, layer_norm_first=False, pre_norm=True, use_gate=True,
                          decoder_depth=args.decoder_depth, decoder_heads=args.decoder_heads,
                          decoder_mlp_ratio=args.decoder_mlp_ratio, decoder_dim=args.decoder_dim).to(local_rank)

    resume_ckpt = None
    resume_step = 0
    if args.resume_checkpoint:
        # Continue a run that was cut off (e.g. by a SLURM time limit) from one of this
        # script's own checkpoints -- a different, richer format than --init-checkpoint's
        # (BAT_base.pt is a flat "student.encoder."-prefixed dict with no optimizer/
        # scheduler/step state; ours below is {'encoder', 'decoder', 'ema_encoder',
        # 'optimizer', 'scheduler', 'ema_scheduler_counter', 'optimized_steps', 'args'}).
        # Takes priority over --init-checkpoint: warm-starting from BAT_base.pt again on
        # top of an already-continued-pretrained encoder would throw away that progress.
        if args.init_checkpoint and is_rank_zero:
            print(f"--resume-checkpoint given; ignoring --init-checkpoint ({args.init_checkpoint}).")
        resume_ckpt = torch.load(args.resume_checkpoint, map_location=f'cuda:{local_rank}')
        student.encoder.load_state_dict(resume_ckpt['encoder'])
        student.decoder.load_state_dict(resume_ckpt['decoder'])
        resume_step = resume_ckpt['optimized_steps']
        if is_rank_zero:
            print(f"Resuming from {args.resume_checkpoint} at step {resume_step}")
    elif args.init_checkpoint:
        # Warm-start continued pretraining from a released checkpoint (e.g. BAT_base.pt)
        # instead of the random init above. Released BAT checkpoints are a flat state
        # dict of the *whole* MLR_Student (encoder + decoder), with keys prefixed
        # "student.encoder."/"student.decoder." -- verified directly against
        # BAT_base.pt: 174 "student.encoder.*" keys, loading with that prefix stripped
        # into student.encoder gives zero missing/unexpected keys. Only the encoder is
        # loaded (the decoder is SSL-pretraining-specific scaffolding, not needed for
        # downstream use, and continued pretraining rebuilds it from scratch here); the
        # teacher's encoder is then copied from this warm-started student encoder by
        # the existing line below, same as the from-scratch path.
        state = torch.load(args.init_checkpoint, map_location=f'cuda:{local_rank}')
        prefix = 'student.encoder.'
        encoder_state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        missing, unexpected = student.encoder.load_state_dict(encoder_state, strict=False)
        assert not missing and not unexpected, (
            f"init_checkpoint mismatch: missing={missing} unexpected={unexpected}"
        )
        if is_rank_zero:
            print(f"Warm-started student encoder from {args.init_checkpoint}")

    teacher = MLR_Teacher(input_shape=(args.time_frame_size, args.n_mels),
                          patch_size=(args.patch_size, args.patch_size),
                          dim=args.dim, depth=args.depth, num_heads=args.num_heads, mlp_ratio=args.mlp_ratio,
                          pos_trainable=False, layer_norm_first=False, pre_norm=True, use_gate=True,
                          instance_norm_target_layer=True, layer_norm_targets=True).to(local_rank)

    if resume_ckpt is not None:
        # Unlike the from-scratch/warm-start paths, the teacher must NOT be copied from
        # the student here -- after any amount of training the EMA teacher has diverged
        # from the student (that divergence is the whole point), so it's restored from
        # its own saved state instead.
        teacher.encoder.load_state_dict(resume_ckpt['ema_encoder'])
    else:
        teacher.encoder.load_state_dict(student.encoder.state_dict())
    teacher.requires_grad_(False)

    print(f'Student params: {sum(p.numel() for p in student.parameters()):_}')
    print(f'Teacher params: {sum(p.numel() for p in teacher.parameters()):_}')

    if args.compile:
        student.compile(mode='default')
        teacher.compile(mode='default')

    student = DDP(student, device_ids=[local_rank], output_device=local_rank)

    ema_scheduler = EMA_Scheduler(decay_start=args.ema_decay_start, decay_end=args.ema_decay_end,
                                  ema_warmup_steps=args.ema_warmup_steps)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scheduler = RiseRunDecay(optimizer, warmup_steps=args.lr_warmup_steps, constant_steps=0,
                             total_steps=args.optimization_steps, min_lr=args.min_lr)

    if resume_ckpt is not None:
        if resume_ckpt['args'].get('grad_accumulation_steps') != args.grad_accumulation_steps:
            print(
                f"[Warning] grad_accumulation_steps changed since this checkpoint was saved "
                f"({resume_ckpt['args'].get('grad_accumulation_steps')} -> {args.grad_accumulation_steps}); "
                "the resumed step count may no longer land on a clean accumulation boundary."
            )
        optimizer.load_state_dict(resume_ckpt['optimizer'])
        scheduler.load_state_dict(resume_ckpt['scheduler'])
        ema_scheduler.counter = resume_ckpt['ema_scheduler_counter']
        del resume_ckpt

    # Training Loop
    total_forward_passes = args.optimization_steps * args.grad_accumulation_steps
    start_forward_pass = resume_step * args.grad_accumulation_steps
    history = {'global_loss': [], 'local_loss': []}
    save_freq = args.save_interval if args.save_interval else args.optimization_steps
    pbar = tqdm(total=args.optimization_steps, initial=resume_step, desc="SSL Pretraining", colour='#87ceeb',
                disable=not is_rank_zero)
    temp_g_loss, temp_l_loss = 0.0, 0.0

    for step in range(start_forward_pass, total_forward_passes):
        x = next(infinite_loader)
        x = x.to(local_rank, non_blocking=True)
        is_accumulating = (step + 1) % args.grad_accumulation_steps != 0

        g_loss, l_loss = ssl_train_step(
            student=student, teacher=teacher, x=x, optimizer=optimizer, data_transforms=data_transforms,
            scheduler=scheduler, ema_scheduler=ema_scheduler, clip_norm=args.clip_norm,
            mask_ratio=args.mask_ratio, num_views=args.num_views, device=local_rank,
            accumulation_steps=args.grad_accumulation_steps, is_accumulating=is_accumulating
        )

        temp_g_loss += g_loss
        temp_l_loss += l_loss

        if not is_accumulating:
            pbar.update(1)

            temp_g_loss = temp_g_loss / args.grad_accumulation_steps
            temp_l_loss = temp_l_loss / args.grad_accumulation_steps

            if is_rank_zero:
                history['global_loss'].append(temp_g_loss)
                history['local_loss'].append(temp_l_loss)

                pbar.set_description(
                    f"Global: {temp_g_loss:.4f} | Local: {temp_l_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

                temp_g_loss, temp_l_loss = 0.0, 0.0

                if pbar.n % save_freq == 0 or pbar.n == args.optimization_steps:
                    current_save_path = log_dir / f'step({pbar.n})_{state_save_path}'
                    torch.save({
                        'encoder': student.module.encoder.state_dict(),
                        'decoder': student.module.decoder.state_dict(),
                        'ema_encoder': teacher.encoder.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'ema_scheduler_counter': ema_scheduler.counter,
                        'optimized_steps': pbar.n,
                        'args': vars(args)
                    }, current_save_path)

                    pd.DataFrame(history).to_csv(history_save_path, index=False)

    if is_rank_zero:
        pbar.close()

    # Cleanup
    torch._dynamo.reset()
    if hasattr(train_loader, '_iterator'):
        del train_loader._iterator
    del infinite_loader, train_loader, student, teacher, optimizer, scheduler, x
    gc.collect()
    torch.cuda.empty_cache()
    cleanup_ddp()


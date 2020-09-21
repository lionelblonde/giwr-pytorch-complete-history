#!/usr/bin/env bash

python main.py \
    --no-cuda \
    --no-render \
    --no-record \
    --task=generate \
    --save_frequency=1000 \
    --num_timesteps=10000000 \
    --training_steps_per_iter=2 \
    --eval_steps_per_iter=10 \
    --eval_frequency=10 \
    --layer_norm \
    --actor_lr=0.00025 \
    --critic_lr=0.00025 \
    --no-with_scheduler \
    --clip_norm=40.0 \
    --wd_scale=0.0 \
    --rollout_len=2 \
    --batch_size=64 \
    --gamma=0.99 \
    --mem_size=100000 \
    --noise_type="normal_0.3" \
    --pn_adapt_frequency=50 \
    --polyak=0.005 \
    --targ_up_freq=100 \
    --n_step_returns \
    --lookahead=10 \
    --no-ret_norm \
    --no-popart \
    --no-clipped_double \
    --no-targ_actor_smoothing \
    --td3_std=0.2 \
    --td3_c=0.5 \
    --actor_update_delay=1 \
    --no-prioritized_replay \
    --alpha=0.3 \
    --beta=1.0 \
    --no-ranked \
    --no-unreal \
    --no-use_c51 \
    --no-use_qr \
    --c51_num_atoms=21 \
    --c51_vmin=-10.0 \
    --c51_vmax=10.0 \
    --num_tau=200 \
    --env_id=Hopper-v3 \
    --seed=0 \
    --iter_num=116_timeout \
    --model_path="/Users/lionelblonde/Code/offline-batch-rl-pytorch/data/checkpoints/teeca_bezoo_puwu.gitSHA_0dd6aec.Hopper-v3.seed02"
```
python train_start.py --actor_critic_share_weights=True --batch_size=16384 --name=POMAPF-v0 --exploration_loss_coeff=0.023 --gamma=0.9756 --hidden_size=512  --learning_rate=0.00022 --lr_schedule=constant  --num_workers=8 --optimizer=adam --ppo_clip_ratio=0.2   --train_for_env_steps=10 --use_rnn=True
```
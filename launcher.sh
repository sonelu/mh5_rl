python trainer.py \
  trainer.num_epochs=4 \
  trainer.total_frames=100000000 \
  trainer.save_trainer_interval=100000 \
  trainer.progress_bar=false \
  optimizer.lr=1e-5 \
  rewards.weights.k_falling=-500 \
  rewards.weights.k_forward=1.25 \
& disown;

python trainer.py \
  trainer.num_epochs=4 \
  trainer.total_frames=100000000 \
  trainer.save_trainer_interval=100000 \
  trainer.progress_bar=false \
  optimizer.lr=1e-5 \
  rewards.weights.k_falling=-200 \
  rewards.weights.k_forward=1.25 \
& disown;

python trainer.py \
  trainer.num_epochs=4 \
  trainer.total_frames=100000000 \
  trainer.save_trainer_interval=100000 \
  trainer.progress_bar=true \
  optimizer.lr=1e-5 \
  rewards.weights.k_falling=-100 \
  rewards.weights.k_forward=1.25 \
& disown;

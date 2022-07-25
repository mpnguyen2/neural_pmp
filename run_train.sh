source ~/anaconda3/etc/profile.d/conda.sh
conda activate NeuralPMP

python train.py cartpole --num_episodes_hnet 1024 --num_episodes_adj 256 --num_hnet_train_max 1000000 --num_adj_train_max 2000 --stop_train_condition 0.005 --num_train 0 --num_warmup 1
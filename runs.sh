# Train
CUDA_VISIBLE_DEVICES=0 python3 -m config_rnn.train --config_name dn2_mnist_even_gp
CUDA_VISIBLE_DEVICES=0 python3 -m config_rnn.train --config_name dn2_mnist_even_gp_tempered
CUDA_VISIBLE_DEVICES=0 python3 -m config_rnn.train --config_name bn2_omniglot_gp

# Generate test samples
CUDA_VISIBLE_DEVICES=0 python3 -m config_rnn.test_samples  --config_name dn2_mnist_even_gp_tempered

# Few shot classification
CUDA_VISIBLE_DEVICES=0 python3 -m config_rnn.test_few_shot_omniglot --config_name bn2_omniglot_gp --seq_len 2 --batch_size 20
CUDA_VISIBLE_DEVICES=0 python3 -m config_rnn.test_few_shot_omniglot --config_name bn2_omniglot_gp_tempered --seq_len 2 --batch_size 20

CUDA_VISIBLE_DEVICES=0 python3 -m config_rnn.test_few_shot_omniglot --config_name bn2_omniglot_tp --seq_len 2 --batch_size 20
CUDA_VISIBLE_DEVICES=0 python3 -m config_rnn.test_few_shot_omniglot --config_name bn2_omniglot_tp_ft_1s_20w --seq_len 2 --batch_size 20

# Fine tuning
CUDA_VISIBLE_DEVICES=0 python3 -m config_rnn.train_finetune  --config_name bn2_omniglot_tp_ft_1s_20w

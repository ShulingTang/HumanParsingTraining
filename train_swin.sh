# base
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.launch --nproc_per_node 8 train_swin_cdg_mutil.py --arch swin_cdg \
--imagenet-pretrain /home/ubuntu/project/HumanParsingTraining/logs/swin_cdg_pretrained_19/checkpoint_230.pth.tar \
--data-dir /workspace/data/full_24 \
--num-classes 24 \
--batch-size 12  \
--learning-rate 7e-3 \
--weight-decay 0 \
--optimizer sgd \
--syncbn \
--lr-divider 500 \
--cyclelr-divider 2 \
--warmup-epochs 15 \
--epochs 120 \
--schp-start 105 \
--cycle-epochs 5 \
--input-size 1024,768 \
--log-dir ./logs/swin_cdg_24 \
--model-restore ./logs/swin_cdg_24/checkpoint.pth.tar

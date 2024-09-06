# base
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.launch --nproc_per_node 8 train_swin_cdg_mutil.py --arch swin_cdg \
--imagenet-pretrain ./pretrained/solider_swin_base.pth \
--data-dir /workspace/data/full_19 \
--num-classes 19 \
--batch-size 12  \
--learning-rate 7e-3 \
--weight-decay 0 \
--optimizer sgd \
--syncbn \
--lr-divider 200 \
--cyclelr-divider 2 \
--warmup-epochs 30 \
--epochs 250 \
--schp-start 210 \
--cycle-epochs 20 \
--input-size 1024,768 \
--log-dir ./logs/swin_cdg_pretrained_19 \
--model-restore ./logs/swin_cdg_pretrained_19/checkpoint_125.pth.tar \
--using-bf16 True

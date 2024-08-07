name='lip_solider_swin_tiny'
python evaluate.py --arch swin_small \
 --data-dir /home/xianzhe.xxz/datasets/HumanParsing/LIP \
 --model-restore ./logs/${name}/schp_4_checkpoint.pth.tar \
 --input-size 512,512 \
 --multi-scales 0.5,0.75,1.0,1.25,1.5 \
 --flip

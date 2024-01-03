

sintr -A MLMI-wa285-SL2-GPU -p ampere --gres=gpu:1 -N1 -n1 -t 1:0:0 --qos=INTR

#request cpu
sintr -A MLMI-wa285-SL2-CPU -p icelake -N1 -n1 -t 0:20:0 --qos=INTR

source /rds/project/rds-xyBFuSj0hm0/MLMI2.M2023/miniconda3/bin/activate

#start jupyter notebook
jupyter notebook --no-browser --ip=* --port=8084

ssh -L 8113:gpu-q-1:8113 -fN wa285@login-p-4.hpc.cam.ac.uk


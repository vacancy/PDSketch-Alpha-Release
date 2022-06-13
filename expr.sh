#! /bin/bash
# rm experiments.txt

jac-run trainval-babyai-mp.py goto-single --use-offline=yes --structure-mode $1 --action-loss-weight 1 --evaluate-interval 0 --iterations 5000 --append-expr
jac-run trainval-babyai-mp.py goto        --use-offline=yes --structure-mode $1 --action-loss-weight 1 --evaluate-interval 0 --iterations 5000 --append-expr --load dumps/$1-goto-single-load=scratch.pth
jac-run trainval-babyai-mp.py goto2       --use-offline=yes --structure-mode $1 --action-loss-weight 1 --evaluate-interval 0 --iterations 5000 --append-expr --load dumps/$1-goto-single-load=scratch.pth

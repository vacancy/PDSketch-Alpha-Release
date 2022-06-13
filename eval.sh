#! /bin/bash
set -x

jac-run trainval-babyai-mp.py goto2      --use-offline=yes --structure-mode $1 --action-loss-weight 1 --evaluate-interval 0 --evaluate --append-result --load dumps/$1-goto2-load=gotosingle.pth
jac-run trainval-babyai-mp.py goto2      --use-offline=yes --structure-mode $1 --action-loss-weight 1 --evaluate-interval 0 --evaluate --append-result --load dumps/$1-goto2-load=gotosingle.pth --evaluate-objects 8
jac-run trainval-babyai-mp.py goto2      --use-offline=yes --structure-mode $1 --action-loss-weight 1 --evaluate-interval 0 --evaluate --append-result --load dumps/$1-goto2-load=gotosingle.pth --discretize
jac-run trainval-babyai-mp.py goto2      --use-offline=yes --structure-mode $1 --action-loss-weight 1 --evaluate-interval 0 --evaluate --append-result --load dumps/$1-goto2-load=gotosingle.pth --evaluate-objects 8 --discretize

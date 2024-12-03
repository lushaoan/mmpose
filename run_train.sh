pip install xtcocotools
pip install munkres

CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh ./configs/barcode/exp1.py 4 --work-dir ./work_dirs/exp1/

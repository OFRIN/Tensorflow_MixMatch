
## 1. fully supervised learning
python Train_with_FSL.py --use_gpu 0 --labels all
python Train_with_FSL.py --use_gpu 0 --labels 4000

## 2. MixMatch
python Train_with_MixMatch.py --use_gpu 2 --labels 4000 
python Train_with_MixMatch.py --use_gpu 3 --labels 250


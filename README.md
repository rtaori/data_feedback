# Data Feedback Loops

This repository contains code for the paper
[Data Feedback Loops: Model-driven Amplification of Dataset Biases](https://arxiv.org/abs/2209.03942).

## Installation

Python 3.9 and PyTorch installation are the most important. All dependencies are marked in `requirements.txt`, installable in a **Python 3.9** environment via: 
```
pip install -r requirements.txt
```

## Downloading and preprocessing data

To automatically download and preprocess all necessary data, run:
```
cd data && ./download_preprocess_all.sh
```
Or, you can download each dataset individually with the provided scripts in the [data](data) directory. 

Note that downloading cifar5m requires the `gsutil` utility. Also note that cinic10 needs to be preprocessed before use; if it's already downloaded, you can run the preprocess script on your download:
```
cd data && python preprocess_cinic10.py --data-dir path/to/your/install
```

## Reproducing main experiments

In the paper, each experiment was run 3 times. Results are logged to wandb for plotting into figures. All programs are meant to be run with 1 GPU only.

Code for image classification experiments (Section 5.1) is in [image_classification](image_classification), code for visual role-labeing experiments (Section 5.2) is in [visual_role_labeling](visual_role_labeling), and code for conditional language generation experiments (Section 5.3) is in [language_generation](language_generation). 

Figure 3 top left:
```
python image_classification/train.py -m 1000 -k 4000 --wandb-group fig3-top --wandb-log
```

Figure 3 top right:
```
python image_classification/train.py -m 2500 -k 2500 --wandb-group fig3-top --wandb-log
```

Figure 3 bottom left:
```
python image_classification/train.py -m 1000 -k 4000 --subsample-train-set-each-round --wandb-group fig3-bottom --wandb-log
```

Figure 3 bottom right:
```
python image_classification/train.py -m 2500 -k 2500 --subsample-train-set-each-round --wandb-group fig3-bottom --wandb-log
```

Figure 4 left:
```
python visual_role_labeling/train.py -m 1000 -k 4000 --wandb-group fig4 --wandb-log
```

Figure 4 right:
```
python visual_role_labeling/train.py -m 2500 -k 2500 --wandb-group fig4 --wandb-log
```

Figure 5 left:
```
python language_generation/train.py -m 1000 -k 4000 --sampling-type nucleus_sampling --wandb-group fig5 --wandb-log
```

Figure 5 right:
```
python language_generation/train.py -m 1000 -k 4000 --sampling-type beam_search --wandb-group fig5 --wandb-log
```

Figure 6 red line (blue & yellow lines are the same as in Figure 5):
```
python language_generation/train.py -m 1000 -k 4000 --sampling-type beam_search --num-train-epochs 5 --learning-rate 5e-4 --wandb-group fig6 --wandb-log
```

## Plotting

All the plotting code is in the [plotting](plotting) directory. The code to reproduce the plots for each figure is in the `plotting/paper_plots.ipynb` notebook. Note that you will first need to run the experiments you want to plot so that data is stored in wandb for the plotting notebook to use.

Theorem 1 upper bound calculations from the paper are implemented in `plotting/feedback_bound.py`. 

## Reproducing appendix experiments

Figure 7 is automatically logged with Figures 3 top left and top right.

Figures 8, 15, 16, and 17 are automatically logged with Figure 4.

Figure 9 left:
```
python image_classification/train.py -m 1000 -k 4000 -a resnet18 --batch-size 128 --wandb-group fig9 --wandb-log
```

Figure 9 right:
```
python image_classification/train.py -m 2500 -k 2500 -a resnet18 --batch-size 128 --wandb-group fig9 --wandb-log
```

Figure 10 left:
```
python image_classification/train.py -d cinic10 -n 20000 -m 1000 -k 4000 --class-imbalance-factor 2 --wandb-group fig9 --wandb-log
```

Figure 10 right:
```
python image_classification/train.py -d cinic10 -n 20000 -m 2500 -k 2500 --class-imbalance-factor 2 --wandb-group fig9 --wandb-log
```

Figure 11 left:
```
python image_classification/train.py -m 1000 -k 4000 --class-imbalance-factor 2 --wandb-group fig11 --wandb-log
```

Figure 11 right:
```
python image_classification/train.py -m 2500 -k 2500 --class-imbalance-factor 2 --wandb-group fig11 --wandb-log
```

Figure 12 left:
```
python image_classification/train.py -m 1000 -k 4000 --class-imbalance-class ship --wandb-group fig12 --wandb-log
```

Figure 12 right:
```
python image_classification/train.py -m 2500 -k 2500 --class-imbalance-class ship --wandb-group fig12 --wandb-log
```

Figure 13 left:
```
python image_classification/train.py -m 1000 -k 4000 --underfit-model --wandb-group fig13 --wandb-log
```

Figure 13 right:
```
python image_classification/train.py -m 2500 -k 2500  --underfit-model --wandb-group fig13 --wandb-log
```

Figure 14 left:
```
python image_classification/train.py -n 20000 -m 1000 -k 4000 --wandb-group fig14 --wandb-log
```

Figure 14 right:
```
python image_classification/train.py -n 20000 -m 2500 -k 2500 --wandb-group fig14 --wandb-log
```

Figure 18 left:
```
python language_generation/train.py -m 2500 -k 2500 --sampling-type nucleus_sampling --wandb-group fig18 --wandb-log
```

Figure 18 right:
```
python language_generation/train.py -m 2500 -k 2500 --sampling-type beam_search --wandb-group fig18 --wandb-log
```

Figure 19 left:
```
python language_generation/train.py -m 1000 -k 4000 --sampling-type nucleus_sampling -a gpt2-medium --wandb-group fig19 --wandb-log
```

Figure 19 right:
```
python language_generation/train.py -m 1000 -k 4000 --sampling-type beam_search -a gpt2-medium --wandb-group fig19 --wandb-log
```

Figure 20 left:
```
python language_generation/train.py -m 1000 -k 4000 --sampling-type nucleus_sampling -a gpt2-large --wandb-group fig20 --wandb-log
```

Figure 20 right:
```
python language_generation/train.py -m 1000 -k 4000 --sampling-type beam_search -a gpt2-large --wandb-group fig20 --wandb-log
```

## Citation

```
@article{taori2022data,
    title={Data Feedback Loops: Model-driven Amplification of Dataset Biases},
    author={Rohan Taori and Tatsunori Hashimoto},
    url={https://arxiv.org/abs/2209.03942},
    year={2022},
}
```

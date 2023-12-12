# AI-SAM
Automatic and Interactive Segment Anything Model. When training, this method learn to generate the segmentation masks for each class as well as the corresponing point prompts using only the segmentation masks as the targets. During inference time, AI-SAM automatically generates a set of point prompts and the segmentation masks for each class. To modify the segmentation masks, the user can modify the point prompts. The overall pipeline is below:
![ai-sam](./assets/ai-sam.png) 

## Installation
The code requires `python>=3.8`, `pytorch>=1.7`, and `torchvision>=0.8`.

You will also need the following packages.
```
scipy
scikit-learn
scikit-image
opencv-python
matplotlib
ipywidgets
notebook
```

## Automatic Evaluation

### ACDC
Prepare the dataset following [MT-UNet](https://github.com/Dootmaan/MT-UNet). Then, download the pretrained [weight](https://pennstateoffice365-my.sharepoint.com/:u:/g/personal/ymp5078_psu_edu/EVHZbBpy07RIr_0ABr3AedgBlgR5wTbgQ8SmDz_9f3n4nA?e=RPn9Mv). Finally, you may run the following code to obtain the scores in the paper:
```sh
python eval_one_gpu.py --dataset acdc --use_amp -checkpoint [path-to-the-downloaded-weight] -model_type vit_h --tr_path [path-to-the-dataset-dir] --use_classification_head --use_lora --use_hard_point
```

#### Synapse
Prepare the dataset following [TransUNet](https://github.com/Beckschen/TransUNet/tree/main). Then, download the pretrained [weight](https://pennstateoffice365-my.sharepoint.com/:u:/g/personal/ymp5078_psu_edu/EdAjEX2E5hNFg8e7t7YetZEBUCGQmfiLN3V1eSiDzvao2A?e=Mw80aw). Finally, you may run the following code to obtain the scores in the paper:
```sh
python eval_one_gpu.py --dataset synapse --use_amp -checkpoint [path-to-the-downloaded-weight] -model_type vit_h --tr_path [path-to-the-dataset-dir] --use_classification_head --use_lora --use_hard_point
```

## Automatic and Interactive Demo
Refer to [this notebook](automatic_interactive_demo.ipynb) for detail. AI-SAM will first generate a set of foreground and background points base on the class of choice and the user can modify the points base on the segmentation result.

## TODO
1. Add code for natural images.

## License

This work is licensed under [Apache 2.0 license](LICENSE).

## Citations
If you find this work useful, please cite:
```
@article{pan2023ai,
  title={AI-SAM: Automatic and Interactive Segment Anything Model},
  author={Pan, Yimu and Zhang, Sitao and Gernand, Alison D and Goldstein, Jeffery A and Wang, James Z},
  journal={arXiv preprint arXiv:2312.03119},
  year={2023}
}
```



## Acknowledgements
The code is modified from [MedSAM](https://github.com/bowang-lab/MedSAM/tree/main) and [SAM](https://github.com/facebookresearch/segment-anything). We also used the LoRA implementation from [SAMed](https://github.com/hitachinsk/SAMed/tree/main).

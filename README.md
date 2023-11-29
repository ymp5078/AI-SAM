# AI-SAM
Automatic and Interactive Segment Anything Model. When training, this method learn to generate the segmentation masks for each class as well as the corresponing point prompts using only the segmentation masks as the targets. During inference time, AI-SAM automatically generates a set of point prompts and the segmentation masks for each class. To modify the segmentation masks, the user can modify the point prompts. The overall pipeline is below:
![ai-sam](./assets/ai-sam-main.pdf) 


## Evaluation

### ACDC
Prepare the dataset following [MT-UNet](https://github.com/Dootmaan/MT-UNet). Then, download the pretrained [weight](.). Finally, you may run the following code to obtain the scores in the paper:
```sh
python eval_one_gpu.py --dataset acdc --use_amp -checkpoint [path-to-the-downloaded-weight] -model_type vit_h --tr_path [path-to-the-dataset-dir] --use_classification_head --use_lora --use_hard_point
```

## Synapse
Prepare the dataset following [TransUNet](https://github.com/Beckschen/TransUNet/tree/main). Then, download the pretrained [weight](.). Finally, you may run the following code to obtain the scores in the paper:
```sh
python eval_one_gpu.py --dataset synapse --use_amp -checkpoint [path-to-the-downloaded-weight] -model_type vit_h --tr_path [path-to-the-dataset-dir] --use_classification_head --use_lora --use_hard_point
```

## TODO
1. Add automatic prompt in the interactive demo.
2. Add code for natural images.

## Citations
If you find this work useful, please cite:


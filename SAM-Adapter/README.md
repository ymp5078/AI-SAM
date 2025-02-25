## Using AI-SAM with SAM-adapter

Please refer to the [original implementation](https://github.com/tianrun-chen/SAM-Adapter-PyTorch) to set up this repo.

## Usage

You can download our pre-trained weights [here](https://pennstateoffice365-my.sharepoint.com/:f:/g/personal/ymp5078_psu_edu/Ehx4vsJMzS5NtaWJtDMfXAYBmLn5Ah1PvqCG8-FClPXM6Q?e=WiDqTA).

# Trainig
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes 1 --nproc_per_node 4 train.py --tag autoprompt_freeze_20 --config ./configs/cod-sam-vit-h-ai-sam.yaml
```

# Evaluation
Automatic evaluation, plot points
```bash
python test.py --config <checkpoint-path>/config.yaml --model <checkpoint-path>/model_epoch_best.pth
```

Interactive evaluation, plot bbox and points
```bash
python test_interactive.py --config <checkpoint-path>/config.yaml --model <checkpoint-path>/model_epoch_last.pth --use_bbox --save_path <save-dir> --return_points
```

Interactive evaluation, plot bbox
```bash
python test_interactive.py --config <checkpoint-path>/config.yaml --model <checkpoint-path>/model_epoch_last.pth --use_bbox --save_path <save-dir>
```

Interactive evaluation for SAM, plot bbox
```bash
python test_interactive.py --config <checkpoint-path>/config.yaml --model <checkpoint-path>/sam_vit_h_4b8939.pth --use_bbox --save_path  <save-dir> --use_base_sam

```

To train and test shadow detection, use shadow-sam-vit-h-ai-sam.yaml. If you need to test different dataset, just modify the dataset path in the config file.






# weldseg

## Welding Segmentation 

For segmentation training see `yolo8` notebook.

## Synthetic data generation

SD utils aggregator: https://github.com/LykosAI/StabilityMatrix

For train, install the kohya_ss package. Config for sdxl in this repo, for sd15 can be used by default. Just change the data source paths and play with lr and batch size, and it should be fine.
You can also try the new OneTrainer utility instead of kohya_ss.

For generation, install any UI with Lora support. 
Simple workflow provided for ComfyUI. Just drag&drop it into the webUI workspace.

Models weights here: https://huggingface.co/Mangaboba/LaserSeamMacrograph

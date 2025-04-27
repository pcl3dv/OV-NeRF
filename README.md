<div align="center">

## OV-NeRF: Open-vocabulary Neural Radiance Fields with Vision and Language Foundation Models for 3D Semantic Understanding 
Guibiao Liao<sup>1,2</sup>, Kaichen Zhou<sup>3</sup>, Zhenyu Bao<sup>1,2</sup>, Kanglin Liu<sup>2, *</sup>, Qing Li<sup>2, *</sup>

<sup>1</sup>Peking University    <sup>2</sup>Pengcheng Laboratory    <sup>3</sup>University of Oxford

<sup>*</sup>Corresponding author

### [Paper](https://ieeexplore.ieee.org/document/10630553)

</div>

![Image](https://github.com/pcl3dv/OV-NeRF/blob/main/images/pipeline.png)


**Abstract**: The development of Neural Radiance Fields (NeRFs) has provided a potent representation for encapsulating the geometric and appearance characteristics of 3D scenes. Enhancing the capabilities of NeRFs in open-vocabulary 3D semantic perception tasks has been a recent focus. However, current methods that extract semantics directly from Contrastive Language-Image Pretraining (CLIP) for semantic field learning encounter difficulties due to noisy and view-inconsistent semantics provided by CLIP. To tackle these limitations, we propose OV-NeRF, which exploits the potential of pre-trained vision and language foundation models to enhance semantic field learning through proposed single-view and cross-view strategies. First, from the *single-view* perspective, we introduce Region Semantic Ranking (RSR) regularization by leveraging 2D mask proposals derived from Segment Anything (SAM) to rectify the noisy semantics of each training view, facilitating accurate semantic field learning. Second, from the *cross-view* perspective, we propose a Cross-view Self-enhancement (CSE) strategy to address the challenge raised by view-inconsistent semantics. Rather than invariably utilizing the 2D inconsistent semantics from CLIP, CSE leverages the 3D consistent semantics generated from the well-trained semantic field itself for semantic field training, aiming to reduce ambiguity and enhance overall semantic consistency across different views. Extensive experiments validate our OV-NeRF outperforms current state-of-the-art methods, achieving a significant improvement of 20.31\% and 18.42\% in mIoU metric on Replica and Scannet, respectively. Furthermore, our approach exhibits consistent superior results across various CLIP configurations, further verifying its robustness. 


## Qualitative Result
### Replica
<img src="https://github.com/pcl3dv/OV-NeRF/blob/main/images/result_replica.png">

### ScanNet
<img src="https://github.com/pcl3dv/OV-NeRF/blob/main/images/result_scannet.png">

### 3DOVS
<img src="https://github.com/pcl3dv/OV-NeRF/blob/main/images/result_3dovs.png">


## Quantitative Result 
### Replica
<div align="center">
<img src="https://github.com/pcl3dv/OV-NeRF/blob/main/images/quantitative_result_replica.png" width="500" >
</div>

### ScanNet
<div align="center">
<img src="https://github.com/pcl3dv/OV-NeRF/blob/main/images/quantitative_result_scannet.png" width="800">
</div>

### 3DOVS
<div align="center">
<img src="https://github.com/pcl3dv/OV-NeRF/blob/main/images/quantitative_result_3dovs.png" width="800">
</div>


## Installation
> Tested on Ubuntu 18.04 + Pytorch 1.12.1+cu116

On default, run the following commands to install the relative packages
```
conda create -n ovnerf python=3.9
conda activate ovnerf
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install ftfy regex tqdm scikit-image opencv-python configargparse lpips imageio-ffmpeg kornia tensorboard
pip install git+https://github.com/openai/CLIP.git
pip install git+https://github.com/facebookresearch/segment-anything.git
```

<!-- ## Datasets -->

<!-- ## Data Preparation -->



## Training
### 1. Train original TensoRF
This step is for reconstructing the TensoRF for the scenes. Please modify the `datadir` and `expname` in `configs/resonstruction.txt` to specify the dataset path and the experiment name. By default we set `datadir` to `data/$scene_name` and `expname` as `$scene_name`. You can then train the original TensoRF by:
```
bash script/reconstruction.sh [GPU_ID]
```
The reconstructed TensoRF will be saved in `log/$dataset/$scene_name`.


### 2. Train segmentation 
We provide the training script for our datasets under `configs` as `$scene_name.txt`. You can train the segmentation by:
```
bash scripts/segmentation.sh [CONFIG_FILE] [GPU_ID] 
```
The trained model will be saved in `log_seg/$dataset/$scene_name`. 

### 3. Evaluate reconstruction
```
bash script/test_reconstruction.sh
```

### 4. Evaluate segmentation
```
bash script/test_segmentation.sh
```


## TODO list
- [x] release the code of the training
- [x] release the code of the evaluation
- [x] update the arxiv link
- [ ] release the code of preprocessing
- [ ] release the preprocessed dataset
- [ ] release the the pretrained model


## Acknowledgements
Some codes are borrowed from [TensoRF](https://github.com/apchenstu/TensoRF), [SAM](https://github.com/facebookresearch/segment-anything) and [3DOVS](https://github.com/Kunhao-Liu/3D-OVS). We thank all the authors for their great work. 


## Citation
Cite below if you find this repository helpful to your project:
```
@article{liao2024ov,
  title={OV-NeRF: Open-vocabulary neural radiance fields with vision and language foundation models for 3D semantic understanding},
  author={Liao, Guibiao and Zhou, Kaichen and Bao, Zhenyu and Liu, Kanglin and Li, Qing},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  volume={34},
  pages={12923--12936},
  year={2024},
  publisher={IEEE}
}
```

<div align="center">

# OV-NeRF: Open-vocabulary Neural Radiance Fields with Vision and Language Foundation Models for 3D Semantic Understanding 
Guibiao Liao<sup>1,2</sup>, Kaichen Zhou<sup>3</sup>, Zhenyu Bao<sup>1,2</sup>, Kanglin Liu<sup>2</sup>, Qing Li<sup>2, *</sup>

<sup>1</sup>Peking University    <sup>2</sup>Pengcheng Laboratory    <sup>3</sup>University of Oxford

<sup>*</sup>Corresponding author: lqing900205@gmail.com

### [Paper](https://arxiv.org/abs/2402.04648)

</div>

![Image](https://github.com/pcl3dv/OV-NeRF/blob/main/images/pipeline.png)


***Abstract**: The development of Neural Radiance Fields (NeRFs) has provided a potent representation for encapsulating the geometric and appearance characteristics of 3D scenes. Enhancing the capabilities of NeRFs in open-vocabulary 3D semantic perception tasks has been a recent focus. However, current methods that extract semantics directly from Contrastive Language-Image Pretraining (CLIP) for semantic field learning encounter difficulties due to noisy and view-inconsistent semantics provided by CLIP. To tackle these limitations, we propose OV-NeRF, which exploits the potential of pre-trained vision and language foundation models to enhance semantic field learning through proposed single-view and cross-view strategies. First, from the *single-view* perspective, we introduce Region Semantic Ranking (RSR) regularization by leveraging 2D mask proposals derived from Segment Anything (SAM) to rectify the noisy semantics of each training view, facilitating accurate semantic field learning. Second, from the *cross-view* perspective, we propose a Cross-view Self-enhancement (CSE) strategy to address the challenge raised by view-inconsistent semantics. Rather than invariably utilizing the 2D inconsistent semantics from CLIP, CSE leverages the 3D consistent semantics generated from the well-trained semantic field itself for semantic field training, aiming to reduce ambiguity and enhance overall semantic consistency across different views. Extensive experiments validate our OV-NeRF outperforms current state-of-the-art methods, achieving a significant improvement of 20.31\% and 18.42\% in mIoU metric on Replica and Scannet, respectively. Furthermore, our approach exhibits consistent superior results across various CLIP configurations, further verifying its robustness. 


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

## Citation
Cite below if you find this repository helpful to your project:
```
@article{liao2024ov,
  title={OV-NeRF: Open-vocabulary Neural Radiance Fields with Vision and Language Foundation Models for 3D Semantic Understanding},
  author={Liao, Guibiao and Zhou, Kaichen and Bao, Zhenyu and Liu, Kanglin and Li, Qing},
  journal={arXiv preprint arXiv:2402.04648},
  year={2024}
}
```

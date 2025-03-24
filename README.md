# 


>**This is the official implementation of the paper with the title “A new lightweight deep learning model optimized with pruning and dynamic quantization to detect freezing gait on wearable devices”by Myung-Kyu Yi and Seong Oun Hwang**
>
>https://doi.org/

## Paper Overview

**Abstract**: 
Freezing of gait (FoG) is a debilitating symptom of Parkinson’s disease that severely impacts patients’ mobility and quality of life. To minimize the risk of falls and injuries associated with FoG, it is crucial to develop highly accurate FoG detection systems for portable wearable devices that enable continuous and real-time monitoring. However, achieving high accuracy in deep learning (DL) models typically requires a large number of parameters and integrating multiple sensors, posing challenges for deployment on resource-constrained wearable devices. To address these challenges, we propose a novel lightweight DL model that combines a convolutional neural network and a gated recurrent unit with residual attention and efficient channel attention mechanisms. Additionally, we incorporate pruning and dynamic quantization techniques, along with an innovative feature selection method, to optimize the model for wearable applications. Experimental results demonstrate that our proposed DL model outperforms state-of-the-art supervised DL models, achieving an F1 score of 0.994 while utilizing 29.9 times fewer parameters than existing models. The model’s maximum memory usage is only 420.91 KB, making it well-suited for wearable devices. Furthermore, optimizations through pruning and dynamic quantization further reduced the model size by an additional 7.84 times, resulting in a final size of just 44.04 KB without sacrificing accuracy. As a result, the proposed DL model achieves high accuracy in FoG detection with minimal memory usage, enabling real-time monitoring on wearable devices and providing a practical solution for managing FoG in Parkinson’s patients.

---
## Dataset
♣ Daphnet Freezing of Gait dataset is available at https://bmi.hmu.gr/the-mobifall-and-mobiact-datasets-2/](https://archive.ics.uci.edu/dataset/245/daphnet+freezing+of+gait)
♣ Multimodal dataset is available at https://data.mendeley.com/datasets/t8j8v4hnm4/1
♣ IMU dataset is available at https://figshare.com/articles/dataset/A_public_dataset_of_video_acceleration_and_angular_velocity_in_individuals_with_Parkinson_s_disease_during_the_turning-in-place_task/14984667

## Codebase Overview
- We note that:
  - <model.py> for the proposed lightweight deep learning models 

Framework uses Tensorflow, Pytorch, tensorflow_addons, numpy, pandas, matplotlib, scikit-learn.  
  
## Citing This Repository

If our project is helpful for your research, please consider citing :

```
@article{yi2024jsen,
  title={A new lightweight deep learning model optimized with pruning and dynamic quantization to detect freezing gait on wearable devices},
  author={Myung-Kyu Yi and Seong Oun Hwang},
  journal={},
  volume={},
  Issue={},
  pages={},
  year={}
  publisher={}
}

```

## Contact

Please feel free to contact via email (<kainos14@hanyang.ac.kr>) if you have further questions.

# Adaptive Partial Transformer for Infrared Small Target Detection
We have submitted the paper for review and will make the code available after publication.

## Network
![outline](Fig/picture1.jpg)

## Datasets
**Our project has the following structure:**
  ```
  ├───dataset/
  │    ├── NUAA-SIRST
  │    │    ├── image
  │    │    │    ├── Misc_1.png
  │    │    │    ├── Misc_2.png
  │    │    │    ├── ...
  │    │    ├── mask
  │    │    │    ├── Misc_1.png
  │    │    │    ├── Misc_2.png
  │    │    │    ├── ...
  │    │    ├── train_NUAA-SIRST.txt
  │    │    │── train_NUAA-SIRST.txt
  │    ├── IRSTD-1K
  │    │    ├── image
  │    │    │    ├── XDU0.png
  │    │    │    ├── XDU1.png
  │    │    │    ├── ...
  │    │    ├── mask
  │    │    │    ├── XDU0.png
  │    │    │    ├── XDU1.png
  │    │    │    ├── ...
  │    │    ├── train_IRSTD-1K.txt
  │    │    ├── train_IRSTD-1K.txt
  │    ├── ...  
  ```
<be>

## Results and Trained Models
#### Qualitative Results

![outline](Fig/picture2.jpg)

#### Quantitative Results on NUAA-SIRST, and IRSTD-1K

| Dataset         | IoU (x10(-2)) | F-measure (x10(-2))| Pd (x10(-2))|  Fa (x10(-6))|
| ------------- |:-------------:|:-----:|:-----:|:-----:|
| NUAA-SIRST    | 76.78  |  86.86 | 98.10 | 11.82 |
| IRSTD-1K      | 70.56  |  82.72 | 92.57 | 9.09 |
| [[weights]]()|


*This code is highly borrowed from [MSHNet](https://github.com/Lliu666/MSHNet). Thanks to Qiankun Liu.
*The overall repository style is highly borrowed from [DNANet](https://github.com/YeRen123455/Infrared-Small-Target-Detection). Thanks to Boyang Li.









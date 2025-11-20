# A Multi-instance Learning Network with Prototype-instance Adversarial Contrastive for Cervix Pathology Grading

## 🧔 Authors [*corresponding author]
- Mingrui Ma, Furong Luo, Binlin Ma, Shuxian Liu, Xiaoyi Lv*, Pan Huang*

## :fire: News
- [2025/11/18] Our manuscript was available online at _Medical Image Analysis (IF 11.8)_, you could click [it](https://www.sciencedirect.com/science/article/pii/S1361841525004268).
- [2025/11/15] Our manuscript was Accepted by _Medical Image Analysis (IF 11.8)_.
- [2025/09/30] Our manuscript was currently "Accept with Minor Revision" in _Medical Image Analysis (IF 11.8)_.
- [2025/05/27] Our manuscript was "Major Revision" in _Medical Image Analysis (IF 11.8)_.
- [2024/12/15] Our manuscript was submitted to _Medical Image Analysis (IF 11.8)_.



## :rocket: Pipeline

Here's an overview of our **Multi-instance Learning Network with Prototype-instance Adversarial Contrastive Learning (PacMIL)** method:

<img src="https://github.com/Baron-Huang/PacMIL/blob/main/Image/Main_Frame_for_PacMIL.png" style="width:80%; height:80%;">



## :mag: TODO
<font color="red">**We are currently organizing all the code. Stay tuned!**</font>
- [x] training code
- [x] Evaluation code
- [x] Model code
- [ ] Pretrained weights
- [ ] Datasets





## 🛠️ Getting Started

To get started with PacMIL, follow the installation instructions below.
1. Random seed fixed

```
The random seed is fixed to 1 for all models.
```

2.  Clone the repo

```sh
git clone https://github.com/Baron-Huang/PacMIL
```

3. Install dependencies
   
```sh
pip install -r requirements.txt
```

4. Training on Swin Transformer-S Backbone
```sh
sh PacMIL_CH_CSCC.sh or PacMIL_CH_LSCC.sh
Modify: --abla_type sota --run_mode train --random_seed ${seed}
```

5. Evaluation
```sh
sh PacMIL_CH_CSCC.sh or PacMIL_CH_LSCC.sh
Modify: --abla_type sota --run_mode test --random_seed ${seed}
```

6. Extract features for plots
```sh
sh PacMIL_CH_CSCC.sh or PacMIL_CH_LSCC.sh
Modify: --abla_type sota --run_mode test --random_seed ${seed} --feat_extract
```

7. Interpretability plots
```sh
sh PacMIL_CH_CSCC.sh or PacMIL_CH_LSCC.sh
Modify: --abla_type sota --run_mode test --random_seed ${seed} --bag_weight
```

## :postbox: Contact
If you have any questions, please contact [Dr. Pan Huang](https://scholar.google.com/citations?user=V_7bX4QAAAAJ&hl=zh-CN) (`mrhuangpan@163.com or panhuang@polyu.edu.hk`).

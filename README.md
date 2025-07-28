# A Multi-instance Learning Network with Prototype-instance Adversarial Contrastive for Cervix Pathology Grading

Mingrui Ma, Furong Luo, Binlin Ma, Shuxian Liu, Xiaoyi Lv*, Pan Huang*, Member, MICCAI

*Corresponding author

Our manuscript is in the peer review，and we will completely share the dataset and code after the peer review.

# Introduction
The pathological grading of cervical squamous cell carcinoma (CSCC) is an fundamental and important index in tumor diagnosis. Pathologists tend to focus on single differentiation areas for grading. Existing multi-instance learning (MIL) divides pathology images, then there are a number of multiple differentiated instances (MDIs) with ambiguous grading patterns. This decreases the ability of the model to represent CSCC pathology grading patterns. Motivated by the above issues, we propose an end-to-end multi-instance learning network with prototype-instance adversarial contrastive, i.e., PacMIL, with three-fold ideas. First, we propose an end-to-end multi-instance nonequilibrium learning algorithm, which addresses the poor matching between MIL feature representations and CSCC pathology grading, and realizes nonequilibrium representation. Second, we propose a prototype-instance adversarial contrastive (PAC) approach that introduces a priori prototype instances and a probability distribution attention mechanism. Additionally, it enhances the model's representation learning ability for single differentiated instances (SDIs). Third, we introduce an adversarial contrast learning approach in the PAC method, which solves that fixed metrics rarely measure the variable MDIs and SDIs. Meanwhile, in the optimization objective function, we embed the correct metric distances of the MDIs and SDIs. Extensive experiments show that our PacMIL model reaches 93.09% and 0.9802 for the mAcc and AUC metrics, respectively, which are better than those of other SOTA models. Moreover, the representation ability of our PacMIL is superior to that of other SOTA models. Overall, our model is more practical. 

---
![image](https://github.com/Baron-Huang/PacMIL/blob/main/Image/Main_Frame_for_PacMI)


# Dataset
You could finds and applies the dataset by the link: https://drive.google.com/drive/folders/1bq8VS7r6Cn9dYqieGe-VzjQYu5dxW9B6?usp=drive_link.


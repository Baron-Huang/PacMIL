# A Multi-instance Learning Network with Prototype-instance Adversarial Contrastive for Cervix Pathology Grading

Mingrui Ma, Furong Luo, Binlin Ma, Shuxian Liu, Xiaoyi Lv*, Pan Huang*, Member, MICCAI

*Corresponding author

Our manuscript is in the peer review，and we will completely share the dataset and code after the peer review.

# Introduction
The pathological grading of cervical squamous cell carcinoma (CSCC) is a fundamental and important index in tumor diagnosis. Pathologists tend to focus on single differentiation areas during the grading process. Existing multi-instance learning (MIL) methods divide pathology images into regions, generating multiple differentiated instances (MDIs) that often exhibit ambiguous grading patterns. These ambiguities reduce the model’s ability to accurately represent CSCC pathological grading patterns. Motivated by these issues, we propose an end-to-end multi-instance learning network with prototype-instance adversarial contrastive learning, termed PacMIL, which incorporates three key ideas. First, we introduce an end-to-end multi-instance nonequilibrium learning algorithm that addresses the mismatch between MIL feature representations and CSCC pathological grading, and enables nonequilibrium representation. Second, we design a prototype-instance adversarial contrastive (PAC) approach that integrates a priori prototype instances and a probability distribution attention mechanism. This enhances the model’s ability to learn representations for single differentiated instances (SDIs). Third, we incorporate an adversarial contrastive learning strategy into the PAC method to overcome the limitation that fixed metrics rarely capture the variability of MDIs and SDIs. In addition, we embed the correct metric distances of the MDIs and SDIs into the optimization objective function to further guide representation learning. Extensive experiments demonstrate that our PacMIL model achieves 93.09% and 0.9802 for the mAcc and AUC metrics, respectively, outperforming other SOTA models. Moreover, the representation ability of PacMIL is superior to that of existing SOTA approaches. Overall, our model offers enhanced practicality in CSCC pathological grading. Our code and dataset will be publicly available at https://github.com/Baron-Huang/PacMIL.

---
![image](https://github.com/Baron-Huang/PacMIL/blob/main/Image/Main_Frame_for_PacMI)


# Dataset
You could finds and applies the dataset by the link: https://drive.google.com/drive/folders/1bq8VS7r6Cn9dYqieGe-VzjQYu5dxW9B6?usp=drive_link.


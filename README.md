# A Multi-instance Learning Network with Prototype-instance Adversarial Contrastive for Cervix Pathology Grading

Mingrui Ma, Furong Luo, Binlin Ma, Shuxian Liu, Pan Huang*, Xiaoyi Lv*

Our manuscript is in the peer review，and we will completely share the dataset and code after the peer review.

# Introduction
The pathological grading of cervical squamous cell carcinoma is an basic and important index in tumor diagnosis. Pathologists often grade by focusing on a single differentiated region. After the existing multi-instance learning divides the pathological image, there are many multidifferentiated instances with blurred grading patterns, which will reduce the ability of the model to represent and learn the CSCC pathological grading pattern. To this end, we propose an end-to-end multi-instance learning network with prototype-instance adversarial contrastive (i.e., PacMIL), which includes three contributions: First, an end-to-end multi-instance nonequilibrium learning algorithm (EMNL) is proposed to solve the problems of poor matching of existing multi-instance learning feature representations to tasks and equalized learning; Second, a prototype-instance adversarial contrastive (PAC) approach is proposed that introduces a priori prototype instances and probability-distributed attention mechanisms to enhance the model's ability to learn feature representations for a single differentiated instance; Third, the adversarial contrast learning method is introduced into the PAC method to solve the problem that contrast learning with fixed metrics is difficult to measure the changing single-differentiated instances versus multidifferentiated instances, and the correct MDIs and SDIs metrics distance are embedded in the optimization objective function. Numerous experiments show that our PacMIL achieves 93.09% and 0.9802 for the mAcc and AUC metrics, which outperforms other SOTA models. Meanwhile, through extensive qualitative experiments, our PacMIL outperforms other SOTA models in its ability to represent the features of Grade 1, Grade 2 and Grade 3. In general, our model will have more clinical application value. 

---
![image](https://github.com/Baron-Huang/LA-ViT/blob/main/Images/Fig_4.png)


# Dataset
You could finds and applies the dataset by the link: https://drive.google.com/drive/folders/1bq8VS7r6Cn9dYqieGe-VzjQYu5dxW9B6?usp=drive_link.


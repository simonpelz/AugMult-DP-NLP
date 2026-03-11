# **Exploring the Impact of Augmentation Multiplicity on DP-NLP Models**

This repository contains the implementation and experimental results for my Bachelor's Thesis at the **Technical University of Munich (TUM)**. The project investigates whether **Augmentation Multiplicity (AugMult)**, which is successful in Computer Vision, can bridge the privacy-utility gap in Differentially Private for the Natural Language Processing domain.

## **The Core Challenge: Privacy vs. Utility**

Training Machine Learning models on sensitive data (like medical records or private chats) carries a risk: models can "memorize" specific training examples. **Differential Privacy (DP)** provides a mathematical guarantee that the model doesn't lean too heavily on any single individual's data.  
**The Problem:** Applying DP usually makes models less accurate (**the Utility Gap**). While "Data Augmentation" (creating variations of data) helps normal models, doing it naively in DP training consumes your "privacy budget" much faster.

### **The Solution: Augmentation Multiplicity (AugMult)**

Instead of treating every augmented sentence as a new data point, **AugMult** creates $K$ versions of a sentence, computes their gradients, and **averages them before** any privacy-preserving steps (clipping and noising) occur. This allows the model to learn from diverse variations without "paying" extra in terms of privacy.  
<!-- add visualization of augmult processing-->

## **Experimental Setup**

We fine-tuned **BERT-base** using Parameter-Efficient Fine-Tuning across two distinct domains:

| Dataset | Language | Task | Size |
| :---- | :---- | :---- | :---- |
| **SST-2** | Easy | Sentiment | 67k samples |
| **MedNLI** | Specialized (Medical) | Inference | 11k samples |

### **Augmentation Strategies**

1. **EDA (Easy Data Augmentation):** Rule-based swaps, deletions, and insertions. Fast but can be "destructive" to meaning.  
2. **Backtranslation:** Translating English \-\> Intermediate Language \-\> English. Computationally expensive but preserves semantic nuance better.

## **Key Results**

### **1\. The Domain Matters**

We found that the impact of AugMult is highly dependent on the dataset. In **SST-2**, the model is already so efficient that augmentations provided no significant boost. However, in the medical domain (**MedNLI**), the results were more nuanced.
MedNLI | SST-2  
:-------------------------:|:-------------------------:
![mednli_main](figures/mednli_main.png)  |   ![sst2](figures/sst2.png) 



### **2\. Augmentation Complexity & Inherent Error**

A major challenge in NLP data augmentation is Label Preservation: if an augmentation changes the meaning too much, the model is effectively trained on "noisy" or incorrect labels. We call this the Inherent Error of the augmentation. The graph below shows the accuracy deficit of runs with augmentations compared to the baseline of no augmentations (averaged over many runs each on the MedNLI task)

![aug_acc_lost_svg](figures/aug_acc_lost.svg)

The Validation accuracy on the left (without augmentations) is significantly higher than training accuracy. When only comparing accuracy on unaugmented samples, training accuracy is very close for the models trained on Backtranslation and no augmentations respectively.

![MedNLI Detail](figures/MedNLI%20Detail.png) 


### **3\. Multiplicity Scaling**

Increasing the number of augmentations ($K$) generally showed a positive trend in accuracy, suggesting that higher values of $K$ (beyond 16\) might yield even better results. But the higher computational cost with effective augmentations are likely prohibitive.  
<!--![honest_k_scaling](figures/honest_k_scaling.png)-->  

![increasing_k](figures/increasing_k.png) 



### **4\. Training Stability & "Rising Loss"**

One of the most interesting findings was that AugMult helps stabilize training when using very small **Clipping Norms (C)**. In standard DP-SGD, small $C$ values often cause the loss to rise which is a sign of gradient polarization. AugMult appears to mitigate this. It is important to note that the accuracy is  best for the highest loss values.
Further research could go into investigating differences in performance on "hard" samples.
 ![Rising_loss](figures/Rising_loss.png) 

## **Findings & Conclusion**

* **Not a "Drop-in" Solution:** Unlike in Computer Vision, AugMult for NLP requires very careful selection of augmentations and pretrained model.
* **Quality \> Quantity:** Backtranslation (semantic-aware) outperformed EDA (rule-based) because it preserved medical jargon more effectively.  
* **Computational Trade-off:** AugMult effectively increases the batch size $K$ times. Additionally, good augmentations are computationally expensive as well. The modest accuracy gains might not justify the significantly increased training time.  
* **Best Use Case:** AugMult is most promising for **specialized, small-scale datasets** where data is scarce and privacy is paramount.

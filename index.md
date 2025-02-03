---
title: "Zero-Shot and Few-Shot Classification of Biomedical Articles"
description: "An in-depth explanation of the paper discussing Zero-Shot and Few-Shot Learning for Biomedical Article Classification using BioBERT."
---

# **Zero-Shot and Few-Shot Classification of Biomedical Articles**

## **Introduction**
- The COVID-19 pandemic led to a rapid increase in biomedical publications.
- **MeSH (Medical Subject Headings)** helps classify articles into categories, but manual labeling is slow.
- **Zero-Shot Learning (ZSL)**: The model classifies unseen categories.
- **Few-Shot Learning (FSL)**: The model learns with very few samples.
- The authors use **BioBERT**, a specialized version of BERT trained on biomedical data, to improve classification.

---

## **Related Work**
- **Zero-Shot Learning**: Uses pretrained models like BERT/BioBERT.
- **Fine-Grained Biomedical Classification**: Most prior work focuses on simple MeSH term retrieval.
- **Probing**: Used to analyze how much hierarchical knowledge a model learns.

---

## **Proposed Approach**
### **Zero-Shot Learning Architectures**
1. **BioBERT Single-Task Learning (STL)**:
   - Uses BioBERT to match **MeSH descriptors** with **paper abstracts**.
   - Output: A probability score indicating relevance.

2. **BioBERT Multi-Task Learning (MTL)**:
   - Adds a **decoder module** to predict **MeSH hierarchy position**.
   - **Goal**: Improve the model's understanding of hierarchical relationships.

### **Mathematical Formulations**
#### **Multi-Task Learning Loss**
$$
\text{losstot} = \frac{1}{2\sigma_1^2} \text{loss}_1 + \frac{1}{2\sigma_2^2} \text{loss}_2 + \log(\sigma_1\sigma_2)
$$
- \( \text{loss}_1 \): Binary classification loss.
- \( \text{loss}_2 \): Hierarchical generation loss.
- \( \sigma_1, \sigma_2 \): Learnable weights to balance losses.

#### **Attention-Based Hierarchy Prediction**
1. **Compute attention scores:**
$$
\text{att}_j = \text{bert}_h \times h_j
$$
2. **Normalize attention weights:**
$$
\hat{\text{att}}_j = \text{softmax}(\text{att}_j)
$$
3. **Apply attention to BioBERT output:**
$$
\text{attn\_applied}_j = \hat{\text{att}}_j^T \times \text{bert}_h
$$
4. **GRU-based decoder update:**
$$
\text{input}_j = \text{embed}_j + \text{attn\_applied}_j
$$
$$
 h_{j+1}, \text{out}_{j+1} = \text{GRU}(h_j, \text{input}_j)
$$

---

## **Probing Hierarchical Knowledge**
- **Does the model encode MeSH term hierarchy?**
- **Two probing tasks:**
  1. **Shortest-Path Probe:** Predicts the distance between two MeSH terms.
  2. **Common-Ancestors Probe:** Checks how many ancestors two terms share.

#### **Hierarchy Distance Calculation**
$$
 d_B(h_i, h_j) = (B(h_i - h_j))^T (B(h_i - h_j))
$$
- **\(B\)** is a learnable projection matrix.
- The model learns a representation where similar MeSH terms are closer.

#### **Loss Function for Probing**
$$
\min_B \sum_{i,j} |d_T(h_i, h_j) - d_B(h_i, h_j)|^2
$$
- **\(d_T\)**: True distance in MeSH hierarchy.
- **\(d_B\)**: Model’s predicted distance.

---

## **Experimental Settings**
- **Datasets:**
  - **Medline/MeSH**: Contains biomedical articles with MeSH term labels.
  - **LitCovid**: A COVID-19 specific subset with 8 broad categories.
- **Evaluation Metrics:**
  - **Balanced Dataset**: Equal number of positive/negative MeSH term labels.
  - **Siblings Dataset**: Includes MeSH hierarchy relations.

---

## **Results & Discussion**
### **Zero/Few-Shot Performance**
- **BioBERT STL outperforms MTL in general classification.**
- **BioBERT MTL is better when distinguishing closely related MeSH terms.**
- **Performance improves with more training data.**

### **Probing Results**
- **MTL model better encodes hierarchical relations.**
- **F1-score increases when using hierarchical knowledge.**

### **Limitations and Future Work**
- **Multi-Task Learning Convergence:** The classification task converges too fast.
- **Annotation Issues:** Inconsistent MeSH labeling affects performance.
- **Large-Scale Zero-Shot Learning:** Requires retrieval-based methods.

---

## **Conclusion**
- **Zero-shot classification is feasible but challenging.**
- **Multi-task learning improves hierarchical encoding but has marginal impact on classification.**
- **Future work should explore better retrieval methods and loss functions.**

---

## **References**
- Hewitt, J., & Manning, C. D. (2019). *A Structural Probe for Finding Syntax in Word Representations.*
- Lee, J., Yoon, W., Kim, S., et al. (2019). *BioBERT: a pre-trained biomedical language representation model for biomedical text mining.*
- Wang, W., Zheng, V. W., Yu, H., & Miao, C. (2019). *A survey of zero-shot learning: Settings, methods, and applications.*


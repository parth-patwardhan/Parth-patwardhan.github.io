# Zero-Shot and Few-Shot Classification of Biomedical Articles: A Comprehensive Analysis

## Introduction
The rapid advancement of biomedical research has resulted in an overwhelming volume of scientific publications. Efficiently classifying and indexing these publications is essential for researchers to find relevant studies quickly. Traditionally, articles are manually annotated with Medical Subject Headings (MeSH) to improve searchability. However, this process is time-consuming and inefficient, especially during crises like the COVID-19 pandemic, when new terms emerge frequently.

To address this challenge, **Zero-Shot Learning (ZSL) and Few-Shot Learning (FSL)** techniques are being explored to automatically classify biomedical articles without requiring large labeled datasets. This report delves into the study of how **BioBERT**, a biomedical adaptation of BERT, facilitates zero-shot classification by leveraging hierarchical and semantic structures inherent in MeSH terms.

## Understanding the Role of MeSH in Biomedical Classification
### What is MeSH?
MeSH (Medical Subject Headings) is a controlled vocabulary developed by the National Library of Medicine (NLM). It is used for indexing biomedical literature systematically. The hierarchy allows researchers to find related topics easily. 

https://github.com/parth-patwardhan/Parth-patwardhan.github.io/blob/main/images/mesh_descriptors.png?raw=true

### Why is MeSH Important?
- It provides structured categorization.
- It enables efficient search and retrieval.
- It facilitates systematic review and meta-analysis.

During the COVID-19 pandemic, new biomedical concepts emerged rapidly, necessitating fast and accurate classification of research articles. Traditional methods struggled to keep up, making automated classification models highly valuable.

https://github.com/parth-patwardhan/Parth-patwardhan.github.io/blob/main/images/mesh_dataset_hierarchy.png?raw=true

## Zero-Shot Learning (ZSL) and Few-Shot Learning (FSL) in Biomedical Text Classification
### What is Zero-Shot Learning?
ZSL is a classification approach where the model correctly assigns labels it has never seen before. It achieves this by leveraging semantic relationships between known and unknown classes. This is particularly useful in biomedical research, where new diseases, treatments, and conditions are constantly emerging.

### How Does BioBERT Enhance ZSL?
BioBERT is a pre-trained transformer model specialized for biomedical text. It incorporates domain-specific knowledge to improve classification accuracy. The study explores two main approaches:
1. **Single-Task Learning (STL):** BioBERT encodes abstracts and MeSH descriptions together.
2. **Multi-Task Learning (MTL):** This approach integrates the MeSH hierarchy, enhancing classification accuracy by incorporating additional context.


https://github.com/parth-patwardhan/Parth-patwardhan.github.io/blob/main/images/bio_bert_model.png?raw=true


## Hierarchical Probing: Understanding the Model’s Knowledge
To assess whether BioBERT effectively learns the MeSH hierarchy, researchers used two **probing tasks**:
1. **Shortest-Path Probe:** Measures whether the model encodes distances between MeSH terms accurately.
2. **Common-Ancestors Probe:** Evaluates whether the model understands shared hierarchical ancestry between terms.

These probing tasks help determine how well the model captures biomedical relationships beyond simple keyword matching.

## Experimental Setup and Evaluation
### Datasets Used
1. **Medline/MeSH:** A large dataset with extensive MeSH annotations for biomedical research.
2. **LitCovid:** A specialized subset focused on COVID-19-related publications.

### Evaluation Metrics
- **Zero-Shot Testing:** Evaluates the model’s ability to generalize to unseen labels.
- **Few-Shot Testing:** Tests performance with very limited labeled examples.
- **Hierarchical Probing:** Analyzes how well the model captures structured relationships in the dataset.

### Performance Comparison
- **STL Model:** Shows strong generalization for broader MeSH categories.
- **MTL Model:** Performs better in distinguishing fine-grained, closely related terms by leveraging hierarchical relationships.

https://github.com/parth-patwardhan/Parth-patwardhan.github.io/blob/main/images/metrics.png?raw=true

## Challenges and Suggested Improvements
### Challenges Faced
1. **Handling Rare Categories:** Certain MeSH terms appear infrequently, making it difficult for the model to generalize.
2. **Computational Complexity:** Training on all possible label-document pairs is computationally expensive.
3. **Data Annotation Inconsistencies:** Different datasets may use slightly different labeling schemes, affecting classification performance.

### Suggested Improvements
1. **Balanced Training:** More sophisticated sampling methods could improve the model’s ability to handle rare categories.
2. **Hierarchical Sampling:** Training with closely related negative examples can enhance the differentiation of terms.
3. **Retrieval-Augmented Classification:** Integrating models like BM25 or ColBERT to first narrow down candidate labels before classification.

## Why This Study Matters
Automating biomedical classification can significantly improve research efficiency, especially during health crises. Zero-Shot Learning allows us to categorize new findings swiftly, aiding healthcare professionals and researchers in making data-driven decisions. This study provides a foundation for further enhancements in automatic text classification for biomedical literature.

## Conclusion
This study highlights the power of Zero-Shot Learning for biomedical article classification. By leveraging hierarchical relationships and sophisticated NLP models like BioBERT, researchers can improve classification accuracy and scalability. Future research should focus on enhancing hierarchical sampling techniques and integrating retrieval-based methods to optimize performance further.


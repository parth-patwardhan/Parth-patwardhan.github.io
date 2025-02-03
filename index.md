# Zero-Shot and Few-Shot Classification of Biomedical Articles in the Context of the COVID-19 Pandemic

## Introduction

The rapid emergence of the COVID-19 pandemic has led to an unprecedented influx of biomedical literature. Efficiently categorizing and indexing this vast amount of information is crucial for researchers and healthcare professionals. Traditional manual annotation methods, such as assigning Medical Subject Headings (MeSH) to articles, are labor-intensive and time-consuming. To address this challenge, the application of zero-shot and few-shot learning approaches has gained attention. These methods aim to classify biomedical articles into categories with little to no labeled data, leveraging advanced natural language processing (NLP) models and the rich semantic structure of MeSH.

## Background

### Medical Subject Headings (MeSH)

MeSH is a comprehensive controlled vocabulary used by the National Library of Medicine for indexing articles in PubMed. It organizes biomedical terms into a hierarchical structure, allowing for precise and consistent categorization of topics. Each MeSH term is associated with a definition and a unique tree number that indicates its position in the hierarchy. This structure facilitates detailed indexing and retrieval of biomedical information.

### Zero-Shot and Few-Shot Learning

In machine learning, zero-shot learning (ZSL) refers to the ability of a model to classify instances from classes that it has never seen during training. This is achieved by leveraging auxiliary information, such as semantic embeddings or descriptions of the unseen classes. Few-shot learning (FSL), on the other hand, deals with scenarios where only a limited number of labeled examples are available for certain classes. Both approaches are particularly valuable in biomedical domains, where new diseases or concepts may emerge rapidly, and labeled data may be scarce.

### BioBERT

BioBERT is a pre-trained language representation model specifically designed for biomedical text mining. It is based on the BERT architecture and has been trained on large-scale biomedical corpora, including PubMed abstracts and PMC full-text articles. BioBERT has demonstrated superior performance in various biomedical NLP tasks, such as named entity recognition, relation extraction, and question answering.

## Proposed Approach

The study proposes a method that enhances BioBERT representations by incorporating the semantic information available in MeSH. The approach involves framing the problem as determining whether the concatenation of a MeSH term definition and a paper abstract constitutes a valid instance. Additionally, a multi-task learning framework is employed to induce the MeSH hierarchy into the representations through a sequence-to-sequence task.

### Model Architecture

The model architecture consists of two main components:

1. **BioBERT Encoder**: This component encodes both the MeSH term definitions and the paper abstracts into dense vector representations. By leveraging the contextual embeddings from BioBERT, the model captures the semantic nuances of biomedical text.

2. **Multi-Task Learning Framework**: In addition to the primary classification task, the model is trained to predict the hierarchical position of MeSH terms. This is achieved by introducing an auxiliary sequence-to-sequence task that generates the tree numbers associated with MeSH terms. The multi-task learning objective encourages the model to learn representations that are aware of the hierarchical structure of MeSH.

### Loss Function

The total loss for the multi-task learning framework is defined as:

\[
\text{loss}_{\text{tot}} = \frac{1}{2\sigma_1^2} \text{loss}_1 + \frac{1}{2\sigma_2^2} \text{loss}_2 + \log(\sigma_1\sigma_2)
\]

Where:

- \(\text{loss}_1\) is the binary cross-entropy loss for the primary classification task.
- \(\text{loss}_2\) is the negative log-likelihood loss for the auxiliary sequence-to-sequence task.
- \(\sigma_1\) and \(\sigma_2\) are learnable parameters that balance the contributions of the two loss components.

The inclusion of the \(\log(\sigma_1\sigma_2)\) term serves as a regularization factor, preventing the model from assigning excessively high values to \(\sigma_1\) or \(\sigma_2\), which could otherwise lead to trivial solutions.

## Experimental Setup

### Datasets

The experiments are conducted on two datasets:

1. **MedLine**: A comprehensive dataset containing biomedical articles with associated MeSH annotations. It serves as a standard benchmark for evaluating biomedical text classification models.

2. **LitCovid**: A specialized subset of PubMed that focuses on COVID-19 literature. It includes articles categorized into general topics related to the pandemic, providing a relevant testbed for the proposed approach.

### Evaluation Metrics

The performance of the model is evaluated using standard classification metrics, including precision, recall, and F1-score. Additionally, hierarchical evaluation metrics are employed to assess the model's ability to capture the MeSH hierarchy in its predictions.

## Results and Discussion

### Zero-Shot Classification Performance

The proposed approach demonstrates promising results in zero-shot classification scenarios. By leveraging the semantic information in MeSH and the contextual embeddings from BioBERT, the model effectively assigns appropriate MeSH terms to articles, even when specific terms were not seen during training.

### Few-Shot Classification Performance

In few-shot settings, where limited labeled examples are available, the model benefits from the multi-task learning framework. The auxiliary task of predicting MeSH tree numbers aids in learning more generalized representations, leading to improved performance compared to baseline models.

### Hierarchical Probing

To assess the extent to which the model captures the hierarchical structure of MeSH, probing tasks are conducted. The results indicate that the multi-task learning framework enables the model to encode hierarchical relations, as evidenced by its ability to predict the shortest path distances and common ancestors between MeSH terms.

## Related Work

The integration of hierarchical information into biomedical text classification has been explored in various studies. For instance, the Hierarchical Deep Neural Network (HDNN) architecture has been proposed to exploit the label hierarchy in PubMed article classification tasks. By aligning the network topology with the hierarchical structure of labels, HDNN enhances performance in extreme multi-label text classification scenarios.

Another approach, BERTMeSH, introduces a deep contextual representation learning method for large-scale biomedical MeSH indexing. This method leverages the flexibility of BERT-based models to handle the diverse section organization in full-text articles, improving the accuracy of MeSH term assignment.

## Conclusion

The study presents a novel approach to zero-shot and few-shot classification of biomedical articles by enhancing BioBERT representations with MeSH semantic information. The multi-task learning framework effectively incorporates the hierarchical structure of MeSH into the model, leading to improved performance in both zero-shot and few-shot scenarios. This approach holds promise for efficient and accurate indexing of rapidly emerging biomedical literature, such as that related to the COVID-19 pandemic.

## Future Work

Future research directions include exploring alternative methods for integrating hierarchical information into language models, such as graph neural networks. Additionally, investigating the applicability of the proposed approach to other domains with hierarchical label structures could further validate its effectiveness.

## References

1. Lupart, S., Favre, B., Nikoulina, V., & Ait-Mokhtar, S. (2022). Zero-Shot and Few-Shot Classification of Biomedical Articles in Context of the COVID-19 Pandemic. arXiv preprint arXiv:2201.03017.

2. Zhang, Y., & Lee, J. (2021). BERTMeSH: deep contextual representation learning for large-scale biomedical MeSH indexing 

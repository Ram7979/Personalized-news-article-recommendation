# News Article Recommendation System


## Detailed README (Full Explanation)

A **hybrid, NLP-driven news recommendation system** that delivers personalized and relevant news articles by combining **content-based filtering**, **collaborative insights**, and **unsupervised clustering**. This project demonstrates an end-to-end machine learning pipeline—from raw text preprocessing to recommendation evaluation—implemented with strong theoretical grounding and practical results.

---

## Project Overview

With the exponential growth of online news, users often face information overload. This project aims to **intelligently filter and recommend news articles** that align with a user’s interests while maintaining topic diversity. The system is designed to be **scalable, modular, and adaptable** for real-world news platforms.

**Key Highlights:**

* Hybrid recommendation architecture
* NLP-based semantic understanding of news headlines
* Topic clustering using K-Means
* Personalized user profiling
* Strong evaluation using clustering and recommendation metrics

---

## Core Concepts Used

* Natural Language Processing (NLP)
* Word Embeddings (Word2Vec – 300D)
* K-Means Clustering
* Cosine Similarity
* Content-Based Filtering
* Collaborative Filtering (implicit behavior)
* Information Retrieval Metrics

---

## Dataset

* **Source:** Kaggle (News Article Dataset)
* **Data Includes:**

  * News headlines
  * Categories (Politics, Entertainment, World News, etc.)
  * Publication timestamps

The dataset captures a wide range of news topics and temporal patterns, making it suitable for personalization and clustering-based analysis.

---

## System Architecture

```
Raw News Data
     ↓
Text Preprocessing (NLP)
     ↓
Word2Vec Embeddings
     ↓
K-Means Clustering (Topic Modeling)
     ↓
User Profile Generation
     ↓
Cosine Similarity Matching
     ↓
Personalized News Recommendations
```

---

## Methodology (Step-by-Step)

### 1)Data Preprocessing

Each news headline undergoes rigorous NLP preprocessing:

* Removal of URLs, numbers, and special characters
* Conversion to lowercase
* Tokenization
* Stopword removal
* Lemmatization

This ensures clean and standardized text suitable for embedding generation.

---

### 2)Exploratory Data Analysis (EDA)

Key insights extracted:

* **Category-wise distribution:** Politics dominates the dataset
* **Month-wise distribution:** Higher article volume from January to May
* **Headline length distribution:** Most headlines fall between 40–90 characters

These insights help tune recommendation diversity and embedding strategies.

---

### 3)Word Embedding Generation (Word2Vec)

* Pretrained **300-dimensional Word2Vec model** is used
* Each headline vector is computed as the **average of its word vectors**
* Captures semantic similarity beyond keyword matching

---

### 4)Article Clustering (K-Means)

* Headlines are clustered into **15 topic-based clusters**
* Optimal number of clusters chosen using **Silhouette Score & Inertia**
* Each cluster represents a generalized news theme

This step enables topic-aware recommendations.

---

### 5)User Profile Creation

* A user profile is created by averaging vectors of articles previously read
* Represents the user’s **interest space** in vector form

---

### 6)Recommendation Generation

* Articles from the user’s preferred clusters are shortlisted
* **Cosine similarity** is computed between user profile and article vectors
* Top-N most similar articles are recommended

This ensures both **relevance and personalization**.

---

## Evaluation Metrics & Results

### Clustering Evaluation

* **Silhouette Score:** `0.65`
* Indicates strong intra-cluster similarity and inter-cluster separation

### Recommendation Evaluation

| Metric       | Value |
| ------------ | ----- |
| Precision@5  | 0.78  |
| Precision@10 | 0.73  |
| Recall@5     | 0.61  |
| Recall@10    | 0.78  |

High precision ensures relevant top recommendations
High recall ensures broader interest coverage

---

## Technologies Used

* **Programming Language:** Python
* **Libraries & Tools:**

  * NumPy, Pandas
  * Scikit-learn
  * NLTK
  * Gensim (Word2Vec)
  * Matplotlib, Seaborn
  * Jupyter Notebook

---

## Project Structure

```
 News-Article-Recommendation
 ┣  News_Article_Recommendation.ipynb
 ┣  Research_Paper.pdf
 ┣  dataset/
 ┣  visualizations/
 ┗  README.md
```

---

##  Key Contributions

* Designed an end-to-end **hybrid recommendation pipeline**
* Implemented semantic article understanding using Word2Vec
* Achieved strong recommendation accuracy with measurable metrics
* Built a modular system adaptable to real-world platforms

---

##  Future Enhancements

* Real-time recommendation updates
* User feedback-based learning loops
* Integration of contextual features (location, device, time)
* Transformer-based embeddings (BERT)
* Online A/B testing for recommendation quality

---


## If you find this project useful

Give it a on GitHub and feel free to fork or contribute!

---

**This project demonstrates strong fundamentals in NLP, machine learning, and recommender systems, making it suitable for academic research, interviews, and real-world applications.**

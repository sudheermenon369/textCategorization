# textCategorization
An NLP tast of assigning various documents to different categories based on the similarity measure.
---Calculating weights---
Here I have used TF-IDF weights to calculate the cosine similarity of documents. Then a comparison of first document to all other documents are made. After the metric has been created, a k-means clustering is done to gather similar documents together. From the cluster table its is clear that documents of same category are floaking together.
---Calculating weights using POS tags---
Here also the clustering is done using metric - cosine value using POS tags.
---Clustering using NER tags as features---
The result shows that using similarity using NER tags is not at all a good measure of finding similar documents. The documents under consideration does not have enough NER tags for proper clustering of documents.

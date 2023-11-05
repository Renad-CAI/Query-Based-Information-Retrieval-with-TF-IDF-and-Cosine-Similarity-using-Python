# Information Retrieval Project

This project is focused on information retrieval using Python and various libraries. It involves reading and preprocessing a set of text documents, calculating TF-IDF scores for both the documents and a query, and then using cosine similarity to rank and retrieve relevant documents for a given query.

## Project Overview

1. **Mount Google Drive**: The project begins by mounting Google Drive to access the dataset and save the results.

2. **Import Libraries**: Import necessary libraries, such as pandas, NLTK, and more.

3. **Read Documents**: Read the text documents from Google Drive and store each document in specific variables (d1 to d10).

4. **Preprocessing Documents**: Perform preprocessing on each document, including lowercasing, tokenization, removing punctuation, removing stop words, lemmatization, and stemming.

5. **Write Preprocessed Documents**: Write the preprocessed documents to new files in the "dataOut" directory on Google Drive.

6. **Read Preprocessed Documents**: Read the preprocessed documents for further analysis.

7. **Compute Most Frequent Words**: Calculate the most frequent words in each document and display the top 4 words for each.

8. **Create a Bag of Words**: Create a wordset from all documents, which will be used to create a dictionary for calculating TF, IDF, and TF-IDF.

9. **Calculate TF for Documents**: Compute Term Frequency (TF) for each word in each document.

10. **Calculate IDF for Documents**: Compute Inverse Document Frequency (IDF) for each word in all documents.

11. **Calculate TF-IDF for Documents**: Compute TF-IDF scores for each word in all documents.

12. **Query Preprocessing**: Preprocess a query by lowercasing, tokenization, removing punctuation, removing stop words, lemmatization, and stemming.

13. **Compute TF-IDF for the Query**: Calculate TF-IDF scores for the query.

14. **Rank Documents by Similarity**: Calculate the cosine similarity between the query and all documents, ranking the documents based on similarity scores.

15. **Display Relevant Documents**: Display relevant and non-relevant documents based on the cosine similarity scores. Documents with NaN scores are considered non-relevant.

This project allows you to search for relevant documents in your dataset using a provided query. 

The documents are ranked based on their similarity to the query, providing a simple information retrieval system.



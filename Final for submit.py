#The purpose of the code below is for my final project
#The book recommendation script for having better sell
#At the beginning I tried to import some librarie
#let's import some libraries

import pandas as pd
import numpy as np
from jupyterlab.semver import test_set
from menuinst.utils import user_is_admin
from networkx.algorithms.bipartite.basic import is_bipartite_node_set
from numpy.core.defchararray import title
from psutil import users

# I used surprise, it is library for building recommendation systems
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
#sklearn helps with text analysis and math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import seaborn as sns
import matplotlib.pyplot as plt
#I chose my dataset from Kaggle
#let's import it and load the csv files
#In my datasets we have 3 csv files
books = pd.read_csv("Books.csv", encoding='latin-1', low_memory=False)
ratings = pd.read_csv("Ratings.csv", encoding='latin-1')
users = pd.read_csv("Users.csv", encoding='latin-1')
#let's see how many rows and columns are in each file
#For showing data shapes(sizes)
print(f"Books: {books.shape}")
print(f"Ratings: {ratings.shape}")
print(f"Users: {users.shape}")
#For cleaning the Data
#I have to remove any repeated data to avoid confusion
books = books.drop_duplicates()
ratings = ratings.drop_duplicates()
users = users.drop_duplicates()
#It is important to remove any books that do not have Title or ISBN
books = books.dropna(subset=["ISBN", "Book-Title"])
#It is important to keep only useful columns we need from the dataset
books[["ISBN", "Book-Title", "Book-Author", "Year-Of-Publication"]]
users = users[["User-ID", "Location", "Age"]]
#It is also important to fill missing ages with the middle value, then remove any user with unrealistic age.
#I mean below 5 or above 100
#so,
users["Age"] = users["Age"].fillna(users["Age"].median())
users = users[(users["Age"] > 5) & (users["Age"] < 100)]
#Let's combine the ratings and book info into one table
# I mean by using the ISBN number to match them
merged_df = ratings.merge(books, on="ISBN")
print(f"Merged data: {merged_df.shape}")
#Let's build collaborative Filtering using surprise library
#It is important to tell surprise what data format we are using
#I mean user ID, book ID, and rating between 1 and 10
reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(merged_df[["User-ID", "ISBN", "Book-Rating"]], reader)
#Based on our aggrement I split data
#I mean 80% for training the model and 20% for testing
trainset, testset= train_test_split(data, test_size=0.2, random_state=42)
#It is better to use an algorithm called SVD to learn from the training data
algo = SVD()
algo.fit(trainset)
#Let's test how good the model is using RMSE and MAE
# The important point is: smaller values mean better predictions
predictions = algo.test(testset)
rmse = accuracy.rmse(predictions, verbose=False)
mae = accuracy.mae(predictions, verbose=False)
# It it time to recommend Books to a user
#First i get a list of books the user has not rated yet
#so,
user_id = 120548
user_books = merged_df[merged_df["User-ID"] == user_id]["ISBN"].unique()
all_books = books["ISBN"].unique()
unseen_books= [book for book in all_books if book not in user_books]
#Predict how much the user would like each of those books
preds = []
for book in unseen_books:
    pred = algo.predict(user_id, book)
    preds.append((book, pred.est))
#Let's sort predictions from best to worst and pick the top 5
preds.sort(key=lambda x: x[1], reverse=True)
top5 = preds[:5]
# showing the top 5 recommended books for the user with predicted scores
print(f"\n📘📘📘Recommendation for user {user_id}:")
for isbn, rating in top5:
    title = books[books["ISBN"] == isbn]["Book-Title"].values[0]
    print(f"{title}: {rating:.2f}")
#Let's talk about Content based filtering using the book info
#Following this, I am comibing the book title and author name into one string for each book
#It is so important to have a string
books['combined'] = books['Book-Title'].fillna('') + books['Book-Author'].fillna('')
#We have learned TF-IDF method so,
#I converted all book text into numbers using TF-IDF ,
# It is important to know common words like "the" are ignored
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(books['combined'])
#Then, I measured similarity between all books
#I mean that how close their text is
#so,
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
#8-Recommend similar books
#Let's define a function: you give it a book title, it returns other similar books
def get_content_recommendations(title, cosine_sim, books_df=books, top_n=5):
    indices = pd.Series(books_df.index, index=books_df["Book-Title"]).drop_duplicates()
    idx = indices.get[title]
    if idx is None:
        print(f"Book '{title}' not found in dataset.")
        return[]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    book_indices = [i[0] for i in sim_scores]
    return book_df['Book-Title'].iloc[book_indices].tolist()
#Let's use that function to get 5 similar books to a specific one
example_title = 'The Lovely Bones: A Novel'
print(f"\n📗📗📗 Content based Recommendation for '{example_title}':")
print(get_content_recommendations(example_title))
#9-Visualize Most Rated Books
#Following this, count which books were rated the most and show them in a pretty graph
top_books = merged_df['Book-Title'].value_counts().head(10)
plt.figure(figsize=(10,6))
sns.barplot(x=top_books.values, y=top_books.index, palette='Blues_d')
plt.title("Top 10 Most Rated Books")
plt.xlabel("Number of Ratings")
plt.ylabel("Book Title")
plt.tight_layout()
plt.show()

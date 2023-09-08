import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Sample book dataset
data = {
    'Title': [
        'Two States', 'To Kill a Mockingbird', '1984', 'The Great Gatsby', 'Pride and Prejudice',
        'Harry Potter', 'The Hobbit', 'Five Point Someone', 'War and Peace', 'The Odyssey'
        # Add more book titles here
    ],
    'Author': [
        'Chetan Bhagat', 'Harper Lee', 'George Orwell', 'F. Scott Fitzgerald', 'Jane Austen',
        'J.K Rowling', 'J.R.R. Tolkien', 'Chetan Bhagat', 'Leo Tolstoy', 'Homer'
        # Add more author names here
    ],
    'Genre': [
        'Romance', 'Fiction', 'Science Fiction', 'Fiction', 'Romance',
        'Fantasy', 'Fantasy', 'Fiction', 'Historical Fiction', 'Classics'
        # Add more genres here
    ],
    'Description': [
        'A classic novel about a young man\'s journey of self-discovery.',
        'A powerful story of racial injustice and moral growth in the American South.',
        'A dystopian novel exploring totalitarianism and thought control.',
        'A tale of wealth, privilege, and the American Dream during the Roaring Twenties.',
        'A timeless love story set in the English countryside.',
        'A futuristic world where conformity and stability are maintained at any cost.',
        'An epic fantasy adventure filled with dragons and magic.',
        'A high-fantasy masterpiece featuring hobbits and a quest to save Middle-earth.',
        'A historical novel set against the backdrop of the Napoleonic era.',
        'An ancient Greek epic poem chronicling the adventures of Odysseus.'
        # Add more book descriptions here
    ]
}

books_df = pd.DataFrame(data)

# Create a TF-IDF vectorizer to convert book descriptions into numerical vectors
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(books_df['Description'])

# Calculate cosine similarity between books
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to get book recommendations based on user preferences
def get_recommendations(book_title, cosine_sim=cosine_sim):
    idx = books_df.index[books_df['Title'] == book_title].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:4]  # Get the top 3 similar books (excluding itself)
    book_indices = [i[0] for i in sim_scores]
    return books_df['Title'].iloc[book_indices]

#Get book recommendations based on user preferences
user_book = input("Enter a book title, author, or genre for recommendations:")
recommendations = get_recommendations(user_book)

print(f"Recommendations for '{user_book}':")
for book in recommendations:
    print(book)

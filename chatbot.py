import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Sample synthetic book dataset
data = {
    'Title': [
        'The Catcher in the Rye', 'To Kill a Mockingbird', '1984', 'The Great Gatsby', 'Pride and Prejudice',
        'Brave New World', 'The Hobbit', 'The Lord of the Rings', 'War and Peace', 'The Odyssey'
        # Add more book titles here
    ],
    'Author': [
        'J.D. Salinger', 'Harper Lee', 'George Orwell', 'F. Scott Fitzgerald', 'Jane Austen',
        'Aldous Huxley', 'J.R.R. Tolkien', 'J.R.R. Tolkien', 'Leo Tolstoy', 'Homer'
        # Add more author names here
    ],
    'Genre': [
        'Fiction', 'Fiction', 'Science Fiction', 'Fiction', 'Romance',
        'Science Fiction', 'Fantasy', 'Fantasy', 'Historical Fiction', 'Classics'
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
def get_recommendations(user_input, cosine_sim=cosine_sim):
    # Check if the user input matches any book titles, authors, or genres
    matches = books_df[books_df['Title'].str.contains(user_input) | 
                       books_df['Author'].str.contains(user_input) | 
                       books_df['Genre'].str.contains(user_input)]
    
    if not matches.empty:
        idx = matches.index
        sim_scores = []
        
        for i in idx:
            sim_scores.extend(list(enumerate(cosine_sim[i])))
        
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:4]  # Get the top 3 similar books (excluding itself)
        book_indices = [i[0] for i in sim_scores]
        return books_df['Title'].iloc[book_indices]
    else:
        return []

# Allow the user to enter a book title, author, or genre for recommendations
user_input = input("Enter a book title, author, or genre for recommendations: ")
recommendations = get_recommendations(user_input)

if recommendations:
    print(f"Recommendations for '{user_input}':")
    for book in recommendations:
        print(book)
else:
    print(f"No recommendations found for '{user_input}'.")

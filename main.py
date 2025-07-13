# Import required libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import random

# === Step 1: Custom Q&A Training Dataset ===
# You can add more entries to improve the chatbot's capabilities
qa_pairs = {
    "hello": "Hi there! How can I help you?",
    "hi": "Hello! What can I do for you?",
    "how are you": "I'm just a bunch of code, but I'm doing great!",
    "what is machine learning": "Machine learning is a field of AI that lets systems learn from data.",
    "what is python": "Python is a popular programming language for AI and more.",
    "bye": "Goodbye! Have a great day!",
    "thanks": "You're welcome!",
    "who are you": "I'm a simple chatbot built with Python and scikit-learn."
}

# === Step 2: Split the data into inputs (X) and outputs (y) ===
X = list(qa_pairs.keys())         # User questions
y = list(qa_pairs.values())       # Corresponding chatbot responses

# === Step 3: Convert text input into numerical vectors ===
# TF-IDF (Term Frequency-Inverse Document Frequency) helps give weight to important words
vectorizer = TfidfVectorizer()
X_vectors = vectorizer.fit_transform(X)  # Learn vocabulary and transform the training data

# === Step 4: Choose your model here ===
# Change this variable to 'nb' for Naive Bayes or 'logreg' for Logistic Regression
model_type = 'logreg'  # 'logreg' or 'nb'

# Initialize the selected model
if model_type == 'logreg':
    model = LogisticRegression()
elif model_type == 'nb':
    model = MultinomialNB()
else:
    raise ValueError("Invalid model_type. Use 'logreg' or 'nb'.")

# Train the model on the vectorized text data
model.fit(X_vectors, y)

# === Step 5: Function to generate chatbot responses ===
def chatbot_response(user_input):
    user_vector = vectorizer.transform([user_input])  # Convert input to vector
    prediction = model.predict(user_vector)           # Predict the closest response
    return prediction[0]

# === Step 6: Run a simple text-based chat loop ===
print("🤖 Chatbot is ready to talk! (Type 'quit' to exit)\n")
while True:
    user_input = input("You: ").strip().lower()
    if user_input == "quit":
        print("🤖 Chatbot: Goodbye!")
        break
    response = chatbot_response(user_input)
    print("🤖 Chatbot:", response)

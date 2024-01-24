<<<<<<< HEAD
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load the dataset
df = pd.read_csv('social_media_data.csv')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train the classifier
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = clf.predict(X_test)

# Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
=======
import instaloader
from textblob import TextBlob
name="main"
def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

def profile_user(username):
    try:
        # Fetch user profile information
        loader = instaloader.Instaloader()
        profile = instaloader.Profile.from_username(loader.context, username)

        # Fetch user posts
        user_posts = [post.caption for post in profile.get_posts() if post.caption is not None]

        # Check if the user has any non-empty posts
        if not user_posts:
            return "No Posts"

        # Profile user based on sentiment analysis
        risk_level = sum(analyze_sentiment(post) for post in user_posts) / len(user_posts)

        if risk_level >= 0.2:
            return "High Risk"
        elif -0.2 < risk_level < 0.2:
            return "Medium Risk"
        else:
            return "Low Risk"

    except instaloader.exceptions.ProfileNotExistsException as e:
        return f"Error: {e}"

if name == "main":
    # Replace 'target_username' with the Instagram username you want to analyze
    username_to_analyze = 'nhce_official'

    # Profile the user and print the risk level
    risk_level = profile_user(username_to_analyze)
    print(f"User: {username_to_analyze}")
    print(f"Risk Level: {risk_level}")
>>>>>>> 08938255751bb4dc8665c2cd56bda7a722e21d6f

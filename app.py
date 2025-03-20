import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for matplotlib to save plots as files
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('vader_lexicon')
nltk.download('wordnet')
nltk.download('stopwords')

# Initialize Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

# Ensure upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Function to check if file has allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Route for main page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle file uploads
@app.route('/uploads', methods=['POST'])
def upload_file():
    if 'comments-file' not in request.files:
        return redirect(request.url)

    comments_file = request.files['comments-file']
    if comments_file.filename == '':
        return redirect(request.url)

    if comments_file and allowed_file(comments_file.filename):
        comments_filename = secure_filename(comments_file.filename)
        comments_filepath = os.path.join(app.config['UPLOAD_FOLDER'], comments_filename)
        comments_file.save(comments_filepath)

        try:
            # Perform sentiment analysis on uploaded file
            results = analyze_sentiment(comments_filepath)
            return render_template('index.html', results=results)

        except Exception as e:
            return f"An error occurred during sentiment analysis: {e}"

    return redirect(request.url)

# Function to perform sentiment analysis
def analyze_sentiment(comments_filepath):
    # Read comments from CSV file into pandas DataFrame
    US_Comments = pd.read_csv(comments_filepath, on_bad_lines='warn')

    # Check for required columns
    if 'likes' not in US_Comments.columns or 'replies' not in US_Comments.columns or 'comment_text' not in US_Comments.columns:
        raise ValueError("The comments file is missing required columns: 'likes', 'replies', 'comment_text'.")

    # Clean and preprocess comment text
    US_Comments.dropna(inplace=True)
    US_Comments = US_Comments.reset_index(drop=True)

    US_Comments['likes'] = pd.to_numeric(US_Comments['likes'], errors='coerce').fillna(0).astype(int)
    US_Comments['replies'] = pd.to_numeric(US_Comments['replies'], errors='coerce').fillna(0).astype(int)

    # Preserve original comments and create a new column for preprocessed comments if needed
    US_Comments['processed_text'] = US_Comments['comment_text'].str.replace("[^a-zA-Z#]", " ")
    US_Comments['processed_text'] = US_Comments['processed_text'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 3]))
    US_Comments['processed_text'] = US_Comments['processed_text'].apply(lambda x: x.lower())

    tokenized_comments = US_Comments['processed_text'].apply(lambda x: x.split())
    wnl = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    tokenized_comments = tokenized_comments.apply(lambda x: [wnl.lemmatize(i) for i in x if i not in stop_words])

    for i in range(len(tokenized_comments)):
        tokenized_comments[i] = ' '.join(tokenized_comments[i])

    US_Comments['processed_text'] = tokenized_comments

    # Perform sentiment analysis using NLTK Vader on original comments
    sia = SentimentIntensityAnalyzer()
    US_Comments['Sentiment Scores'] = US_Comments['comment_text'].apply(lambda x: sia.polarity_scores(x)['compound'])
    US_Comments['Sentiment'] = US_Comments['Sentiment Scores'].apply(lambda s: 'Positive' if s > 0 else ('Neutral' if s == 0 else 'Negative'))

    # Generate sentiment distribution plot
    sentiment_distribution = US_Comments['Sentiment'].value_counts()
    plt.figure(figsize=(8, 5))
    sns.barplot(x=sentiment_distribution.index, y=sentiment_distribution.values)
    plt.title('Distribution of Sentiment Categories')
    plt.xlabel('Sentiment Category')
    plt.ylabel('Number of Comments')
    sentiment_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'sentiment_distribution.png')
    plt.savefig(sentiment_image_path)
    plt.close()

    # Return list of comments with their associated sentiment
    return list(zip(US_Comments['comment_text'], US_Comments['Sentiment']))

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)

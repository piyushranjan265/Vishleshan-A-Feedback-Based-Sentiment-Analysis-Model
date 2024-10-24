import pandas as pd
import re
from transformers import RobertaTokenizer, RobertaForSequenceClassification, pipeline
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# Load the data
data = pd.read_csv('feedback_responses.csv')

# Check for missing columns
required_columns = [
    "What did you like the most about the course content?",
    "What did you like least about the course content?",
    "What improvements would you suggest for the instructor?",
    "Any additional comments or suggestions?"
]
if not all(col in data.columns for col in required_columns):
    raise ValueError("Some required columns are missing from the dataset.")

# Combine relevant text columns into a single column for analysis
data['feedback'] = data.apply(lambda x: ' '.join(str(x[col]) for col in required_columns).strip(), axis=1)

# Basic text preprocessing function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation (optional, depending on needs)
    text = re.sub(r'\d+', '', text)  # Remove numbers
    return text

data['cleaned_feedback'] = data['feedback'].apply(preprocess_text)

# Load the pre-trained RoBERTa model and tokenizer
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name)

# Create a sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Apply the sentiment analysis pipeline to the feedback data and map labels
label_mapping = {
    'LABEL_0': 'Negative',
    'LABEL_1': 'Neutral',
    'LABEL_2': 'Positive'
}
data['sentiment'] = data['cleaned_feedback'].apply(lambda x: label_mapping.get(sentiment_pipeline(x)[0]['label'], 'Unknown'))

# LDA for topic modeling
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
dtm = vectorizer.fit_transform(data['cleaned_feedback'])
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(dtm)
topics = lda.transform(dtm).argmax(axis=1)

# Add topics to dataframe
data['topic'] = topics

# Save the processed data to a new CSV for use in the Streamlit app
data.to_csv('processed_feedback.csv', index=False)

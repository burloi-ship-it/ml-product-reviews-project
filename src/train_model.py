import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib

# load data from dataset
url = "https://raw.githubusercontent.com/burloi-ship-it/ml-product-reviews-project/refs/heads/main/data/product_reviews_full.csv"

# read csv
df = pd.read_csv(url)

# drop all rows with missing values
df = df.dropna()

# convert all sentiment values to lower case and strip spaces
df['sentiment'] = df['sentiment'].astype(str).str.lower().str.strip()

# convert sentiment column type to 'category'
df['sentiment'] = df['sentiment'].astype('category')

# drop columns that are useless for analysis
df = df.drop(columns=['review_uuid', 'product_name', 'product_price'])

# create new column with length of each review text
df['review_length'] = df['review_text'].astype(str).str.len()

# define features and labels
x = df[['review_title', 'review_text', 'review_length']]
y = df['sentiment']

# define preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('title', TfidfVectorizer(), 'review_title'),
        ('text', TfidfVectorizer(), 'review_text'),
        ('length', MinMaxScaler(), ['review_length'])
    ]
)

# define pipeline with the best model
pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('classifier', RandomForestClassifier())
])

# train the model on entire dataset
pipeline.fit(x, y)

# save the model to a file
joblib.dump(pipeline, 'model/trained_model.pkl')

print('Model saved as model/trained_model.pkl')
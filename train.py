import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# 1. Example Data (तू तुमचा data वापर)
data = {
    'text': ["हे खरे आहे", "हे फेक आहे", "सरकारने नवीन योजना जाहीर केली", "फेक बातमी: कोणाचा स्कँडल"],
    'label': [1, 0, 1, 0]  # 1=Real, 0=Fake
}
df = pd.DataFrame(data)

# 2. Vectorizer तयार करा
vectorizer = TfidfVectorizer(stop_words='english')

# 3. Text ला vector मध्ये convert करा
X = vectorizer.fit_transform(df['text'])

# 4. Model train करा
model = LogisticRegression()
model.fit(X, df['label'])

# 5. Model आणि Vectorizer save करा
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("Model आणि Vectorizer तयार झाले.")

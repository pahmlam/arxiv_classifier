from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer

# Khởi tạo App
app = FastAPI()

# Mount thư mục static để phục vụ file HTML
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load Resources (Chạy 1 lần khi start app để tối ưu tốc độ)
print("Đang tải tài nguyên...")
svm_model = joblib.load("models/svm_model.pkl")
label_map = joblib.load("models/label_map.pkl")
id_to_label = {v: k for k, v in label_map.items()}
embedding_model = SentenceTransformer('intfloat/multilingual-e5-base')

# Setup NLTK
nltk.download('stopwords')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

class AbstractInput(BaseModel):
    abstract: str

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/predict")
async def predict_topic(input_data: AbstractInput):
    # 1. Preprocess
    clean_text = preprocess_text(input_data.abstract)
    
    # 2. Vectorize (Embeddings)
    # Lưu ý format của e5 model 
    formatted_text = f"query: {clean_text}"
    vector = embedding_model.encode([formatted_text])
    
    # 3. Predict
    prediction_id = svm_model.predict(vector)[0]
    predicted_label = id_to_label[prediction_id]
    
    return {
        "topic": predicted_label,
        "processed_text": clean_text
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
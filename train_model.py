import re
import numpy as np
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

# Tải dữ liệu cần thiết cho NLTK (Cải tiến tiền xử lý)
nltk.download('stopwords')
nltk.download('wordnet')

# --- CẤU HÌNH ---
MODEL_PATH = "models/svm_model.pkl"
LABEL_MAP_PATH = "models/label_map.pkl"
EMBEDDING_MODEL_NAME = 'intfloat/multilingual-e5-base' # Model dùng trong PDF 
CATEGORIES_TO_SELECT = ['astro-ph', 'cond-mat', 'cs', 'math', 'physics'] 

# --- 1. TIỀN XỬ LÝ NÂNG CAO (IMPROVEMENT)  ---
def advanced_preprocessing(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Chuyển về chữ thường & bỏ ký tự đặc biệt 
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Loại bỏ stopwords và Lemmatization (đưa từ về dạng gốc)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    return " ".join(words)

# --- 2. TẢI VÀ CHUẨN BỊ DỮ LIỆU ---
def load_and_prepare_data():
    print("Đang tải dataset...")
    # Dataset từ Hugging Face 
    ds = load_dataset("UniverseTBD/arxiv-abstracts-large", split="train", streaming=True)
    
    samples = []
    print("Đang lọc dữ liệu (lấy 2000 mẫu để demo nhanh)...")
    
    count = 0
    for s in ds:
        cats = s['categories'].split()
        primary_cat = cats[0].split('.')[0]
        
        if primary_cat in CATEGORIES_TO_SELECT:
            samples.append({
                "text": s['abstract'],
                "label": primary_cat
            })
            count += 1
            if count >= 2000: # Tăng số lượng mẫu so với PDF (1000) để model tốt hơn
                break
    
    return samples

# --- 3. HUẤN LUYỆN ---
def train():
    data = load_and_prepare_data()
    
    # Preprocessing
    print("Đang tiền xử lý...")
    texts = [advanced_preprocessing(item['text']) for item in data]
    labels = [item['label'] for item in data]
    
    # Label Encoding
    unique_labels = sorted(list(set(labels)))
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    y = np.array([label_to_id[label] for label in labels])
    
    # Vectorization (Sentence Embeddings) [cite: 374]
    print("Đang mã hóa văn bản (Embeddings)...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    # Thêm tiền tố "query: " nếu dùng model e5 
    formatted_texts = [f"query: {t}" for t in texts] 
    X = embedding_model.encode(formatted_texts, show_progress_bar=True)
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Model Tuning (Cải tiến: GridSearch cho SVM) 
    print("Đang huấn luyện và tối ưu SVM...")
    param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
    grid.fit(X_train, y_train)
    
    best_model = grid.best_estimator_
    
    # Đánh giá
    y_pred = best_model.predict(X_test)
    print(f"Accuracy tốt nhất: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, target_names=unique_labels))
    
    # Lưu model
    print("Đang lưu model...")
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(label_to_id, LABEL_MAP_PATH)
    print("Hoàn tất!")

if __name__ == "__main__":
    train()
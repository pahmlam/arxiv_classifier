# ArXiv Abstract Classification Project

Dự án Web App phân loại chủ đề của các bài báo khoa học (ArXiv Papers) dựa trên đoạn tóm tắt (Abstract). Hệ thống sử dụng **FastAPI** cho backend, **Sentence Embeddings** để mã hóa văn bản và thuật toán **SVM** để phân loại.

## 🚀 Tính năng chính

* **Phân loại đa lớp:** Dự đoán thuộc về 1 trong 5 nhóm chủ đề: `astro-ph`, `cond-mat`, `cs`, `math`, `physics`.
* **Web Interface:** Giao diện HTML/CSS đơn giản, thân thiện để nhập liệu và xem kết quả.
* **API Service:** Backend FastAPI hiệu năng cao.
* **Advanced Preprocessing:** Tích hợp xử lý ngôn ngữ tự nhiên nâng cao (Lemmatization, Stopwords removal).
* **Hyperparameter Tuning:** Tự động tối ưu tham số mô hình SVM bằng Grid Search.

## 📂 Cấu trúc dự án

```text
arxiv_project/
├── models/               # Chứa model SVM và Label map đã huấn luyện (.pkl)
├── static/               # Chứa giao diện Frontend
│   └── index.html
├── app.py                # Web Server (FastAPI)
├── train_model.py        # Pipeline huấn luyện & Đánh giá mô hình
├── requirements.txt      # Các thư viện cần thiết
├── .gitignore            # Cấu hình Git ignore
└── README.md             # Hướng dẫn sử dụng
````

## 🛠️ Cài đặt

1.  **Clone dự án hoặc tải về máy:**

    ```bash
    git clone <your-repo-url>
    cd arxiv_project
    ```

2.  **Tạo và kích hoạt môi trường ảo (Khuyên dùng):**

    ```bash
    python -m venv venv
    # Windows:
    venv\Scripts\activate
    # Mac/Linux:
    source venv/bin/activate
    ```

3.  **Cài đặt các thư viện:**

    ```bash
    pip install -r requirements.txt
    ```

## 🧠 Huấn luyện Mô hình

Trước khi chạy web app, bạn cần huấn luyện mô hình để tạo ra file `.pkl` trong thư mục `models/`.

Chạy lệnh sau:

```bash
python train_model.py
```

**Quá trình này bao gồm:**

1.  Tải dataset `UniverseTBD/arxiv-abstracts-large` từ Hugging Face.
2.  [cite_start]**Tiền xử lý nâng cao (Improvement):** Loại bỏ ký tự đặc biệt, chuyển chữ thường, Lemmatization (đưa từ về nguyên mẫu) và loại bỏ Stopwords (từ vô nghĩa)[cite: 2180].
3.  [cite_start]**Mã hóa (Vectorization):** Sử dụng mô hình `intfloat/multilingual-e5-base` (Sentence Embeddings) để hiểu ngữ nghĩa tốt hơn so với Bag-of-Words truyền thống[cite: 2181].
4.  [cite_start]**Tối ưu hóa (Tuning):** Sử dụng `GridSearchCV` để tìm tham số tốt nhất cho SVM[cite: 2184].
5.  Lưu model vào thư mục `models/`.

## 🌐 Khởi chạy Web App

Sau khi huấn luyện xong, khởi động server FastAPI:

```bash
python app.py
```

  * Truy cập địa chỉ: `http://localhost:8000` trên trình duyệt.
  * Nhập một đoạn abstract tiếng Anh và nhấn **Dự đoán**.

## 📝 Ví dụ Input

Bạn có thể thử nhập đoạn văn bản sau (thuộc về Computer Science - CS):

> "We propose a novel deep learning architecture for image recognition tasks. The model utilizes convolutional neural networks combined with attention mechanisms to improve feature extraction."

## 📈 Cải tiến đã thực hiện

1.  **Tiền xử lý chuyên sâu:** Sử dụng thư viện NLTK để thực hiện Lemmatization thay vì chỉ cắt từ đơn giản.
2.  [cite_start]**Embedding hiện đại:** Thay thế TF-IDF bằng Sentence Transformers (S-BERT approach) giúp độ chính xác tăng lên đáng kể (\~88% so với \~60-70% của các phương pháp cũ)[cite: 2055, 2182].
3.  **Model Tuning:** Tích hợp Grid Search để không phải chọn tham số `C` và `kernel` thủ công.

## 🤝 Đóng góp

Mọi đóng góp xin vui lòng tạo Pull Request hoặc mở Issue.

## LICENSE

Distributed under the MIT License. See LICENSE.txt for more information.

Copyright (c) 2025 Pham Tung Lam

```
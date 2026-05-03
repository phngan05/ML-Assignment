# Machine Learning Assignment
## Course Information
- **Course**: Machine Learning  
- **Semester**: 252 (2025–2026)  
- **University**: Ho Chi Minh City University of Technology (HCMUT)  
- **Instructor**: Dr. Trương Vĩnh Lân 
---

## Team Members
| Name | Student ID | Email |
|------|-----------|-------|
| Phạm Hồng Ngân | 2312232 | ngan.phamhong@hcmut.edu.vn |
| Nguyễn Khánh Trinh | 2313579 | trinh.nguyenkhanh@hcmut.edu.vn|
| Phan Lê Thiên | 2313221 | thien.phan2411@hcmut.edu.vn |
| Nguyễn Tấn Lộc | 2311957 | loc.nguyen2311957@hcmut.edu.vn |
---

## Objective
The primary goal of this project is to apply theoretical machine learning knowledge to solve real-world problems using a structured pipeline. Our team focuses on the following objectives: 

- Implement a Traditional Text ML Pipeline: Develop a structured workflow including specialized Text EDA, preprocessing (tokenization, stopword removal), feature extraction, and supervised learning.  

- Advanced Text Preprocessing: Execute rigorous cleaning of academic text, including handling scientific notations, lemmatization, and padding to ensure data quality.  

- Feature Extraction: Compare different vectorization techniques to transform text into numerical data:
    + Traditional Methods: Bag-of-Words (BoW) or TF-IDF.  
    + Modern Embeddings: Utilizing pre-trained models (SciBERT) and saving them as .npy files for downstream tasks.  
    + Model Optimization: Evaluate the performance of various classification algorithms to find the most accurate model for classification.  
    + Exploratory Data Analysis (Text-focused): Conduct in-depth EDA to analyze word frequency distributions, document length statistics, and label balance within the article dataset.  
    + Reproducible Research: Deliver a fully functional Google Colab notebook that executes the entire pipeline seamlessly and provides a comparative analysis of the results.

---

## Project Structure
```sh
root/
│
├── notebooks/
│ └── main.ipynb
│
├── modules/
│ ├── preprocessing.py
│ ├── feature_engineering.py
│ └── model.py
│
├── features/ # Extracted features (.npy / .h5)
│
├── reports/
│ └── report.pdf
│
├── requirements.txt
└── README.md
```
---

## How to Run

### Run on Google Colab (Recommended)

1. Open the notebook.

2. Run all cells.

3. The dataset will be automatically downloaded from public sources.

---

### Run locally

1. Clone the repository:
```sh
git clone https://github.com/phngan05/ML-Assignment
cd ML-Assignment
```

2. Run the main.ipynb with Colab Environment

## Link
- **Colab Notebook**: [Open In Colab](https://colab.research.google.com/github/phngan05/ML-Assignment/blob/main/notebooks/main.ipynb)
- **Report PDF**: 

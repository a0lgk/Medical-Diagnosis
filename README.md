# 🏥 AI Medical Diagnosis System

Interactive web application that predicts medical diagnoses from natural language symptom descriptions.

---

## ⚠️ IMPORTANT DISCLAIMER

**THIS TOOL IS FOR EDUCATIONAL PURPOSES ONLY**

- ❌ Does **NOT** replace professional medical diagnosis
- ❌ Should **NOT** be used to make medical decisions
- ✅ **ALWAYS** consult a qualified doctor for health issues
- ✅ In case of medical emergency, call emergency services immediately

---

## 📋 Table of Contents

1. [Description](#description)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Architecture](#architecture)
6. [Results](#results)
7. [Possible Improvements](#possible-improvements)

---

## 📖 Description

This project implements a text classification system to predict medical diagnoses from symptom descriptions. It uses:

- **Machine Learning**: Multi-class classification with Scikit-learn
- **NLP**: TF-IDF vectorization for text analysis
- **Interface**: Interactive web application with Gradio

### Features

✅ Prediction of 22 different diagnoses  
✅ Confidence scores for each prediction  
✅ Intuitive user interface  
✅ Consultation history  
✅ Pre-loaded examples  
✅ Probability charts  

---

## 📊 Dataset

**Source**: [gretelai/symptom_to_diagnosis](https://huggingface.co/datasets/gretelai/symptom_to_diagnosis)

- **Total**: 1,065 examples
- **Train**: 853 examples
- **Test**: 212 examples
- **Classes**: 22 medical diagnoses
- **Language**: English
- **Format**: Natural language descriptions

### Included Diagnoses

The 22 possible diagnoses are:

1. Allergy
2. Arthritis
3. Bronchial Asthma
4. Cervical Spondylosis
5. Chicken Pox
6. Common Cold
7. Dengue
8. Diabetes
9. Drug Reaction
10. Fungal Infection
11. Gastroesophageal Reflux Disease
12. Hypertension
13. Impetigo
14. Jaundice
15. Malaria
16. Migraine
17. Peptic Ulcer Disease
18. Pneumonia
19. Psoriasis
20. Typhoid
21. Urinary Tract Infection
22. Varicose Veins

---

## 🚀 Installation

### Prerequisites

- Python 3.8+
- pip
- Jupyter Notebook (recommended)

### Steps

1. **Download the files**

2. **Install dependencies**

Using requirements.txt:
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install datasets scikit-learn pandas numpy matplotlib seaborn joblib gradio
```

---

## 💻 Usage

### Option 1: All-in-One Jupyter Notebook (Recommended)

Open and run the complete notebook:

```
medical_diagnosis_complete.ipynb
```

This notebook contains everything in one place:
1. **Data Exploration**: Load and visualize the dataset
2. **Model Training**: Train and compare 5 different models
3. **Web Interface**: Launch the interactive Gradio app

**Simply run all cells in order!**

### Option 2: Separate Files

If you prefer to work with separate files:

#### Step 1: Explore the Dataset (Optional)

```
medical_symptom_diagnosis_exploration.ipynb
```

#### Step 2: Train the Model

```
medical_diagnosis_modeling.ipynb
```

This will generate:
- `best_model.pkl`
- `tfidf_vectorizer.pkl`
- `model_metadata.json`

#### Step 3: Launch the Web Interface

```bash
python medical_diagnosis_app.py
```

Access at: **http://localhost:7860**

---

## 🏗️ Architecture

### 1. Preprocessing (TF-IDF)

```python
TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 3),    # Unigrams, bigrams, trigrams
    min_df=2,
    max_df=0.8,
    stop_words='english'
)
```

### 2. Models Tested

| Model | Type | Key Hyperparameters |
|-------|------|---------------------|
| Logistic Regression | Linear | `max_iter=1000`, `class_weight='balanced'` |
| Random Forest | Ensemble | `n_estimators=200`, `max_depth=20` |
| SVM | Kernel-based | `kernel='linear'`, `C=1.0` |
| Naive Bayes | Probabilistic | `alpha=0.1` |
| Gradient Boosting | Ensemble | `n_estimators=100`, `learning_rate=0.1` |

### 3. Prediction Pipeline

```
Text → TF-IDF Vectorization → Model → Probabilities → Top N Diagnoses
```

---

## 📈 Results

### Typical Performance

Exact performance depends on the selected model, but generally:

- **Accuracy**: ~0.90-0.95
- **F1-Score**: ~0.90-0.95
- **Precision**: ~0.90-0.95
- **Recall**: ~0.90-0.95

### Evaluation Metrics

The modeling notebook provides:

✅ Complete Classification Report  
✅ Confusion Matrix  
✅ Comparison of all models  
✅ Error analysis  
✅ Confidence distribution  

---

## 🎯 User Interface

### Features

1. **Text Area**
   - Describe your symptoms in English
   - Support for detailed descriptions

2. **Slider**
   - Choose the number of diagnoses to display (1-10)

3. **Results**
   - Top N diagnoses with probabilities
   - Visual confidence bars
   - Medical recommendations

4. **Chart**
   - Probability distribution as a bar plot

5. **History**
   - Previous consultations with timestamps

6. **Examples**
   - 8 pre-loaded examples for testing

---

## 🔧 Possible Improvements

### Short Term

- [ ] Multi-language support (French, Spanish, etc.)
- [ ] Information about each diagnosis
- [ ] Examination recommendation system
- [ ] Export results as PDF

### Medium Term

- [ ] Multi-label classification (multiple simultaneous diagnoses)
- [ ] Advanced models (BERT, BioBERT)
- [ ] Consider age, gender, medical history
- [ ] Database for persistent history

### Long Term

- [ ] Integration with medical APIs
- [ ] Q&A system to refine diagnosis
- [ ] User feedback to improve the model
- [ ] Cloud deployment (AWS, GCP, Azure)

---

## 📁 File Structure

```
.
├── README.md                                      # This file
├── requirements.txt                               # Python dependencies
├── medical_diagnosis_complete.ipynb               # All-in-one notebook (RECOMMENDED)
├── medical_symptom_diagnosis_exploration.ipynb    # Dataset exploration (optional)
├── medical_diagnosis_modeling.ipynb               # Model training (optional)
├── medical_diagnosis_app.py                       # Gradio app (optional)
├── best_model.pkl                                 # Trained model (generated)
├── tfidf_vectorizer.pkl                           # Vectorizer (generated)
└── model_metadata.json                            # Metadata (generated)
```

---

## 🛠️ Technologies Used

- **Python** 3.8+
- **Scikit-learn**: Machine learning
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Matplotlib / Seaborn**: Visualizations
- **Gradio**: Web interface
- **Hugging Face Datasets**: Dataset loading
- **Joblib**: Model serialization

---

## 📚 Resources

- [Hugging Face Dataset](https://huggingface.co/datasets/gretelai/symptom_to_diagnosis)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Gradio Documentation](https://www.gradio.app/)
- [TF-IDF Explained](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)

---

## 🤝 Contributing

This project is for educational purposes. Feel free to:

- Test different models
- Improve preprocessing
- Add features to the interface
- Experiment with other medical datasets

---

## ⚖️ License

Dataset: Apache 2.0 (gretelai/symptom_to_diagnosis)

---

## 👨‍💻 Author

Project developed as part of learning Machine Learning applied to healthcare.

---

## 🎓 Key Learnings

1. **NLP**: Text vectorization with TF-IDF
2. **Classification**: Multi-class with multiple algorithms
3. **Evaluation**: Metrics, confusion matrix, error analysis
4. **Deployment**: User interface with Gradio
5. **Responsibility**: Importance of disclaimers in healthcare

---

## 🚀 Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Open the complete notebook**:
   ```bash
   jupyter notebook medical_diagnosis_complete.ipynb
   ```

3. **Run all cells** in order

4. **Test the interface** that appears at the bottom of the notebook!

---

**🚨 FINAL REMINDER**: This tool does NOT replace professional medical diagnosis. Always consult a qualified doctor.

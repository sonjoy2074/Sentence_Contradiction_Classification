# Sentence Contradiction Classification

## Project Overview
This project aims to classify pairs of sentences into one of three categories based on their semantic relationships:
- **Contradiction**: Sentences have opposite meanings.
- **Entailment**: One sentence logically follows from the other.
- **Neutral**: Sentences are related but do not imply each other.

The goal is to build and evaluate machine learning models to achieve high accuracy in this classification task.

## Dataset Description
The dataset consists of labeled and unlabeled data files:
- **train.csv**: Contains labeled training data with the following columns:
  - `id`: Unique identifier for each sentence pair.
  - `sentence1`: The first sentence (Premise).
  - `sentence2`: The second sentence (Hypothesis).
  - `label`: The relationship classification:
    - `0` = Contradiction
    - `1` = Neutral
    - `2` = Entailment
- **test.csv**: Unlabeled data for predictions.

## Model Implementation Details

### Step 1: Exploratory Data Analysis (EDA)
- Analyzed class distribution and text patterns.
- Visualized the distribution of `Contradiction`, `Entailment`, and `Neutral` labels.
- Inspected sentence lengths, word distributions, and common words.
- Checked for missing values and outliers.

### Step 2: Text Preprocessing
- Tokenized sentences into words.
- Converted text to lowercase.
- Removed stop words, special characters, and punctuation.
- Applied stemming/lemmatization to normalize words.
- Used TF-IDF for feature extraction, creating a numerical representation of the text.

### Step 3: Model Creation
Implemented the following models:

#### Baseline Models
1. **Random Forest**: A tree-based ensemble learning algorithm.
2. **Decision Tree**: A simple tree-based model for classification.
3. **XGBoost**: An advanced gradient boosting method.

#### Neural Networks
- **Custom Artificial Neural Network (ANN)**: Includes dense layers with dropout for regularization.

#### Advanced Models
- **LSTM**: A sequence-based model for handling text input with memory capabilities.

#### Transformer-Based Models
- **BERT (Bidirectional Encoder Representations from Transformers)**: Fine-tuned for contextual understanding and high accuracy.

### Step 4: Model Evaluation
- Evaluated models using accuracy, precision, recall, F1-score, and confusion matrix.
- Visualized model performance using confusion matrices.

## Steps to Run the Code
1. Clone the repository:
   ```bash
   git clone <repository_url>
   ```
2. Navigate to the project directory:
   ```bash
   cd sentence-classification
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Python script or Jupyter Notebook for model training and evaluation.
   ```bash
   python main.py
   ```

## Model Evaluation Results
1. **Random Forest**: Achieved baseline performance.
2. **Decision Tree**: Provided interpretable results but lower accuracy.
3. **XGBoost**: Performed better with optimized hyperparameters.
4. **ANN**: Showed good performance with tuned parameters.
5. **LSTM**: Captured sequential information and improved classification accuracy.
6. **BERT**: Delivered the highest accuracy with contextual embeddings and fine-tuning.

### Confusion Matrices
- Plotted confusion matrices for each model to analyze misclassifications.

## Additional Observations
- Fine-tuning hyperparameters and using contextual embeddings significantly improved model performance.
- Data preprocessing and feature extraction steps had a major impact on accuracy.

---

For further details, refer to the included performance evaluation report and the source code in the repository.


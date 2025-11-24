# Titanic Survival Prediction Using Logistic Regression

![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-success)

Binary classification project using Logistic Regression to predict survival on the Titanic.

## Overview

This project implements a complete machine learning pipeline for binary classification on the famous Titanic dataset from Kaggle. The goal is to predict whether a passenger survived the Titanic disaster based on various features such as age, sex, passenger class, and more.

**Course:** Basics of AI  
**Task:** Binary Classification  
**Algorithm:** Logistic Regression  
**Dataset:** [Titanic - Machine Learning from Disaster (Kaggle)](https://www.kaggle.com/c/titanic)

## Dataset Information

The Titanic dataset contains information about passengers aboard the RMS Titanic, which sank on April 15, 1912. The dataset includes:

- **Total Passengers:** 891 (in training set)
- **Features:** 11 (including passenger demographics, ticket information, etc.)
- **Target Variable:** Survived (0 = No, 1 = Yes)
- **Survival Rate:** ~38.4%

### Key Features:
- `Pclass`: Passenger class (1st, 2nd, 3rd)
- `Sex`: Gender of passenger
- `Age`: Age in years
- `SibSp`: Number of siblings/spouses aboard
- `Parch`: Number of parents/children aboard
- `Fare`: Passenger fare
- `Embarked`: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

## Methodology

The project follows a comprehensive 12-step machine learning pipeline:

1. **Data Loading** - Load the Titanic dataset from CSV
2. **Data Exploration** - Understand data structure, distributions, and missing values
3. **Data Visualization** - Create visual representations of data patterns
4. **Data Preparation** - Handle missing values and drop irrelevant columns
5. **Feature Engineering** - Create new features and encode categorical variables
6. **Train/Test Split** - Split data into training (80%) and testing (20%) sets
7. **Feature Normalization** - Standardize features using StandardScaler
8. **Model Training** - Train Logistic Regression model
9. **Cross-Validation** - Perform 5-fold cross-validation
10. **Hyperparameter Tuning** - Optimize model using GridSearchCV with regularization
11. **Model Evaluation** - Comprehensive evaluation with multiple metrics
12. **Results Visualization** - Generate visualizations of model performance

## Requirements

### Dependencies

```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
```

### System Requirements
- Python 3.7 or higher
- 2GB RAM minimum
- Operating System: Windows, macOS, or Linux

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/wutangrepo/titanic-logistic-regression.git
cd titanic-logistic-regression
```

2. **Create a virtual environment (recommended):**
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install required packages:**
```bash
pip install -r requirements.txt
```

4. **Download the Titanic dataset:**
   - Visit [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic/data)
   - Download `train.csv`
   - Place it in the project root directory

## Usage

Run the main script:

```bash
python titanic_logistic_regression.py
```

The script will execute the complete pipeline and generate output files automatically.

For detailed usage instructions, see [USAGE.md](USAGE.md).

## Project Structure

```
titanic-logistic-regression/
│
├── titanic_logistic_regression.py    # Main script with complete pipeline
├── README.md                          # This file
├── USAGE.md                          # Detailed usage guide
├── requirements.txt                   # Python dependencies
├── LICENSE                           # MIT License
├── .gitignore                        # Git ignore rules
│
├── train.csv                         # Titanic dataset (download separately)
│
└── Generated Output Files:
    ├── data_visualization.png        # Data exploration plots
    ├── model_evaluation.png          # Model performance visualizations
    ├── feature_coefficients.csv      # Feature importance rankings
    └── model_results.txt             # Summary of results
```

## Expected Outputs

When you run the script, it generates the following files:

1. **data_visualization.png** - Six plots showing:
   - Survival distribution
   - Age distribution
   - Survival by sex
   - Survival by passenger class
   - Fare distribution
   - Feature correlation heatmap

2. **model_evaluation.png** - Four plots showing:
   - Confusion matrix
   - ROC curve with AUC score
   - Top 10 feature coefficients
   - Distribution of predicted probabilities

3. **feature_coefficients.csv** - Complete list of features and their coefficients, sorted by importance

4. **model_results.txt** - Text file containing:
   - Cross-validation results
   - Final model performance metrics
   - Training and testing accuracy

## Key Features Implemented

### Data Preprocessing
- ✅ Missing value imputation (Age, Embarked, Fare)
- ✅ Irrelevant column removal (PassengerId, Name, Ticket, Cabin)
- ✅ Feature encoding (Sex, Embarked)

### Feature Engineering
- ✅ FamilySize (SibSp + Parch + 1)
- ✅ IsAlone (binary indicator)
- ✅ AgeGroup (categorical age bins)
- ✅ FareGroup (categorical fare bins)
- ✅ One-hot encoding for categorical variables

### Model Training & Evaluation
- ✅ Logistic Regression with L1/L2 regularization
- ✅ 5-fold cross-validation
- ✅ Hyperparameter tuning (GridSearchCV)
- ✅ StandardScaler normalization
- ✅ Stratified train/test split (80/20)

### Evaluation Metrics
- ✅ Accuracy (training and testing)
- ✅ Precision
- ✅ Recall
- ✅ F1-Score
- ✅ ROC-AUC Score
- ✅ Confusion Matrix
- ✅ Classification Report

### Visualizations
- ✅ Data distribution plots
- ✅ Survival analysis by features
- ✅ Correlation heatmap
- ✅ ROC curve
- ✅ Feature importance plot
- ✅ Prediction probability distribution

## Results Summary

*Results will be generated after running the script. Expected performance metrics:*

- **Cross-Validation Accuracy:** ~79-82%
- **Testing Accuracy:** ~78-82%
- **Precision:** ~75-80%
- **Recall:** ~70-75%
- **F1-Score:** ~73-77%
- **ROC-AUC Score:** ~84-87%

*Note: Actual results may vary slightly due to random initialization and data splits.*

## Course Information

**Course:** Basics of AI  
**Institution:** [Your Institution]  
**Academic Year:** [Your Year]  
**Topic:** Binary Classification with Logistic Regression

This project demonstrates fundamental concepts in machine learning including:
- Data preprocessing and cleaning
- Feature engineering and selection
- Model training and validation
- Hyperparameter optimization
- Model evaluation and interpretation

## Future Enhancements

Potential improvements for this project:

- [ ] Add ensemble methods (Random Forest, Gradient Boosting)
- [ ] Implement feature selection algorithms
- [ ] Add more advanced feature engineering
- [ ] Create an interactive dashboard (Streamlit/Dash)
- [ ] Deploy model as a web API
- [ ] Add CI/CD pipeline with GitHub Actions
- [ ] Implement model interpretability (SHAP values)
- [ ] Create a Jupyter notebook version

## Contributing

This is an academic project, but suggestions and improvements are welcome! Feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -m 'Add some improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Kaggle](https://www.kaggle.com) for providing the Titanic dataset
- Scikit-learn team for excellent machine learning tools
- Course instructors and teaching assistants
- The data science community

## Contact

For questions or feedback about this project:

- **Repository:** [github.com/wutangrepo/titanic-logistic-regression](https://github.com/wutangrepo/titanic-logistic-regression)
- **Issues:** [Report an issue](https://github.com/wutangrepo/titanic-logistic-regression/issues)

---

**Made with ❤️ for the Basics of AI course**

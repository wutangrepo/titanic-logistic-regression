# Usage Guide

This guide provides detailed instructions on how to set up and run the Titanic Logistic Regression project.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Setup Instructions](#setup-instructions)
3. [Downloading the Dataset](#downloading-the-dataset)
4. [Running the Script](#running-the-script)
5. [Understanding the Output](#understanding-the-output)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Usage](#advanced-usage)

## Prerequisites

Before you begin, ensure you have the following installed on your system:

- **Python 3.7 or higher** - [Download Python](https://www.python.org/downloads/)
- **pip** (Python package manager) - Usually comes with Python
- **Git** (optional, for cloning the repository) - [Download Git](https://git-scm.com/downloads)

### Verify Your Installation

Open a terminal/command prompt and run:

```bash
python --version
# or
python3 --version
```

You should see output like: `Python 3.7.x` or higher.

```bash
pip --version
# or
pip3 --version
```

You should see pip version information.

## Setup Instructions

### Step 1: Get the Project

#### Option A: Clone with Git (Recommended)

```bash
git clone https://github.com/wutangrepo/titanic-logistic-regression.git
cd titanic-logistic-regression
```

#### Option B: Download ZIP

1. Go to the [repository page](https://github.com/wutangrepo/titanic-logistic-regression)
2. Click the green "Code" button
3. Select "Download ZIP"
4. Extract the ZIP file
5. Navigate to the extracted folder in your terminal

### Step 2: Create a Virtual Environment (Recommended)

Creating a virtual environment keeps your project dependencies isolated.

#### On Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

#### On macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` appear in your terminal prompt, indicating the virtual environment is active.

### Step 3: Install Dependencies

With the virtual environment activated, install the required packages:

```bash
pip install -r requirements.txt
```

This will install:
- numpy (numerical computing)
- pandas (data manipulation)
- matplotlib (plotting)
- seaborn (statistical visualizations)
- scikit-learn (machine learning)

### Step 4: Verify Installation

Check that all packages are installed correctly:

```bash
pip list
```

You should see all the required packages listed.

## Downloading the Dataset

The Titanic dataset must be downloaded separately from Kaggle.

### Method 1: Manual Download (Easiest)

1. **Go to the Kaggle competition page:**
   - Visit: https://www.kaggle.com/c/titanic/data

2. **Sign in or create a Kaggle account** (free)

3. **Download the data:**
   - Click on "Download All" or download `train.csv` specifically
   - If you downloaded a ZIP file, extract it

4. **Place the file:**
   - Move or copy `train.csv` to your project directory
   - The file should be in the same folder as `titanic_logistic_regression.py`

### Method 2: Using Kaggle API (Advanced)

If you prefer using the command line:

1. **Install Kaggle API:**
   ```bash
   pip install kaggle
   ```

2. **Set up API credentials:**
   - Go to your Kaggle account settings
   - Scroll to "API" section
   - Click "Create New API Token"
   - This downloads `kaggle.json`
   - Place it in:
     - Windows: `C:\Users\<YourUsername>\.kaggle\kaggle.json`
     - macOS/Linux: `~/.kaggle/kaggle.json`

3. **Download the dataset:**
   ```bash
   kaggle competitions download -c titanic
   unzip titanic.zip
   ```

### Verify Dataset

Check that `train.csv` is in your project directory:

```bash
# Windows
dir train.csv

# macOS/Linux
ls -l train.csv
```

## Running the Script

Once you have the dataset in place, you're ready to run the script!

### Basic Execution

Simply run:

```bash
python titanic_logistic_regression.py
```

### What Happens When You Run It

The script will execute 12 steps sequentially:

1. **Data Loading** - Loads train.csv
2. **Data Exploration** - Displays dataset statistics
3. **Data Visualization** - Creates data_visualization.png
4. **Data Preparation** - Cleans the data
5. **Feature Engineering** - Creates new features
6. **Train/Test Split** - Splits data 80/20
7. **Feature Normalization** - Standardizes features
8. **Model Training** - Trains logistic regression
9. **Cross-Validation** - Performs 5-fold CV
10. **Hyperparameter Tuning** - Optimizes parameters
11. **Model Evaluation** - Calculates metrics
12. **Results Visualization** - Creates model_evaluation.png

### Expected Runtime

- **Normal execution:** 30-60 seconds
- **With hyperparameter tuning:** 1-3 minutes

The script will display progress information for each step in the terminal.

## Understanding the Output

After the script completes, you'll find four new files:

### 1. data_visualization.png

A 2x3 grid of plots showing:
- **Survival Distribution:** Bar chart of survived vs. not survived
- **Age Distribution:** Histogram of passenger ages
- **Survival by Sex:** Bar chart comparing male vs. female survival
- **Survival by Pclass:** Bar chart showing survival by passenger class
- **Fare Distribution:** Histogram of ticket fares
- **Correlation Heatmap:** Feature correlations

**Use:** Understand the dataset and identify important patterns

### 2. model_evaluation.png

A 2x2 grid of plots showing:
- **Confusion Matrix:** True/False Positives/Negatives
- **ROC Curve:** Model discrimination ability with AUC score
- **Feature Coefficients:** Top 10 most important features
- **Prediction Distribution:** Histogram of predicted probabilities

**Use:** Assess model performance and understand predictions

### 3. feature_coefficients.csv

A CSV file containing:
- All feature names
- Their coefficients (positive = increases survival probability)
- Absolute coefficient values for ranking

**Use:** Identify which features most influence predictions

Example:
```
Feature,Coefficient,Abs_Coefficient
Sex,2.456,2.456
Pclass_3,-0.987,0.987
Age,-0.543,0.543
...
```

### 4. model_results.txt

A text summary containing:
- Cross-validation results (5-fold)
- Training accuracy
- Testing accuracy
- Precision, Recall, F1-Score
- ROC-AUC Score

**Use:** Quick reference for model performance metrics

## Troubleshooting

### Issue: "FileNotFoundError: train.csv"

**Problem:** The dataset file is not in the correct location.

**Solution:**
1. Verify `train.csv` is in the same directory as the script
2. Check the filename is exactly `train.csv` (case-sensitive on Linux/macOS)
3. Make sure you're running the script from the project directory

### Issue: "ModuleNotFoundError: No module named 'numpy'"

**Problem:** Dependencies are not installed.

**Solution:**
```bash
pip install -r requirements.txt
```

If using Python 3 specifically:
```bash
pip3 install -r requirements.txt
```

### Issue: "Permission Denied" when creating files

**Problem:** No write permissions in the directory.

**Solution:**
1. Ensure you have write permissions in the project directory
2. On Linux/macOS, try: `chmod +w .`
3. On Windows, check folder permissions in Properties

### Issue: Script runs but no visualizations appear

**Problem:** Running in a headless environment or matplotlib backend issue.

**Solution:**
The script saves visualizations as PNG files automatically. You don't need a display. Just check for the output files:
- `data_visualization.png`
- `model_evaluation.png`

### Issue: Low accuracy or poor results

**Problem:** This is expected! The Titanic dataset is challenging.

**Typical Results:**
- Accuracy: 78-82%
- ROC-AUC: 84-87%

**Note:** Results will vary slightly due to random initialization. This is normal!

### Issue: Very slow execution

**Problem:** Hyperparameter tuning is computationally intensive.

**What's Normal:**
- Grid search can take 1-3 minutes
- Progress may pause during Step 10

**If it's too slow:**
You can modify the script to reduce the parameter grid size in the `hyperparameter_tuning()` function.

### Issue: "MemoryError"

**Problem:** Insufficient RAM (rare for this dataset).

**Solution:**
1. Close other applications
2. Ensure you have at least 2GB free RAM
3. Try restarting your computer

## Advanced Usage

### Running in Jupyter Notebook

You can adapt this code for Jupyter:

1. Create a new notebook
2. Split the `main()` function into separate cells
3. Execute cells one by one for interactive exploration

### Modifying Parameters

You can modify various parameters in the script:

**Train/Test Split Ratio:**
```python
# In split_data function, change test_size
X_train, X_test, y_train, y_test = split_data(df_feat, test_size=0.3)
```

**Cross-Validation Folds:**
```python
# In cross_validate_model function, change cv
cv_scores = cross_validate_model(model, X_train_scaled, y_train, cv=10)
```

**Hyperparameter Grid:**
```python
# In hyperparameter_tuning function, modify param_grid
param_grid = {
    'C': [0.1, 1, 10],  # Fewer values = faster
    'penalty': ['l2'],  # Only L2 regularization
    'solver': ['lbfgs']  # Only one solver
}
```

### Using the Test Set

To make predictions on the Kaggle test set:

1. Download `test.csv` from Kaggle
2. Modify the script to load test.csv after training
3. Use `model.predict()` to generate predictions
4. Create a submission file for Kaggle

### Extending the Project

Ideas for enhancements:

1. **Add more features:**
   - Extract titles from names (Mr., Mrs., Dr., etc.)
   - Create deck information from cabin numbers
   - Calculate ticket frequency

2. **Try different models:**
   - Random Forest
   - Gradient Boosting
   - Neural Networks

3. **Feature selection:**
   - Use Recursive Feature Elimination (RFE)
   - Try LASSO regularization for automatic selection

4. **Ensemble methods:**
   - Combine multiple models
   - Use voting classifiers

## Getting Help

If you encounter issues not covered in this guide:

1. **Check the error message carefully** - It often tells you exactly what's wrong
2. **Search for the error online** - Stack Overflow is your friend
3. **Review the code** - Comments explain each step
4. **Open an issue** - [Create an issue](https://github.com/wutangrepo/titanic-logistic-regression/issues) on GitHub

## Next Steps

After successfully running the script:

1. **Analyze the results** - Look at the visualizations and metrics
2. **Experiment with features** - Try creating new ones
3. **Tune hyperparameters** - Expand the search grid
4. **Compare models** - Implement other algorithms
5. **Write a report** - Document your findings for your course

## Additional Resources

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/)
- [Kaggle Titanic Tutorial](https://www.kaggle.com/c/titanic/overview/tutorials)

---

**Happy Learning! 🚢📊**

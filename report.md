# **Preprocessing Steps:**

### **1. Loading and Exploring Data:**

#### **Columns:**
- **Train and Test Data:**
  - The dataset is loaded using `pd.read_csv` for both `train.csv` and `test.csv`. 
  - The first few rows of the training data are displayed using `sample(5)` to understand the structure.
  - The `info()` and `describe()` methods give insights into the data types, non-null values, and statistics for numerical columns.

#### **Reasoning:**
- Initial exploration is necessary to understand the data structure, check for missing values, and identify any anomalies that may need further handling.

---

### **2. Handling Missing Data:**

#### **Columns Used/Omitted:**
- **Numerical Columns:**
  - Columns like `Age`, `Income_Annual`, `Base_Salary_PerMonth`, `Total_Bank_Accounts`, and others are considered.
  - Missing values in these columns are filled using the median, grouped by `Customer_ID`, to preserve customer-specific characteristics.
  
- **Categorical Columns:**
  - Categorical columns such as `Credit_Mix`, `Payment_Behaviour`, `Profession`, and `Payment_of_Min_Amount` are handled using imputation based on the mode, grouped by `Customer_ID`.

#### **Reasoning:**
- **Imputation with Median (Numerical):**  
  The median is used for imputation as it is less sensitive to outliers compared to the mean. Grouping by `Customer_ID` helps ensure that the imputation reflects individual customer characteristics.
  
- **Imputation with Mode (Categorical):**  
  The mode represents the most frequent category and is a reasonable choice for filling missing categorical values. If the mode is not available for a specific group, the overall mode is used as a fallback.

---

### **3. Feature Engineering:**

#### **Columns Used/Omitted:**
- **Loan_Count Feature:**
  - Derived from the `Loan_Type` column by counting the types of loans each customer has. This feature captures the number of loans a customer holds and may provide valuable information related to their financial behavior.

- **Credit_History_Age:**
  - The `Credit_History_Age` column, which is in a "years and months" format, is transformed into total months to standardize the data for further processing.

#### **Reasoning:**
- **Loan_Count:**  
  By counting the types of loans (from the `Loan_Type` column), we create a new feature that directly captures how diverse the customer's financial activity is, which can be an important predictor for their credit behavior.
  
- **Credit_History_Age:**  
  Converting `Credit_History_Age` into months makes the column more consistent and comparable to other numerical features, ensuring it can be used effectively in machine learning models.

---

### **4. Cleaning and Converting Data Types:**

#### **Columns Used/Omitted:**
- **Numeric Columns:**  
  Columns such as `Age`, `Income_Annual`, `Base_Salary_PerMonth`, etc., are cleaned to remove non-numeric characters (e.g., special symbols) and converted to numeric values.
  
- **Categorical Columns:**  
  Categorical columns are processed using label encoding.

#### **Reasoning:**
- **Cleaning Numeric Columns:**  
  Removing non-numeric characters ensures that numerical columns can be properly handled by machine learning models. The conversion to numeric values (with `pd.to_numeric`) ensures that any non-numeric values are coerced to `NaN` and subsequently handled.

- **Label Encoding (Categorical):**  
  Categorical columns (e.g., `Credit_Mix`, `Payment_Behaviour`, `Profession`) are encoded using `LabelEncoder`. Machine learning models typically require numerical input, so this transformation makes the data compatible with algorithms like decision trees or logistic regression.

---

### **5. Handling Categorical Columns with Issues:**

#### **Columns Used/Omitted:**
- **Categorical Columns with Specific Issues:**  
  Columns like `Credit_Mix`, `Payment_Behaviour`, `Payment_of_Min_Amount`, and `Profession` contain problematic values (`_`, `!@9#%8`, `NM`, etc.) that are replaced or imputed.

#### **Reasoning:**
- The goal is to clean problematic values (e.g., `_`, `!@9#%8`) by imputing missing values based on the mode within groups (grouped by `Customer_ID`), or if no mode exists, the overall mode is used. This ensures that these columns contain meaningful values for downstream modeling while preserving group-based patterns.

---

### **6. Outlier Removal:**

#### **Columns Used/Omitted:**
- **Numerical Columns:**  
  Numerical columns like `Age`, `Income_Annual`, `Base_Salary_PerMonth`, `Total_Bank_Accounts`, and others are considered for outlier removal.
  
#### **Reasoning:**
- **Outlier Removal (IQR Method):**  
  The IQR method is used to filter out extreme values (outliers). Outliers can skew model performance and lead to less accurate predictions, particularly in models sensitive to extreme values (e.g., linear regression). The IQR method is a robust and widely used approach for identifying and removing outliers.

---

### **7. Label Encoding:**

#### **Columns Used/Omitted:**
- **Categorical Columns:**  
  Columns like `Profession`, `Credit_Mix`, and `Payment_Behaviour` are encoded into numerical labels.

#### **Reasoning:**
- **Label Encoding:**  
  Label encoding is applied to categorical columns to convert them into numerical representations suitable for machine learning models. It is important for algorithms like decision trees, logistic regression, and others that cannot handle categorical variables directly.

---

### **8. Dropping Unnecessary Columns:**

#### **Columns Omitted:**
- **Dropped Columns:**  
  Columns like `ID`, `Name`, `Number`, `Loan_Type`, and `Customer_ID` are removed.
  
#### **Reasoning:**
- **Removing Non-Predictive Columns:**  
  The columns dropped (e.g., `ID`, `Name`, `Customer_ID`) are either unique identifiers or irrelevant to predicting the target variable (`Credit_Score`). Including these in the model could lead to overfitting or unnecessary complexity.

---

### **9. Target Variable Encoding:**

#### **Columns Used/Omitted:**
- **Target Variable (`Credit_Score`):**  
  The target variable `Credit_Score`, which is categorical (e.g., `Poor`, `Standard`, `Good`), is mapped to numerical values (0, 1, 2).

#### **Reasoning:**
- **Target Encoding:**  
  Converting the target variable (`Credit_Score`) into numerical values allows machine learning models to interpret it as a continuous variable. This step is crucial when training models like decision trees, random forests, and logistic regression.

---

### **Summary of Experimental Design:**

1. **Data Transformation:**
   - The dataset is cleaned and transformed into a format suitable for machine learning, ensuring that both numerical and categorical features are appropriately processed and transformed.
   
2. **Handling Missing Data:**
   - Missing values are imputed using robust methods like the median (for numerical features) and the mode (for categorical features), grouped by `Customer_ID` to ensure individual customer characteristics are maintained.

3. **Feature Engineering:**
   - New features like `Loan_Count` are created to capture important customer behaviors (e.g., the number of loans a customer holds), which may be important predictors for credit scoring.

4. **Outlier Removal:**
   - Outliers are removed using the IQR method to prevent extreme values from influencing the model and to ensure that the data used for training is more representative of typical customer behavior.

5. **Categorical Data Handling:**
   - Categorical columns are label encoded, allowing them to be processed by machine learning algorithms that require numerical input.

6. **Column Selection:**
   - Columns irrelevant to predictive modeling (e.g., `ID`, `Name`, `Customer_ID`) are omitted, while essential features are retained or derived (e.g., `Loan_Count`, `Credit_History_Age`).

7. **Target Variable Encoding:**
   - The target variable `Credit_Score` is encoded into numeric values to facilitate model training.

---

This preprocessing pipeline ensures that the dataset is clean, consistent, and ready for machine learning models, with a focus on handling missing data, feature engineering, outlier removal, and encoding categorical variables.

---
# **Models:**

For this classification task, we aimed to predict the **Credit Score** (`Credit_Score`), a categorical variable with three classes: `Poor`, `Standard`, and `Good`. We tested several machine learning models to evaluate which one performs best. The models chosen are as follows:

- **Logistic Regression**
- **K-Nearest Neighbors (KNN)**
- **Random Forest Classifier**
- **XGBoost**
- **Support Vector Machine (SVM)**
- **Artificial Neural Network (ANN)**

---

### 1. Logistic Regression:

#### **Initial Logistic Regression:**

1. **Data Splitting**: The data was split into training and testing sets using an 80-20 split.
2. **Model Training**: The logistic regression model was initialized with a maximum of 1000 iterations to ensure convergence.
3. **Model Evaluation**: Accuracy and classification report were used to evaluate the model's performance.

**Initial Results:**
- **Accuracy**: 55.12%
- **Classification Report**: The classification report provided detailed metrics for precision, recall, and F1-score for each class (`Poor`, `Standard`, `Good`).

#### **Improved Logistic Regression (After Feature Selection):**

To enhance the performance of the model, we decided to remove features that had a correlation of less than 0.1 with the target variable (`Credit_Score`). This step was based on the assumption that features with low correlation contribute less to the prediction and might add noise to the model.

1. **Correlation Analysis**: We analyzed the correlation of all features with the target variable (`Credit_Score`) using a correlation matrix.
2. **Feature Selection**: We kept only the features that had a correlation of 0.1 or greater with the target variable, removing those that had low correlation.
3. **Model Training**: The logistic regression model was retrained with the selected features.

**Improved Results:**
- **Accuracy**: 58.93%
- **Classification Report**: The classification report showed improved performance across all classes after removing features with low correlation.

#### **Summary:**
- **Initial Accuracy**: 55.12%
- **Improved Accuracy**: 58.93%
- **Performance Improvement**: The accuracy improved by 3.81% after removing columns with low correlation to the target variable. This shows that feature selection can significantly improve model performance by eliminating noise and irrelevant features.

---

### 2. KNN:

#### **KNN Model (Before Feature Selection):**

1. **Data Splitting**: We split the data into training and testing sets using an 80-20 split.
2. **Feature Scaling**: Since KNN is sensitive to the scale of the features, we standardized the features using **StandardScaler** to ensure all features contribute equally to the distance calculation.
3. **Model Training**: We trained the KNN model on the scaled features.
4. **Model Evaluation**: We evaluated the model using accuracy, as the target variable is categorical.

**Initial Results:**
- **Accuracy**: The model's accuracy is evaluated, and the results showed a certain baseline performance.

#### **KNN Model (After Feature Selection):**

To improve the model’s performance, we decided to apply **feature selection** by removing features that had low correlation (less than 0.3) with the target variable (`Credit_Score`). This helps eliminate noisy features and focus on the ones that have stronger predictive power.

1. **Correlation Analysis**: We performed a correlation analysis to identify features with low correlation to the target variable.
2. **Feature Selection**: Columns with correlation less than 0.3 were dropped from the dataset.
3. **Data Splitting and Standardization**: We split the data again and standardized the features before training the model.

**Improved Results:**
- **Accuracy**: 69%
- After removing the low-correlation features, the accuracy of the model improved significantly, demonstrating the importance of feature selection for enhancing model performance.

#### **Summary:**
- **Initial Accuracy**: The model's initial performance was tested without filtering features.
- **Improved Accuracy**: The accuracy increased to **69%** after filtering out low-correlation features, showcasing the benefit of focusing on relevant features.
- **Feature Selection Impact**: The improved performance with feature selection emphasizes the significance of reducing dimensionality by removing irrelevant or weakly correlated features.

---

### 3. Random Forest:

#### **Random Forest Model (With Hyperparameter Tuning):**

1. **Data Splitting**: The dataset was split into training and validation sets using an 80-20 ratio to evaluate the performance during hyperparameter optimization.
2. **Hyperparameter Tuning**: **Optuna** was used to tune the hyperparameters of the Random Forest model. The following hyperparameters were optimized:
   - `n_estimators`: The number of trees in the forest.
   - `max_depth`: The maximum depth of each tree.
   - `max_features`: The number of features considered for each split.
   - `max_leaf_nodes`: The maximum number of leaf nodes in the trees.
   - `min_samples_split`: The minimum number of samples required to split a node.
   - `min_samples_leaf`: The minimum number of samples required at each leaf node.
   - `bootstrap`: Whether bootstrap sampling is used when building trees.
3. **Model Training**: After identifying the best hyperparameters using Optuna, the model was trained on the entire training dataset to leverage the full data for improved accuracy.
4. **Model Evaluation**: The optimized model was then tested on the test set to evaluate its performance. The accuracy achieved on the test set was **71.44%**.

**Results:**
- **Best Hyperparameters**: 
  - `n_estimators`: 138  
  - `max_depth`: 9  
  - `max_features`: None  
  - `max_leaf_nodes`: 50  
  - `min_samples_split`: 6  
  - `min_samples_leaf`: 4  
  - `bootstrap`: True  

- **Best Validation Accuracy**: 71.48%

#### **Summary:**
- **Optuna Hyperparameter Tuning**: The use of **Optuna** to optimize hyperparameters significantly boosted the performance of the Random Forest model.
- **Accuracy**: The model achieved an accuracy of **71.44%** on the test set, demonstrating its strong predictive power.
- **Hyperparameters**: The optimal hyperparameters helped improve the model’s ability to generalize and reduce overfitting.

---

### 4. XGBoost:

#### **XGBoost Model (With Hyperparameter Tuning):**

1. **Data Splitting**: The dataset was split into training and validation sets using an 80-20 ratio. This was done to allow model evaluation during hyperparameter optimization.
2. **Hyperparameter Tuning**: Using **Optuna**, we optimized several hyperparameters of the XGBoost model. These included:
   - `n_estimators`: The number of boosting rounds.
   - `max_depth`: The maximum depth of each tree.
   - `learning_rate`: The step size shrinking to prevent overfitting.
   - `subsample`: The fraction of training data used for fitting.
   - `colsample_bytree`: The fraction of features used for each boosting round.
   - `gamma`: The regularization parameter that controls overfitting.
   - `min_child_weight`: The minimum weight of a child node.
   - `reg_alpha` and `reg_lambda`: L1 and L2 regularization terms.
   - `max_delta_step`: Used to make the model more robust.
3. **Model Training**: The best hyperparameters found by Optuna were used to train the final XGBoost model on the entire training dataset.
4. **Model Evaluation**: The model was evaluated using the test set, and the final accuracy was calculated.

**Results:**
- **Best Hyperparameters**: 
  - `n_estimators`: 250  
  - `max_depth`: 12  
  - `learning_rate`: 0.05  
  - `subsample`: 0.85  
  - `colsample_bytree`: 0.95  
  - `gamma`: 0.15  
  - `min_child_weight`: 2  
  - `reg_alpha`: 1e-4  
  - `reg_lambda`: 1e-4  
  - `max_delta_step`: 3  

- **Best Validation Accuracy**: **80.48%**

#### **Summary:**
- **Optuna Hyperparameter Tuning**: The use of **Optuna** significantly improved the performance of the XGBoost model by fine-tuning its hyperparameters.
- **Accuracy**: The optimized model achieved an accuracy of **80.47%** on the test set, demonstrating excellent predictive power.
- **Key Hyperparameters**: The combination of a high number of estimators, moderate learning rate, and careful regularization helped the model generalize well while avoiding overfitting.

---

### 5. Support Vector Machine (SVM)


#### **SVM Model (With Data Preprocessing):**

1. **Data Preparation**:
   - The training dataset (`train_data`) was used to define the feature columns (`X_train`) and the target column (`y_train`), which represents the **Credit_Score**.
   - We handled missing values in the features by applying **mean imputation**, filling missing values with the mean of the respective columns.

2. **Feature Scaling**:
   - Since SVM is sensitive to the scale of the features, we standardized the data using **StandardScaler**, which scales features to have a mean of 0 and a standard deviation of 1. This step was performed on both the training and validation sets.

3. **Model Training**:
   - We used the **SVC (Support Vector Classifier)** with an RBF kernel. The `C` parameter was set to 1.0, and `gamma` was set to 'scale', which automatically adjusts the kernel coefficient based on the number of features.
   - The model was trained using the training data and evaluated on both the training and validation sets.

#### **Model Evaluation**:

1. **Accuracy on Training Data**:  
   The SVM model achieved a training accuracy of **68.95%**, indicating that the model is able to learn patterns from the training set.

2. **Accuracy on Validation Data**:  
   The model achieved a validation accuracy of **67.60%**, which suggests that the model generalizes well to unseen data, though there is still room for improvement in terms of accuracy.

3. **Predictions on Test Data**:  
   After training, the model made predictions on the test data (which did not include the **Credit_Score** column). The predictions were then appended to the test dataset for further analysis.

#### **Summary**:

- **Hyperparameters**: The SVM model was trained using the **RBF kernel** with default settings for `C` (1.0) and `gamma` ('scale'). These values were sufficient for obtaining decent performance.
- **Model Performance**: The SVM model performed reasonably well, with a training accuracy of **68.95%** and validation accuracy of **67.60%**.
- **Prediction Results**: The model successfully predicted **Credit_Score** for the test dataset, and these predictions were saved for future analysis.


---

### 6. Artificial Neural Network (ANN):

#### **Preprocessing:**

1. **SMOTE** was applied to address class imbalance by generating synthetic samples for the minority class (`Standard`) to balance the distribution of target classes.
2. **StandardScaler** was used to normalize the features before feeding them into the model, which ensures that the input data has a mean of 0 and a standard deviation of 1.
3. Data was split into **training** and **test sets** with an 85/15 ratio using **stratified sampling** to preserve the class distribution.

#### **Model Architecture:**

- **Input Layer**: Matches the number of features in the training data.
- **Hidden Layers**: 
  - 5 dense layers with units ranging from 512 to 64.
  - Each hidden layer is followed by **BatchNormalization** and **Dropout** to improve generalization and prevent overfitting.
- **Output Layer**: A dense layer with 3 units (one for each credit score category) and a **softmax activation** function.

#### **Training:**

- The model was trained for a maximum of **800 epochs**, using **EarlyStopping** to prevent overfitting by halting training if the validation accuracy did not improve after 70 epochs.
- **Batch size** was set to 512 for better training efficiency.
- **Training Time**: The model took multiple iterations to converge, and the best weights were restored once the model stopped improving.

#### **Evaluation:**

**Training Set Results:**
- **Accuracy**: 92%

**Test Set Results:**
- **Accuracy**: 83%

#### **Summary:**
- The **ANN model** demonstrated strong performance on both the training and test datasets.
- **Class 2 (Good)** had the highest recall and precision across both training and test sets.
- **Class 1 (Standard)** showed lower recall on the test set, which suggests potential issues with prediction due to class imbalance or inherent difficulty in distinguishing this class.
- The **final accuracy** of **83%** on the test set shows that the model is a strong classifier for predicting the credit scores, but some room for improvement exists, especially in handling **Class 1 (Standard)**.

---


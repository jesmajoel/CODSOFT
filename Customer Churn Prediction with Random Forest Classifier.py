#importing libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

#Loading dataset
dataset = pd.read_csv('Churn_Modelling.csv')

#Preparing features and target
x = dataset.drop(['RowNumber', 'CustomerId', 'Surname', 'Exited'], axis=1)
y = dataset['Exited'].values

#One-Hot Encoding categorical columns
x = pd.get_dummies(x, columns=['Geography', 'Gender'], drop_first=True)

#Splitting dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

#Feature Scaling
sc = StandardScaler()
x_train_sc = sc.fit_transform(x_train)
x_test_sc = sc.transform(x_test)

#Training Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(x_train_sc, y_train)

#Making predictions
y_pred = model.predict(x_test_sc)
y_pred_proba = model.predict_proba(x_test_sc)[:, 1]

#Evaluating model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, y_pred_proba)

#Displaying metrics
print(f'The Accuracy: {accuracy * 100:.2f}%')
print(f'The Precision Score: {precision * 100:.2f}%')
print(f'The Recall Score: {recall * 100:.2f}%')
print(f'The F1-Score: {f1 * 100:.2f}%')
print(f'The AUC-ROC Score: {auc_roc:.2f}')

#Ploting ROC 
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC-ROC Curve (AUC = {auc_roc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()

#Feature Importance Plot
feature_importances = model.feature_importances_

#Create DataFrame for feature importance 
importance_df = pd.DataFrame({
    'Feature': x.columns,
    'Importance': feature_importances
})

#Sort features by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

#Ploting feature importance
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='teal')
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.gca().invert_yaxis()  
plt.show()

#Taking user input for custom prediction
def get_user_input():
    print("\nEnter the details for the customer:")
    
    #Collect user input
    CreditScore = int(input("Enter CreditScore: "))
    Age = int(input("Enter Age: "))
    Tenure = int(input("Enter Tenure (years): "))
    Balance = float(input("Enter Balance: "))
    NumOfProducts = int(input("Enter Number of Products: "))
    HasCrCard = int(input("Has Credit Card? (1 for Yes, 0 for No): "))
    IsActiveMember = int(input("Is Active Member? (1 for Yes, 0 for No): "))
    EstimatedSalary = float(input("Enter Estimated Salary: "))
    Geography_France = int(input("Is the Geography France? (1 for Yes, 0 for No): "))
    Geography_Germany = int(input("Is the Geography Germany? (1 for Yes, 0 for No): "))
    Gender_Male = int(input("Is the Gender Male? (1 for Yes, 0 for No): "))
    
    #Create DataFrame with input values
    custom_data = pd.DataFrame(
        [[CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary,
          Geography_France, Geography_Germany, Gender_Male]],
        columns=[
            'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
            'EstimatedSalary', 'Geography_France', 'Geography_Germany', 'Gender_Male'
        ]
    )
    
    #Ensure custom data has the same columns as the training data
    for col in x.columns:
        if col not in custom_data.columns:
            custom_data[col] = 0  # Add missing columns with 0 values

    #Reordering custom data to match the feature order used in training
    custom_data = custom_data[x.columns]

    #Scaling custom data
    custom_data_scaled = sc.transform(custom_data)

    return custom_data_scaled

#Get user input and make a prediction
custom_data_scaled = get_user_input()
custom_prediction = model.predict(custom_data_scaled)

#Displaying prediction result
print("\nCustom Prediction Result:", "Churned" if custom_prediction[0] == 1 else "Not Churned")

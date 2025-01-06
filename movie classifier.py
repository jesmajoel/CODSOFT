import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.decomposition import TruncatedSVD
import warnings
import joblib

#Ignore warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#Load the datasets
def load_data():
    try:
        df_train = pd.read_csv("train_data.txt", sep=':::', names=['ID', 'TITLE', 'GENRE', 'DESCRIPTION'], engine='python')
        x_test = pd.read_csv("test_data.txt", sep=':::', names=['ID', 'TITLE', 'DESCRIPTION'], engine='python')
        df_test_sol = pd.read_csv("test_data_solution.txt", sep=':::', names=['ID', 'TITLE', 'GENRE', 'DESCRIPTION'], engine='python')
        return df_train, x_test, df_test_sol
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None

df_train, x_test, df_test_sol = load_data()

if df_train is not None:
    #Display basic info
    print(f"Train data shape: {df_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    print(f"Test solution data shape: {df_test_sol.shape}")
    print(df_train.head(3))
    print(x_test.head(3))
    print(df_test_sol.head(3))
    df_train.info()

    #Genre distribution visualization
    def plot_genre_distribution(df):
        genre_counts = df['GENRE'].value_counts()
        plt.figure(figsize=(12, 6))
        sns.barplot(x=genre_counts.index, y=genre_counts.values, palette="viridis")
        plt.title("Number of Movies per Genre", fontsize=16)
        plt.xlabel("Genre", fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    plot_genre_distribution(df_train)

    # Preprocessing
    def preprocess_data(df_train, x_test, df_test_sol):
        df_train.drop(columns=['ID'], inplace=True)
        x_test.drop(columns=['ID'], inplace=True)

        le = LabelEncoder()
        df_train['GENRE'] = le.fit_transform(df_train['GENRE'])
        df_test_sol['GENRE'] = le.transform(df_test_sol['GENRE'])

        df_train['combined_text'] = df_train['TITLE'] + ' ' + df_train['DESCRIPTION']
        x_test['combined_text'] = x_test['TITLE'] + ' ' + x_test['DESCRIPTION']

        return df_train, x_test, df_test_sol, le

    df_train, x_test, df_test_sol, le = preprocess_data(df_train, x_test, df_test_sol)

    #Sample 1% of data for faster experimentation
    df_train = df_train.sample(frac=0.01, random_state=42)
    x_test = x_test.sample(frac=0.01, random_state=42)
    df_test_sol = df_test_sol.iloc[x_test.index]

    #TF-IDF Vectorizer
    print("Starting TF-IDF Vectorization...")
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 3), max_df=0.85, min_df=2, sublinear_tf=True)
    tfidf_vectorizer.fit(df_train['combined_text'])
    print("TF-IDF Vectorization complete.")

    X_train_tfidf = tfidf_vectorizer.transform(df_train['combined_text'])
    X_test_tfidf = tfidf_vectorizer.transform(x_test['combined_text'])

    #Use RandomOverSampler for oversampling
    ros = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train_tfidf, df_train['GENRE'])

    #Dimensionality Reduction (SVD)
    print("Starting Dimensionality Reduction (SVD)...")
    svd = TruncatedSVD(n_components=50, random_state=42)  # Reduced components for testing
    X_train_reduced = svd.fit_transform(X_train_resampled)
    X_test_reduced = svd.transform(X_test_tfidf)
    print("Dimensionality Reduction complete.")

    #Apply MinMaxScaler to ensure non-negative values
    scaler = MinMaxScaler()

    #Fit the scaler on the reduced training data and transform both training and test data
    X_train_reduced = scaler.fit_transform(X_train_reduced)
    X_test_reduced = scaler.transform(X_test_reduced)

    #Train-test split for validation
    X_train, X_val, y_train, y_val = train_test_split(X_train_reduced, y_train_resampled, test_size=0.1, random_state=42)

    #Hyperparameter tuning for Random Forest Classifier
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    }

    print("Starting GridSearchCV...")
    grid_search = GridSearchCV(RandomForestClassifier(class_weight='balanced', random_state=42), param_grid, cv=2, scoring='accuracy', n_jobs=-1)  # Reduced cv for quicker testing
    grid_search.fit(X_train, y_train)
    print("GridSearchCV complete.")
    rf_model = grid_search.best_estimator_

    #Evaluate the model
    def evaluate_model(y_true, y_pred, set_name="Set"):
        print(f"{set_name} Classification Report")
        print(classification_report(y_true, y_pred, target_names=le.classes_, zero_division=1))
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
        plt.title(f"Confusion Matrix - {set_name}", fontsize=16)
        plt.xlabel("Predicted", fontsize=12)
        plt.ylabel("Actual", fontsize=12)
        plt.show()

    #Train and validation set evaluation
    evaluate_model(y_train, rf_model.predict(X_train), "Train Set")
    evaluate_model(y_val, rf_model.predict(X_val), "Validation Set")

    #Test Set Evaluation
    y_test_pred = rf_model.predict(X_test_reduced)
    evaluate_model(df_test_sol['GENRE'], y_test_pred, "Test Set")

    #Save the model and vectorizer
    joblib.dump(rf_model, "random_forest_model.pkl")
    joblib.dump(tfidf_vectorizer, "tfidf_vectorizer.pkl")
    joblib.dump(svd, "svd_transformer.pkl")

    #Prediction function
    def predict_genre(title, description):
        try:
            data = pd.DataFrame({'TITLE': [title], 'DESCRIPTION': [description]})
            data['combined_text'] = data['TITLE'] + ' ' + data['DESCRIPTION']
            X_new_tfidf = tfidf_vectorizer.transform(data['combined_text'])
            X_new_reduced = svd.transform(X_new_tfidf)
            X_new_reduced = scaler.transform(X_new_reduced)  # Apply MinMaxScaler
            y_pred = rf_model.predict(X_new_reduced)
            return le.inverse_transform(y_pred)[0]
        except Exception as e:
            return f"Error in prediction: {str(e)}"

    #Example prediction
    example_title = "Edgar's Lunch (1998)"
    example_description = "L.R. Brane loves his life..."
    predicted_genre = predict_genre(example_title, example_description)
    print(f"The predicted genre is: {predicted_genre}")

else:
    print("Data loading failed. Please check the file paths.")

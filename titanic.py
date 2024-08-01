import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

@st.cache_data
def load_data():
    df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')
    return df

def preprocess_data(df):
    imputer = SimpleImputer(strategy='median')
    df['Age'] = imputer.fit_transform(df[['Age']])
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df.drop('Cabin', axis=1, inplace=True)
    df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
    label_encoder = LabelEncoder()
    df['Sex'] = label_encoder.fit_transform(df['Sex'])
    df['Embarked'] = label_encoder.fit_transform(df['Embarked'])
    return df

def split_and_scale(df):
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def main():
    st.title("Titanic Dataset Analysis")

    df = load_data()
    
    st.sidebar.title("Navigation")
    options = ["Exploratory Data Analysis", "Model Training", "Model Evaluation", "Feature Importance", "Make Predictions"]
    choice = st.sidebar.selectbox("Select an option", options)

    if choice == "Exploratory Data Analysis":
        st.subheader("Exploratory Data Analysis")
        if st.checkbox("Show Data Info"):
            st.write(df.info())
        if st.checkbox("Show Data Description"):
            st.write(df.describe())
        if st.checkbox("Show Missing Values"):
            st.write(df.isnull().sum())
        if st.checkbox("Show Data"):
            st.write(df.head())
        
        st.subheader("Visualizations")
        if st.checkbox("Show Correlation Heatmap"):
            plt.figure(figsize=(10, 6))
            sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
            st.pyplot(plt)

    elif choice == "Model Training":
        st.subheader("Data Preprocessing")
        df = preprocess_data(df)
        st.write("Data after preprocessing:")
        st.write(df.head())
        
        st.subheader("Model Training")
        X_train, X_test, y_train, y_test = split_and_scale(df)
        model = train_model(X_train, y_train)
        st.write("Model trained successfully!")

    elif choice == "Model Evaluation":
        st.subheader("Model Evaluation")
        df = preprocess_data(df)
        X_train, X_test, y_train, y_test = split_and_scale(df)
        model = train_model(X_train, y_train)
        y_pred = model.predict(X_test)
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred)}")
        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred))
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred))
        
    elif choice == "Feature Importance":
        st.subheader("Feature Importance")
        df = preprocess_data(df)
        X_train, X_test, y_train, y_test = split_and_scale(df)
        model = train_model(X_train, y_train)
        feature_importance = pd.DataFrame({
            'Feature': df.drop('Survived', axis=1).columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        st.write(feature_importance)
        
        st.write("Feature Importance Visualization:")
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance)
        plt.title('Feature Importance')
        st.pyplot(plt)

    elif choice == "Make Predictions":
        st.subheader("Make Predictions")
        
        pclass = st.selectbox("Pclass", [1, 2, 3])
        sex = st.selectbox("Sex", ["male", "female"])
        age = st.slider("Age", 0, 100, 25)
        sibsp = st.slider("SibSp", 0, 10, 0)
        parch = st.slider("Parch", 0, 10, 0)
        fare = st.slider("Fare", 0, 100, 50)
        embarked = st.selectbox("Embarked", ["C", "Q", "S"])

        sex = 1 if sex == "male" else 0
        embarked = {"C": 0, "Q": 1, "S": 2}[embarked]
        
        user_data = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])
        scaler = StandardScaler()
        user_data = scaler.fit_transform(user_data)
        
        df = preprocess_data(df)
        X_train, X_test, y_train, y_test = split_and_scale(df)
        model = train_model(X_train, y_train)
        prediction = model.predict(user_data)
        prediction_prob = model.predict_proba(user_data)
        
        st.write("Prediction (0 = Not Survived, 1 = Survived):", prediction[0])
        st.write("Prediction Probability:", prediction_prob)

if __name__ == "__main__":
    main()

import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from feature_extractor import extract_features

def read_labeled_clips(kill_dir, non_kill_dir):
    "Read labeled clips from kill and non_kill directories"
    
    X = []
    y = []
    
    for file in os.listdir(kill_dir):
        if file.endswith('.mp3'):
            features = extract_features(os.path.join(kill_dir, file))
            X.append(features)
            y.append(1)  # Label as 1 for kill clips
    
    for file in os.listdir(non_kill_dir):
        if file.endswith('.mp3'):
            features = extract_features(os.path.join(non_kill_dir, file))
            X.append(features)
            y.append(0)  # Label as 0 for non-kill clips
    
    return np.array(X), np.array(y)

def train_and_evaluate_model(X, y):
    "Train a RandomForestClassifier and evaluate with train/test split"
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    
    return clf

def save_model(clf, model_path):
    "Save the trained model to a file"
    
    import pickle
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)

if __name__ == "__main__":
    kill_dir = "kill_detector/data/samples/kill"
    non_kill_dir = "kill_detector/data/samples/non_kill"
    model_path = "kill_detector/models/kill_model.pkl"

    X, y = read_labeled_clips(kill_dir, non_kill_dir)
    clf = train_and_evaluate_model(X, y)
    save_model(clf, model_path)

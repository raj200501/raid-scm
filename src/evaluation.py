from sklearn.metrics import accuracy_score, classification_report

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    y_pred = predictions.argmax(axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    print(f'Classification Report:\n{report}')

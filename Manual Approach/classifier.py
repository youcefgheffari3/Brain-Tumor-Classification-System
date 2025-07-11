from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_and_evaluate(X, y, model_type='svm'):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_type == 'svm':
        clf = SVC()
    elif model_type == 'knn':
        clf = KNeighborsClassifier(n_neighbors=3)
    else:
        raise ValueError("Unsupported model type")

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return acc, report

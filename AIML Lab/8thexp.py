from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score


class Ensemble:
    def __init__(self):
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        x, y = load_breast_cancer(return_X_y=True)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=0.25, random_state=23
        )

    @staticmethod
    def __Classifiers__(name=None):
        random_state = 23

        if name == 'decision_tree':
            return DecisionTreeClassifier(random_state=random_state)

        if name == 'kneighbors':
            return KNeighborsClassifier()

        if name == 'logistic_regression':
            return LogisticRegression(
                random_state=random_state,
                solver='liblinear'
            )

    def __DecisionTreeClassifier__(self):
        decision_tree = Ensemble.__Classifiers__(name='decision_tree')
        decision_tree.fit(self.x_train, self.y_train)

    def __KNearestNeighborsClassifier__(self):
        knn = Ensemble.__Classifiers__(name='kneighbors')
        knn.fit(self.x_train, self.y_train)

    def __LogisticRegression__(self):
        logistic_regression = Ensemble.__Classifiers__(
            name='logistic_regression'
        )
        logistic_regression.fit(self.x_train, self.y_train)

    def __VotingClassifier__(self):
        # Instantiate classifiers
        decision_tree = Ensemble.__Classifiers__(name='decision_tree')
        knn = Ensemble.__Classifiers__(name='kneighbors')
        logistic_regression = Ensemble.__Classifiers__(
            name='logistic_regression'
        )

        # Voting Classifier
        vc = VotingClassifier(
            estimators=[
                ('decision_tree', decision_tree),
                ('knn', knn),
                ('logistic_regression', logistic_regression)
            ],
            voting='soft'
        )

        # Train model
        vc.fit(self.x_train, self.y_train)

        # Predictions
        y_pred_train = vc.predict(self.x_train)
        y_pred_test = vc.predict(self.x_test)

        print("Train accuracy:", accuracy_score(self.y_train, y_pred_train))
        print("Test accuracy:", accuracy_score(self.y_test, y_pred_test))


if __name__ == "__main__":
    ensemble = Ensemble()
    ensemble.load_data()
    ensemble.__VotingClassifier__()

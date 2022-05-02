from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import time

seed = 12345
n_jobs = 1 

init_time = time.time()
X, y = make_classification(n_samples=1000, random_state=seed)
clf = RandomForestClassifier(random_state=seed)
params = {'n_estimators': [10, 100]}
cv = GridSearchCV(clf, params, cv=5, n_jobs=1)
cv.fit(X, y)
print(f"Best params: {cv.best_params_} score: {cv.best_score_}")
haty = cv.best_estimator_.predict(X)
print(f"Full set accuracy: {accuracy_score(y, haty)}")
print(f"{time.time() - init_time:0.4f} [s]")



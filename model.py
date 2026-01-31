import logging
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, GridSearchCV

logger = logging.getLogger(__name__)

def logistic_regression(
    *,
    X_train,
    y_train,
    param_grid: dict,
    random_state: int,
    scoring: str="f1_macro"
) -> GridSearchCV:

    # Build a training pipeline
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1
        ))
    ])

    # Preserve the percentage of samples for each class
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state
    )

    # Grid search
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=scoring,
        n_jobs=-1,
        cv=cv
    )

    grid.fit(X_train, y_train)

    return grid


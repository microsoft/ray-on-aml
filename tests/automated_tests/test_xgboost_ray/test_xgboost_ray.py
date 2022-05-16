
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'../../../'))

from src.ray_on_aml.core import Ray_On_AML

#dask
from azureml.core import Run

from xgboost_ray import RayXGBClassifier, RayParams
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
def train_xgboost():
    seed = 42

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.25, random_state=42
    )

    clf = RayXGBClassifier(
        n_jobs=10,  # In XGBoost-Ray, n_jobs sets the number of actors
        random_state=seed
    )

    # scikit-learn API will automatically conver the data
    # to RayDMatrix format as needed.
    # You can also pass X as a RayDMatrix, in which case
    # y will be ignored.

    clf.fit(X_train, y_train)

    pred_ray = clf.predict(X_test)
    print(pred_ray.shape)

    pred_proba_ray = clf.predict_proba(X_test)
    print(pred_proba_ray.shape)

    # It is also possible to pass a RayParams object
    # to fit/predict/predict_proba methods - will override
    # n_jobs set during initialization

    clf.fit(X_train, y_train, ray_params=RayParams(num_actors=10))

    pred_ray = clf.predict(X_test, ray_params=RayParams(num_actors=10))
    print(pred_ray.shape)



if __name__ == "__main__":
    run = Run.get_context()
    ws = run.experiment.workspace
    ray_on_aml =Ray_On_AML()
    ray = ray_on_aml.getRay(additional_ray_start_head_args="--temp-dir=outputs",additional_ray_start_worker_args="--temp-dir=outputs")

    for item, value in os.environ.items():
        print('{}: {}'.format(item, value))

    if ray: #in the headnode
        print("head node detected")

        train_xgboost   

    else:
        print("in worker node")

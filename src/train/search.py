from __future__ import annotations
from initialize_environment import RANDOM_STATE
import numpy as np

from train.models import (
    plot_delta_scatter,
    metrics_rsquared,
    metrics_relative,
    fit_linear_regression,
    fit_lasso_regression,
    fit_decision_tree_regression,
    fit_gbm_regression,
    # fit_random_forest_regression
)


def gbm_grid_search(
    RANDOM_STATE,
    x_train,
    y_train,
    x_test,
    y_test,
    n_estimators_range,
    learning_rate_range,
    max_depth_range,
):
    # Parameter ranges
    # n_estimators_range = [16, 20, 24, 28, 32, 36, 40, 44]

    best_score = -np.inf
    best_params = [-np.inf, -np.inf, -np.inf]

    for n_estimators in n_estimators_range:
        for learning_rate in learning_rate_range:
            for max_depth in max_depth_range:
                model = fit_gbm_regression(
                    x_train,
                    y_train,
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    random_state=RANDOM_STATE,
                )
                # model.fit(x_train, y_train)
                r2_test = metrics_rsquared(model, x_test, y_test)

                print(
                    f"Params: n_estimators={n_estimators}, "
                    f"learning_rate={learning_rate:.3f}, max_depth={max_depth} | "
                    f"Test R²={r2_test:.4f}"
                )

                if r2_test > best_score:
                    best_score = r2_test
                    best_params = (
                        n_estimators,
                        learning_rate,
                        max_depth,
                    )

    print("\nBest parameters found:")
    print(
        f"n_estimators={best_params[0]}, learning_rate={best_params[1]:.3f}, max_depth={best_params[2]}"
    )
    print(f"Best Test R²={best_score:.4f}")

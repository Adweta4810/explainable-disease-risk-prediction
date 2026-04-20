import pandas as pd

from src.data.preprocess import (
    load_diabetes_data,
    preprocess_diabetes,
    scale_diabetes_features,
    save_diabetes_cleaned
)
from src.models.train import (
    prepare_train_test,
    train_logistic_regression,
    train_random_forest,
    train_xgboost,
    save_model
)
from src.models.evaluate import (
    cv_report,
    evaluate_model,
    print_evaluation,
    make_summary_table
)


def run_diabetes_pipeline():
    # Load and preprocess
    diabetes_df = load_diabetes_data("data/diabetes/diabetes.csv")
    diabetes_df = preprocess_diabetes(diabetes_df)
    diabetes_df = scale_diabetes_features(diabetes_df, target_col="Outcome")
    save_diabetes_cleaned(diabetes_df, "data/diabetes/diabetes_cleaned.csv")

    # Split
    X, y, X_train, X_test, y_train, y_test = prepare_train_test(diabetes_df, "Outcome")

    # Train models
    lr_model, lr_params, lr_cv = train_logistic_regression(X_train, y_train)
    rf_model, rf_params, rf_cv = train_random_forest(X_train, y_train)
    xgb_model, xgb_params, xgb_cv = train_xgboost(X_train, y_train)

    # Evaluate
    lr_results = evaluate_model(lr_model, X_test, y_test, target_names=["No Diabetes", "Diabetes"])
    rf_results = evaluate_model(rf_model, X_test, y_test, target_names=["No Diabetes", "Diabetes"])
    xgb_results = evaluate_model(xgb_model, X_test, y_test, target_names=["No Diabetes", "Diabetes"])

    print_evaluation(lr_results, "Logistic Regression")
    print_evaluation(rf_results, "Random Forest")
    print_evaluation(xgb_results, "XGBoost")

    summary = make_summary_table({
        "Logistic Regression": lr_results,
        "Random Forest": rf_results,
        "XGBoost": xgb_results
    })

    print("\nFINAL TEST SET SUMMARY — DIABETES")
    print(summary.round(4))

    # Save best model manually after checking summary
    save_model(rf_model, "models/diabetes/best_model.pkl")


if __name__ == "__main__":
    run_diabetes_pipeline()
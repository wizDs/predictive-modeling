import functools
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn import model_selection
from xgboost import XGBClassifier


def feature_generation(monthly_payments: pd.DataFrame, user_table: pd.DataFrame):
    def _next_user_payment(months: int) -> pd.DataFrame:

        return (
            df.groupby("user_id")["date"]
            .apply(lambda s: s.shift(-months))
            .reset_index(level=0)["date"]
            .rename(f"next_{months}m")
        )

    # mutate dataframe
    df = monthly_payments.copy()
    df = df.sort_values(by=["user_id", "date"])
    # identify next payments
    next_1m = _next_user_payment(1)
    next_2m = _next_user_payment(2)
    next_3m = _next_user_payment(3)
    next_months = (next_1m, next_2m, next_3m)

    # add columns
    df = functools.reduce(lambda df1, df2: df1.join(df2), next_months, initial=df)
    # define label
    df["churned_3m"] = (
        df["next_1m"].isnull() & df["next_2m"].isnull() & df["next_3m"].isnull()
    )

    # merge festures and labels
    training_data = user_table.merge(
        df[["user_id", "date", "churned_3m"]], on="user_id", how="left"
    ).set_index(["user_id", "date"])

    # TODO: handle censored data (1/1-2018 was last observed payment date)
    return training_data


def create_dataset(
    monthly_payments: pd.DataFrame, user_table: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:

    training_data = feature_generation(
        monthly_payments=monthly_payments, user_table=user_table
    )

    X = training_data.drop(columns="churned_3m")
    y = training_data["churned_3m"]

    return X, y


def split_kfold_and_evaluate_each(
    X: pd.DataFrame, y: pd.DataFrame, kfold: model_selection.KFold
) -> list[dict[str, float]]:
    output = []

    xgb = XGBClassifier()
    splits = kfold.split(X, y)
    for idx_train, idx_test in splits:
        X_train, y_train = X.iloc[idx_train], y.iloc[idx_train]
        X_test, y_test = X.iloc[idx_test], y.iloc[idx_test]
        xgb.fit(X_train, y_train)
        y_pred = xgb.predict_proba(X_test)[:, 1]
        residuals = y_test - y_pred
        acc = metrics.accuracy_score(y_test, y_pred >= 0.5)
        mse = float(np.mean((residuals) ** 2))
        r2 = metrics.r2_score(y_test, y_pred)
        fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
        auc = float(metrics.auc(fpr, tpr))
        output += [{"mse": mse, "r2": r2, "acc": acc, "auc": auc}]

    return output


if __name__ == "__main__":
    monthly_payments_df = pd.read_csv("data/monthly_payments.csv")
    user_table_df = pd.read_csv("data/user_table.csv")

    kfold = model_selection.KFold(n_splits=5, random_state=9565687, shuffle=True)
    features, labels = create_dataset(monthly_payments_df, user_table_df)
    metrics_data_xgb = split_kfold_and_evaluate_each(features, labels, kfold)
    print(metrics_data_xgb)

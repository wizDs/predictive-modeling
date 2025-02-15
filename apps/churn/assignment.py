import functools
import pandas as pd
import numpy as np
import enum
from sklearn import metrics
from sklearn import model_selection
from xgboost import XGBClassifier


class FeatureColumn(enum.StrEnum):
    NEXT_1M = "next_1m"
    NEXT_2M = "next_2m"
    NEXT_3M = "next_3m"
    LABEL_3M = "churned_3m"


def feature_generation(monthly_payments: pd.DataFrame, user_table: pd.DataFrame):
    def _next_user_payment(months: int, column_name: FeatureColumn) -> pd.DataFrame:

        return (
            df.groupby("user_id")["date"]
            .apply(lambda s: s.shift(-months))
            .reset_index(level=0)["date"]
            .rename(column_name)
        )

    # mutate dataframe
    df = monthly_payments.copy()
    df = df.sort_values(by=["user_id", "date"])
    # identify next payments
    next_1m = _next_user_payment(1, FeatureColumn.NEXT_1M)
    next_2m = _next_user_payment(2, FeatureColumn.NEXT_2M)
    next_3m = _next_user_payment(3, FeatureColumn.NEXT_3M)
    next_months = (next_1m, next_2m, next_3m)

    # add columns
    df = functools.reduce(lambda df1, df2: df1.join(df2), next_months, initial=df)
    # define label
    df[FeatureColumn.LABEL_3M] = (
        df[FeatureColumn.NEXT_1M].isnull()
        & df[FeatureColumn.NEXT_2M].isnull()
        & df[FeatureColumn.NEXT_3M].isnull()
    )

    # merge festures and labels
    training_data = user_table.merge(
        df[["user_id", "date", FeatureColumn.LABEL_3M]], on="user_id", how="left"
    ).set_index(["user_id", "date"])

    # TODO: handle censored data (1/1-2018 was last observed payment date)
    return training_data


def create_dataset(
    monthly_payments: pd.DataFrame, user_table: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:

    training_data = feature_generation(
        monthly_payments=monthly_payments, user_table=user_table
    )

    features = training_data.drop(columns="churned_3m")
    targets = training_data["churned_3m"]

    return features, targets


def split_kfold_and_evaluate_each(
    features: pd.DataFrame, targets: pd.DataFrame, kfold: model_selection.KFold
) -> list[dict[str, float]]:
    output = []

    xgb = XGBClassifier()
    splits = kfold.split(features, targets)
    for idx_train, idx_test in splits:
        features_train, targets_train = (
            features.iloc[idx_train],
            targets.iloc[idx_train],
        )
        features_test, targets_test = features.iloc[idx_test], targets.iloc[idx_test]
        xgb.fit(features_train, targets_train)
        y_pred = xgb.predict_proba(features_test)[:, 1]
        residuals = targets_test - y_pred
        acc = metrics.accuracy_score(targets_test, y_pred >= 0.5)
        mse = float(np.mean((residuals) ** 2))
        r2 = metrics.r2_score(targets_test, y_pred)
        fpr, tpr, _ = metrics.roc_curve(targets_test, y_pred)
        auc = float(metrics.auc(fpr, tpr))
        output += [{"mse": mse, "r2": r2, "acc": acc, "auc": auc}]

    return output


if __name__ == "__main__":
    monthly_payments_df = pd.read_csv("data/monthly_payments.csv")
    user_table_df = pd.read_csv("data/user_table.csv")

    k_fold = model_selection.KFold(n_splits=5, random_state=9565687, shuffle=True)
    features_df, labels_df = create_dataset(monthly_payments_df, user_table_df)
    metrics_data_xgb = split_kfold_and_evaluate_each(features_df, labels_df, k_fold)
    print(metrics_data_xgb)

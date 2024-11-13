import pandas as pd
from sklearn import metrics
import numpy as np
from sklearn.model_selection import KFold
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score


def create_dataset(monthly_payments: pd.DataFrame, user_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # mutate dataframe
    df = monthly_payments.copy()
    df = df.sort_values(by=['user_id', 'date'])
    # identify next payments
    next_1m = df.groupby('user_id')['date'].apply(lambda s: s.shift(-1)).reset_index(level=0)['date']
    next_2m = df.groupby('user_id')['date'].apply(lambda s: s.shift(-2)).reset_index(level=0)['date']
    next_3m = df.groupby('user_id')['date'].apply(lambda s: s.shift(-3)).reset_index(level=0)['date']
    # add columns
    df = (df
        .join(next_1m.rename('next_1m'))
        .join(next_2m.rename('next_2m'))
        .join(next_3m.rename('next_3m')))
    # define label
    df['churned_3m'] = df['next_1m'].isnull() & df['next_2m'].isnull() & df['next_3m'].isnull()

    # merge festures and labels
    training_data = (user_table
                     .merge(df[['user_id', 'date', 'churned_3m']],
                            on='user_id', how='left')
                     .set_index(['user_id', 'date']))
    #TODO: handle censored data (1/1-2018 was last observed payment date)

    X = training_data.drop(columns='churned_3m')
    y = training_data['churned_3m']

    return X, y


def split_kfold_and_evaluate_each(X: pd.DataFrame, y: pd.DataFrame, kfold: KFold) -> list[dict[str, float]]:
    output = []

    xgb = XGBClassifier()
    splits = kfold.split(X, y)
    for idx_train, idx_test in splits:
        X_train, y_train = X.iloc[idx_train], y.iloc[idx_train]
        X_test, y_test = X.iloc[idx_test], y.iloc[idx_test]
        xgb.fit(X_train, y_train)
        y_pred = xgb.predict_proba(X_test)[:,1]
        residuals = y_test - y_pred
        acc = accuracy_score(y_test, y_pred >= 0.5)
        mse = float(np.mean((residuals) **2))
        r2 = r2_score(y_test, y_pred)
        fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
        auc = float(metrics.auc(fpr, tpr))
        output += [{'mse': mse,
                    'r2': r2,
                    'acc': acc, 
                    'auc': auc}]
        
    return output



if __name__ == '__main__':
    monthly_payments_df = pd.read_csv('data/monthly_payments.csv')
    user_table_df = pd.read_csv('data/user_table.csv')
    
    kfold = KFold(n_splits=5, random_state=9565687, shuffle=True)
    features, labels = create_dataset(monthly_payments_df, user_table_df)
    metrics_data_xgb = split_kfold_and_evaluate_each(features, labels, kfold)
    print(metrics_data_xgb)
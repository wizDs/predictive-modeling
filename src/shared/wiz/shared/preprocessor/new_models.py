# import pathlib
# from functools import partial
# import json
# import numpy as np
# from wiz.interface import (
#     estimator_interface,
#     preproc_interface,
#     target_interface,
#     modeling_interface,
# )
# from wiz.shared import get_model
# import xgboost
# from toolz.itertoolz import pluck
# import polars as pl
# from sklearn.pipeline import Pipeline
# from sklearn import metrics
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.model_selection import KFold
# from wiz.shared.estimator import estimator
# from wiz.shared.preprocessor import preprocessor
# from wiz.shared.target_transformer import target_transformer
# from wiz.evaluation import helpers2 as helpers

# # from .eval_regression import ModelReport, ModelReportBuilder


# # read data
# def read_data(path: pathlib.Path):
#     return (
#         pl.read_csv(path, null_values="NA")
#         .rename(
#             mapping=lambda s: s.lower()
#             .replace(".", "_")
#             .replace("(", "")
#             .replace(")", "")
#         )
#         .with_columns(pl.col(pl.String).replace("None", None))
#     )


# if __name__ == "__main__":

#     # num_cols = make_column_selector(dtype_include=np.number)
#     # cat_cols = make_column_selector(dtype_include=object)

#     curr_path = pathlib.Path(".").absolute()
#     data_path = curr_path / "data" / "train.csv"
#     print(data_path)

#     # read feature descriptions
#     with open(
#         curr_path / "data" / "feature_description.txt", "r", encoding="utf-8"
#     ) as f:
#         feature_descriptions = json.loads(f.read())["features"]

#     # mapper from name to description
#     description_mapper = dict(pluck(["name", "desc"], feature_descriptions))

#     data_df = read_data(data_path)

#     # for col in features_df[cat_cols].columns:
#     #     features_df[col] = features_df[col].fillna("")
#     # X[num_cols].pipe(num_ppl.fit_transform).describe().transpose().round(2)
#     _data_df = data_df.drop("id", "saleprice")
#     basic_columns = preproc_interface.BasicColumns(
#         numerical_columns=_data_df.select(pl.selectors.numeric()).columns,
#         categorical_columns=_data_df.select(pl.selectors.string()).columns,
#         target_column="saleprice",
#     )
#     num_columns_only = preproc_interface.BasicColumns(
#         numerical_columns=_data_df.select(pl.selectors.numeric()).columns,
#         categorical_columns=[],
#         target_column="saleprice",
#     )

#     output = []
#     prediction_datasets = []
#     for i in range(20):
#         train_validation_df = helpers.split_train_test(data_df)

#         runs = [
#             # (
#             #     "num_columns_only",
#             #     num_columns_only,
#             #     preproc_interface.CategoricalProcessor.ONE_HOT_ENCODER,
#             # ),
#             # (
#             #     "basic_columns",
#             #     basic_columns,
#             #     preproc_interface.CategoricalProcessor.ONE_HOT_ENCODER,
#             # ),
#             (
#                 "basic_columns",
#                 basic_columns,
#                 preproc_interface.CategoricalProcessor.ORDINAL_ENCODER,
#             ),
#             (
#                 "basic_columns",
#                 basic_columns,
#                 preproc_interface.CategoricalProcessor.TARGET_ENCODER,
#             ),
#         ]

#         for name, _basic_columns, proc_type in runs:
#             for tt in (
#                 target_interface.DummyTransformer(),
#                 target_interface.PowerTransformer(l=1.5),
#                 target_interface.PowerTransformer(l=0.5),
#                 target_interface.LogTransformer(),
#             ):
#                 for est in (
#                     estimator_interface.XGBoostRegressor(),
#                     # estimator_interface.LinearRegression(),
#                     # estimator_interface.LGBMRegressor(),
#                 ):
#                     train_interface = modeling_interface.TrainInputInterface(
#                         preprocessor=preproc_interface.PreProcInterface(
#                             basic_columns=_basic_columns,
#                             categorical_processor_type=proc_type,
#                         ),
#                         target=tt,
#                         estimator=est,
#                     )
#                     prediction_set = helpers.get_predictions(
#                         interface=train_interface,
#                         data=train_validation_df,
#                     )
#                     prediction_dataset = pl.DataFrame(prediction_set).with_columns(
#                         [
#                             (pl.col("prediction") - pl.col("target")).alias("error"),
#                             pl.lit(est.estimator_type).alias("estimator"),
#                             pl.lit(proc_type.value).alias("type"),
#                             pl.lit(
#                                 tt.target_type
#                                 if not isinstance(tt, target_interface.PowerTransformer)
#                                 else f"{tt.target_type}({tt.l})"
#                             ).alias("target_transformer"),
#                             pl.col("target")
#                             .qcut(
#                                 quantiles=np.arange(0.1, 1.1, 0.1),
#                                 labels=list(map(str, range(1, 12, 1))),
#                             )
#                             .cast(pl.Int8())
#                             .alias("target_grp"),
#                         ]
#                     )
#                     prediction_datasets += [prediction_dataset]
#                     eval_metrics = helpers.evaluate_estimator(
#                         interface=train_interface,
#                         data=train_validation_df,
#                     )

#                     output += [
#                         {
#                             # "name": name,
#                             "estimator": est.estimator_type,
#                             "type": proc_type.value,
#                             "target_transformer": (
#                                 tt.target_type
#                                 if not isinstance(tt, target_interface.PowerTransformer)
#                                 else f"{tt.target_type}({tt.l})"
#                             ),
#                             **eval_metrics,
#                         }
#                     ]

#     _prediction_datasets: pl.DataFrame = (
#         pl.concat(prediction_datasets)
#         .group_by(
#             [
#                 "estimator",
#                 "type",
#                 "target_transformer",
#                 pl.col("target_grp").cut(breaks=[-1, 3, 9, 10]),
#             ]
#         )
#         .agg(pl.mean("error"))
#         .pivot(
#             on="target_grp",
#             index=[
#                 "estimator",
#                 "type",
#                 "target_transformer",
#             ],
#             values="error",
#         )
#     )
#     print(_prediction_datasets)
#     _df = (
#         pl.DataFrame(output)
#         .group_by(["estimator", "type", "target_transformer"])
#         .agg(
#             [
#                 pl.mean("mape").round(3).alias("avg_mape"),
#                 pl.mean("mean_error").cast(pl.Int32).alias("mean_error"),
#                 pl.mean("median_error").cast(pl.Int32).alias("median_error"),
#                 pl.mean("mae").cast(pl.Int32).alias("avg_mae"),
#                 pl.std("mae").cast(pl.Int32).alias("std_mae"),
#             ]
#         )
#         .sort("avg_mape")
#     )
#     print(_df.to_pandas())

#     log_transformer = target_transformer.LogTransformer()
#     _log_targets_df = log_transformer.transform(targets_df["saleprice"].to_numpy())

#     # where
#     test_features_df.with_columns(
#         [
#             pl.Series(name="prediction", values=_pred, dtype=pl.Float64),
#             pl.Series(name="target", values=_target, dtype=pl.Float64),
#         ]
#     ).with_columns(
#         (pl.col("target") / 1_000)
#         .qcut(quantiles=np.arange(0, 1.01, 0.1), labels=np.arange(0, 1.01, 0.1))
#         .alias("category")
#     )

#     # kfold = KFold(random_state=423, shuffle=True)
#     # ksplit = kfold.split(features_df)
#     # next(ksplit)

#     distr_df = error_by_actual_price(features_df, targets_df, create_xgboost(), kfold)
#     print(distr_df)

#     evaluation_set = [
#         EvaluationSet(
#             features=features_df[num_cols],
#             labels=targets_df,
#             model_constructor=create_xgboost,
#         ),
#         EvaluationSet(
#             features=features_df, labels=targets_df, model_constructor=create_xgboost
#         ),
#         EvaluationSet(
#             features=features_df,
#             labels=targets_df,
#             model_constructor=partial(
#                 create_xgboost, preprocessor=create_preprocessor_oe
#             ),
#         ),
#         EvaluationSet(
#             features=features_df[num_cols],
#             labels=targets_df,
#             model_constructor=partial(
#                 create_log_linear, preprocessor=create_preprocessor_oe
#             ),
#         ),
#         EvaluationSet(
#             features=features_df[
#                 features_df[num_cols].columns.tolist()
#                 + ["poolqc", "fence", "paveddrive"]
#             ],
#             labels=targets_df,
#             model_constructor=partial(
#                 create_log_linear, preprocessor=create_preprocessor_oe
#             ),
#         ),
#         EvaluationSet(
#             features=features_df[
#                 features_df[num_cols].columns.tolist()
#                 + ["poolqc", "fence", "paveddrive"]
#             ],
#             labels=targets_df,
#             model_constructor=create_log_linear,
#         ),
#         EvaluationSet(
#             features=features_df[num_cols],
#             labels=targets_df,
#             model_constructor=create_lasso,
#         ),
#         EvaluationSet(
#             features=features_df, labels=targets_df, model_constructor=create_knn
#         ),
#         EvaluationSet(
#             features=features_df,
#             labels=targets_df,
#             model_constructor=partial(create_knn, preprocessor=create_preprocessor_oe),
#         ),
#     ]

#     reports = evaluate_multiple_models(kfold, *evaluation_set)
#     for eval_set, r in zip(evaluation_set, reports):
#         print("\n\n")
#         print(
#             eval_set.model_constructor
#             if isinstance(eval_set.model_constructor, partial)
#             else eval_set.model_constructor.__name__
#         )
#         print("column counts: ", len(eval_set.features.columns.tolist()))
#         print(json.dumps(r.summary.model_dump(), indent=4))

#     # test data
#     test_path = curr_path / "data" / "test.csv"
#     test_df = pd.read_csv(test_path)
#     test_df.set_index("Id", inplace=True)
#     X_validation = test_df.rename(
#         columns=lambda s: (
#             s.lower().replace(".", "_").replace("(", "").replace(")", "")
#         )
#     )

#     model = create_xgboost()
#     model.fit(features_df, targets_df)
#     y_validation_pred = model.predict(X_validation).round().astype(int)
#     y_validation_pred

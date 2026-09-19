# Graph Report - .  (2026-09-19)

## Corpus Check
- Corpus is ~40,837 words - fits in a single context window. You may not need a graph.

## Summary
- 734 nodes · 1271 edges · 73 communities (50 shown, 23 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 66 edges (avg confidence: 0.54)
- Token cost: 80,743 input · 0 output

## Community Hubs (Navigation)
- Stock Data Pair Features
- Estimator Base Classes
- Power-App Energy Price Loading
- Weather API Client (DMI)
- Stock Buy-Score Optimization
- Budget Payment Date Tests
- Regression Model Evaluation
- Job Skill NER Extraction (Ollama)
- Estimator Evaluation Helpers
- Notebooks Energy Data Loading
- Estimator Preprocessing Interfaces
- Classifier Evaluation Reports
- Job-App Session & Anonymization
- Target Value Transformers
- CI Affected-Apps Workflow & Dev Skills
- Stock Trade Date Matching
- Budget App Main Flow
- MinIO Storage Tests
- LightGBM Wrapper
- Evaluation Metrics
- Churn Feature Engineering
- MinIO Client Interface
- Storage Object Response Protocol
- Session Storage (S3/MinIO)
- Mock MinIO Response (tests)
- Pants Affected-Apps Script
- House-Prices Feature Descriptions
- Power System Schemas
- Transport Deduction Calculator
- Whisper Model Loader
- Stock Model Performance Chart
- Claude CLI Session Scratch Dir
- Shared Library Packages
- CAPM Filename Cleaner
- Danish Tax Calculator
- Whisper Transcriber Setup
- Handover Skill Principles
- App Dependency Stacks
- Job-App MinIO Service
- Job-App Backend / Tools-App
- CI-Gate Required Check
- Ruff Job
- SF Crime Analysis
- SF Crime Conda Env
- Weather ETL (MSSQL)
- Budget Package
- Budget-App2
- Churn Package
- Finance Package
- Github Scripts
- Notebooks
- Power-App
- Stock-Analyzer Package
- Transcribe Package
- Weather Package
- Repo Root

## God Nodes (most connected - your core abstractions)
1. `TrainingData` - 25 edges
2. `StockPrice` - 20 edges
3. `EvaluationData` - 20 edges
4. `StockProcess` - 18 edges
5. `RandomForrestModel` - 16 edges
6. `SessionStorage` - 16 edges
7. `Measurement` - 15 edges
8. `TrainingDataGenerator` - 15 edges
9. `Regressor` - 15 edges
10. `LinearModel` - 14 edges

## Surprising Connections (you probably didn't know these)
- `EvaluationSet` --uses--> `KNeighborsRegressor`  [INFERRED]
  apps/house-prices/src/model.py → src/shared/wiz/shared/estimator/neighbors.py
- `create_log_linear()` --calls--> `LinearRegression`  [INFERRED]
  apps/house-prices/src/model.py → src/interface/wiz/interface/estimator_interface.py
- `error_by_actual_price()` --references--> `BaseEstimator`  [EXTRACTED]
  apps/house-prices/src/model.py → src/shared/wiz/shared/estimator/estimator.py
- `LinearModel` --inherits--> `LinearRegression`  [EXTRACTED]
  apps/ml-stocks/source/stock_model/Class/Regression.py → src/interface/wiz/interface/estimator_interface.py
- `mypy (src) job` --references--> `python-default Pants resolve (hand-mirrored requirements)`  [INFERRED]
  .github/workflows/ci.yml → 3rdparty/python/requirements.txt

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **ci-gate aggregates ruff/changes/mypy/mypy-apps into one required check** — _github_workflows_ci_ci_gate, _github_workflows_ci_ruff, _github_workflows_ci_changes, _github_workflows_ci_mypy, _github_workflows_ci_mypy_apps [EXTRACTED 1.00]
- **Skills drive changes through local checks then the CI workflow** — _claude_skills_implement_issue_skill_validation_pipeline, _claude_skills_address_pr_comments_skill_address_pr_comments, _github_workflows_ci_ci_workflow [INFERRED 0.85]

## Communities (73 total, 23 thin omitted)

### Community 0 - "Stock Data Pair Features"
Cohesion: 0.07
Nodes (25): MoneyStream, object, CurrentAndFuturePair, CurrentAndPastPair, DataPair, Features, date, getFictiveData() (+17 more)

### Community 1 - "Estimator Base Classes"
Cohesion: 0.06
Nodes (33): BaseEstimator, BinaryClassifier, DoubleArray, FeatureArray, FeatureImportance, Abstract base class for estimators (classifiers or regressors)., Train the model on data X and labels y., Predict outputs for input data X. (+25 more)

### Community 2 - "Power-App Energy Price Loading"
Cohesion: 0.07
Nodes (41): BaseRequestParams, Column, cost_type_short_name(), Elspotprices, EnergyDataClient, FeatureColumn, Frequency, get_expected_record_count() (+33 more)

### Community 3 - "Weather API Client (DMI)"
Cohesion: 0.09
Nodes (27): get_apikey(), get_args(), get_interface_from_args(), date, TimeDelta, WeatherInterface, DMIClientWrapper, Geometry (+19 more)

### Community 4 - "Stock Buy-Score Optimization"
Cohesion: 0.09
Nodes (18): buy_score(), lr_eval(), numbers_of_stocks_to_buy(), rf_eval(), linear_programming(), mixed_integer_problem(), n_stocks_to_buy(), ndarray (+10 more)

### Community 5 - "Budget Payment Date Tests"
Cohesion: 0.09
Nodes (26): field_validator, NamedTuple, test_equal(), test_in_far_future(), test_in_far_past(), test_in_near_future(), test_just_payed(), test_last_month() (+18 more)

### Community 6 - "Regression Model Evaluation"
Cohesion: 0.10
Nodes (28): ModelEvaluation, ModelReport, ModelReportBuilder, BaseModel, DataFrame, KFold, Pipeline, Series (+20 more)

### Community 7 - "Job Skill NER Extraction (Ollama)"
Cohesion: 0.11
Nodes (28): Language, is_available(), _locate_spans(), _ollama_generate(), _parse_response(), predict(), LLM-based skill extractor using a local Ollama model (qwen2.5:7b).  Returns the, Extract a JSON array from the LLM response, tolerating markdown fences. (+20 more)

### Community 8 - "Estimator Evaluation Helpers"
Cohesion: 0.09
Nodes (24): evaluate_estimator(), EvaluationSet, get_predictions(), PredictionSet, DataFrame, split_train_test(), evaluate_estimator(), EvaluationSet (+16 more)

### Community 9 - "Notebooks Energy Data Loading"
Cohesion: 0.15
Nodes (18): BaseRequestParams, Column, Elspotprices, EnergyDataClient, FeatureColumn, Frequency, get_expected_record_count(), join_prices_and_consumption_data() (+10 more)

### Community 10 - "Estimator Preprocessing Interfaces"
Cohesion: 0.14
Nodes (12): EstimatorType, CategoricalProcessor, PreProcInterface, model_from_type(), preprocessor_from_type(), target_from_type(), DefaultPreProcessor, PreProcessor (+4 more)

### Community 11 - "Classifier Evaluation Reports"
Cohesion: 0.20
Nodes (11): ClassificationReportBuilder, ConfusionMatrix, ModelEvaluation, ModelReport, BaseModel, DataFrame, KFold, object (+3 more)

### Community 12 - "Job-App Session & Anonymization"
Cohesion: 0.18
Nodes (12): _get_storage(), _llm_available(), _load_skill_model(), _predict_skills_llm(), _predict_skills_spacy(), cache_data, cache_resource, bucket_from_env() (+4 more)

### Community 13 - "Target Value Transformers"
Cohesion: 0.18
Nodes (6): DummyTransformer, LogTransformer, PowerTransformer, ndarray, TargetTransformer, Target

### Community 14 - "CI Affected-Apps Workflow & Dev Skills"
Cohesion: 0.18
Nodes (15): python-default Pants resolve (hand-mirrored requirements), address-pr-comments skill, PR comment classification (question/actionable/future/ambiguous), Reply-as-resumability marker, implement-issue skill, Local-then-CI validation pipeline with bounded retry loops, changes job (detect changes), CI workflow (ci.yml) (+7 more)

### Community 15 - "Stock Trade Date Matching"
Cohesion: 0.20
Nodes (8): identify_selling_date(), match_curr_price(), match_next_opening_price(), months_between_two_dates(), date, try_find_next_working_day(), relativedelta, timedelta

### Community 16 - "Budget App Main Flow"
Cohesion: 0.29
Nodes (13): calculate(), collect_payment_interface_inputs(), load_expected_income_to_polars(), load_expected_payments_to_polars(), load_state(), main(), DataFrame, Save the payment interface state to a temp file. (+5 more)

### Community 17 - "MinIO Storage Tests"
Cohesion: 0.21
Nodes (9): object_key(), _MockMinio, _MockObject, In-memory mock implementing job_app.storage.MinioClient, for tests., _storage(), test_list_sessions_and_versions(), test_load_missing_returns_empty_string(), test_object_key() (+1 more)

### Community 18 - "LightGBM Wrapper"
Cohesion: 0.26
Nodes (4): LGBMClassifier, LGBMRegressor, DoubleArray, FeatureArray

### Community 19 - "Evaluation Metrics"
Cohesion: 0.22
Nodes (5): ClassifierMetric, ndarray, Common regression evaluation metrics., Common classifier evaluation metrics., RegressorMetric

### Community 20 - "Churn Feature Engineering"
Cohesion: 0.33
Nodes (8): create_dataset(), feature_generation(), FeatureColumn, DataFrame, datetime, KFold, # TODO: handle censored data (1/1-2018 was last observed payment date), split_kfold_and_evaluate_each()

### Community 21 - "MinIO Client Interface"
Cohesion: 0.32
Nodes (3): MinioClient, BinaryIO, The subset of minio.Minio's interface SessionStorage relies on, for testability.

### Community 22 - "Storage Object Response Protocol"
Cohesion: 0.36
Nodes (3): _ObjectResponse, _StorageObject, Protocol

### Community 23 - "Session Storage (S3/MinIO)"
Cohesion: 0.36
Nodes (4): Path, Session/version file storage backed by an S3-compatible (MinIO) bucket., Write every session/version file in this bucket under local_dir, mirroring the S, SessionStorage

### Community 25 - "Pants Affected-Apps Script"
Cohesion: 0.46
Nodes (6): main(), resolve_app_dir(), ResolveCase, test_main_dedupes_and_sorts_output(), test_resolve_app_dir(), parametrize

### Community 26 - "House-Prices Feature Descriptions"
Cohesion: 0.38
Nodes (6): FeatureDescription, generate_streamlit_app(), BaseModel, DataFrame, Path, read_feature_descriptions()

### Community 27 - "Power System Schemas"
Cohesion: 0.38
Nodes (4): PowerSystem, BaseModel, BaseModel, SpotPrice

### Community 28 - "Transport Deduction Calculator"
Cohesion: 0.33
Nodes (3): date, Transport fradrag calculator — working days commuted per month., working_days_in_month()

### Community 29 - "Whisper Model Loader"
Cohesion: 0.47
Nodes (5): load_model(), ModelType, cache_resource, Load and cache the Whisper model., Whisper

### Community 30 - "Stock Model Performance Chart"
Cohesion: 1.00
Nodes (4): Model Performance Chart: GN Store Price vs Model Score, High Score = Buy Recommendation Thesis, GN Store Stock (ticker/security), Model Score (buy recommendation signal)

### Community 31 - "Claude CLI Session Scratch Dir"
Cohesion: 0.50
Nodes (4): _mirror_dir(), Path, A per-browser-session local scratch directory the Claude CLI can read/write agai, _run_claude()

### Community 32 - "Shared Library Packages"
Cohesion: 0.67
Nodes (4): evaluation, house-prices, interface, shared

### Community 35 - "Whisper Transcriber Setup"
Cohesion: 0.67
Nodes (3): Whisper Transcriber app, Python 3.13 audioop removal -> switch to st.audio_input, ffmpeg system dependency for Whisper

## Knowledge Gaps
- **40 isolated node(s):** `github-scripts`, `budget-app2`, `FeatureColumn`, `churn`, `finance` (+35 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **23 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `load_model()` connect `Whisper Model Loader` to `Job-App Session & Anonymization`?**
  _High betweenness centrality (0.132) - this node is a cross-community bridge._
- **Why does `_load_skill_model()` connect `Job-App Session & Anonymization` to `Whisper Model Loader`?**
  _High betweenness centrality (0.127) - this node is a cross-community bridge._
- **Why does `LinearRegression` connect `Regression Model Evaluation` to `Estimator Evaluation Helpers`, `Stock Buy-Score Optimization`?**
  _High betweenness centrality (0.089) - this node is a cross-community bridge._
- **Are the 5 inferred relationships involving `TrainingData` (e.g. with `DataPair` and `Features`) actually correct?**
  _`TrainingData` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `StockPrice` (e.g. with `CurrentAndFuturePair` and `CurrentAndPastPair`) actually correct?**
  _`StockPrice` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `EvaluationData` (e.g. with `LinearModel` and `RandomForrestModel`) actually correct?**
  _`EvaluationData` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `StockProcess` (e.g. with `CurrentAndFuturePair` and `CurrentAndPastPair`) actually correct?**
  _`StockProcess` has 6 INFERRED edges - model-reasoned connections that need verification._
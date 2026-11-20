# TabPilot report

Run 01M7X1P8BYZM75STVD4JGC114E · passengers.csv · status completed


## 1. Summary


| Item | Value |
| --- | --- |
| Task | binary |
| Target | survived |
| Final model | random\_forest (0-random\_forest-b0a4624d) |
| Primary metric | roc\_auc (maximize) |
| CV score (DEV, mean ± std) | 0.8231 ± 0.0278 |
| Independent test score | none (no holdout) |
| Run status | completed |


## 2. Dataset summary


| Item | Value |
| --- | --- |
| File | passengers.csv |
| Format | csv |
| Rows (raw / valid) | 891 / 891 |
| Columns | 10 |
| Development rows | 891 |
| Feature columns | 8 |
| Duplicate rows | 0 |
| Class counts (DEV) | 0: 527, 1: 364 |
| Validation | repeated\_stratified\_kfold, k = 5, repeats = 3, holdout fraction = 0.0 |


## 3. Data decisions


| Excluded column | Normalized name | Reason |
| --- | --- | --- |
| PassengerId | passengerid | identifier\_integer\_unique |


| Original column name | Normalized name |
| --- | --- |
| PassengerId | passengerid |
| Pclass | pclass |
| Sex | sex |
| Age | age |
| SibSp | sibsp |
| Parch | parch |
| Fare | fare |
| Cabin Deck | cabin\_deck |
| Embarked | embarked |
| Survived | survived |


## 4. Data-quality and leakage warnings


- W-HIGH-MISSING:cabin\_deck


## 5. Modeling plan


Plan source: rule-based planner.


| Model | Priority | Class weight | Tuning share |
| --- | --- | --- | --- |
| lightgbm | 1 | none | 0.60 |
| random\_forest | 2 | none | 0.25 |
| linear | 3 | none | 0.15 |


Validation status: valid.


## 6. Preprocessing decisions


| Decision | Value |
| --- | --- |
| Numeric imputation | median |
| Categorical imputation | most\_frequent |
| Missing indicators | yes |
| Low-cardinality encoding | onehot |
| High-cardinality encoding | frequency |
| Target transform | none |


> Every learned transformation is fitted inside the pipeline on training-fold rows only.


## 7. Features


Baseline features: datetime parts, missing-value indicators and rare-category grouping, built inside the pipeline.


No LLM-proposed features were used.


## 8. Leaderboard (cross-validation)


*Cross-validation on development data (DEV) only*

| Entry | Model | Features | CV mean | CV std | Fold scores | Fit s | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0-random\_forest-b0a4624d | random\_forest | baseline | 0.8231 | 0.0278 | 0.7812, 0.8760, 0.8648, 0.8177, 0.7722, 0.8177, 0.8153, 0.8237, 0.8267, 0.8224, 0.8186, 0.8239, 0.7954, 0.8419, 0.8488 | 0.35 | ok |
| 0-linear-83a8a00c | linear | baseline | 0.8210 | 0.0318 | 0.7741, 0.8846, 0.8526, 0.8056, 0.7721, 0.8355, 0.7935, 0.8416, 0.8372, 0.8050, 0.8015, 0.8145, 0.8008, 0.8401, 0.8558 | 0.04 | ok |
| 0-lightgbm-64a38e9e | lightgbm | baseline | 0.8127 | 0.0328 | 0.7663, 0.8783, 0.8544, 0.8069, 0.7482, 0.8138, 0.7925, 0.8333, 0.8194, 0.8050, 0.8117, 0.7943, 0.8003, 0.8193, 0.8472 | 0.10 | ok |


![CV mean score with ±std error bars](figures/leaderboard.png)


![CV score of every fold](figures/fold-scores.png)


*Threshold metrics on OOF predictions, 0-lightgbm-64a38e9e*

| Threshold rule | Threshold | Accuracy | F1 |
| --- | --- | --- | --- |
| threshold 0 5 | 0.500 | 0.7553 | 0.7038 |
| f1 optimal | 0.339 | 0.7576 | 0.7346 |


*Threshold metrics on OOF predictions, 0-linear-83a8a00c*

| Threshold rule | Threshold | Accuracy | F1 |
| --- | --- | --- | --- |
| threshold 0 5 | 0.500 | 0.7722 | 0.7374 |
| f1 optimal | 0.499 | 0.7733 | 0.7390 |


*Threshold metrics on OOF predictions, 0-random\_forest-b0a4624d*

| Threshold rule | Threshold | Accuracy | F1 |
| --- | --- | --- | --- |
| threshold 0 5 | 0.500 | 0.7621 | 0.7143 |
| f1 optimal | 0.303 | 0.7508 | 0.7363 |


## 9. Reflection history


No reflection rounds were run.


## 10. Ensemble decision


Ensemble kept: no. Reason: relative gain 0.0680% below 0.2%


| Member | Weight |
| --- | --- |
| 0-random\_forest-b0a4624d | 0.667 |
| 0-linear-83a8a00c | 0.333 |


## 11. Why this model was selected


Rule: max mean CV score; near-ties (0.25 SEM): simpler, fewer features, faster, id. Winner: 0-random\_forest-b0a4624d.


> Cross-validation scores of the selected model are optimistically biased, because the same folds were used for tuning and selection.


> 3 configurations were compared in total (every model and retrained variant on the leaderboard, including failed and skipped ones). The more configurations are compared, the more optimistic the best score tends to be.


| Runner-up | Mean difference | Min | Max | Folds won |
| --- | --- | --- | --- | --- |
| 0-linear-83a8a00c | 0.0021 | -0.0179 | 0.0218 | 9/15 |


> Paired fold differences are descriptive: with this few folds they are not a significance test.


## 12. Independent test (used once, after selection)


No independent test results exist: no holdout: fewer than 2,000 valid rows; CV only (§11.5).


## 13. Feature importance and SHAP


Method: shap\_tree. Rows explained: 300.


![Mean \|SHAP\| per source column](figures/shap-importance.png)


![Native model importance](figures/gain-importance.png)


![SHAP values](figures/shap-beeswarm-1.png)


| Rank | Source column | Importance |
| --- | --- | --- |
| 1 | sex | 0.1822 |
| 2 | pclass | 0.0796 |
| 3 | fare | 0.0402 |
| 4 | age | 0.0207 |
| 5 | sibsp | 0.0199 |
| 6 | cabin\_deck | 0.0118 |
| 7 | embarked | 0.0057 |
| 8 | parch | 0.0055 |


## 14. Error analysis


Overall OOF error rate on DEV: 0.2379.


| Class | Precision | Recall | Support |
| --- | --- | --- | --- |
| 0 | 0.8070 | 0.7856 | 527 |
| 1 | 0.7011 | 0.7280 | 364 |


*Worst segments (categories with at least 30 DEV rows)*

| Column | Rank | Category | Rows | Error rate |
| --- | --- | --- | --- | --- |
| sex | 1 | female | 306 | 0.2745 |
| sex | 2 | male | 585 | 0.2188 |
| cabin\_deck | 1 | E | 43 | 0.3023 |
| cabin\_deck | 2 | D | 62 | 0.2581 |
| cabin\_deck | 3 | F | 40 | 0.2250 |
| cabin\_deck | 4 | (missing) | 683 | 0.2225 |
| cabin\_deck | 5 | G | 32 | 0.1562 |
| embarked | 1 | S | 637 | 0.2606 |
| embarked | 2 | C | 160 | 0.2250 |
| embarked | 3 | Q | 87 | 0.1034 |


![Confusion matrix (OOF, row-normalized)](figures/confusion-matrix.png)


![ROC curve (OOF)](figures/roc-curve.png)


![Precision–recall (OOF)](figures/pr-curve.png)


![Reliability curve (OOF, 10 bins)](figures/reliability.png)


## 15. Limitations


- Importance and SHAP values describe how this model uses features, not causal effects in the world.
- Correlated features share or swap importance; a low importance does not mean a feature is irrelevant.
- SHAP values are computed on a sample and on preprocessed features; one-hot columns are aggregated.
- Permutation importance can be misleading with correlated features.
- Cross-validation scores of the selected model are optimistically biased, because the same folds were used for tuning and selection.
- No independent test set: no holdout: fewer than 2,000 valid rows; CV only (§11.5).


## 16. Reproducibility information


| Item | Value |
| --- | --- |
| Run id | 01M7X1P8BYZM75STVD4JGC114E |
| TabPilot version | 0.1.0 |
| Execution profile | hosted |
| Split id | fa8e261dfcad2e67651f1d4e60c861704ce8c0a4ebef9b50eafff97b5c2294a9 |
| Seeds | root=42, split=42, model=43, optuna=44, profile=45, explain=46 |
| Libraries | catboost 1.2.10, lightgbm 4.7.0, numpy 2.4.6, optuna 5.0.0, pandas 3.0.6, scikit-learn 1.9.1, xgboost 2.1.4 |
| Time budget (s) | 240 |
| Elapsed at report (s) | 115.5 |
| Trials used | 60 of 60 |


## 17. Narrative


> Template narrative, built deterministically from the facts table. Reason: no LLM provider available.


This run predicts \`survived\` \[F2\] as a binary \[F1\] task. TabPilot compared 3 \[F9\] model types with cross-validation over 15 \[F7\] folds on the development rows and selected random\_forest \[F4\] by its roc\_auc \[F3\] score.


The selected model's mean cross-validated score is 0.8231 \[F5\], with a fold-to-fold standard deviation of 0.0278 \[F6\]. The runner-up, linear \[F10\], averaged 0.8210 \[F11\]; the selected model scored higher on 9 \[F13\] of 15 \[F7\] folds. With this few folds the comparison is descriptive, not a significance test. Cross-validated scores of the selected model are optimistic, because the same folds were used to choose it.


No rows were set aside for a final check, so only cross-validated scores are available for this run.


The inputs this model relied on most were \`sex\` \[F18\], \`pclass\` \[F19\], \`fare\` \[F20\], \`age\` \[F21\] and \`sibsp\` \[F22\]. Importance describes how this model uses its inputs, not what would happen in the world if a value changed; correlated inputs can share or swap importance.


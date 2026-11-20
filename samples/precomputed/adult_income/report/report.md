# TabPilot report

Run 01M7X1VW0WMQZFAJB17K8N4P6W · adult\_income.csv · status completed


## 1. Summary


| Item | Value |
| --- | --- |
| Task | binary |
| Target | income |
| Final model | linear (0-linear-00160141) |
| Primary metric | roc\_auc (maximize) |
| CV score (DEV, mean ± std) | 0.7730 ± 0.0124 |
| Independent test score | 0.7791 |
| Run status | completed |


## 2. Dataset summary


| Item | Value |
| --- | --- |
| File | adult\_income.csv |
| Format | csv |
| Rows (raw / valid) | 4000 / 4000 |
| Columns | 10 |
| Development rows | 3200 |
| Feature columns | 9 |
| Duplicate rows | 2 |
| Class counts (DEV) | \<=50K: 1866, \>50K: 1334 |
| Validation | stratified\_kfold, k = 5, repeats = 1, holdout fraction = 0.2 |


## 3. Data decisions


| Original column name | Normalized name |
| --- | --- |
| education-num | education\_num |
| marital-status | marital\_status |
| hours-per-week | hours\_per\_week |
| capital-gain | capital\_gain |
| native-region | native\_region |


## 4. Data-quality and leakage warnings


- W-CONFLICTING-DUPLICATES:4
- W-CONFLICTING-DUPLICATES:6
- W-DUPLICATES:2
- W-DUPLICATES:4


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
| 0-linear-00160141 | linear | baseline | 0.7730 | 0.0124 | 0.7553, 0.7676, 0.7886, 0.7752, 0.7784 | 0.06 | ok |
| 0-lightgbm-584c35aa | lightgbm | baseline | 0.7667 | 0.0146 | 0.7555, 0.7608, 0.7787, 0.7528, 0.7855 | 0.15 | ok |
| 0-random\_forest-b0a4624d | random\_forest | baseline | 0.7559 | 0.0114 | 0.7438, 0.7447, 0.7661, 0.7573, 0.7677 | 0.38 | ok |


![CV mean score with ±std error bars](figures/leaderboard.png)


![CV score of every fold](figures/fold-scores.png)


*Threshold metrics on OOF predictions, 0-lightgbm-584c35aa*

| Threshold rule | Threshold | Accuracy | F1 |
| --- | --- | --- | --- |
| threshold 0 5 | 0.500 | 0.7100 | 0.6237 |
| f1 optimal | 0.320 | 0.6706 | 0.6777 |


*Threshold metrics on OOF predictions, 0-linear-00160141*

| Threshold rule | Threshold | Accuracy | F1 |
| --- | --- | --- | --- |
| threshold 0 5 | 0.500 | 0.7078 | 0.6219 |
| f1 optimal | 0.305 | 0.6675 | 0.6768 |


*Threshold metrics on OOF predictions, 0-random\_forest-b0a4624d*

| Threshold rule | Threshold | Accuracy | F1 |
| --- | --- | --- | --- |
| threshold 0 5 | 0.500 | 0.6966 | 0.6009 |
| f1 optimal | 0.327 | 0.6431 | 0.6615 |


## 9. Reflection history


No reflection rounds were run.


## 10. Ensemble decision


Ensemble kept: no. Reason: hill climbing selected a single model


| Member | Weight |
| --- | --- |
| 0-linear-00160141 | 1.000 |


## 11. Why this model was selected


Rule: max mean CV score; near-ties (0.25 SEM): simpler, fewer features, faster, id. Winner: 0-linear-00160141.


> Cross-validation scores of the selected model are optimistically biased, because the same folds were used for tuning and selection.


> 3 configurations were compared in total (every model and retrained variant on the leaderboard, including failed and skipped ones). The more configurations are compared, the more optimistic the best score tends to be.


| Runner-up | Mean difference | Min | Max | Folds won |
| --- | --- | --- | --- | --- |
| 0-lightgbm-584c35aa | 0.0063 | -0.0072 | 0.0223 | 3/5 |


> Paired fold differences are descriptive: with this few folds they are not a significance test.


## 12. Independent test (used once, after selection)


| Model | roc\_auc | Bootstrap 95% interval |
| --- | --- | --- |
| Final model | 0.7791 | 0.7477 – 0.8104 |
| Reference baseline (lightgbm) | 0.7547 | — |


> These rows were used once, after the selection was frozen, and never for training, tuning or selection.


## 13. Feature importance and SHAP


Method: shap\_linear. Rows explained: 1000.


![Mean \|SHAP\| per source column](figures/shap-importance.png)


![SHAP values](figures/shap-beeswarm-1.png)


| Rank | Source column | Importance |
| --- | --- | --- |
| 1 | education\_num | 0.5599 |
| 2 | age | 0.5578 |
| 3 | marital\_status | 0.5428 |
| 4 | hours\_per\_week | 0.2400 |
| 5 | occupation | 0.2194 |
| 6 | capital\_gain | 0.1067 |
| 7 | education | 0.0352 |
| 8 | native\_region | 0.0254 |
| 9 | sex | 0.0223 |


## 14. Error analysis


Overall OOF error rate on DEV: 0.2922.


| Class | Precision | Recall | Support |
| --- | --- | --- | --- |
| \<=50K | 0.7259 | 0.8017 | 1866 |
| \>50K | 0.6752 | 0.5765 | 1334 |


*Worst segments (categories with at least 30 DEV rows)*

| Column | Rank | Category | Rows | Error rate |
| --- | --- | --- | --- | --- |
| marital\_status | 1 | Married | 1457 | 0.3157 |
| marital\_status | 2 | Divorced | 474 | 0.3017 |
| marital\_status | 3 | Never-married | 1086 | 0.2689 |
| marital\_status | 4 | Widowed | 183 | 0.2186 |
| occupation | 1 | Transport | 362 | 0.3343 |
| occupation | 2 | Exec-managerial | 376 | 0.3085 |
| occupation | 3 | (missing) | 159 | 0.3082 |
| occupation | 4 | Machine-op | 375 | 0.2907 |
| occupation | 5 | Sales | 390 | 0.2897 |
| education | 1 | Bachelors | 550 | 0.3436 |
| education | 2 | Assoc | 275 | 0.3382 |
| education | 3 | Some-college | 742 | 0.3032 |
| education | 4 | HS-grad | 1049 | 0.2908 |
| education | 5 | Masters | 175 | 0.2571 |


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


## 16. Reproducibility information


| Item | Value |
| --- | --- |
| Run id | 01M7X1VW0WMQZFAJB17K8N4P6W |
| TabPilot version | 0.1.0 |
| Execution profile | hosted |
| Split id | 9fd678b9b9825874d1692b623adc5e98da2951203f67fe0a58a0fc6539a4aa88 |
| Seeds | root=42, split=42, model=43, optuna=44, profile=45, explain=46 |
| Libraries | catboost 1.2.10, lightgbm 4.7.0, numpy 2.4.6, optuna 5.0.0, pandas 3.0.6, scikit-learn 1.9.1, xgboost 2.1.4 |
| Time budget (s) | 240 |
| Elapsed at report (s) | 68.7 |
| Trials used | 60 of 60 |


## 17. Narrative


> Template narrative, built deterministically from the facts table. Reason: no LLM provider available.


This run predicts \`income\` \[F2\] as a binary \[F1\] task. TabPilot compared 3 \[F8\] model types with cross-validation over 5 \[F7\] folds on the development rows and selected linear \[F4\] by its roc\_auc \[F3\] score.


The selected model's mean cross-validated score is 0.7730 \[F5\], with a fold-to-fold standard deviation of 0.0124 \[F6\]. The runner-up, lightgbm \[F9\], averaged 0.7667 \[F10\]; the selected model scored higher on 3 \[F12\] of 5 \[F7\] folds. With this few folds the comparison is descriptive, not a significance test. Cross-validated scores of the selected model are optimistic, because the same folds were used to choose it.


On the independent test rows, used once after selection, the selected model scored 0.7791 \[F13\]. The reference baseline, lightgbm \[F14\], scored 0.7547 \[F15\]. The 95% \[F16\] bootstrap interval for the selected model runs from 0.7477 \[F17\] to 0.8104 \[F18\].


The inputs this model relied on most were \`education\_num\` \[F22\], \`age\` \[F23\], \`marital\_status\` \[F24\], \`hours\_per\_week\` \[F25\] and \`occupation\` \[F26\]. Importance describes how this model uses its inputs, not what would happen in the world if a value changed; correlated inputs can share or swap importance.


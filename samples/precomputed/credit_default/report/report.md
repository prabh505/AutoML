# TabPilot report

Run 01M7X207B9N4PZA9ASZ7HX4RF8 · credit\_default.csv · status completed


## 1. Summary


| Item | Value |
| --- | --- |
| Task | binary |
| Target | default\_payment\_next\_month |
| Final model | linear (0-linear-00160141) |
| Primary metric | roc\_auc (maximize) |
| CV score (DEV, mean ± std) | 0.7353 ± 0.0118 |
| Independent test score | 0.7453 |
| Run status | completed |


## 2. Dataset summary


| Item | Value |
| --- | --- |
| File | credit\_default.csv |
| Format | csv |
| Rows (raw / valid) | 4000 / 4000 |
| Columns | 16 |
| Development rows | 3200 |
| Feature columns | 14 |
| Duplicate rows | 0 |
| Class counts (DEV) | 0: 2499, 1: 701 |
| Validation | stratified\_kfold, k = 5, repeats = 1, holdout fraction = 0.2 |


## 3. Data decisions


| Excluded column | Normalized name | Reason |
| --- | --- | --- |
| ID | id | identifier\_integer\_unique |


| Original column name | Normalized name |
| --- | --- |
| ID | id |
| LIMIT\_BAL | limit\_bal |
| SEX | sex |
| EDUCATION | education |
| MARRIAGE | marriage |
| AGE | age |
| PAY\_1 | pay\_1 |
| BILL\_AMT1 | bill\_amt1 |
| PAY\_AMT1 | pay\_amt1 |
| PAY\_2 | pay\_2 |
| BILL\_AMT2 | bill\_amt2 |
| PAY\_AMT2 | pay\_amt2 |
| PAY\_3 | pay\_3 |
| BILL\_AMT3 | bill\_amt3 |
| PAY\_AMT3 | pay\_amt3 |
| default.payment.next.month | default\_payment\_next\_month |


## 4. Data-quality and leakage warnings


No data-quality or leakage warnings were raised.


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
| 0-linear-00160141 | linear | baseline | 0.7353 | 0.0118 | 0.7499, 0.7362, 0.7253, 0.7434, 0.7219 | 0.04 | ok |
| 0-random\_forest-94dafae1 | random\_forest | baseline | 0.7236 | 0.0066 | 0.7339, 0.7253, 0.7164, 0.7205, 0.7216 | 0.35 | ok |
| 0-lightgbm-69f7d29e | lightgbm | baseline | 0.7236 | 0.0101 | 0.7337, 0.7325, 0.7093, 0.7234, 0.7189 | 0.09 | ok |


![CV mean score with ±std error bars](figures/leaderboard.png)


![CV score of every fold](figures/fold-scores.png)


*Threshold metrics on OOF predictions, 0-lightgbm-69f7d29e*

| Threshold rule | Threshold | Accuracy | F1 |
| --- | --- | --- | --- |
| threshold 0 5 | 0.500 | 0.7872 | 0.1498 |
| f1 optimal | 0.200 | 0.6366 | 0.4663 |


*Threshold metrics on OOF predictions, 0-linear-00160141*

| Threshold rule | Threshold | Accuracy | F1 |
| --- | --- | --- | --- |
| threshold 0 5 | 0.500 | 0.7947 | 0.2835 |
| f1 optimal | 0.235 | 0.6906 | 0.4762 |


*Threshold metrics on OOF predictions, 0-random\_forest-94dafae1*

| Threshold rule | Threshold | Accuracy | F1 |
| --- | --- | --- | --- |
| threshold 0 5 | 0.500 | 0.7856 | 0.1552 |
| f1 optimal | 0.250 | 0.6987 | 0.4668 |


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
| 0-random\_forest-94dafae1 | 0.0118 | 0.0003 | 0.0229 | 5/5 |


> Paired fold differences are descriptive: with this few folds they are not a significance test.


## 12. Independent test (used once, after selection)


| Model | roc\_auc | Bootstrap 95% interval |
| --- | --- | --- |
| Final model | 0.7453 | 0.7028 – 0.7884 |
| Reference baseline (lightgbm) | 0.6672 | — |


> These rows were used once, after the selection was frozen, and never for training, tuning or selection.


## 13. Feature importance and SHAP


Method: shap\_linear. Rows explained: 1000.


![Mean \|SHAP\| per source column](figures/shap-importance.png)


![SHAP values](figures/shap-beeswarm-1.png)


| Rank | Source column | Importance |
| --- | --- | --- |
| 1 | pay\_2 | 0.3133 |
| 2 | pay\_1 | 0.2990 |
| 3 | limit\_bal | 0.2659 |
| 4 | pay\_3 | 0.2268 |
| 5 | bill\_amt3 | 0.1140 |
| 6 | pay\_amt3 | 0.0761 |
| 7 | bill\_amt2 | 0.0345 |
| 8 | pay\_amt1 | 0.0344 |
| 9 | pay\_amt2 | 0.0320 |
| 10 | age | 0.0277 |
| 11 | bill\_amt1 | 0.0260 |
| 12 | education | 0.0211 |
| 13 | marriage | 0.0139 |
| 14 | sex | 0.0035 |


## 14. Error analysis


Overall OOF error rate on DEV: 0.2053.


| Class | Precision | Recall | Support |
| --- | --- | --- | --- |
| 0 | 0.8086 | 0.9656 | 2499 |
| 1 | 0.6019 | 0.1854 | 701 |


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
| Run id | 01M7X207B9N4PZA9ASZ7HX4RF8 |
| TabPilot version | 0.1.0 |
| Execution profile | hosted |
| Split id | e4814c321dfa3aaf4423029d86b89415da1f8326f04dbdc24b451f97855acd52 |
| Seeds | root=42, split=42, model=43, optuna=44, profile=45, explain=46 |
| Libraries | catboost 1.2.10, lightgbm 4.7.0, numpy 2.4.6, optuna 5.0.0, pandas 3.0.6, scikit-learn 1.9.1, xgboost 2.1.4 |
| Time budget (s) | 240 |
| Elapsed at report (s) | 77.1 |
| Trials used | 60 of 60 |


## 17. Narrative


> Template narrative, built deterministically from the facts table. Reason: no LLM provider available.


This run predicts \`default\_payment\_next\_month\` \[F2\] as a binary \[F1\] task. TabPilot compared 3 \[F8\] model types with cross-validation over 5 \[F7\] folds on the development rows and selected linear \[F4\] by its roc\_auc \[F3\] score.


The selected model's mean cross-validated score is 0.7353 \[F5\], with a fold-to-fold standard deviation of 0.0118 \[F6\]. The runner-up, random\_forest \[F9\], averaged 0.7236 \[F10\]; the selected model scored higher on 5 \[F12\] of 5 \[F7\] folds. With this few folds the comparison is descriptive, not a significance test. Cross-validated scores of the selected model are optimistic, because the same folds were used to choose it.


On the independent test rows, used once after selection, the selected model scored 0.7453 \[F13\]. The reference baseline, lightgbm \[F14\], scored 0.6672 \[F15\]. The 95% \[F16\] bootstrap interval for the selected model runs from 0.7028 \[F17\] to 0.7884 \[F18\].


The inputs this model relied on most were \`pay\_2\` \[F23\], \`pay\_1\` \[F24\], \`limit\_bal\` \[F25\], \`pay\_3\` \[F26\] and \`bill\_amt3\` \[F27\]. Importance describes how this model uses its inputs, not what would happen in the world if a value changed; correlated inputs can share or swap importance.


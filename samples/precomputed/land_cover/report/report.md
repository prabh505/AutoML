# TabPilot report

Run 01M7X22Q266XRRPM58QTQ985JG · land\_cover.csv · status completed


## 1. Summary


| Item | Value |
| --- | --- |
| Task | multiclass |
| Target | land\_cover |
| Final model | linear (0-linear-3d8f3e9d) |
| Primary metric | log\_loss (minimize) |
| CV score (DEV, mean ± std) | 0.2004 ± 0.0144 |
| Independent test score | 0.2175 |
| Run status | completed |


## 2. Dataset summary


| Item | Value |
| --- | --- |
| File | land\_cover.csv |
| Format | csv |
| Rows (raw / valid) | 2500 / 2500 |
| Columns | 10 |
| Development rows | 2000 |
| Feature columns | 8 |
| Duplicate rows | 0 |
| Class counts (DEV) | forest: 532, cropland: 484, grassland: 451, urban: 278, water: 255 |
| Validation | stratified\_kfold, k = 5, repeats = 1, holdout fraction = 0.2 |


## 3. Data decisions


| Excluded column | Normalized name | Reason |
| --- | --- | --- |
| tile\_id | tile\_id | identifier\_integer\_unique |


## 4. Data-quality and leakage warnings


- W-LEAK-STRONG:red


| Column | Score | Metric | Level |
| --- | --- | --- | --- |
| red | 0.914 | auc | strong |


## 5. Modeling plan


Plan source: rule-based planner.


| Model | Priority | Class weight | Tuning share |
| --- | --- | --- | --- |
| lightgbm | 1 | balanced | 0.60 |
| random\_forest | 2 | balanced | 0.25 |
| linear | 3 | balanced | 0.15 |


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
| 0-ensemble-37301ae2 | ensemble | ensemble | 0.1953 | 0.0228 | 0.1684, 0.1941, 0.1912, 0.2317, 0.1911 | 0.30 | ok |
| 0-linear-3d8f3e9d | linear | baseline | 0.2004 | 0.0144 | 0.1780, 0.2005, 0.1997, 0.2175, 0.2064 | 0.03 | ok |
| 0-lightgbm-6283f9f8 | lightgbm | baseline | 0.2051 | 0.0352 | 0.1732, 0.2010, 0.1963, 0.2652, 0.1898 | 0.26 | ok |
| 0-random\_forest-1ceab2e8 | random\_forest | baseline | 0.2199 | 0.0188 | 0.2034, 0.2170, 0.2115, 0.2522, 0.2152 | 0.40 | ok |


![CV mean score with ±std error bars](figures/leaderboard.png)


![CV score of every fold](figures/fold-scores.png)


## 9. Reflection history


No reflection rounds were run.


## 10. Ensemble decision


Ensemble kept: yes. Reason: improved by 2.55% and better in 4 of 5 folds


| Member | Weight |
| --- | --- |
| 0-linear-3d8f3e9d | 0.500 |
| 0-lightgbm-6283f9f8 | 0.500 |


## 11. Why this model was selected


Rule: max mean CV score; near-ties (0.25 SEM): simpler, fewer features, faster, id. Winner: 0-linear-3d8f3e9d.


> Cross-validation scores of the selected model are optimistically biased, because the same folds were used for tuning and selection.


> 3 configurations were compared in total (every model and retrained variant on the leaderboard, including failed and skipped ones). The more configurations are compared, the more optimistic the best score tends to be.


| Runner-up | Mean difference | Min | Max | Folds won |
| --- | --- | --- | --- | --- |
| 0-lightgbm-6283f9f8 | 0.0047 | -0.0166 | 0.0478 | 2/5 |


> Paired fold differences are descriptive: with this few folds they are not a significance test.


## 12. Independent test (used once, after selection)


| Model | log\_loss | Bootstrap 95% interval |
| --- | --- | --- |
| Final model | 0.2175 | 0.1738 – 0.2640 |
| Reference baseline (lightgbm) | 0.3982 | — |


> These rows were used once, after the selection was frozen, and never for training, tuning or selection.


## 13. Feature importance and SHAP


Method: shap\_member\_plus\_permutation. Rows explained: 1000.


> SHAP values are for the highest-weight ensemble member.


![Mean \|SHAP\| per source column](figures/shap-importance.png)


![Permutation importance (one DEV fold)](figures/permutation-importance.png)


![SHAP values per row, class forest](figures/shap-beeswarm-1.png)


![SHAP values per row, class cropland](figures/shap-beeswarm-2.png)


![SHAP values per row, class grassland](figures/shap-beeswarm-3.png)


| Rank | Source column | Importance |
| --- | --- | --- |
| 1 | red | 2.3972 |
| 2 | nir | 1.7333 |
| 3 | blue | 1.5879 |
| 4 | green | 1.1080 |
| 5 | ndvi | 0.1644 |
| 6 | region | 0.0964 |
| 7 | elevation\_m | 0.0893 |
| 8 | slope\_deg | 0.0429 |


## 14. Error analysis


Overall OOF error rate on DEV: 0.0785.


| Class | Precision | Recall | Support |
| --- | --- | --- | --- |
| cropland | 0.8632 | 0.8471 | 484 |
| forest | 0.9811 | 0.9774 | 532 |
| grassland | 0.8290 | 0.8492 | 451 |
| urban | 0.9892 | 0.9892 | 278 |
| water | 1.0000 | 1.0000 | 255 |


*Worst segments (categories with at least 30 DEV rows)*

| Column | Rank | Category | Rows | Error rate |
| --- | --- | --- | --- | --- |
| region | 1 | west | 498 | 0.0863 |
| region | 2 | south | 472 | 0.0826 |
| region | 3 | east | 537 | 0.0764 |
| region | 4 | north | 493 | 0.0690 |


![Confusion matrix (OOF, row-normalized)](figures/confusion-matrix.png)


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
| Run id | 01M7X22Q266XRRPM58QTQ985JG |
| TabPilot version | 0.1.0 |
| Execution profile | hosted |
| Split id | 1e8481ca520060aaee4b302e0f5467a40d323a60d8feacd803f3af6f10e4c962 |
| Seeds | root=42, split=42, model=43, optuna=44, profile=45, explain=46 |
| Libraries | catboost 1.2.10, lightgbm 4.7.0, numpy 2.4.6, optuna 5.0.0, pandas 3.0.6, scikit-learn 1.9.1, xgboost 2.1.4 |
| Time budget (s) | 240 |
| Elapsed at report (s) | 80.7 |
| Trials used | 60 of 60 |


## 17. Narrative


> Template narrative, built deterministically from the facts table. Reason: no LLM provider available.


This run predicts \`land\_cover\` \[F2\] as a multiclass \[F1\] task. TabPilot compared 3 \[F8\] model types with cross-validation over 5 \[F7\] folds on the development rows and selected linear \[F4\] by its log\_loss \[F3\] score.


The selected model's mean cross-validated score is 0.2004 \[F5\], with a fold-to-fold standard deviation of 0.0144 \[F6\]. The runner-up, lightgbm \[F9\], averaged 0.2051 \[F10\]; the selected model scored higher on 2 \[F12\] of 5 \[F7\] folds. With this few folds the comparison is descriptive, not a significance test. Cross-validated scores of the selected model are optimistic, because the same folds were used to choose it.


On the independent test rows, used once after selection, the selected model scored 0.2175 \[F13\]. The reference baseline, lightgbm \[F14\], scored 0.3982 \[F15\]. The 95% \[F16\] bootstrap interval for the selected model runs from 0.1738 \[F17\] to 0.2640 \[F18\].


The inputs this model relied on most were \`red\` \[F23\], \`nir\` \[F24\], \`blue\` \[F25\], \`green\` \[F26\] and \`ndvi\` \[F27\]. Importance describes how this model uses its inputs, not what would happen in the world if a value changed; correlated inputs can share or swap importance. The final model blends 2 \[F28\] models, kept because the blend scored higher on most folds.


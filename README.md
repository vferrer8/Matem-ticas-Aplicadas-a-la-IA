# Mathematics Applied to AI: Regression Models for Energy Demand

Comparison of linear, polynomial and regularised regression models to predict energy demand from economic and climate variables. The model choice is backed by information criteria (AIC / BIC) and cross-validation.

Coursework for the "Mathematics Applied to AI" module of the MSc in Artificial Intelligence (UTAMED). The accompanying PDF contains the theoretical development and the discussion of results.

## The dataset

`datos.csv`: 200 observations, all numeric.

| Variable | Role |
|---|---|
| `temperatura_media` (mean temperature) | Predictor |
| `gasto_consumidor` (consumer spending) | Predictor |
| `inflacion` (inflation) | Predictor |
| `desempleo` (unemployment) | Predictor |
| `consumo_historico` (historical consumption) | Predictor |
| `demanda_energia` (energy demand) | **Target** |

## What the notebook does

**1. Exploration.** Pair plot, correlation heatmap and box plots. The extreme values are kept: they reflect seasonal peaks, such as air conditioning in summer, rather than errors.

**2. Full linear model.** The five predictors are standardised and their coefficients inspected. Temperature (negative) and historical consumption (positive) dominate. The final models keep three predictors: temperature, unemployment and historical consumption.

**3. Model comparison on a held-out split (85 / 15).**

| Model | R² | RMSE | AIC | BIC |
|---|---|---|---|---|
| Linear regression | 0.919 | 217.4 | **1542.9** | **1555.5** |
| Polynomial regression (degree 2) | 0.923 | 211.3 | 1545.2 | 1576.5 |

The polynomial model gains very little fit for three times as many parameters. Both AIC and BIC prefer the linear model.

**4. Regularisation and 5-fold cross-validation** (Ridge with α chosen by `RidgeCV`):

| Model | CV R² | CV RMSE |
|---|---|---|
| Linear regression | 0.818 | 271.8 |
| Polynomial (degree 2) | 0.808 | 278.5 |
| Ridge | **0.818** | **271.8** |
| Polynomial + Ridge | 0.810 | 277.7 |

**Conclusion.** The relationship is essentially linear. Polynomial terms add complexity without improving out-of-sample error, and Ridge matches plain OLS because the three predictors are not strongly collinear. The simplest model is the right choice.

## Repository contents

```
├── RAA_Mates_Clean_VF.ipynb     # Notebook: EDA, regression models, AIC/BIC, cross-validation
└── RAA_Mates_VictorFerrer.pdf   # Report: theory and discussion of results
```

> The notebook and the report are in Spanish. The source CSV is not included in the repository.

## Tech stack

Python · pandas · NumPy · scikit-learn (`LinearRegression`, `PolynomialFeatures`, `RidgeCV`, `cross_validate`) · Seaborn · Jupyter

## Author

**Víctor Ferrer**, MSc in Artificial Intelligence, UTAMED

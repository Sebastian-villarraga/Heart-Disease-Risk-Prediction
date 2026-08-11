# Heart Disease Risk Prediction

## Exercise Summary
This project implements logistic regression from scratch to predict heart disease using the Kaggle Heart Disease dataset. The notebook includes exploratory data analysis, binary logistic regression training, decision boundary visualization, L2 regularization, and a SageMaker preparation section for managed cloud training and testing.

## Dataset Description
The dataset is taken from the Kaggle Heart Disease dataset:
- Source: https://www.kaggle.com/datasets/neurocipher/heartdisease
- Records: 1025 rows
- Features: age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, resting ECG, maximum heart rate achieved, exercise-induced angina, ST depression, slope, number of major vessels, thalassemia, and a binary heart disease target.
- Target: 0 = no disease, 1 = disease presence.

## Files
- `heart_disease_lr_analysis.ipynb`: Jupyter notebook with the full analysis.
- `heart.csv`: Dataset used for training and evaluation.
- `sagemaker_train.py`: A training script template that can be adapted for Amazon SageMaker training.
- `images/`: Placeholders for SageMaker evidence images.

## SageMaker Evidence
This repository includes a SageMaker training and testing preparation section in the notebook. Add actual SageMaker screenshots to the `images/` folder and update the README image links if needed.

![SageMaker training screenshot](images/sagemaker_training.png)
![SageMaker training completion screenshot](images/sagemaker_completion.png)
![SageMaker metrics screenshot](images/sagemaker_metrics.png)

## Notes
- The notebook trains logistic regression from scratch without scikit-learn for the model implementation.
- The SageMaker section prepares training data and a script for upload into the AWS Academy SageMaker environment.

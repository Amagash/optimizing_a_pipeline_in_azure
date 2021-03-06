# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run. 

The main steps are described in the following diagram:

![img_1](img/img_1.png)

## Summary
**In 1-2 sentences, explain the problem statement: e.g "This dataset contains data about... we seek to predict..."**
### Problem statement

This dataset contains data about telemarketing campaigns of banks and is part of the following research paper: [A Data-Driven Approach to Predict the Success of Bank Telemarketing](https://repositorio.iscte-iul.pt/bitstream/10071/9499/5/dss_v3.pdf)

The goal is to predict the success of telemarketing calls for selling bank long-term deposits.

The dataset is a tabular dataset containing 32951 rows, 20 features and one target variable.

|    | Variable name | Type            | Description                                      | Example           |
|----|---------------|-----------------|--------------------------------------------------|-------------------|
| 1  | age           | numerical (int) | age of the customer                              | 25                |
| 2  | job           | categorical     | job of the customer                              | technician        |
| 3  | marital       | categorical     | marital status of the customer                   | married           |
| 4  | education     | categorical     | education level of the customer                  | university.degree |
| 5  | default       | categorical     | Unknown. Can be 'yes', 'no', 'unknown'           | no                |
| 6  | housing       | categorical     | Has housing loan? can be 'yes', 'no', 'unknown'. | yes               |
| 7  | loan          | categorical     | Has personal loan? Can be 'yes', 'no', 'unknown' | no                |
| 8  | contact       | categorical     | Type of communication to contact the customer    | cellular          |
| 9  | month         | categorical     | Month the customer was contacted                 | jun               |
| 10 | day_of_week   | categorical     | The day of the week the customer was contacted   | fri               |
| 11 | duration      | numeric         | communication duration in seconds                | 285               |
| 12 | campaign      | numeric         | Number of times the customer was contacted       | 4                 |
| 13 | pdays         | numeric         | Unknown.                                         | 999               |
| 14 | previous      | numeric         | Unknown.                                         | 0                 |
| 15 | poutcome      | categorical     | outcome of the campaign                          | nonexistent       |
| 16 | emp.var.rate  | numeric         | employment variation rate - quarterly indicator  | 1.4               |
| 17 | cons.price.id | numeric         | consumer price index - monthly indicator         | 92.8929999999999  |
| 18 | cons.conf.idx | numeric         | consumer confidence index - monthly indicator    | -46.2             |
| 19 | euribor3m     | numeric         | euribor 3 month rate - daily indicator           | 1.299             |
| 20 | nr.employed   | numeric         | number of employees - quarterly indicator        | 5099.1            |
|----|---------------|-----------------|--------------------------------------------------|-------------------|
| 21 | y             | biniary         | target                                           | 'yes' or 'no'     |

**In 1-2 sentences, explain the solution: e.g. "The best performing model was a ..."**

Two models were built using two different approaches, Scikit-learn and Azure AutoML.

| Model        | Best score |
|--------------|------------|
| Scikit-learn | 91.50%     |
| Azure AutoML | 91.80%     |

The best performing model was built using Azure AutoML.
## Scikit-learn Pipeline
**Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**

Those are the steps of the Scikit-learn pipeline:
1. Initialize the Workspace and get the workspace config
2. Create the Experiment in the workspace called `udacity-project`
3. Create the compute cluster on the workspace called `udacity-cluster`
4. Create a `HyperDriveConfig` specifying 
   - SKLearn estimator in the `train.py`: `LogisticRegression` specifying the following parameters
      - `C` : Inverse of regularization strength. Smaller values cause stronger regularization
      - `max_iter` : Maximum number of iterations to converge
   - Policy:  An early termination policy based on slack factor/slack amount and evaluation interval
   - Parameter sampler: Defines random sampling over a hyperparameter search space
5. Submit the `HyperDriveConfig` to the experiment and run it:
   1. Loading the data
   2. Cleaning the data (drop NaN values, One Hot encoding, Convertion of some categorical variable into dummy variables...)
   3. Split the data into training and test sets
   4. Parse the arguments (C, max_iter) for the Logistic Regression parameters
   5. Train the model on the training set
   6. Compute the accuracy on the test set
   7. Log the results
6.  Get the best run and save the model

The best run yields to an accuracy of 91.50% with C=2.79 and max_iter=400.

### Pipeline architecture

**What are the benefits of the parameter sampler you chose?**

The `azureml.train.hyperdrive` package contains modules and classes supporting hyperparameter tuning. It is possible to define the parameter search space as discrete or continuous, and a sampling method over the search space as random (`RandomParameterSampling`), grid (`GridParameterSampling`), or Bayesian (`BayesianParameterSampling`).

The `GridParameterSampling` method defines a grid of hyperparameter values. The tuning algorithm exhaustively searches this space in a sequential manner and trains a model for every possible combination of hyperparameter values. This method has the advantage of being exhaustive, but not very efficient.

The `RandomParameterSampling` method differs from grid search in that we no longer provide an explicit set of possible values for each hyperparameter; rather, we provide a statistical distribution for each hyperparameter from which values are sampled. Essentially, we define a sampling distribution for each hyperparameter to carry out a randomized search. Early termination uses knowledge from previous runs to determine poorly performing runs.

The `BayesianParameterSampling` method is a sequential model-based optimization (SMBO) algorithm that uses the results from the previous iteration to decide the next hyperparameter value candidates.

Random and Bayesian methods are more efficient than grid search. For this project, the `RandomParameterSampling` was chosen as the parameter sampler because it is more optimized than grid search and and it supports early termination policy which is not the case with the Bayesian method.

**What are the benefits of the early stopping policy you chose?**

Early termination policies can be applied to HyperDrive runs to improve computational efficiency. A run is cancelled when the criteria of a specified policy are met. The `azureml.train.hyperdrive` package contains three different early termination policy classes. 

The `BanditPolicy` Class defines an early termination policy based on slack criteria, and a frequency and delay interval for evaluation. Any run that doesn't fall within the slack factor or slack amount of the evaluation metric with respect to the best performing run will be terminated.

The `MedianStoppingPolicy` Class defines an early termination policy based on running averages of the primary metric of all runs.The Median Stopping policy computes running averages across all runs and cancels runs whose best performance is worse than the median of the running averages. Specifically, a run will be canceled at interval N if its best primary metric reported up to interval N is worse than the median of the running averages for intervals 1:N across all runs.

The `TruncationSelectionPolicy` Class defines an early termination policy that cancels a given percentage of runs at each evaluation interval. This policy periodically cancels the given percentage of runs that rank the lowest for their performance on the primary metric. The policy strives for fairness in ranking the runs by accounting for improving model performance with training time. When ranking a relatively young run, the policy uses the corresponding (and earlier) performance of older runs for comparison. Therefore, runs aren't terminated for having a lower performance because they have run for less time than other runs.

The Bandit policy was chosen because it has the advantage of having more aggressive savings compared to the Median policy that is more conservative.

## AutoML
**In 1-2 sentences, describe the model and hyperparameters generated by AutoML.**

The AutoML run was designed with the following parameters:
- `experiment_timeout_minutes=30`: Maximum time in minutes that each iteration can run for before it terminates is 30 min
- `task='classification'`: The type of task to run here is classification
- `primary_metric='accuracy'`: The metric that Automated Machine Learning will optimize for model selection is the accuracy.
- `training_data=dataset`: The training data to be used within the experiment. It contains both training features and a label column.
- `label_column_name='target'`: The name of the label column is 'target'.
- `n_cross_validations=5`: The number of cross validation to perform is 5.

The best model generated by this AutoML run is a Voting Ensemble with an accuracy of 91.80%.

![img_2](img/img_2.PNG)

A voting ensemble (or a ???majority voting ensemble???) is an ensemble machine learning model that combines the predictions from multiple other models. The algorithms selected in the ensemble are the XGBoostClassifier and the LightGBM.

AutoML generated the following steps in the pipeline.

![img_3](img/img_3.PNG)

Here is a list of the metrics measured.

![img_4](img/img_4.PNG)

## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**

Architectures were pretty different with a Logistic Regression model for Scikit-learn and a Voting Ensemble model for AutoML. Despite this difference, the two models have comparable performances with 91.80% accuracy for AutoML versus 91.50% for Scikit-learn and a running time of 34 min 44 sec for AutoML and 26 min 45 sec for Scikit-learn. 

The real difference is in the total time it takes to get the model. With Scikit-learn, we need to generate the train.py and specify all the hyperparameter tuning while only having to give the dataset to AutoML. It is definitely a time saver without loosing in accuracy and compute time.

## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**

One flag was raised during the AutoML run as shown in the image below.

![img_5](img/img_5.PNG)

It is noticed that there are imbalanced classes in the input. The algorithms used by automated ML detect imbalance when the number of samples in the minority class is equal to or fewer than 20% of the number of samples in the majority class, where minority class refers to the one with fewest samples and majority class refers to the one with most samples.

This imbalance can be handled by using technique like **Synthetic Minority Oversampling Technique** (SMOTE) during the data preparation step. It could be interesting to preprocess the data with SMOTE in futur work.

Also, additional hypeparameters could be tested in the Sklearn Logistic Regression along with different parameter sampling techniques and tuning the arguments of the BanditPolicy. 

The AutoML run could also be extended with different experiment timeout to see if the performance could be improved.

## Proof of cluster clean up
**If you did not delete your compute cluster in the code, please complete this section. Otherwise, delete this section.**
**Image of cluster marked for deletion**

`aml_compute.delete()`

![img_6](img/img_6.PNG)


Sources : 
- [A Data-Driven Approach to Predict the Success of Bank Telemarketing](https://repositorio.iscte-iul.pt/bitstream/10071/9499/5/dss_v3.pdf)
- [Comparison of Hyperparameter Tuning algorithms: Grid search, Random search, Bayesian optimization](https://medium.com/analytics-vidhya/comparison-of-hyperparameter-tuning-algorithms-grid-search-random-search-bayesian-optimization-5326aaef1bd1)
- [Hyperparameter tuning a model with Azure Machine Learning](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)
- [hyperdrive Package](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive?view=azure-ml-py)
- [Handle imbalanced data](https://docs.microsoft.com/en-us/azure/machine-learning/concept-manage-ml-pitfalls#handle-imbalanced-data)
- [SMOTE](https://docs.microsoft.com/en-us/azure/machine-learning/studio-module-reference/smote)

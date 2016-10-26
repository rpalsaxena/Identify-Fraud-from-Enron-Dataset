# Identify-Fraud-from-Enron-Dataset

In 2000, Enron was one of the largest companies in the United States. 
By 2002, it had collapsed into bankruptcy due to widespread corporate fraud.
In the resulting Federal investigation, a significant amount of typically confidential information entered into the public record,
including tens of thousands of emails and detailed financial data for top executives.
In this project, I am building a person of interest identifier based on financial and email data made public as a result of the Enron scandal.


# > Goal of this project

I am trying to build a predictive model that can identify _Persons of Interest_  based on the features of Enron Dataset. 

This model can also be used to investigate frauds at other companies.

It contains 146 records with 21 features/ attributes.

I found 3 outliers in this dataset, removed them by using pop().

`data_dict.pop("TOTAL",0)` , `data_dict.pop("THE TRAVEL AGENCY IN THE PARK",0)` : These two values can not be considered as the names of person. So, it is a wrong value in dataset and can be considered as outliers.

`data_dict.pop("LOCKHART EUGENE E",0)`: This value be considered as name of a person but it didn't contain any values in its features set.

All features have 'NaN' values and "poi" feature is also `False`. Its an empty record in our data. So, this can also be considered as an outlier.

# > Feature Selection, Scaling

I added all the `financial_features`, `email_features` to `feature_list`. I also added 3 new features in this data set:

* `poi_to_person_rate` : Ratio of no. of emails recieved from POI and total no. of emails recieved.
* `person_to_poi_rate` : Ratio of no. of emails sent to POI and total no. of emails sent.
* `total`              : Sum of all the financial features for each record.

I used SelectKBest() to select best 6 features from this dataset. It selected `'exercised_stock_options'`, `'total_stock_value'`, `'bonus'`, `'salary'`,`'total'`, `'person_to_poi_rate'` features from the dataset.

I have used MinMaxScaler to scale all the feature values in a range of [0-1]. It helps to avoid problems caused by different units in the data. Helps in maintaining consistency among values range.

# > Algorithm Selection

I tested my dataset on 5 different algorithms, used GridSearchCV() for tuning my algorithm's parameters.

**GaussianNB :**

`Accuracy: 0.86073	Precision: 0.46846	Recall: 0.33050	F1: 0.38757	F2: 0.35118`

**Decision Tree :**

`Accuracy: 0.86573	Precision: 0.48643	Recall: 0.12550	F1: 0.19952	F2: 0.14737`

Best parameters : 

`class_weight=None, criterion='entropy', max_depth=2,
max_features=None, max_leaf_nodes=10, min_samples_leaf=1,
min_samples_split=20, min_weight_fraction_leaf=0.0,
presort=False, random_state=None, splitter='random'
`
**Support Vector Machine :**

`Got a divide by zero when trying out. Precision or recall may be undefined due to a lack of true positive predicitons.
`

This algo didn't work well here. Its showing precision , recall undefined due to lack of positive predictions.

**AdaBoost :**

`Accuracy: 0.84847	Precision: 0.19734	Recall: 0.04450	F1: 0.07262	F2: 0.05266`

Best parameters :

`
algorithm='SAMME', base_estimator=None, learning_rate=5.0,
n_estimators=5, random_state=None
`

**RandomForest :**

`Accuracy: 0.84927	Precision: 0.37391	Recall: 0.19350	F1: 0.25502	F2: 0.21417`

Best parameters :

`bootstrap=True, class_weight=None, criterion='entropy',
max_depth=None, max_features='auto', max_leaf_nodes=10,
min_samples_leaf=1, min_samples_split=2,
min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
oob_score=False, random_state=None, verbose=0,
warm_start=False
'



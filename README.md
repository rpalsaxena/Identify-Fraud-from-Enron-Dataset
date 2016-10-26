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



#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import matplotlib.pyplot
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from tester import test_classifier
from sklearn.feature_selection import SelectKBest
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report	
from sklearn.cross_validation import StratifiedShuffleSplit

	
sys.stdout=open("test.txt","w")

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
financial_features = ['salary', 'deferral_payments', 'total_payments',
					'loan_advances', 'bonus', 'restricted_stock_deferred',
					'deferred_income', 'total_stock_value','expenses',
					'exercised_stock_options', 'other', 'long_term_incentive',
                      'restricted_stock', 'director_fees']

email_features = ['to_messages', 'from_poi_to_this_person', 'from_messages',
                  'from_this_person_to_poi', 'shared_receipt_with_poi']

features_list =  ['poi'] + financial_features + email_features


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop("TOTAL",0)
data_dict.pop("THE TRAVEL AGENCY IN THE PARK",0)
	

my_dataset = data_dict

### Task 3: Create new feature(s)
for keys,values in data_dict.iteritems():
	if values["to_messages"] != "NaN" and \
		values["from_messages"] != "NaN" and \
		values["from_poi_to_this_person"] != "NaN" and \
		values["from_this_person_to_poi"] != "NaN":
		
		values["poi_to_person_rate"] = float(values["from_poi_to_this_person"]) / \
			values["to_messages"]
		values["person_to_poi_rate"] = float(values["from_this_person_to_poi"]) / \
			values["from_messages"]
	else:
		values["poi_to_person_rate"] = 0
		values["person_to_poi_rate"] = 0
	
	values["total"] = sum([values[f] for f in financial_features 
		if (values[f] != "NaN")])
	
features_list.append("poi_to_person_rate")
features_list.append("person_to_poi_rate")
features_list.append("total")

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

scaler = MinMaxScaler()
features = scaler.fit_transform(features)

# K-best features
kbest = SelectKBest(k = 6)
kbest.fit(features, labels)
f_list = zip(kbest.get_support(), features_list[1:], kbest.scores_)
f_list = sorted(f_list, key=lambda x: x[2], reverse = True)
'''
import pprint
print "K-best features:", 
pprint.pprint(f_list)
#Output shows that 'exercised_stock_options', 'total_stock_value', 'bonus', 'salary','total', 'person_to_poi_rate' are the best 5 features of this dataset
'''
#updated feature list
features_list = ['poi', 'exercised_stock_options', \
				'total_stock_value', 'bonus', \
				'salary','total', 'person_to_poi_rate']

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

feature_train, feature_test, labels_train, labels_test = \
	train_test_split( features, labels, test_size=0.3, random_state=42)	


# from sklearn.naive_bayes import GaussianNB
#Naive bayes
gnb_clf = GaussianNB()
parameters = {}
algo = GridSearchCV(gnb_clf, parameters)
print '\nGaussianNB:'
algo.fit(feature_train, labels_train)
test_classifier(algo.best_estimator_, my_dataset, features_list)


#Decision Tree
print '\nDecision Tree:'
dt_clf = tree.DecisionTreeClassifier()
parameters = {'criterion': ['gini', 'entropy'], \
			'min_samples_split': [2, 5, 10, 20], \
			'max_depth': [None, 2, 5, 10], \
			'splitter': ['random', 'best'], \
			'max_leaf_nodes': [None, 5, 10, 20]   }
algo = GridSearchCV(dt_clf, parameters)	
algo.fit(feature_train, labels_train)
test_classifier(algo.best_estimator_, my_dataset, features_list)


###SVC
print '\nSVM:'
svc_clf=SVC()
parameters = {'C': [0.001, 0.01, 0.1, 1, 10], \
			'kernel': ['rbf', 'linear', 'poly'], \
			'gamma': [0.001, 0.01, 0.1, 1], \
			'max_iter': [-1, 5, 10, 50]   }
algo = GridSearchCV(svc_clf, parameters)
algo.fit(feature_train, labels_train)
test_classifier(algo.best_estimator_, my_dataset, features_list)



###AdaBoost
print '\nAdaBoost:'
ada_clf = AdaBoostClassifier(algorithm='SAMME')
parameters = {'learning_rate': [0.1, 0.5, 1.0, 5.0], \
			'algorithm': ['SAMME', 'SAMME.R'], \
			'n_estimators': [1, 5, 10, 50, 100]}
algo = GridSearchCV(ada_clf, parameters)
algo.fit(feature_train, labels_train)
test_classifier(algo.best_estimator_, my_dataset, features_list)

###RandomForest
print '\nRandomForest\n'
rf_clf = RandomForestClassifier()
parameters = {'criterion': ['gini', 'entropy'], \
			'max_depth': [None, 2, 5, 10], \
			'max_leaf_nodes': [None, 5, 10, 20], \
			'n_estimators': [1, 5, 10, 50, 100]}
algo = GridSearchCV(rf_clf, parameters)
algo.fit(feature_train, labels_train)
test_classifier(algo.best_estimator_, my_dataset, features_list)



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

clf = gnb_clf
dump_classifier_and_data(clf, my_dataset, features_list)


sys.stdout.close()
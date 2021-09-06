from datetime import time
import numpy as np
# np.random.seed(..) # for reproducibility
import pandas as pd

import os, datetime
import pickle

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, confusion_matrix, classification_report
from sklearn import svm, tree
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import *
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import xgboost

from numpy import quantile, where, random

import matplotlib.pyplot as plt
import matplotlib

NUM_SPLIT = 10
ALL_RESULTS_FILE = 'all_results.csv'
VENN_DATA_FILE = 'venn_diagram_data.csv'
reader = open(ALL_RESULTS_FILE, 'w')
v_reader = open(VENN_DATA_FILE, 'w')

def load_data(filename, delimiter='|'):
    df = pd.read_csv(filename, delimiter=delimiter) # , nrows=500)
    all_unique_klasses = df['raw_label'].unique()
    klasses_dict = dict()

    for i, klass in enumerate(all_unique_klasses):
        klasses_dict[klass] = i
    
    df['doc_vector'] = df.apply(lambda row: process_doc_raw_vectors(row), axis=1)
    df['code_vector'] = df.apply(lambda row: process_source_raw_vectors(row), axis=1)
    df['klass'] = df.apply(lambda row: process_raw_klasses(row, klasses_dict), axis=1)

    # features
    doc_features = ['d-%d' % i for i, v in enumerate(df['doc_vector'][0])]
    code_features = ['c-%d' % i for i, v in enumerate(df['code_vector'][0])]
    
    df[doc_features] = pd.DataFrame(df.doc_vector.tolist(), index= df.index)
    df[code_features] = pd.DataFrame(df.code_vector.tolist(), index= df.index)

    return df, klasses_dict, doc_features, code_features

def process_doc_raw_vectors(row):
    vector = []
    for value in row['documentation_raw_vector'].split(' '):
        vector.append(float(value))
    
    vector = np.asarray(vector)
    return vector

def process_source_raw_vectors(row):
    vector = []
    for value in row['source_code_raw_vector'].split(' '):
        vector.append(float(value))
    
    vector = np.asarray(vector)
    return vector

def process_raw_klasses(row, klasses_dict):
    return klasses_dict[row['raw_label']]

def get_classifiers(multi_class=False, input_shape=(), num_klasses=0):
    classifiers = []

    model0 = svm.OneClassSVM(kernel='rbf', gamma=0.001, nu=0.03)
    classifiers.append((model0, 'SVM-1C'))

    model7 = KNeighborsClassifier()
    classifiers.append((model7, 'KNN'))

    model1 = xgboost.XGBClassifier(random_state=23)
    classifiers.append((model1, 'XGB'))

    model2 = svm.SVC(kernel='linear', probability=True)
    classifiers.append((model2, 'SVC'))

    model3 = tree.DecisionTreeClassifier(random_state=23)
    classifiers.append((model3, 'DT'))
    
    model4 = RandomForestClassifier(random_state=23)
    classifiers.append((model4, 'RF'))

    model5 = GaussianNB()
    classifiers.append((model5, 'GNB'))

    model6 = SGDClassifier(random_state=23)
    classifiers.append((model6, 'SGD'))

    model7 = KNeighborsClassifier()
    classifiers.append((model7, 'KNN'))

    model8 = BaggingClassifier(tree.DecisionTreeClassifier(random_state=23), random_state=23)
    classifiers.append((model8, 'BDT1'))

    model9 = BaggingClassifier(tree.DecisionTreeClassifier(random_state=23), random_state=23)
    classifiers.append((model9, 'BDT2'))

    model10 = ExtraTreesClassifier(random_state=23)
    classifiers.append((model10, 'ET'))

    model11 = AdaBoostClassifier(random_state=23)
    classifiers.append((model11, 'ADA1'))

    model12 = AdaBoostClassifier(random_state=23)
    classifiers.append((model12, 'ADA2'))

    model13 = GradientBoostingClassifier(random_state=23)
    classifiers.append((model13, 'GB'))

    model14 = MLPClassifier(random_state=1, max_iter=1000, activation='logistic', early_stopping=True, verbose=True)
    classifiers.append((model14, 'NN'))

    return classifiers

def train_and_test_one_class(X, features, y, result_filename, num_splits=NUM_SPLIT, input_shape=(384, 1), num_klasses=6, addon=''):

    y = np.array([sample for sample in y])

    classifiers = get_classifiers()[0:1]
    res = {}  # Dictionary for results
    cms = {}  # Dictionary for CM

    # SCORINGS
    scoring = { 'accuracy'  : make_scorer(accuracy_score), 
                'precision' : make_scorer(precision_score),
                'recall'    : make_scorer(recall_score), 
                'f1_score'  : make_scorer(f1_score),
                'kappa'     : make_scorer(cohen_kappa_score) }

    def report_scoring(y_true, y_p):
        print(classification_report(y_true, y_p))
        return accuracy_score(y_true, y_p)

    kfold = None
    X = X[features]
    X_train, X_test, y_train, y_test = X, X, y, y # train_test_split(X, y, test_size=0.15, random_state=23)

    output_classifiers = []
    kfold_models_and_testing_data = dict()
    for i, clf in enumerate(classifiers):

        clf_name = clf[1]  # Classifier name to identify which classifier is used
        res[clf_name] = {}
        cms[clf_name] = ''

        print('*' * 5, clf_name, '*' * 5)

        # pred = clf[0].fit_predict(X_train)
        # scores = clf[0].score_samples(X_train)
        # thresh = quantile(scores, 0.03)
        # index = where(scores <= thresh)
        # tmp_X = X_train.to_numpy()
        # values = tmp_X[index]

        # print(values)

        # plt.scatter(tmp_X[:,0], tmp_X[:,1])
        # plt.scatter(values[:,0], values[:,1], color='r')
        # plt.show()

        # NO NO
        clf[0].fit(X_train)
        y_pred = clf[0].predict(X_test)
        y_pred = [int(yp >= 1) for yp in y_pred]
        accuracy, f1_score_m, precision, recall = accuracy_score(y_test, y_pred), f1_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)
        cms[clf_name] = confusion_matrix(y_test, y_pred)

        # GET THE MEAN OF ALL SCORINGS
        res[clf_name]['acc'] = accuracy
        res[clf_name]['prec'] = precision
        res[clf_name]['rec'] = recall
        res[clf_name]['fsco'] = f1_score_m
        res[clf_name]['kappa'] = kappa

        # output classifier
        output_classifiers.append((clf[0], clf_name))
    
    with open(result_filename, 'w') as f:
        line = '\n{:32s} {:10s}{:10s}{:10s}    {:10s}{:10s}{:10s}'.format('', 'Accuracy', 'Precision', 'Recall', 'F-score', 'Kappa', 'Eperiment')
        f.write(line + '\n')
        print(line)

        for clf in res.keys():
            curr = res[clf]
            line = '%s,%f,%f,%f,%f,%f,%s' % (clf, curr['acc'], curr['prec'], curr['rec'], curr['fsco'], curr['kappa'], addon)
            reader.write(line + '\n')
            line = '{:30s} {:10f} {:10f} {:10f} {:10f} {:10f}   {:10s}'.format(clf, curr['acc'], curr['prec'], curr['rec'], curr['fsco'], curr['kappa'], addon)
            f.write(line + '\n')
            print(line)

    # TO REMOVE
    if kfold:
        return kfold_models_and_testing_data
    else:
        return output_classifiers

def train_and_test_binary(X, features, y, result_filename, num_splits=NUM_SPLIT, input_shape=(384, 1), num_klasses=6, addon=''):

    y = np.array([sample for sample in y])

    classifiers = get_classifiers()[1:]
    res = {}  # Dictionary for results
    cms = {}  # Dictionary for CM

    # SCORINGS
    scoring = { 'accuracy'  : make_scorer(accuracy_score), 
                'precision' : make_scorer(precision_score),
                'recall'    : make_scorer(recall_score), 
                'f1_score'  : make_scorer(f1_score),
                'kappa'     : make_scorer(cohen_kappa_score) }

    def report_scoring(y_true, y_p):
        print(classification_report(y_true, y_p))
        return accuracy_score(y_true, y_p)

    if num_splits > 1:
        kfold = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=23)
    else:
        kfold = None
        X = X[features]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=23)

    output_classifiers = []
    kfold_models_and_testing_data = dict()
    for i, clf in enumerate(classifiers):

        clf_name = clf[1]  # Classifier name to identify which classifier is used
        res[clf_name] = {}
        cms[clf_name] = ''

        print('*' * 5, clf_name, '*' * 5)

        if kfold:
            results = {
                'test_accuracy': [],
                'test_precision': [],
                'test_recall': [],
                'test_f1_score': [],
                'test_kappa': []
            }
            kfold_models_and_testing_data[clf_name] = []
            for train_index, test_index in kfold.split(X, y): # kfold.split(X)
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y[train_index], y[test_index]

                X_train = X_train[features]
                clf[0].fit(X_train, y_train)

                kfold_models_and_testing_data[clf_name].append((clf[0], X_test, y_test))
                
                X_test = X_test[features]
                y_pred = clf[0].predict(X_test)
                accuracy, f1_score_m, precision, recall = accuracy_score(y_test, y_pred), f1_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred)
                kappa = cohen_kappa_score(y_test, y_pred)
                cms[clf_name] = confusion_matrix(y_test, y_pred)

                results['test_accuracy'].append(accuracy)
                results['test_precision'].append(precision)
                results['test_recall'].append(recall)
                results['test_f1_score'].append(f1_score_m)
                results['test_kappa'].append(kappa)
            
            # GET THE MEAN OF ALL SCORINGS
            res[clf_name]['acc'] = np.mean(results['test_accuracy'])
            res[clf_name]['prec'] = np.mean(results['test_precision'])
            res[clf_name]['rec'] = np.mean(results['test_recall'])
            res[clf_name]['fsco'] = np.mean(results['test_f1_score'])
            res[clf_name]['kappa'] = np.mean(results['test_kappa'])
        else:
            clf[0].fit(X_train, y_train)
            y_pred = clf[0].predict(X_test)
            accuracy, f1_score_m, precision, recall = accuracy_score(y_test, y_pred), f1_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred)
            kappa = cohen_kappa_score(y_test, y_pred)
            cms[clf_name] = confusion_matrix(y_test, y_pred)

            # GET THE MEAN OF ALL SCORINGS
            res[clf_name]['acc'] = accuracy
            res[clf_name]['prec'] = precision
            res[clf_name]['rec'] = recall
            res[clf_name]['fsco'] = f1_score_m
            res[clf_name]['kappa'] = kappa

        # output classifier
        output_classifiers.append((clf[0], clf_name))
    
    with open(result_filename, 'w') as f:
        line = '\n{:32s} {:10s}{:10s}{:10s}    {:10s}{:10s}{:10s}'.format('', 'Accuracy', 'Precision', 'Recall', 'F-score', 'Kappa', 'Eperiment')
        f.write(line + '\n')
        print(line)

        for clf in res.keys():
            curr = res[clf]
            line = '%s,%f,%f,%f,%f,%f,%s' % (clf, curr['acc'], curr['prec'], curr['rec'], curr['fsco'], curr['kappa'], addon)
            reader.write(line + '\n')
            line = '{:30s} {:10f} {:10f} {:10f} {:10f} {:10f}   {:10s}'.format(clf, curr['acc'], curr['prec'], curr['rec'], curr['fsco'], curr['kappa'], addon)
            f.write(line + '\n')
            print(line)

    # TO REMOVE
    if kfold:
        return kfold_models_and_testing_data
    else:
        return output_classifiers

def train_and_test_multiclass(X, y, result_filename, num_splits=5, input_shape=(384, 1), num_klasses=6, addon=''):

    y = np.array([sample for sample in y])

    classifiers = get_classifiers(multi_class=True)[1:]
    res = {}  # Dictionary for results
    cms = {}  # Dictionary for CM

    # SCORINGS
    scoring = { 'accuracy'  : make_scorer(accuracy_score), 
                'precision' : make_scorer(precision_score, average='weighted'),
                'recall'    : make_scorer(recall_score, average='weighted'), 
                'f1_score'  : make_scorer(f1_score, average='weighted'),
                'kappa'     : make_scorer(cohen_kappa_score) }

    if num_splits > 1:
        kfold = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=23)
    else:
        kfold = None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=23)

    for i, clf in enumerate(classifiers):

        clf_name = clf[1]  # Classifier name to identify which classifier is used
        res[clf_name] = {}
        cms[clf_name] = ''

        if kfold:
            results = cross_validate(
                estimator           = clf[0],
                X                   = X,
                y                   = y,
                cv                  = kfold,
                scoring             = scoring,
                return_train_score  = True)

            # GET THE MEAN OF ALL SCORINGS
            res[clf_name]['acc'] = np.mean(results['test_accuracy'])
            res[clf_name]['prec'] = np.mean(results['test_precision'])
            res[clf_name]['rec'] = np.mean(results['test_recall'])
            res[clf_name]['fsco'] = np.mean(results['test_f1_score'])
            res[clf_name]['kappa'] = np.mean(results['test_kappa'])
        else:
            clf[0].fit(X_train, y_train)
            y_pred = clf[0].predict(X_test)
            accuracy, f1_score_m, precision, recall = accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='weighted'), precision_score(y_test, y_pred, average='weighted'), recall_score(y_test, y_pred, average='weighted')
            kappa = cohen_kappa_score(y_test, y_pred)
            cms[clf_name] = confusion_matrix(y_test, y_pred)
                
            # GET THE MEAN OF ALL SCORINGS
            res[clf_name]['acc'] = accuracy
            res[clf_name]['prec'] = precision
            res[clf_name]['rec'] = recall
            res[clf_name]['fsco'] = f1_score_m
            res[clf_name]['kappa'] = kappa
    
    with open(result_filename, 'w') as f:
        line = '\n{:32s} {:10s}{:10s}{:10s}    {:10s}{:10s}{:10s}'.format('', 'Accuracy', 'Precision', 'Recall', 'F-score', 'Kappa', 'Eperiment')
        f.write(line + '\n')
        print(line)

        for clf in res.keys():
            curr = res[clf]
            line = '%s,%f,%f,%f,%f,%f,%s' % (clf, curr['acc'], curr['prec'], curr['rec'], curr['fsco'], curr['kappa'], addon)
            reader.write(line + '\n')
            line = '{:30s} {:10f} {:10f} {:10f} {:10f} {:10f}   {:10s}'.format(clf, curr['acc'], curr['prec'], curr['rec'], curr['fsco'], curr['kappa'], addon)
            f.write(line + '\n')
            print(line)

def exp1_one_class_sink(data, label_mapping, doc_features, code_features, vector, doc_first=False):
    result_filename = 'results__exp1_one_class_sink'
    positive = label_mapping['SINK']
    dataset = data[data['klass'] == positive].reset_index()
    
    X = []
    y = []

    if isinstance(vector, str):
        features = doc_features if vector == 'doc_vector' else code_features
        result_filename = result_filename + '__%s' % vector
        addon = 'ONE CLASS SINK. With %s' % (vector)
    else:
        features = []
        result_filename = result_filename + '__concat'
        addon = 'ONE CLASS SINK. ALL Vectors'
        if doc_first:
            features.extend(doc_features)
            features.extend(code_features)
            result_filename = result_filename + '__doc_first'
            addon = addon + ' - DOC FIRST'
        else:
            features.extend(code_features)
            features.extend(doc_features)

    X = dataset
    y = dataset['klass'].apply(lambda k: int(k == positive))

    input_shape, num_klasses = (len(features), 1), len(np.unique(y)) - 1
    return train_and_test_one_class(X, features, y, result_filename + '.txt', num_splits=NUM_SPLIT, input_shape=input_shape, num_klasses=num_klasses, addon=addon)

def exp1_one_class_source(data, label_mapping, doc_features, code_features, vector, doc_first=False):
    result_filename = 'results__exp1_one_class_sourcek'
    positive = label_mapping['SOURCE']
    dataset = data[data['klass'] == positive].reset_index()
    
    X = []
    y = []

    if isinstance(vector, str):
        features = doc_features if vector == 'doc_vector' else code_features
        result_filename = result_filename + '__%s' % vector
        addon = 'ONE CLASS SOURCE. With %s' % (vector)
    else:
        features = []
        result_filename = result_filename + '__concat'
        addon = 'ONE CLASS SOURCE. ALL Vectors'
        if doc_first:
            features.extend(doc_features)
            features.extend(code_features)
            result_filename = result_filename + '__doc_first'
            addon = addon + ' - DOC FIRST'
        else:
            features.extend(code_features)
            features.extend(doc_features)

    X = dataset
    y = dataset['klass'].apply(lambda k: int(k == positive))

    input_shape, num_klasses = (len(features), 1), len(np.unique(y)) - 1
    return train_and_test_one_class(X, features, y, result_filename + '.txt', num_splits=NUM_SPLIT, input_shape=input_shape, num_klasses=num_klasses, addon=addon)

def exp1_bin_sink_and_not_sink(data, label_mapping, doc_features, code_features, vector, doc_first=False):
    result_filename = 'results__exp1_bin_sink_and_not_sink'
    positive = label_mapping['SINK']
    try:
        dataset = data[data['klass'] != label_mapping['UNKNOWN']].reset_index()
    except:
        dataset = data[data['klass'] != 0].reset_index()
    
    X = []
    y = []

    if isinstance(vector, str):
        features = doc_features if vector == 'doc_vector' else code_features
        result_filename = result_filename + '__%s' % vector
        addon = 'BINARY (SINK AND NON SINK). With %s' % (vector)
    else:
        features = []
        result_filename = result_filename + '__concat'
        addon = 'BINARY (SINK AND NON SINK). ALL Vectors'
        if doc_first:
            features.extend(doc_features)
            features.extend(code_features)
            result_filename = result_filename + '__doc_first'
            addon = addon + ' - DOC FIRST'
        else:
            features.extend(code_features)
            features.extend(doc_features)

    X = dataset
    y = dataset['klass'].apply(lambda k: int(k == positive))

    input_shape, num_klasses = (len(features), 1), len(np.unique(y)) - 1
    return train_and_test_binary(X, features, y, result_filename + '.txt', num_splits=NUM_SPLIT, input_shape=input_shape, num_klasses=num_klasses, addon=addon)

def exp1_bin_source_and_not_source(data, label_mapping, doc_features, code_features, vector, doc_first=False):
    result_filename = 'results__exp1_bin_source_and_not_source'
    positive = label_mapping['SOURCE']
    try:
        dataset = data[data['klass'] != label_mapping['UNKNOWN']].reset_index()
    except:
        dataset = data[data['klass'] != 0].reset_index()
    
    X = []
    y = []

    if isinstance(vector, str):
        features = doc_features if vector == 'doc_vector' else code_features
        result_filename = result_filename + '__%s' % vector
        addon = 'BINARY (SOURCE AND NON SOURCE). With %s' % (vector)
    else:
        features = []
        result_filename = result_filename + '__concat'
        addon = 'BINARY (SOURCE AND NON SOURCE). ALL Vectors'
        if doc_first:
            features.extend(doc_features)
            features.extend(code_features)
            result_filename = result_filename + '__doc_first'
            addon = addon + ' - DOC FIRST'
        else:
            features.extend(code_features)
            features.extend(doc_features)

    X = dataset
    y = dataset['klass'].apply(lambda k: int(k == positive))

    input_shape, num_klasses = (len(features), 1), len(np.unique(y)) - 1
    return train_and_test_binary(X, features, y, result_filename + '.txt', num_splits=NUM_SPLIT, input_shape=input_shape, num_klasses=num_klasses, addon=addon)

def exp1_multiclass(data, label_mapping, doc_features, code_features, vector, doc_first=False):
    result_filename = 'results__exp1_multiclass'
    try:
        dataset = data[data['klass'] != label_mapping['UNKNOWN']].reset_index()
        label_mapping.pop('UNKNOWN')
    except:
        dataset = data[data['klass'] != 0].reset_index()
    
    X = []
    y = []

    if isinstance(vector, str):
        features = doc_features if vector == 'doc_vector' else code_features
        result_filename = result_filename + '__%s' % vector
        addon = 'MULTICLASS. With %s' % (vector)
    else:
        features = []
        result_filename = result_filename + '__concat'
        addon = 'MULTICLASS. ALL Vectors'
        if doc_first:
            features.extend(doc_features)
            features.extend(code_features)
            result_filename = result_filename + '__doc_first'
            addon = addon + ' - DOC FIRST'
        else:
            features.extend(code_features)
            features.extend(doc_features)

    X = dataset[features]
    y = dataset['klass'].apply(lambda k: int(k) - 1)
    
    input_shape, num_klasses = (len(features), 1), len(np.unique(y))
    train_and_test_multiclass(X, y, result_filename + '.txt', num_splits=NUM_SPLIT, input_shape=input_shape, num_klasses=num_klasses, addon=addon)

def get_data_4_venn_diagram(models_and_data, my_type, features, label_mapping):
    vector = 'code' if 'c' in features[0] else 'doc'
    for clf_name, mds in models_and_data.items():
        for i, md in enumerate(mds):
            clf = md[0]
            test_dataset = md[1]
            methods_name = test_dataset['method'].tolist()
            X_test = test_dataset[features]
            y_true = md[2]
            y_pred = clf.predict(X_test)
            for j, y in enumerate(y_pred):
                current_y = y_true[j]
                if current_y == 1: # INTEREST
                    if y == current_y: # CORRECT PREDICT
                        model_prediction = 'CORRECT'
                    else: # WRONG PREDICT
                        model_prediction = 'WRONG'
                    line = '"%s",%d,%s,%s,%s,%s\n' % (methods_name[j], i, my_type, clf_name, vector, model_prediction)
                else:
                    if y == current_y: # CORRECT PREDICT
                        model_prediction = 'CORRECT'
                    else: # WRONG PREDICT
                        model_prediction = 'WRONG'
                    line = '"%s",%d,%s,%s,%s,%s\n' % (methods_name[j], i, 'NON_%s' % my_type, clf_name, vector, model_prediction)
                v_reader.write(line)

if __name__ == '__main__':
    input_filename = '../../data/matrix_with_new_dataset_confirmed.lst'
    output_filename = '../../data/labeled_vectors.csv'
    data_bundle_filename = '../../data/data_bundle.pkl'
    vector = ['code_vector', 'doc_vector']

    if not os.path.exists(data_bundle_filename):
        data, categories_mapping, doc_features, code_features = load_data(input_filename)
        pickle.dump([data, categories_mapping, doc_features, code_features], open(data_bundle_filename, 'wb'))
    else:
        data, categories_mapping, doc_features, code_features = pickle.load(open(data_bundle_filename, 'rb'))

    # ONE-CLASS CLASSIFICATION
    exp1_one_class_sink(data, categories_mapping, doc_features, code_features, vector)
    exp1_one_class_source(data, categories_mapping, doc_features, code_features, vector)
    exp1_one_class_sink(data, categories_mapping, doc_features, code_features, vector, True)
    exp1_one_class_source(data, categories_mapping, doc_features, code_features, vector, True)

    # BINARY CLASSIFICATION
    exp1_bin_source_and_not_source(data, categories_mapping, doc_features, code_features, vector)
    exp1_bin_sink_and_not_sink(data, categories_mapping, doc_features, code_features, vector)
    exp1_bin_source_and_not_source(data, categories_mapping, doc_features, code_features, vector, True)
    exp1_bin_sink_and_not_sink(data, categories_mapping, doc_features, code_features, vector, True)

    # MULTICLASS CLASSIFICATION
    exp1_multiclass(data, categories_mapping, doc_features, code_features, vector)
    exp1_multiclass(data, categories_mapping, doc_features, code_features, vector, True)

    # CLASSIFICATIONS WITH SINGLE VECTOR
    for v in vector:
        exp1_one_class_sink(data, categories_mapping, doc_features, code_features, v)
        exp1_one_class_source(data, categories_mapping, doc_features, code_features, v)
        
        features = doc_features if v == 'doc_vector' else code_features

        source_bin_models_and_testing_data = exp1_bin_source_and_not_source(data, categories_mapping, doc_features, code_features, v)
        get_data_4_venn_diagram(source_bin_models_and_testing_data, "SOURCE", features, categories_mapping)
        
        sink_bin_models_and_testing_data = exp1_bin_sink_and_not_sink(data, categories_mapping, doc_features, code_features, v)
        get_data_4_venn_diagram(sink_bin_models_and_testing_data, "SINK", features, categories_mapping)
        
        exp1_multiclass(data, categories_mapping, doc_features, code_features, v)

    v_reader.close()
    reader.close()
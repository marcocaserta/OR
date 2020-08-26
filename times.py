import sys
import numpy as np
import pandas as pd
import logging

import argparse

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV

import time

def setup_custom_logger(name):
    handler = logging.FileHandler('log/times.log')
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.addHandler(screen_handler)
    return logger

logger = setup_custom_logger('Regression Times OR')
#  logger.disabled = True

def parse_command_line():
    parser = argparse.ArgumentParser(description='Define Regressor:: ')
    parser.add_argument('-r', '--regressor', default='rf', type=str,
    help="Choose a valid regressor type.", choices=['rf', 'svr', 'gbr', 'br', 'hgbr', 'nn'],
    required=True)

    args = parser.parse_args()
    return args

def create_df():
    df = pd.read_excel('data/data_2018_scores.xlsx')
    df['shift'] = np.where(df['Hora planificación'].dt.hour< 12, "M", np.where(df['Hora planificación'].dt.hour  <= 18, "A", "E"))
    
    features = ['Especialidad', 'Intervención (Cod. OMC)','Grupo OMC', 'Quirófano','Tipo', 'Urgente', 'AP', 'Cirujano', 'Anestesista', 'Ayudante', 'Instrumentista', 'Circulante', 'Auxiliar', 'or_time', 'leader', 'team', 'team_size', 'shift']
#features = ['Especialidad', 'Intervención (Cod. OMC)','Grupo OMC', 'Tipo', 'Urgente', 'AP', 'Cirujano', 'Anestesista', 'or_time', 'leader', 'team']

# extract featured columns only
    df_tf = df[features]
    logger.info("NaN in columns :: \n{}".format(df_tf.isna().sum()))

# transform categorical columns in dummy vars
    df_tf = pd.get_dummies(df_tf, prefix='', prefix_sep='')
    logger.info("Df dummy variables created. Df size = {} x {}.".format(df_tf.shape[0], df_tf.shape[1]))

    return df_tf

def get_training_testing(df_tf):

    # separate Training and Testing
    train_dataset = df_tf.sample(frac=0.8,random_state=0)
    test_dataset = df_tf.drop(train_dataset.index)
    logger.info("Training dataset size \t= {:5d} x {}.".format(train_dataset.shape[0], train_dataset.shape[1]))
    logger.info("Testing dataset size \t= {:5d} x {}.".format(test_dataset.shape[0], test_dataset.shape[1]))

    # normalized columns (this is beneficial for some of the methods)
    normed_train_data = train_dataset
    normed_test_data = test_dataset
    cols = ['leader', 'team', 'team_size'] 
    for col in cols:
      normed_train_data[col] = (train_dataset[col] - train_dataset[col].mean())/train_dataset[col].std()
      normed_test_data[col] = (test_dataset[col] - train_dataset[col].mean())/train_dataset[col].std()

    return train_dataset, test_dataset, normed_train_data, normed_test_data

def get_parameters_svr():
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                         'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
                        {'kernel': ['sigmoid'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                         'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
                        {'kernel': ['linear'], 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]}
                       ]

    return tuned_parameters

def get_parameters_rf():
    n_estimators = np.arange(150, 350, 50)
    n_estimators = [10]
    max_features = ['auto', 'sqrt']
    max_depth = np.arange(80, 210, 10)
    max_depth = [90]
    min_samples_split = [2, 5]
    min_samples_leaf = [4, 8, 12]
    min_sample_leaf=[4]
    random_grid = {'n_estimators' : n_estimators,
                   'max_depth' : max_depth,
                   'min_samples_split' : min_samples_split,
                   'min_samples_leaf' : min_samples_leaf}

    return random_grid

def get_parameters_gbr():
    n_estimators = np.arange(150, 350, 50)
    loss = ['ls', 'lad', 'huber']
    criterion = ['mse', 'friedman_mse', 'mae']
    min_samples_split = [2, 5]
    min_samples_leaf = [4, 8, 12]
    max_depth = np.arange(80, 210, 10)

    tuned_parameters = {'n_estimators' : n_estimators,
                        'loss' : loss,
                        'criterion': criterion,
                        'max_depth' : max_depth,
                        'min_samples_split' : min_samples_split,
                        'min_samples_leaf' : min_samples_leaf}

    return tuned_parameters

def get_parameters_br():
    
    n_estimators = np.arange(150, 350, 50)
    bootstrap_features = ['False', 'True']

    tuned_parameters = {'n_estimators' : n_estimators,
                        'bootstrap_features' : bootstrap_features}

    return tuned_parameters

def get_parameters_hgbr():
    loss = ['least_squares', 'least_absolute_deviation', 'poisson']
    learning_rate = [0.1, 0.25, 1.0]
    max_iter = np.arange(150, 350, 50)
    max_depth = np.arange(80, 210, 10)
    l2_regularization = [0, 1]

    tuned_parameters = {'loss' : loss,
                        'learning_rate' : learning_rate,
                        'max_iter' : max_iter,
                        'max_depth' : max_depth,
                        'l2_regularization' : l2_regularization
                        }

    return tuned_parameters

def get_parameters_nn():
    hidden_layer_sizes = [(50, ), (100, ), (200, ), (500, )]
    activation = ['relu', 'logistic']
    alpha = [0.01, 0.001, 0.0001, 0.00001]
    solver = ['adam', 'sgd', 'lbfgs']
    learning_rate = ['constant', 'adaptive', 'invscaling']
    max_iter = np.arange(150, 350, 50)

    tuned_parameters = {'hidden_layer_sizes' : hidden_layer_sizes,
                        'activation' : activation,
                        'alpha' : alpha,
                        'solver' : solver,
                        'learning_rate' : learning_rate,
                        'max_iter' : max_iter}

    return tuned_parameters

                   
def select_model(args):
    if args.regressor == 'rf':
        model = RandomForestRegressor(n_jobs=-1, criterion='mse')
        tuned_parameters = get_parameters_rf()
    elif args.regressor == 'svr':
        model = SVR()
        tuned_parameters = get_parameters_svr()
    elif args.regressor == 'gbr':
        model = GradientBoostingRegressor(criterion='friedman_mse')
        tuned_parameters = get_parameters_gbr()
    elif args.regressor == 'br':
        model = BaggingRegressor()
        tuned_parameters = get_parameters_br()
    elif args.regressor == 'hgbr':
        model = HistGradientBoostingRegressor()
        tuned_parameters = get_parameters_hgbr()
    elif args.regressor == 'nn':
        model = MLPRegressor()
        tuned_parameters = get_parameters_nn()
    else:
        logger.error('Regressor %s not defined.', args.regressor)

    return model, tuned_parameters


def main():
    logger.info('')
    logger.info('*'*80)
    args = parse_command_line()

    df = create_df()
    train_dataset, test_dataset, normed_train_data, normed_test_data = get_training_testing(df)

    # extract labels (DV)
    train_labels = train_dataset.pop('or_time')
    test_labels = test_dataset.pop("or_time")

    X = normed_train_data
    y = train_labels


    logger.info("Tuning Parameters for ** '%s' ** via Grid Search", args.regressor)

    model, tuned_parameters = select_model(args)

    #grid = GridSearchCV(model, tuned_parameters,cv=5, scoring = 'neg_mean_absolute_error', verbose=10, n_jobs=-1)
    grid = RandomizedSearchCV(model, tuned_parameters, cv=5, scoring = 'neg_mean_absolute_error', verbose=10)
    start = time.time()
    grid.fit(X, y)

    logger.info("Best score obtained in the search = {}".format(grid.best_score_))
    logger.info("Best parameters = {}".format(grid.best_params_))
    logger.info("Best estimator = \n{}".format(grid.best_estimator_))

# evaluate the best grid searched model on the testing data
    logger.info("Grid search took {:.2f} seconds".format( time.time() - start))

    yhat = grid.predict(normed_test_data)
    mae_mod = sum(abs(test_labels - yhat))/len(test_labels)
    logger.info("MAE {} \t = {}".format(args.regressor, mae_mod))

if __name__ == '__main__':
    main()


import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
import csv
from datetime import datetime


def lgb_train(X_train, X_test, Y_train, lgb_params_, log_filepath, test_prediction=False, num_folds=5):
    NUM_FOLDS = num_folds
    RANDOM_SEED = 2017
    np.random.seed(RANDOM_SEED)
    kfold = StratifiedKFold(
        n_splits=NUM_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED
    )

    y_test_pred = np.zeros((len(X_test), NUM_FOLDS))
    cv_val_scores = []
    cv_train_scores = []

    X_train_feat = X_train
    X_train_values=X_train.values
    X_test_values=X_test.values
    for fold_num, (ix_train, ix_val) in enumerate(kfold.split(X_train_values, Y_train)):
        print('Fitting fold {fold_num + 1} of {kfold.n_splits}')
        X_fold_train = X_train_values[ix_train,:]
        X_fold_val = X_train_values[ix_val,:]

        y_fold_train = Y_train[ix_train]
        y_fold_val = Y_train[ix_val]
        
        lgb_params = lgb_params_.copy()

        lgb_data_train = lgb.Dataset(X_fold_train, y_fold_train)
        lgb_data_val = lgb.Dataset(X_fold_val, y_fold_val)    
        evals_result = {}
        
        model = lgb.train(
            lgb_params,
            lgb_data_train,
            valid_sets=[lgb_data_train, lgb_data_val],
            evals_result=evals_result,
            num_boost_round=lgb_params['num_boost_round'],
            early_stopping_rounds=lgb_params['early_stopping_rounds'],
            verbose_eval=False,
        )
        
        fold_train_scores = evals_result['training'][lgb_params['metric']]
        fold_val_scores = evals_result['valid_1'][lgb_params['metric']]
        
        print('Fold {}: {} rounds, training loss {:.6f}, validation loss {:.6f}'.format(
            fold_num + 1,
            len(fold_train_scores),
            fold_train_scores[-1],
            fold_val_scores[-1],
        ))
        cv_train_scores.append(fold_train_scores[-1])
        cv_val_scores.append(fold_val_scores[-1])
        y_test_pred[:, fold_num] = model.predict(X_test_values).reshape(-1)


    feat_imp = pd.DataFrame({
    'column': list(X_train.columns),
    'importance': model.feature_importance()}).sort_values(by='importance')

    print('Final CV val score:', cv_val_scores)
    print('Final mean CV val score:', np.mean(cv_val_scores))
    with open(log_filepath, 'a') as fp:
        a = csv.writer(fp, delimiter=',')
        data = [[datetime.now().strftime('%d-%m-%Y_%H-%M-%S'), 'lightgb', 
                 cv_val_scores, np.mean(cv_val_scores), 
                 cv_train_scores, np.mean(cv_train_scores),
                 lgb_params_, feat_imp.to_dict()]]
        a.writerows(data)
    print()

    if test_prediction:
        print('Make submission file...')
        y_test = np.mean(y_test_pred, axis=1)
        with open("submission_file.csv", 'w') as f:
            f.write("Id,Score\n")
            for i in range(y_test.shape[0]):
                f.write(str(i)+','+str(y_test[i])+'\n')
        print('Submission file written !')

    return feat_imp


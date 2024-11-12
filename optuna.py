def lgbm_objective(trial):
    
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1200),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 20, 256),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 200),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'bagging_freq' : trial.suggest_int('bagging_freq', 1, 3),
        'is_unbalance' : trial.suggest_categorical('is_unbalance', ["+", "-"]),
    }

    # creating LightGBM classifier with suggested hyperparameters
    lgbm_model = LGBMClassifier(**params, 
                                objective = 'binary',
                                random_state=42, 
                                n_jobs=-1, 
                                verbosity = -1, 
                                metric = 'binary_logloss', 
                                early_stopping_round=50)

    # stratified k-fold cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []

    for train_index, test_index in skf.split(X_train, y_train):
        X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

        lgbm_model.fit(X_train_fold, y_train_fold, eval_set = [(X_test_fold, y_test_fold)], eval_metric = 'binary_logloss')
        y_pred_proba = lgbm_model.predict_proba(X_test_fold)[:, 1]
        auc_scores.append(roc_auc_score(y_test_fold, y_pred_proba))

    print(f'Trial {trial.number}: {auc_scores} , mean : {np.mean(auc_scores)}')
    return np.mean(auc_scores)

def catboost_objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 500, 5000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'depth': trial.suggest_int('depth', 3, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0, 5),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'random_strength': trial.suggest_float('random_strength', 0, 2),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 5),
        'auto_class_weights' : trial.suggest_categorical('auto_class_weights', ['Balanced', None]),
        
    }


    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # creating LightGBM classifier with suggested hyperparameters
        model = CatBoostClassifier(**params,
                                   random_seed=42,
                                   loss_function='Logloss',
                                   cat_features=cat_cols,
                                   thread_count=-1,
                                   eval_metric = 'AUC'
                                  )
        model.fit(X_train_fold, y_train_fold, eval_set=(X_val_fold, y_val_fold), early_stopping_rounds=150, verbose=False)
        y_pred_proba = model.predict_proba(X_val_fold)[:, 1]
        auc = roc_auc_score(y_val_fold, y_pred_proba)
        auc_scores.append(auc)
    
    print(f'Trial {trial.number}: {auc_scores} , mean : {np.mean(auc_scores)}')
    return np.mean(auc_scores)

def xgb_objective(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1200),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_samples', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # creating LightGBM classifier with suggested hyperparameters
        model = XGBClassifier(**params,
                                   random_seed=42,
                                   eval_metric = 'AUC'
                                  )
        model.fit(X_train_fold, y_train_fold, eval_set=(X_val_fold, y_val_fold), early_stopping_rounds=150, verbose=False)
        y_pred_proba = model.predict_proba(X_val_fold)[:, 1]
        auc = roc_auc_score(y_val_fold, y_pred_proba)
        auc_scores.append(auc)
    
    print(f'Trial {trial.number}: {auc_scores} , mean : {np.mean(auc_scores)}')
    return np.mean(auc_scores)

lgbm_study = optuna.create_study(direction='maximize', sampler=TPESampler(n_startup_trials=30, seed=42, multivariate=True))
lgbm_study.optimize(lgbm_objective, n_trials=100)

print('Best Hyperparameters:', lgbm_study.best_params)
print('Best AUC:', lgbm_study.best_value)
lgbm_combined_data_best_params = lgbm_study.best_params

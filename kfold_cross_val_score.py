def kfold_cross_val_score(X, y, use_model, scale = True):
    
    model = use_model
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_auc = []
    
    for fold, (train_index, test_index) in enumerate(skf.split(X, y), 1):

        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        if scale:
            sc = StandardScaler()
            sc.fit_transform(X_train)
            sc.transform(X_test)
            
        model.fit(X_train, y_train)

        y_pred = model.predict_proba(X_test)[:, -1]

        auc = roc_auc_score(y_test, y_pred)

        fold_auc.append(auc)
    
    return np.mean(fold_auc)

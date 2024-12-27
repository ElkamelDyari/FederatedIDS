from imblearn.over_sampling import SMOTE

def over_sample(X_train, y_train):

    smote=SMOTE(n_jobs=-1,sampling_strategy={2:1000, 4:1000, 5:1000})
    return smote.fit_resample(X_train, y_train)

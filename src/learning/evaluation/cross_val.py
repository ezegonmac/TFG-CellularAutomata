from sklearn.model_selection import cross_val_score

# Cross val scores
# (whole dataset, not just test data)

def get_cross_val_MSE(X, y, model):
    scores = cross_val_score(model, X, y, cv=10, scoring='neg_mean_squared_error')
    return scores.mean()


def get_cross_val_R2(X, y, model):
    scores = cross_val_score(model, X, y, cv=10, scoring='r2')
    return scores.mean()

from constants import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from learning.scoring import *


def main():
    df = load_dataset_density(DATASET8)
    
    iterations = [str(i) for i in range(1, 10)]
    X = df[['B', 'S', '0']]
    y = df[iterations]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SKLEARN_RANDOM_SEED)
    
    knn_model = KNeighborsRegressor(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    
    y_pred = knn_model.predict(X_test)

    print_random_checks(X_test, y_test, knn_model)
    print_overall_score(X_test, y_test, knn_model)
    print_cross_val_score(X, y, knn_model)
    mse_by_iteration = get_mse_by_iteration(y_test, y_pred)
    score_by_iteration = get_r2_by_iteration(y_test, y_pred)
    
    print('MSE by iteration:' + str(mse_by_iteration))
    print('R2 by iteration:' + str(score_by_iteration))


def print_cross_val_score(X, y, model):
    scores = cross_val_score(model, X, y, cv=10)
    
    print(f'Cross validation scores: {scores}')
    print(f'Cross validation mean score: {scores.mean()}')


def print_overall_score(X_test, y_test, model):
    score = model.score(X_test, y_test)
    
    print(f'Score: {score}')


def print_random_checks(X_test, y_test, model):
    random_idx = np.random.choice(len(X_test), 20)
    predicts = model.predict(X_test)
    predict_df = pd.DataFrame()
    predict_df['Real'] = y_test.iloc[random_idx]['1']
    predict_df['Predict'] = predicts[random_idx, 0]
    predict_df['Error'] = predict_df['Real'] - predict_df['Predict']
    
    print('Some random checks:')
    print(predict_df)


def generate_BS_scatter_plot(df):
    iteration = '0'
    df.rename(columns={iteration: 'Density'}, inplace=True)
    df.plot.scatter(x='B', y='S', c='Density', colormap='jet', alpha=0.5)
    plt.show()


def load_dataset_density(dataset):
    dataset_name = dataset + '_density'
    dataset_folder = f'{DATA_FOLDER}/{dataset_name}'
    file = f'{dataset_folder}/density_dataset.csv'
    
    df = pd.read_csv(file)
    
    return df

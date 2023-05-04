
# Print all scores

from learning.test.files_scores import get_scores_by_dataset_and_model


def print_scores(dataset, model_name):
    print('--------------------------------------')
    print(f'Statistics: {model_name} - {dataset}')
    print('--------------------------------------')
    
    row = get_scores_by_dataset_and_model(dataset, model_name)
    
    print("Individuals: " + str(row['Number of individuals'].values[0]))
    
    print("RMSE: " + str(row['RMSE mean'].values[0]))
    print("R2: " + str(row['R2 mean'].values[0]))
    print("RMSE std: " + str(row['RMSE std'].values[0]))
    print("R2 std: " + str(row['R2 std'].values[0]))
    print("RMSE by iteration: " + str(row['RMSE mean by iteration'].values[0]))
    print("R2 by iteration: " + str(row['R2 mean by iteration'].values[0]))
    print("RMSE std by iteration: " + str(row['RMSE std by iteration'].values[0]))
    print("R2 std by iteration: " + str(row['R2 std by iteration'].values[0]))
    print("\n")

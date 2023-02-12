from learning.test.files import get_scores_by_dataset_and_model

# Print all scores

def print_scores(dataset, model_name):
    print('--------------------------------------')
    print(f'{model_name} - {dataset}')
    print('--------------------------------------')
    
    row = get_scores_by_dataset_and_model(dataset, model_name)
    
    print("Individuals: " + str(row['Number of individuals'].values[0]))
    
    print("RMSE: " + str(row['RMSE'].values[0]))
    print("R2: " + str(row['R2'].values[0]))
    print("RMSE std: " + str(row['RMSE std'].values[0]))
    print("R2 std: " + str(row['R2 std'].values[0]))
    print("RMSE by iteration: " + str(row['RMSE by iteration'].values[0]))
    print("R2 by iteration: " + str(row['R2 by iteration'].values[0]))
    print("RMSE std by iteration: " + str(row['RMSE std by iteration'].values[0]))
    print("R2 std by iteration: " + str(row['R2 std by iteration'].values[0]))
    print("\n")

from learning.test.files_scores import generate_scores_file_from_executions
from learning.test.files_executions import generate_execution_scores_file

from learning.test.scores_print import print_scores


def generate_model_and_scores_files(model, dataset, model_name, model_variation, save_model=True, num_individuals=None, num_executions=10):
    
    print('# Generating execution files')
    
    generate_execution_scores_file(
        model=model,
        dataset=dataset,
        model_name=model_name,
        model_variation=model_variation,
        num_individuals=num_individuals,
        num_executions=num_executions,
        save_model=save_model,
    )
    
    print('# Generating scores file')
    
    generate_scores_file_from_executions(
        dataset=dataset,
        model_name=model_name,
        model_variation=model_variation,
        num_individuals=num_individuals,
        )
    
    print_scores(dataset, model_name)

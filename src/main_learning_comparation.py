from constants import *
from learning.misc.comparation import *

if __name__ == '__main__':
    
    generate_rule_type_comparation_plot(rule_type='BTST')
    generate_rule_type_comparation_plot(rule_type='BISI')
    generate_rule_type_comparation_plot(rule_type='BS')
    
    # train_and_test_models_ds8(num_executions=10, num_individuals=1000, save_models=False)
    # generate_models_score_plots_ds8(num_individuals=1000)
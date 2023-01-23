from constants import *
from learning.scoring import get_scores_by_dataset


def generate_score_by_iteration_table(dataset) -> None:
    """
    Generate a table of scores by iteration for each model and prints it in console.
    """
    
    # Write #L and #T as { and } to avoid conflicts with latex
    # Write \ as \\ to avoid conflicts with latex
    base = '''
\\begin{table}[!htp]
    \\centering
    {\\footnotesize
    \\begin{tabular}{|c|#Lcolumns#T|}
        \\hline
        \\textbf{Modelo} #Llabels#T\\\\[1.1ex]
        \\hline
        #Ldata#T
    \\end{tabular}
    }
    \\caption{Métricas de predicción por iteración para el #Ldataset#T}
    \\label{tab:metrics_#Ldataset#T}
\\end{table}
    '''
    # scape characters
    base = base.replace('{', '{{').replace('}', '}}')
    # base = base.replace('\\', '\\\\')
    
    base = base.replace('#L', '{').replace('#T', '}')
    
    scores_df = get_scores_by_dataset(dataset)
    
    models = scores_df['Model'].unique()
    
    mse_data = ''
    r2_data = ''
    mses = []
    r2s = []
    for i in range(len(models)):
        mses = []
        r2s = []
        model = models[i]
        mse = scores_df[scores_df['Model'] == model]['MSE by iteration'].values[0]
        r2 = scores_df[scores_df['Model'] == model]['R2 by iteration'].values[0]
        
        for j in range(1, len(mse)+1):
            j = str(j)
            mses.append('%.2f' % mse[j])
            r2s.append('%.2f' % r2[j])
        
        mse = ' & '.join(mses)
        r2 = ' & '.join(r2s)
        
        mse_data += '''{model} & {mse} \\\\
        \hline
        '''.format(model=models[i], mse=mse)
        
        r2_data += '''{model} & {r2} \\\\
        \hline
        '''.format(model=models[i], r2=r2)

    iterations = len(mses)
    columns = '*{' + str(iterations) + '}{p{0.08cm}}'
    labels = ''
    for i in range(1, iterations+1):
        labels += '& \\textbf{{{i}}}'.format(i=i)

    mse_table = base.format(data=mse_data, dataset=dataset, columns=columns, labels=labels)
    r2_table = base.format(data=r2_data, dataset=dataset, columns=columns, labels=labels)
    
    print('MSE TABLE')
    print('\n\n')
    
    print(mse_table)
    
    print('R2 TABLE')
    print('\n\n')
    
    print(r2_table)

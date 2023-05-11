from learning.test.files_scores import get_scores_by_dataset


def generate_scores_table(dataset) -> None:
    """
    Generate a table of scores for each model and prints it in console.
    """
    
    # Write #L and #T as { and } to avoid conflicts with latex
    # Write \ as \\ to avoid conflicts with latex
    base = '''
\\begin{table}[!htp]
    \\renewcommand{\\arraystretch}{2}
    \\centering
    \\begin{tabular}{|c|cc|}
        \\hline
        \\textbf{Modelo} & \\textbf{RMSE} & \\textbf{R2} \\\\[1.1ex]
        \\hline
        #Ldata#T
    \\end{tabular}
    \\caption{Métricas para el #Ldataset#T}
    \\label{tab:metrics_#Ldataset#T}
\\end{table}
    '''
    # scape characters
    base = base.replace('{', '{{').replace('}', '}}')
    # base = base.replace('\\', '\\\\')
    
    base = base.replace('#L', '{').replace('#T', '}')
    
    scores_df = get_scores_by_dataset(dataset)
    # filter by Number of individuals
    scores_df = scores_df[scores_df['Number of individuals'] == 1000]
    
    models = scores_df['Model'].unique()
    
    data = ''
    rmses = []
    r2s = []
    for i in range(len(models)):
        rmses = []
        r2s = []
        model = models[i]
        rmse = scores_df[scores_df['Model'] == model]['RMSE mean'].values[0]
        r2 = scores_df[scores_df['Model'] == model]['R2 mean'].values[0]
        rmse_std = scores_df[scores_df['Model'] == model]['RMSE std'].values[0]
        r2_std = scores_df[scores_df['Model'] == model]['R2 std'].values[0]

        # mean
        rmse_val = '{:.3e}'.format(rmse)
        r2_val = '{:.3e}'.format(r2)
        rmse_val = rmse_val.replace('e-0', 'e-').replace('e+0', 'e+')
        r2_val = r2_val.replace('e-0', 'e-').replace('e+0', 'e+')
        # std
        rmse_std = '{:.2e}'.format(rmse_std)
        r2_std = '{:.2e}'.format(r2_std)
        rmse_std = rmse_std.replace('e-0', 'e-').replace('e+0', 'e+')
        r2_std = r2_std.replace('e-0', 'e-').replace('e+0', 'e+')
        # sum
        rmse_val = rmse_val + '$\\pm$' + rmse_std
        r2_val = r2_val + '$\\pm$' + r2_std
        rmses.append(rmse_val)
        r2s.append(r2_val)
        
        data += '''{model} & {rmse} & {r2} \\\\
        \\hline
        '''.format(model=models[i], rmse=rmses[-1], r2=r2s[-1])

    iterations = len(rmses)
    labels = ''
    for i in range(1, iterations+1):
        labels += '& \\textbf{{{i}}}'.format(i=i)

    table = base.format(data=data, dataset=dataset, labels=labels, metric='RMSE')

    print('TABLE')
    print('\n\n')
    
    print(table)


def generate_score_by_iteration_table(dataset) -> None:
    """
    Generate a table of RMSE and R2 scores by iteration for each model and prints it in console.
    """
    
    # Write #L and #T as { and } to avoid conflicts with latex
    # Write \ as \\ to avoid conflicts with latex
    base = '''
\\begin{table}[!htp]
    \\centering
    {\\footnotesize
    \\setlength\\tabcolsep{7pt}
    \\begin{tabular}{|c|#Lcolumns#T|}
        \\hline
        \\textbf{Modelo} #Llabels#T\\\\[1.1ex]
        \\hline
        #Ldata#T
    \\end{tabular}
    }
    \\caption{#Lmetric#T por iteración para el #Ldataset#T}
    \\label{tab:#Lmetric#T_#Ldataset#T}
\\end{table}
    '''
    # scape characters
    base = base.replace('{', '{{').replace('}', '}}')
    # base = base.replace('\\', '\\\\')
    
    base = base.replace('#L', '{').replace('#T', '}')
    
    scores_df = get_scores_by_dataset(dataset)
    # filter by Number of individuals
    scores_df = scores_df[scores_df['Number of individuals'] == 1000]
    
    models = scores_df['Model'].unique()
    
    mse_data = ''
    r2_data = ''
    rmses = []
    r2s = []
    for i in range(len(models)):
        rmses = []
        r2s = []
        model = models[i]
        rmse = scores_df[scores_df['Model'] == model]['RMSE mean by iteration'].values[0]
        r2 = scores_df[scores_df['Model'] == model]['R2 mean by iteration'].values[0]
        
        for j in range(1, len(rmse)+1):

            rmse_val = '{:.2e}'.format(rmse[j])
            r2_val = '{:.2e}'.format(r2[j])
            rmse_val = rmse_val.replace('e-0', 'e-').replace('e+0', 'e+')
            r2_val = r2_val.replace('e-0', 'e-').replace('e+0', 'e+')
            rmses.append(rmse_val)
            r2s.append(r2_val)
        
        # & \mbox{rmse} & \mbox{rmse}
        rmse = '} & \\mbox{'.join(rmses)
        r2 = '} & \\mbox{'.join(r2s)
        rmse = '\\mbox{' + rmse + '}'
        r2 = '\\mbox{' + r2 + '}'
        
        mse_data += '''{model} & {rmse} \\\\
        \hline
        '''.format(model=models[i], rmse=rmse)
        
        r2_data += '''{model} & {r2} \\\\
        \hline
        '''.format(model=models[i], r2=r2)

    iterations = len(rmses)
    columns = '*{' + str(iterations) + '}{p{1.08cm}}'
    labels = ''
    for i in range(1, iterations+1):
        labels += '& \\textbf{{{i}}}'.format(i=i)

    rmse_table = base.format(data=mse_data, dataset=dataset, columns=columns, labels=labels, metric='RMSE')
    r2_table = base.format(data=r2_data, dataset=dataset, columns=columns, labels=labels, metric='R2')
    
    print('rmse TABLE')
    print('\n\n')
    
    print(rmse_table)
    
    print('R2 TABLE')
    print('\n\n')
    
    print(r2_table)

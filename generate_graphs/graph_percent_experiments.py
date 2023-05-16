import matplotlib.pyplot as plt
import numpy as np
import argparse
import pandas as pd
import scipy.stats as stats

def main(args):
    calc_all_percent_improvement(args)

    calc_all_raw_values(args)

    return

def calc_all_percent_improvement(args):
    levels = sorted([int(the_level) for the_level in args.levels.split()])
    percent_improvement = list()
    for level in levels:
        improvement = calc_improvement_over_baseline(args, level)
        percent_improvement.append(improvement)
        plt.text(level + 0.5, improvement, ('+' if improvement > 0 else '') + str(improvement) + '%', fontsize=7)
    plt.plot(levels, percent_improvement, marker='o')
    plt.xlabel('Percent of Dataset')
    plt.ylabel('Percent Improvement in {} of\nBernoulli Auxiliary Labels over Baseline'.format(args.metric_name))
    plt.title(args.dataset)
    plt.axhline(y=0, color = 'r', linestyle = '--', alpha=0.5)
    plt.grid()
    plt.savefig('{}_transfer_lowData.pdf'.format(args.dataset))
    plt.show()

    return

def calc_all_raw_values(args):
    baseline_results_lowLr = list()
    metabalance_results_lowLr = list()
    baseline_results_highLr = list()
    metabalance_results_highLr = list()

    levels = sorted([int(the_level) for the_level in args.levels.split()])
    for level in levels:
        metabalance_results_highLr.append(calc_avg_performance_for_technique_and_lr(args, level, 'metabalance', '2e-3'))
        baseline_results_highLr.append(calc_avg_performance_for_technique_and_lr(args, level, 'baseline', '2e-3'))
        metabalance_results_lowLr.append(calc_avg_performance_for_technique_and_lr(args, level, 'metabalance', '2e-4'))
        baseline_results_lowLr.append(calc_avg_performance_for_technique_and_lr(args, level, 'baseline', '2e-4'))

    plt.errorbar(levels, [x[0] for x in metabalance_results_lowLr], [x[1] for x in metabalance_results_lowLr], capsize=2, marker='o', color='royalblue', label='Metabalance (lr=2e-4)')
    plt.errorbar(levels, [x[0] for x in baseline_results_lowLr], [x[1] for x in baseline_results_lowLr], capsize=2, marker='.', color='cornflowerblue', label='Baseline (lr=2e-4)')
    plt.errorbar(levels, [x[0] for x in metabalance_results_highLr], [x[1] for x in metabalance_results_highLr], capsize=2, marker='X', color='indianred', label='Metabalance (lr=2e-3)')
    plt.errorbar(levels, [x[0] for x in baseline_results_highLr], [x[1] for x in baseline_results_highLr], capsize=2, marker='x', color='lightcoral', label='Baseline (lr=2e-3)')

    plt.xlabel('Percent of Dataset')
    plt.ylabel('Accuracy (%)')
    plt.title(args.dataset)

    plt.grid()
    plt.legend()
    plt.savefig('performance_transfer_{}'.format(args.dataset))
    plt.show()

    return

def calc_improvement_over_baseline(args, level):
    csv_name = generate_csv_name(args.dataset, level)
    df = pd.read_csv(csv_name, index_col='Step')
    avg_metabalance, std_metabalance = get_performance(df, ['metabalance'])
    avg_baseline, std_baseline = get_performance(df, ['baseline'])
    return round((avg_metabalance / avg_baseline - 1) * 100, 2)

def calc_avg_performance_for_technique_and_lr(args, level, technique, lr):
    csv_name = generate_csv_name(args.dataset, level)
    df = pd.read_csv(csv_name, index_col='Step')
    avg_acc, std_acc = get_performance(df, [technique, lr])
    return round(avg_acc, 3), round(std_acc, 3)

def generate_csv_name(dataset_name, level):
    return 'data/{}_level{}.csv'.format(dataset_name, level)

def get_performance(df, keywords):
    #print(df.columns)
    desired_columns = [x for x in df.columns \
        if all(keyword in x and 'MIN' not in x and 'MAX' not in x for keyword in keywords)]
    #print(desired_columns)
    if len(keywords) == 1:
        assert len(desired_columns) == 6
    performance_by_run = df.loc[59, desired_columns]
    #print(performance_by_run)
    avg_performance = performance_by_run.mean()
    std_performance = stats.sem(performance_by_run)
    #print(avg_performance)
    return avg_performance, std_performance

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='graph_percent_experiments.py',
        description='Generates graph for a dataset with x-axis percent of dataset, y-axis percent improvement over baseline',
        epilog='Generates graph for a dataset with x-axis percent of dataset, y-axis percent improvement over baseline')
    parser.add_argument('--dataset', type=str, default='cifar10', help='What dataset this is on')
    parser.add_argument('--levels', type=str, default='1 2 5 10 25', help='What levels to calculate')
    parser.add_argument('--metric_name', type=str, default='Validation Accuracy', help='What we\'re calling our metric')
    args = parser.parse_args()
    main(args)
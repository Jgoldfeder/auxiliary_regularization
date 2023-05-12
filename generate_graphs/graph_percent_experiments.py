import matplotlib.pyplot as plt
import numpy as np
import argparse
import pandas as pd

def main(args):
    levels = sorted([int(the_level) for the_level in args.levels.split()])
    percent_improvement = list()
    for level in levels:
        improvement = calc_improvement_over_baseline(args, level)
        percent_improvement.append(improvement)
        plt.text(level - 0.1, improvement + 0.02, ('+' if improvement > 0 else '') + str(improvement), fontsize=7)
    plt.plot(levels, percent_improvement, marker='o')
    plt.xlabel('Percent of Dataset')
    plt.ylabel('Percent Improvement of\nBernoulli Auxiliary Labels over Baseline')
    plt.title(args.dataset)
    plt.axhline(y=0, color = 'r', linestyle = '--', alpha=0.5)
    plt.grid()
    plt.show()
    return

def calc_improvement_over_baseline(args, level):
    csv_name = generate_csv_name(args.dataset, level)
    df = pd.read_csv(csv_name, index_col='Step')
    avg_metabalance = get_performance(df, 'metabalance')
    avg_baseline = get_performance(df, 'baseline')
    return round((avg_metabalance / avg_baseline - 1) * 100, 2)

def generate_csv_name(dataset_name, level):
    return 'data/{}_level{}.csv'.format(dataset_name, level)

def get_performance(df, technique):
    #print(df.columns)
    desired_columns = [x for x in df.columns if (technique in x and 'MIN' not in x and 'MAX' not in x)]
    #print(desired_columns)
    assert len(desired_columns) == 6
    performance_by_run = df.loc[59, desired_columns]
    #print(performance_by_run)
    avg_performance = performance_by_run.mean()
    #print(avg_performance)
    return avg_performance

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='graph_percent_experiments.py',
        description='Generates graph for a dataset with x-axis percent of dataset, y-axis percent improvement over baseline',
        epilog='Generates graph for a dataset with x-axis percent of dataset, y-axis percent improvement over baseline')
    parser.add_argument('--dataset', type=str, default='cifar10', help='What dataset this is on')
    parser.add_argument('--levels', type=str, default='1 2 5 10 25', help='What levels to calculate')
    parser.add_argument('--metric_name', type=str, default='Best Validation Accuracy', help='What we\'re calling our metric')
    args = parser.parse_args()
    main(args)
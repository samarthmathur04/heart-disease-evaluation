"""Render full confusion matrices from saved metrics; does not fit models."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from train_compare import ROOT


def plot_matrices(rows, path, title):
    fig, axes = plt.subplots(1, len(rows), figsize=(4 * len(rows), 4))
    for ax, row in zip(np.atleast_1d(axes), rows):
        matrix = np.array([[row['tn'], row['fp']], [row['fn'], row['tp']]], dtype=int)
        ax.imshow(matrix, cmap='Blues', vmin=0, vmax=sum(matrix.ravel()))
        for i in range(2):
            for j in range(2):
                tag = [['TN', 'FP'], ['FN', 'TP']][i][j]
                ax.text(j, i, f'{tag}\n{matrix[i,j]}', ha='center', va='center', fontsize=15)
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['Absent', 'Present'],
               yticklabels=['Absent', 'Present'], xlabel='Predicted disease', ylabel='Actual disease',
               title=row['model'])
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    destination = ROOT / 'results' / 'experiments'
    destination.mkdir(exist_ok=True)
    metrics = pd.read_csv(ROOT / 'results' / 'test_metrics.csv')
    rows = metrics.loc[(metrics.operating_point == 'default_0.5') &
                       (metrics.model != 'Constant prevalence reference')].to_dict('records')
    plot_matrices(rows, destination / 'test_confusion_matrices.png',
                  'Previously evaluated test set: n=184, threshold=0.5')
    nested = destination / 'nested_summary.csv'
    if nested.exists():
        metrics = pd.read_csv(nested)
        rows = metrics.loc[(metrics.operating_point == 'inner_target_sensitivity_0.90') &
                           metrics.model.isin(['Baseline logistic', 'Tuned forest', 'Tuned boosting'])].to_dict('records')
        plot_matrices(rows, destination / 'training_confusion_matrices.png',
                      'Training-only outer validation: n=734; inner-selected cutoffs targeting 90% sensitivity')
    print('Confusion-matrix figures regenerated from saved counts.')


if __name__ == '__main__':
    main()

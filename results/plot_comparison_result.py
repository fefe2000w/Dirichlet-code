import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

import scipy.io as scio

database = scio.loadmat("../datasets/benchmarks.mat")
benchmarks = database["benchmarks"][0]


# Initialize an empty list to store the benchmark names
dataset_names = []

# Iterate over the benchmarks
for benchmark in benchmarks:
    name = benchmark[0]
    if name != "image" and name != 'splice':
        dataset_names.append(name)
methods = ['vi', 'db', 'la', 'ep']
result_dir = os.path.join('.', 'evaluation')

all_report = {}
all_errors = {}

# Load reports using pickle
for benchmark_name in dataset_names:
    # 初始化内层字典
    if benchmark_name not in all_report:
        all_report[benchmark_name] = {}
    if benchmark_name not in all_errors:
        all_errors[benchmark_name] = {}

    for method_name in methods:
        report_path = os.path.join(result_dir, f'{benchmark_name}_{method_name}_report.dat')
        error_path = os.path.join(result_dir, f'{benchmark_name}_{method_name}_errors.dat')

        if os.path.exists(report_path):
            with open(report_path, 'rb') as f:
                all_report[benchmark_name][method_name] = pickle.load(f)
        else:
            print(f"Warning: {report_path} does not exist.")

        if os.path.exists(error_path):
            with open(error_path, 'rb') as f:
                all_errors[benchmark_name][method_name] = pickle.load(f)
        else:
            print(f"Warning: {error_path} does not exist.")

print("Loading process completed.")

vi_error_rate = []
db_error_rate = []
la_error_rate = []
ep_error_rate = []

vi_mnll = []
db_mnll = []
la_mnll = []
ep_mnll = []

vi_ece = []
db_ece = []
la_ece = []
ep_ece = []

vi_time = []
db_time = []
la_time = []
ep_time = []

vi_optim_time = []
db_optim_time = []
la_optim_time = []
ep_optim_time = []

vi_pred_time = []
db_pred_time = []
la_pred_time = []
ep_pred_time = []


for benchmark, methods in all_report.items():
    vi_error_rate.append(methods.get('vi', {}).get('vi_error_rate'))
    db_error_rate.append(methods.get('db', {}).get('db_error_rate'))
    la_error_rate.append(methods.get('la', {}).get('la_error_rate'))
    ep_error_rate.append(methods.get('ep', {}).get('ep_error_rate'))

    vi_mnll.append(methods.get('vi', {}).get('vi_mnll'))
    db_mnll.append(methods.get('db', {}).get('db_mnll'))
    la_mnll.append(methods.get('la', {}).get('la_mnll'))
    ep_mnll.append(methods.get('ep', {}).get('ep_mnll'))

    vi_ece.append(methods.get('vi', {}).get('vi_ece'))
    db_ece.append(methods.get('db', {}).get('db_ece'))
    la_ece.append(methods.get('la', {}).get('la_ece'))
    ep_ece.append(methods.get('ep', {}).get('ep_ece'))

    vi_time.append(
        methods.get('vi', {}).get('vi_elapsed_optim')
        + methods.get('vi', {}).get('vi_elapsed_pred'))
    db_time.append(
        methods.get('db', {}).get('db_elapsed_optim')
        + methods.get('db', {}).get('db_elapsed_pred')
    )
    la_time.append(
        methods.get('la', {}).get('la_elapsed_optim')
        + methods.get('la', {}).get('la_elapsed_pred'))
    ep_time.append(
        methods.get('ep', {}).get('ep_elapsed_optim')
        + methods.get('ep', {}).get('ep_elapsed_pred')
    )

    vi_optim_time.append(methods.get('vi', {}).get('vi_elapsed_optim'))
    db_optim_time.append(methods.get('db', {}).get('db_elapsed_optim'))
    la_optim_time.append(methods.get('la', {}).get('la_elapsed_optim'))
    ep_optim_time.append(methods.get('ep', {}).get('ep_elapsed_optim'))

    vi_pred_time.append(methods.get('vi', {}).get('vi_elapsed_pred'))
    db_pred_time.append(methods.get('db', {}).get('db_elapsed_pred'))
    la_pred_time.append(methods.get('la', {}).get('la_elapsed_pred'))
    ep_pred_time.append(methods.get('ep', {}).get('ep_elapsed_pred'))


vi_error_rate_err = []
db_error_rate_err = []
la_error_rate_err = []
ep_error_rate_err = []

vi_mnll_err = []
db_mnll_err = []
la_mnll_err = []
ep_mnll_err = []

vi_ece_err = []
db_ece_err = []
la_ece_err = []
ep_ece_err = []

for benchmark, methods in all_errors.items():
    vi_error_rate_err.append(methods.get('vi', {}).get('vi_error_rate'))
    db_error_rate_err.append(methods.get('db', {}).get('db_error_rate'))
    la_error_rate_err.append(methods.get('la', {}).get('la_error_rate'))
    ep_error_rate_err.append(methods.get('ep', {}).get('ep_error_rate'))

    vi_mnll_err.append(methods.get('vi', {}).get('vi_mnll'))
    db_mnll_err.append(methods.get('db', {}).get('db_mnll'))
    la_mnll_err.append(methods.get('la', {}).get('la_mnll'))
    ep_mnll_err.append(methods.get('ep', {}).get('ep_mnll'))

    vi_ece_err.append(methods.get('vi', {}).get('vi_ece'))
    db_ece_err.append(methods.get('db', {}).get('db_ece'))
    la_ece_err.append(methods.get('la', {}).get('la_ece'))
    ep_ece_err.append(methods.get('ep', {}).get('ep_ece'))


bar_width = 0.2
index = np.arange(len(dataset_names))

fig, ax = plt.subplots(3, 1, figsize=(12, 18))

hatches = ["", "//", "||", "++"]

ax[0].bar(
    index,
    vi_error_rate,
    bar_width,
    yerr=vi_error_rate_err,
    label="Variational Inference",
    hatch=hatches[0],
)
ax[0].bar(
    index + bar_width,
    db_error_rate,
    bar_width,
    yerr=db_error_rate_err,
    label="Dirichlet-Based GPC",
    hatch=hatches[1],
)
ax[0].bar(
    index + 2 * bar_width,
    la_error_rate,
    bar_width,
    yerr=la_error_rate_err,
    label="Laplace Approximation",
    hatch=hatches[2],
)
ax[0].bar(
    index + 3 * bar_width,
    ep_error_rate,
    bar_width,
    yerr=ep_error_rate_err,
    label="Expectation Propagation",
    hatch=hatches[3],
)
ax[0].set_xlabel("Dataset")
ax[0].set_ylabel("Error Rate")
ax[0].set_title("Error Rate by Dataset and Method")
ax[0].set_xticks(index + 1.5 * bar_width)
ax[0].set_xticklabels(dataset_names, rotation=45)
ax[0].legend()

ax[1].bar(
    index,
    vi_mnll,
    bar_width,
    yerr=vi_mnll_err,
    label="Variational Inference",
    hatch=hatches[0],
)
ax[1].bar(
    index + bar_width,
    db_mnll,
    bar_width,
    yerr=db_mnll_err,
    label="Dirichlet-Based GPC",
    hatch=hatches[1],
)
ax[1].bar(
    index + 2 * bar_width,
    la_mnll,
    bar_width,
    yerr=la_mnll_err,
    label="Laplace Approximation",
    hatch=hatches[2],
)
ax[1].bar(
    index + 3 * bar_width,
    ep_mnll,
    bar_width,
    yerr=ep_mnll_err,
    label="Expectation Propagation",
    hatch=hatches[3],
)
ax[1].set_xlabel("Dataset")
ax[1].set_ylabel("MNLL")
ax[1].set_title("MNLL by Dataset and Method")
ax[1].set_xticks(index + 1.5 * bar_width)
ax[1].set_xticklabels(dataset_names, rotation=45)
ax[1].legend()

ax[2].bar(
    index,
    vi_ece,
    bar_width,
    yerr=vi_ece_err,
    label="Variational Inference",
    hatch=hatches[0],
)
ax[2].bar(
    index + bar_width,
    db_ece,
    bar_width,
    yerr=db_ece_err,
    label="Dirichlet-Based GPC",
    hatch=hatches[1],
)
ax[2].bar(
    index + 2 * bar_width,
    la_ece,
    bar_width,
    yerr=la_ece_err,
    label="Laplace Approximation",
    hatch=hatches[2],
)
ax[2].bar(
    index + 3 * bar_width,
    ep_ece,
    bar_width,
    yerr=ep_ece_err,
    label="Expectation Propagation",
    hatch=hatches[3],
)
ax[2].set_xlabel("Dataset")
ax[2].set_ylabel("ECE")
ax[2].set_title("ECE by Dataset and Method")
ax[2].set_xticks(index + 1.5 * bar_width)
ax[2].set_xticklabels(dataset_names, rotation=45)
ax[2].legend()

graph_dir = os.path.join('results', 'graph')
if not os.path.exists(graph_dir):
    os.makedirs(graph_dir)
save_path = os.path.join(graph_dir, 'measurement.png')
#plt.savefig(save_path)

plt.tight_layout()
plt.show()

# Comparison of time
fig, ax = plt.subplots(3, 1, figsize=(12, 18))

ax[0].bar(
    index, vi_time, bar_width, label="Variational Inference", hatch=hatches[0]
)
ax[0].bar(
    index + bar_width,
    db_time,
    bar_width,
    label="Dirichlet-Based GPC",
    hatch=hatches[1],
)
ax[0].bar(
    index + 2 * bar_width,
    la_time,
    bar_width,
    label="Laplace Approximation",
    hatch=hatches[2],
)
ax[0].bar(
    index + 3 * bar_width,
    ep_time,
    bar_width,
    label="Expectation Propagation",
    hatch=hatches[3],
)
ax[0].set_xlabel("Dataset")
ax[0].set_ylabel("Time")
ax[0].set_title("Time by Dataset and Method")
ax[0].set_xticks(index + 1.5 * bar_width)
ax[0].set_xticklabels(dataset_names, rotation=45)
ax[0].legend()

ax[1].bar(
    index,
    vi_optim_time,
    bar_width,
    label="Variational Inference",
    hatch=hatches[0],
)
ax[1].bar(
    index + bar_width,
    db_optim_time,
    bar_width,
    label="Dirichlet-Based GPC",
    hatch=hatches[1],
)
ax[1].bar(
    index + 2 * bar_width,
    la_optim_time,
    bar_width,
    label="Laplace Approximation",
    hatch=hatches[2],
)
ax[1].bar(
    index + 3 * bar_width,
    ep_optim_time,
    bar_width,
    label="Expectation Propagation",
    hatch=hatches[3],
)
ax[1].set_xlabel("Dataset")
ax[1].set_ylabel("Optimisation Time")
ax[1].set_title("Optim Time by Dataset and Method")
ax[1].set_xticks(index + 1.5 * bar_width)
ax[1].set_xticklabels(dataset_names, rotation=45)
ax[1].legend()

ax[2].bar(
    index,
    vi_pred_time,
    bar_width,
    label="Variational Inference",
    hatch=hatches[0],
)
ax[2].bar(
    index + bar_width,
    db_pred_time,
    bar_width,
    label="Dirichlet-Based GPC",
    hatch=hatches[1],
)
ax[2].bar(
    index + 2 * bar_width,
    la_pred_time,
    bar_width,
    label="Laplace Approximation",
    hatch=hatches[2],
)
ax[2].bar(
    index + 3 * bar_width,
    ep_pred_time,
    bar_width,
    label="Expectation Propagation",
    hatch=hatches[3],
)
ax[2].set_xlabel("Dataset")
ax[2].set_ylabel("Prediction Time")
ax[2].set_title("Pred Time by Dataset and Method")
ax[2].set_xticks(index + 1.5 * bar_width)
ax[2].set_xticklabels(dataset_names, rotation=45)
ax[2].legend()

save_path = os.path.join(graph_dir, 'time.png')
#plt.savefig(save_path)

plt.show()


# Measurement by time
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

axs[0].scatter(
    vi_time,
    vi_error_rate,
    color="r",
    marker="o",
    label="Variational Inference",
)
axs[0].scatter(
    db_time, db_error_rate, color="g", marker="x", label="Dirichlet-Based GPC"
)
axs[0].scatter(
    la_time,
    la_error_rate,
    color="b",
    marker="s",
    label="Laplace Approximation",
)
axs[0].scatter(
    ep_time,
    ep_error_rate,
    color="m",
    marker="d",
    label="Expectation Propagation",
)
axs[0].set_xlabel("Time")
axs[0].set_ylabel("Error Rate")
axs[0].set_title("Error Rate vs Time")
axs[0].legend()

axs[1].scatter(
    vi_time, vi_mnll, color="r", marker="o", label="Variational Inference"
)
axs[1].scatter(
    db_time, db_mnll, color="g", marker="x", label="Dirichlet-Based GPC"
)
axs[1].scatter(
    la_time, la_mnll, color="b", marker="s", label="Laplace Approximation"
)
axs[1].scatter(
    ep_time, ep_mnll, color="m", marker="d", label="Expectation Propagation"
)
axs[1].set_xlabel("Time")
axs[1].set_ylabel("MNLL")
axs[1].set_title("MNLL vs Time")
axs[1].legend()

axs[2].scatter(
    vi_time, vi_ece, color="r", marker="o", label="Variational Inference"
)
axs[2].scatter(
    db_time, db_ece, color="g", marker="x", label="Dirichlet-Based GPC"
)
axs[2].scatter(
    la_time, la_ece, color="b", marker="s", label="Laplace Approximation"
)
axs[2].scatter(
    ep_time, ep_ece, color="m", marker="d", label="Expectation Propagation"
)
axs[2].set_xlabel("Time")
axs[2].set_ylabel("ECE")
axs[2].set_title("ECE vs Time")
axs[2].legend()

save_path = os.path.join(graph_dir, 'measurement_by_time.png')
#plt.savefig(save_path)

plt.tight_layout()
plt.show()

##===========================================================================================
measurements = ["error_rate", "mnll", "ece"]
data = {
    "error_rate": {
        "vi": vi_error_rate,
        "db": db_error_rate,
        "la": la_error_rate,
        "ep": ep_error_rate
    },
    "mnll": {
        "vi": vi_mnll,
        "db": db_mnll,
        "la": la_mnll,
        "ep": ep_mnll
    },
    "ece": {
        "vi": vi_ece,
        "db": db_ece,
        "la": la_ece,
        "ep": ep_ece
    }
}

errors = {
    "error_rate": {
        "vi": vi_error_rate_err,
        "db": db_error_rate_err,
        "la": la_error_rate_err,
        "ep": ep_error_rate_err
    },
    "mnll": {
        "vi": vi_mnll_err,
        "db": db_mnll_err,
        "la": la_mnll_err,
        "ep": ep_mnll_err
    },
    "ece": {
        "vi": vi_ece_err,
        "db": db_ece_err,
        "la": la_ece_err,
        "ep": ep_ece_err
    }
}

# Colors and markers for the plots
colors = ['red', 'blue', 'magenta', 'green']
markers = ['o', 's', 'D', '^']

num_datasets = len(dataset_names)
num_cols = 3
num_rows = (num_datasets + num_cols - 1) // num_cols

# For each measurement, create a plot comparing benchmarks
for measurement in measurements:
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(18, 18))
    axs = axs.flatten()

    for i, dataset in enumerate(dataset_names):
        ax = axs[i]

        for method, color, marker in zip(methods, colors, markers):
            value = data[measurement][method][i]
            err = errors[measurement][method][i]

            ax.errorbar(value, method, xerr=err, label=method,
                        fmt=marker, color=color, ecolor=color, elinewidth=2, capsize=4, markersize=8)

        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(methods)
        ax.set_xlabel('')
        ax.set_title(f'{dataset}')

    # Hide any empty subplots
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    # Set the main title for the figure
    fig.suptitle(f'{measurement.replace("_", " ").title()} Comparison', fontsize=20)

    # Add legend only to the first subplot
    axs[0].legend(loc='upper right', fontsize=12)

    # Save the figure to the results directory
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(graph_dir, f'comparison_{measurement}.png')
    #plt.savefig(save_path)

    # Show the plot
    plt.show()
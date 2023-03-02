import argparse
import json
from os import listdir
from os.path import isdir, join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def aggregated_results(
    results_dir,
    ent_types=["Artist", "WoA"],
    metrics=["correct", "incorrect", "missed", "spurious"],
    no_predictors=[1, 2, 3],  # corresponds to model seeds and annotator ids
):
    model_results = {}
    human_results = {}
    for metric in metrics:
        model_results[metric] = {}
        human_results[metric] = {}
        for ent_type in ent_types:
            model_results[metric][ent_type] = []
            human_results[metric][ent_type] = []

    ds_dirs = [
        join(results_dir, d)
        for d in listdir(results_dir)
        if isdir(join(results_dir, d))
    ]
    for ds_dir in ds_dirs:
        for id in no_predictors:
            # Model
            fpath = join(ds_dir, f"seed{id}", "predict_results.json")
            with open(fpath, "r") as _:
                predictions = json.load(_)
                for ent_type in ent_types:
                    for metric in metrics:
                        key = f"predict_{ent_type}_strict_{metric}"
                        model_results[metric][ent_type].append(predictions[key])
                    # compute possible - number annotations in the gold-standard from the values just added
                    possible = (
                        model_results["correct"][ent_type][-1]
                        + model_results["incorrect"][ent_type][-1]
                        + model_results["missed"][ent_type][-1]
                    )
                    # compute actual - number annotations produced by the NER system from the values just added
                    actual = (
                        model_results["correct"][ent_type][-1]
                        + model_results["incorrect"][ent_type][-1]
                        + model_results["spurious"][ent_type][-1]
                    )
                    # update scores as ratios of possibe or actual
                    for metric in ["correct", "missed"]:
                        model_results[metric][ent_type][-1] /= possible
                    for metric in ["incorrect", "spurious"]:
                        model_results[metric][ent_type][-1] /= actual

            # Human
            fpath = join(ds_dir, f"annotator{id}_results.json")
            with open(fpath, "r") as _:
                predictions = json.load(_)
                for ent_type in ent_types:
                    for metric in metrics:
                        key = f"{ent_type}_strict_weak_{metric}"
                        key_pos = f"{ent_type}_strict_weak_possible"
                        key_act = f"{ent_type}_strict_weak_actual"
                        if metric in ["correct", "missed"]:
                            human_results[metric][ent_type].append(
                                predictions[key] / predictions[key_pos]
                            )
                        else:
                            human_results[metric][ent_type].append(
                                predictions[key] / predictions[key_act]
                            )
    return model_results, human_results


def plot(model_results, human_results, title="strict"):
    """
    Plot error bars using data from the dataframe df
    """
    x = []
    # Model
    mean = []
    std = []
    for metric in model_results:
        for ent_type in model_results[metric]:
            x.append(f"{ent_type}_{metric}")
            mean.append(np.mean(model_results[metric][ent_type]))
            std.append(np.std(model_results[metric][ent_type]))
    mline = plt.errorbar(
        x, mean[: len(x)], std[: len(x)], linestyle="None", marker="o", color="#0868ac"
    )
    # Human
    mean = []
    std = []
    for metric in human_results:
        for ent_type in human_results[metric]:
            mean.append(np.mean(human_results[metric][ent_type]))
            std.append(np.std(human_results[metric][ent_type]))
    hline = plt.errorbar(
        x, mean[: len(x)], std[: len(x)], linestyle="None", marker="^", color="#7bccc4"
    )
    plt.legend([mline, hline], ["BERT", "human"])
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        dest="results_dir",
        type=str,
        help="Directory where the results were saved or exported",
        required=True,
    )
    args = parser.parse_args()
    model_results, human_results = aggregated_results(args.results_dir)
    df_model = pd.DataFrame.from_dict(
        model_results, columns=["Artist", "WoA"], orient="index"
    )
    df_human = pd.DataFrame.from_dict(
        human_results, columns=["Artist", "WoA"], orient="index"
    )
    plot(model_results, human_results)

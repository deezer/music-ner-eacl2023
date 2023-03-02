import argparse
import json
from copy import deepcopy
from os import listdir
from os.path import isdir, join

import numpy as np
from scipy import stats

SCENARIOS = {"seen": [], "rare_unseen": []}


def seen_rare_unseen_results(
    results_dir,
    ent_types=["Artist", "WoA"],
    no_predictors=[1, 2, 3],  # corresponds to model seeds and annotator ids)
):
    results = {}
    for ent_type in ent_types:
        results[ent_type] = {"model": deepcopy(SCENARIOS), "human": deepcopy(SCENARIOS)}
        ds_dirs = [
            join(results_dir, d)
            for d in listdir(results_dir)
            if isdir(join(results_dir, d))
        ]
        for ds_dir in ds_dirs:
            for scenario in SCENARIOS:
                for id in no_predictors:
                    # Model
                    fpath = join(ds_dir, scenario, f"seed{id}", "predict_results.json")
                    with open(fpath, "r") as _:
                        predictions = json.load(_)
                        key = f"predict_{ent_type}_strict_recall"
                        results[ent_type]["model"][f"{scenario}"].append(
                            predictions[key]
                        )
                    # Human
                    fpath = join(ds_dir, scenario, f"annotator{id}_results.json")
                    with open(fpath, "r") as _:
                        predictions = json.load(_)
                        key = f"{ent_type}_strict_weak_recall"
                        results[ent_type]["human"][f"{scenario}"].append(
                            predictions[key]
                        )
    return results


def print_latex_table(results):
    for ent_type in results:
        for pred_type in results[ent_type]:
            latex_str = f" {ent_type} & {pred_type} & "
            for scenario in results[ent_type][pred_type]:
                mean = np.mean(results[ent_type][pred_type][scenario])
                std = np.std(results[ent_type][pred_type][scenario])
                latex_str += (
                    str(round(mean, 2)) + " $\\pm$ " + str(round(std, 2)) + " & "
                )
            print(latex_str[:-2] + " \\\\")


def significance_test(results, alpha=0.05):
    for ent_type in results:
        for pred_type in results[ent_type]:
            test_case = f"testing for {pred_type} {ent_type}"
            _, p = stats.mannwhitneyu(
                results[ent_type][pred_type]["seen"],
                results[ent_type][pred_type]["rare_unseen"],
            )
            if p < alpha:
                print(f"{test_case}: reject H0, the distributions are not the same")
            else:
                print(f"{test_case}: accept H0, the distributions are the same")


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
    results = seen_rare_unseen_results(args.results_dir)

    print("\nTable 7:")
    print_latex_table(results)

    print("\nResults statistical significance testing")
    significance_test(results)

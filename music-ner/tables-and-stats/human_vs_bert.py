import argparse
import json
from os import listdir
from os.path import isdir, join

import numpy as np
from scipy import stats


def aggregated_results(
    results_dir,
    ent_types=["Artist", "WoA"],
    eval_schemas=["strict", "exact", "ent_type"],
    metrics=["f1", "precision", "recall"],
    no_predictors=[1, 2, 3],  # corresponds to model seeds and annotator ids
):
    results = {}
    for metric in metrics:
        results[metric] = {}
        for schema in eval_schemas:
            results[metric][schema] = {}
            for ent_type in ent_types:
                results[metric][schema][ent_type] = {"model": [], "human": []}
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
                            key = f"predict_{ent_type}_{schema}_{metric}"
                            results[metric][schema][ent_type]["model"].append(
                                predictions[key]
                            )

                        # Human
                        fpath = join(ds_dir, f"annotator{id}_results.json")
                        with open(fpath, "r") as _:
                            predictions = json.load(_)
                            if schema == "strict":
                                key = f"{ent_type}_{schema}_weak_{metric}"
                            else:
                                key = f"{ent_type}_{schema}_{metric}"
                            results[metric][schema][ent_type]["human"].append(
                                predictions[key]
                            )
    return results


def print_latex_table(results, schemas=[]):
    if schemas == []:
        schemas = list(results.keys())
    ent_types = results[schemas[0]].keys()
    latex_str = {}
    for ent_type in ent_types:
        latex_str["model"] = f"{ent_type} & model & "
        latex_str["human"] = f"{ent_type} & human & "
        for schema in schemas:
            for pred_type in results[schema][ent_type]:
                mean = np.mean(results[schema][ent_type][pred_type])
                std = np.std(results[schema][ent_type][pred_type])
                latex_str[pred_type] += (
                    str(round(mean, 2)) + " $\\pm$ " + str(round(std, 2)) + " & "
                )
        for pred_type in latex_str:
            print(latex_str[pred_type][:-2] + " \\\\")
    print("\n")


def significance_test(results, alpha=0.05):
    for metric in results:
        for schema in results[metric]:
            for ent_type in results[metric][schema]:
                human_results = results[metric][schema][ent_type]["human"]
                model_results = results[metric][schema][ent_type]["model"]
                test_case = f"{metric}, {schema}, {ent_type}"
                _, p = stats.mannwhitneyu(model_results, human_results)
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
    results = aggregated_results(args.results_dir)

    print("\nTable 5:")
    print_latex_table(results["f1"])
    print("Table 6, precision:\n")
    print_latex_table(results["precision"], schemas=["strict"])
    print("Table 6, recall:\n")
    print_latex_table(results["recall"], schemas=["strict"])

    print("\nResults statistical significance testing")
    significance_test(results)
    print("\n")

import argparse
import json
from os import listdir
from os.path import isdir, join

import numpy as np
from scipy import stats

REF_MODEL = "bert-large-uncased"
OTHER_MODELS = ["roberta-large", "mpnet-base"]
MODELS = [REF_MODEL] + OTHER_MODELS


def model_results(results_dir, models, ent_types=["Artist", "WoA", "overall"]):
    results = {}
    for model in models:
        results[model] = {}
        for ent_type in ent_types:
            key = f"predict_{ent_type}_strict_f1"
            if ent_type == "overall":
                key += "_macro"
            results[model][ent_type] = []
            ds_dirs = [
                join(results_dir, d)
                for d in listdir(results_dir)
                if isdir(join(results_dir, d))
            ]
            for ds_dir in ds_dirs:
                fpath = join(ds_dir, model, "predict_results.json")
                with open(fpath, "r") as _:
                    predictions = json.load(_)
                    results[model][ent_type].append(predictions[key])
    return results


def print_latex_table(results):
    for model in results:
        latex_str = model + " & "
        for ent_type in results[model]:
            mean = np.mean(results[model][ent_type])
            std = np.std(results[model][ent_type])
            latex_str += str(round(mean, 2)) + " $\\pm$ " + str(round(std, 2)) + " & "
        print(latex_str[:-2])


def significance_test(ref_model_results, other_model_results, alpha=0.05):
    ent_types = ref_model_results.keys()
    for ent_type in ent_types:
        for model in other_model_results:
            test_case = f"{ent_type} - {REF_MODEL} and {model}"
            _, p = stats.mannwhitneyu(
                ref_model_results[ent_type], other_model_results[model][ent_type]
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
    results = model_results(args.results_dir, MODELS)

    print("\nTable 4:")
    print_latex_table(results)

    ref_model_results = results[REF_MODEL]
    other_model_results = {model: results[model] for model in OTHER_MODELS}
    print("\nResults statistical significance testing")
    significance_test(ref_model_results, other_model_results)
    print("\n")

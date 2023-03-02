from copy import deepcopy

from ner_eval import Evaluator
from tabulate import tabulate


def compute_results(
    true_labels,
    true_predictions,
    ent_types=["Artist", "WoA"],
    eval_schemas=["strict", "ent_type", "exact"],
):
    metrics_results = {
        "precision": [],
        "recall": [],
        "f1": [],
        "correct": [],
        "incorrect": [],
        "partial": [],
        "missed": [],
        "spurious": [],
        "possible": [],
        "actual": [],
    }
    # overall results
    results = {}
    for eval_schema in eval_schemas:
        results[eval_schema] = deepcopy(metrics_results)

    all_labels = ent_types
    target_labels = ["Artist", "WoA"]
    evaluation_agg_entities_type = {e: deepcopy(results) for e in target_labels}
    evaluator = Evaluator(true_labels, true_predictions, all_labels)
    tmp_results, tmp_results_agg = evaluator.evaluate()
    # aggregate overall results
    for eval_schema in results.keys():
        for metric in metrics_results:
            results[eval_schema][metric] = tmp_results[eval_schema][metric]
    for e_type in target_labels:
        for eval_schema in results.keys():
            for metric in metrics_results:
                evaluation_agg_entities_type[e_type][eval_schema][
                    metric
                ] = tmp_results_agg[e_type][eval_schema][metric]

    final_results = {}
    for key, value in results.items():
        for n, v in value.items():
            final_results[f"overall_{key}_{n}_micro"] = v
    for e_type in target_labels:
        for key, value in evaluation_agg_entities_type[e_type].items():
            for n, v in value.items():
                final_results[f"{e_type}_{key}_{n}"] = v
                macro_key = f"overall_{key}_{n}_macro"
                if macro_key not in final_results:
                    final_results[macro_key] = v
                else:
                    final_results[macro_key] += v

    for key in final_results:
        if "macro" in key:
            final_results[key] /= len(target_labels)

    print("\n Overall")
    print_results(results, metrics_results)

    for e_type in target_labels:
        print("\n", e_type)
        print_results(evaluation_agg_entities_type[e_type], metrics_results)

    return final_results


def print_results(results, metrics_results):
    """
    Helper to print the results in a table form
    """
    headers = ["schema"] + list(metrics_results.keys())
    results_tbl = []
    for eval_schema in results.keys():
        row_results_tbl = [eval_schema]
        for metric in metrics_results.keys():
            row_results_tbl.append(results[eval_schema][metric])
        results_tbl.append(row_results_tbl)
    print(tabulate(results_tbl, headers=headers))

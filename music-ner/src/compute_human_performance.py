import argparse
import json
import os
import sys

sys.path.append("music-ner/datasets")
from ds_utils import read_sents
from eval_utils import compute_results

NO_ANNOTATORS = 3


def sent2labels(sent):
    labels = [label for token, label in sent]
    return labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", dest="data_dir", type=str, help="Data directory", required=True
    )
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        help="Directory where to export the results",
        required=True,
    )
    args = parser.parse_args()

    gtruth_fpaths = {}
    gtruth_fpaths[""] = f"{args.data_dir}/test.bio"
    if not os.path.isfile(gtruth_fpaths[""]):
        print("{} file not found".format(gtruth_fpaths[""]))
        sys.exit(1)

    gtruth_fpaths["seen"] = f"{args.data_dir}/seen/test.bio"
    if not os.path.isfile(gtruth_fpaths["seen"]):
        print("{} file not found".format(gtruth_fpaths["seen"]))

    gtruth_fpaths["rare_unseen"] = f"{args.data_dir}/rare_unseen/test.bio"
    if not os.path.isfile(gtruth_fpaths["rare_unseen"]):
        print("{} file not found".format(gtruth_fpaths["rare_unseen"]))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for scenario in gtruth_fpaths:
        if gtruth_fpaths[scenario] == "":
            continue
        print(scenario)
        gtruth_sents = read_sents(gtruth_fpaths[scenario])
        gtruth_labels = [sent2labels(gtruth_sents[s]) for s in gtruth_sents]

        for i in range(1, NO_ANNOTATORS + 1):
            print("Annotator {}".format(i))
            annot_filepath = os.path.join(args.data_dir, "annotator{}.bio".format(i))
            print(annot_filepath)
            if not os.path.isfile(annot_filepath):
                break  # no more annotators
            annot_sents = read_sents(annot_filepath)
            annot_labels = [sent2labels(annot_sents[s]) for s in annot_sents]

            metrics = compute_results(
                gtruth_labels,
                annot_labels,
                ent_types=["Artist", "WoA", "Artist_or_WoA"],
                eval_schemas=["strict_weak", "ent_type", "exact"],
            )
            scenario_dir = os.path.join(args.output_dir, scenario)
            if not os.path.exists(scenario_dir):
                os.makedirs(scenario_dir)
            path = os.path.join(scenario_dir, f"annotator{i}_results.json")
            with open(path, "w") as f:
                json.dump(metrics, f, indent=4, sort_keys=True)

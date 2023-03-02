#!/usr/bin/env python
# coding=utf-8
#
# Original source: https://github.com/davidsbatista/NER-Evaluation/tree/master/ner_evaluation

# Script adapted for the article "A Human Subject Study of Named Entity
# Recognition in Conversational Music Recommendation Queries" submitted
# to EACL 2023

import logging
from collections import namedtuple
from copy import deepcopy
from difflib import SequenceMatcher

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level="DEBUG",
)

Entity = namedtuple("Entity", "e_type start_offset end_offset")


class Evaluator:
    def __init__(self, true, pred, tags):
        if len(true) != len(pred):
            raise ValueError("Number of predicted does not equal true")

        self.true = true
        self.pred = pred
        self.tags = tags

        # Setup dict into which metrics will be stored.
        self.metrics_results = {
            "correct": 0,
            "incorrect": 0,
            "partial": 0,
            "missed": 0,
            "spurious": 0,
            "possible": 0,
            "actual": 0,
            "precision": 0,
            "recall": 0,
            "f1": 0,
        }

        # Copy results dict to cover the four schemes.
        self.results = {
            "strict": deepcopy(self.metrics_results),
            "strict_weak": deepcopy(self.metrics_results),
            "exact": deepcopy(self.metrics_results),
            "ent_type": deepcopy(self.metrics_results),
            "ent_type_weighted": deepcopy(self.metrics_results),
        }

        # Create an accumulator to store results
        self.evaluation_agg_entities_type = {e: deepcopy(self.results) for e in tags}

    def evaluate(self):
        logging.info(
            "Imported %s predictions for %s true examples",
            len(self.pred),
            len(self.true),
        )

        for true_ents, pred_ents in zip(self.true, self.pred):
            # Check that the length of the true and predicted examples are the
            # same. This must be checked here, because another error may not
            # be thrown if the lengths do not match.

            if len(true_ents) != len(pred_ents):
                raise ValueError("Prediction length does not match true example length")

            # Compute results for one message
            tmp_results, tmp_agg_results = compute_metrics(
                collect_named_entities(true_ents),
                collect_named_entities(pred_ents),
                self.tags,
            )

            # Cycle through each result and accumulate
            for eval_schema in self.results:
                for metric in self.results[eval_schema]:
                    self.results[eval_schema][metric] += tmp_results[eval_schema][
                        metric
                    ]

            # Calculate global precision and recall
            self.results = compute_precision_recall(self.results)

            # Aggregate results by entity type
            for e_type in self.tags:
                for eval_schema in tmp_agg_results[e_type]:
                    for metric in tmp_agg_results[e_type][eval_schema]:
                        self.evaluation_agg_entities_type[e_type][eval_schema][
                            metric
                        ] += tmp_agg_results[e_type][eval_schema][metric]

                # Calculate precision recall at the individual entity level
                self.evaluation_agg_entities_type[e_type] = compute_precision_recall(
                    self.evaluation_agg_entities_type[e_type]
                )

        return self.results, self.evaluation_agg_entities_type


def collect_named_entities(tokens):
    """
    Create a list of Entity named-tuples, storing the entity type and the start and end
    offsets of the entity.

    :param tokens: a list of tags
    :return: a list of Entity named-tuples
    """

    named_entities = []
    start_offset = None
    end_offset = None
    ent_type = None

    for offset, token_tag in enumerate(tokens):
        if token_tag == "O":
            if ent_type is not None and start_offset is not None:
                end_offset = offset - 1
                named_entities.append(Entity(ent_type, start_offset, end_offset))
                start_offset = None
                end_offset = None
                ent_type = None

        elif ent_type is None:
            ent_type = token_tag[2:]
            start_offset = offset

        elif ent_type != token_tag[2:] or (
            ent_type == token_tag[2:] and token_tag[:1] == "B"
        ):

            end_offset = offset - 1
            named_entities.append(Entity(ent_type, start_offset, end_offset))

            # start of a new entity
            ent_type = token_tag[2:]
            start_offset = offset
            end_offset = None

    # catches an entity that goes up until the last token
    if ent_type is not None and start_offset is not None and end_offset is None:
        named_entities.append(Entity(ent_type, start_offset, len(tokens) - 1))

    return named_entities


def sim_ratio(a, b):
    return SequenceMatcher(None, a, b).ratio()


def compute_metrics(true_named_entities, pred_named_entities, tags):
    eval_metrics = {
        "correct": 0,
        "incorrect": 0,
        "partial": 0,
        "missed": 0,
        "spurious": 0,
        "precision": 0,
        "recall": 0,
        "f1": 0,
    }

    # overall results
    evaluation = {
        "strict": deepcopy(eval_metrics),
        "strict_weak": deepcopy(eval_metrics),
        "ent_type": deepcopy(eval_metrics),
        "ent_type_weighted": deepcopy(eval_metrics),
        "exact": deepcopy(eval_metrics),
    }

    # results by entity type
    evaluation_agg_entities_type = {e: deepcopy(evaluation) for e in tags}

    # Subset into only the tags that we are interested in.
    # NOTE: we remove the tags we don't want from both the predicted and the
    # true entities. This covers the two cases where mismatches can occur:
    #
    # 1) Where the model predicts a tag that is not present in the true data
    # 2) Where there is a tag in the true data that the model is not capable of
    # predicting.
    true_named_entities = [ent for ent in true_named_entities if ent.e_type in tags]
    pred_named_entities = [ent for ent in pred_named_entities if ent.e_type in tags]

    # keep track of entities that overlapped
    true_which_overlapped_with_pred = []

    # go through each predicted named-entity
    for pred in pred_named_entities:
        found_overlap = False
        # Check each of the potential scenarios in turn. See
        # http://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/
        # for scenario explanation.

        # Scenario I: Exact match between true and pred
        if pred in true_named_entities:
            true_which_overlapped_with_pred.append(pred)

            evaluation["strict"]["correct"] += 1
            evaluation["strict_weak"]["correct"] += 1
            evaluation["exact"]["correct"] += 1
            evaluation["ent_type"]["correct"] += 1
            evaluation["ent_type_weighted"]["correct"] += 1

            # for the agg. by e_type results
            evaluation_agg_entities_type[pred.e_type]["strict"]["correct"] += 1
            evaluation_agg_entities_type[pred.e_type]["strict_weak"]["correct"] += 1
            evaluation_agg_entities_type[pred.e_type]["exact"]["correct"] += 1
            evaluation_agg_entities_type[pred.e_type]["ent_type"]["correct"] += 1
            evaluation_agg_entities_type[pred.e_type]["ent_type_weighted"][
                "correct"
            ] += 1
        else:
            # check for overlaps with any of the true entities
            for true in true_named_entities:
                pred_range = range(pred.start_offset, pred.end_offset)
                true_range = range(true.start_offset, true.end_offset)

                # Scenario IV: Offsets match, but entity type is wrong
                if (
                    true.start_offset == pred.start_offset
                    and pred.end_offset == true.end_offset
                    and true.e_type != pred.e_type
                ):

                    # overall results
                    evaluation["strict"]["incorrect"] += 1
                    evaluation["exact"]["correct"] += 1
                    if pred.e_type == "Artist_or_WoA":
                        evaluation["strict_weak"]["partial"] += 1
                        evaluation["ent_type"]["partial"] += 1
                        evaluation["ent_type_weighted"]["partial"] += 1

                    else:
                        evaluation["strict_weak"]["partial"] += 1
                        evaluation["ent_type"]["incorrect"] += 1
                        evaluation["ent_type_weighted"]["incorrect"] += 1

                    # aggregated by entity type results
                    evaluation_agg_entities_type[true.e_type]["strict"][
                        "incorrect"
                    ] += 1
                    evaluation_agg_entities_type[true.e_type]["exact"]["correct"] += 1
                    if pred.e_type == "Artist_or_WoA":
                        evaluation_agg_entities_type[true.e_type]["strict_weak"][
                            "partial"
                        ] += 1
                        evaluation_agg_entities_type[true.e_type]["ent_type"][
                            "partial"
                        ] += 1
                        evaluation_agg_entities_type[true.e_type]["ent_type_weighted"][
                            "partial"
                        ] += 1
                    else:
                        evaluation_agg_entities_type[true.e_type]["strict_weak"][
                            "incorrect"
                        ] += 1
                        evaluation_agg_entities_type[true.e_type]["ent_type"][
                            "incorrect"
                        ] += 1
                        evaluation_agg_entities_type[true.e_type]["ent_type_weighted"][
                            "incorrect"
                        ] += 1

                    true_which_overlapped_with_pred.append(true)
                    found_overlap = True
                    break

                # check for an overlap i.e. not exact boundary match, with true entities
                elif find_overlap(true_range, pred_range) and (
                    pred.e_type == true.e_type or pred.e_type == "Artist_or_WoA"
                ):
                    # Make sure not to count this true entity twice
                    # This could happen if for this true entity there are multiple predictions that overlap
                    if true in true_which_overlapped_with_pred:
                        # print(true, pred)
                        continue

                    # overall results
                    evaluation["strict"]["incorrect"] += 1
                    evaluation["strict_weak"]["incorrect"] += 1
                    evaluation["exact"]["incorrect"] += 1

                    if pred.e_type == true.e_type:
                        evaluation["ent_type"]["correct"] += 1
                        evaluation["ent_type_weighted"]["correct"] += sim_ratio(
                            true_range, pred_range
                        )
                    elif pred.e_type == "Artist_or_WoA":
                        evaluation["ent_type"]["partial"] += 1
                        evaluation["ent_type_weighted"]["partial"] += 1

                    # aggregated by entity type results
                    evaluation_agg_entities_type[true.e_type]["strict"][
                        "incorrect"
                    ] += 1
                    evaluation_agg_entities_type[true.e_type]["strict_weak"][
                        "incorrect"
                    ] += 1
                    evaluation_agg_entities_type[true.e_type]["exact"]["incorrect"] += 1

                    if pred.e_type == true.e_type:
                        evaluation_agg_entities_type[true.e_type]["ent_type"][
                            "correct"
                        ] += 1
                        evaluation_agg_entities_type[true.e_type]["ent_type_weighted"][
                            "correct"
                        ] += sim_ratio(true_range, pred_range)
                    elif pred.e_type == "Artist_or_WoA":
                        evaluation_agg_entities_type[true.e_type]["ent_type"][
                            "partial"
                        ] += 1
                        evaluation_agg_entities_type[true.e_type]["ent_type_weighted"][
                            "partial"
                        ] += 1

                    true_which_overlapped_with_pred.append(true)
                    found_overlap = True
                    break

            if not found_overlap:
                # Overall results
                evaluation["strict"]["spurious"] += 1
                evaluation["strict_weak"]["spurious"] += 1
                evaluation["ent_type"]["spurious"] += 1
                evaluation["ent_type_weighted"]["spurious"] += 1
                evaluation["exact"]["spurious"] += 1

                # Aggregated by entity type results
                ptype = pred.e_type
                evaluation_agg_entities_type[ptype]["strict"]["spurious"] += 1
                evaluation_agg_entities_type[ptype]["strict_weak"]["spurious"] += 1
                evaluation_agg_entities_type[ptype]["ent_type"]["spurious"] += 1
                evaluation_agg_entities_type[ptype]["ent_type_weighted"][
                    "spurious"
                ] += 1
                evaluation_agg_entities_type[ptype]["exact"]["spurious"] += 1

    # Scenario III: Entity was missed entirely.
    for true in true_named_entities:
        # if true not in pred_named_entities:
        if true in true_which_overlapped_with_pred:
            continue
        else:
            # overall results
            evaluation["strict"]["missed"] += 1
            evaluation["strict_weak"]["missed"] += 1
            evaluation["ent_type"]["missed"] += 1
            evaluation["ent_type_weighted"]["missed"] += 1
            evaluation["exact"]["missed"] += 1

            # for the agg. by e_type
            evaluation_agg_entities_type[true.e_type]["strict"]["missed"] += 1
            evaluation_agg_entities_type[true.e_type]["strict_weak"]["missed"] += 1
            evaluation_agg_entities_type[true.e_type]["ent_type"]["missed"] += 1
            evaluation_agg_entities_type[true.e_type]["ent_type_weighted"][
                "missed"
            ] += 1
            evaluation_agg_entities_type[true.e_type]["exact"]["missed"] += 1

    # Compute 'possible', 'actual' according to SemEval-2013 Task 9.1 on the
    # overall results, and use these to calculate precision and recall.
    for eval_type in evaluation:
        evaluation[eval_type] = compute_actual_possible(evaluation[eval_type])

    # Compute 'possible', 'actual', and precision and recall on entity level
    # results. Start by cycling through the accumulated results.
    for entity_type, entity_level in evaluation_agg_entities_type.items():
        # Cycle through the evaluation types for each dict containing entity
        # level results.
        for eval_type in entity_level:
            evaluation_agg_entities_type[entity_type][
                eval_type
            ] = compute_actual_possible(entity_level[eval_type])
    return evaluation, evaluation_agg_entities_type


def find_overlap(true_range, pred_range):
    """Find the overlap between two ranges
    Find the overlap between two ranges. Return the overlapping values if
    present, else return an empty set().
    Examples:
    >>> find_overlap((1, 2), (2, 3))
    2
    >>> find_overlap((1, 2), (3, 4))
    set()
    """
    true_set = set(true_range)
    pred_set = set(pred_range)
    overlaps = true_set.intersection(pred_set)
    return overlaps


def compute_actual_possible(results):
    """
    Takes a result dict that has been output by compute metrics.
    Returns the results dict with actual, possible populated.

    When the results dicts is from partial or ent_type metrics, then
    partial_or_type=True to ensure the right calculation is used for
    calculating precision and recall.
    """
    correct = results["correct"]
    incorrect = results["incorrect"]
    partial = results["partial"]
    missed = results["missed"]
    spurious = results["spurious"]
    # Possible: number annotations in the gold-standard which contribute to the
    # final score
    possible = correct + incorrect + missed + partial
    # Actual: number of annotations produced by the NER system
    actual = correct + incorrect + spurious + partial
    results["actual"] = actual
    results["possible"] = possible
    return results


def compute_precision_recall(results):
    """
    Takes a result dict that has been output by compute metrics.
    Returns the results dict with precison and recall populated.

    When the results dicts is from partial or ent_type metrics, then
    partial_or_type=True to ensure the right calculation is used for
    calculating precision and recall.
    """
    for key, value in results.items():
        actual = value["actual"]
        possible = value["possible"]
        correct = value["correct"]
        partial = value["partial"]
        results[key]["precision"] = (
            (correct + 0.5 * partial) / actual if actual > 0 else 0
        )
        results[key]["recall"] = (
            (correct + 0.5 * partial) / possible if possible > 0 else 0
        )
        if results[key]["precision"] + results[key]["recall"] == 0:
            results[key]["f1"]
        else:
            results[key]["f1"] = (
                2
                * results[key]["precision"]
                * results[key]["recall"]
                / (results[key]["precision"] + results[key]["recall"])
            )
    return results

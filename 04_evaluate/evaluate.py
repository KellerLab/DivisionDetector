from __future__ import print_function
from scipy.optimize import linear_sum_assignment
import json
import sys
import numpy as np

# maximal distance in world units to consider a detection to match with a
# ground-truth annotation
matching_threshold = 100

def create_matching_costs(rec_divisions, gt_divisions):

    num_recs = len(rec_divisions)
    num_gts = len(gt_divisions)

    rec_labels = {
        i: l
        for i, l in enumerate(rec_divisions.keys())
    }
    gt_labels = {
        i: l
        for i, l in enumerate(gt_divisions.keys())
    }

    rec_locations = np.array(
        [
            rec_divisions[l]['center']
            for l in rec_divisions.keys()
        ], dtype=np.float32)
    gt_locations = np.array(
        [
            gt_divisions[l]['center']
            for l in gt_divisions.keys()
        ], dtype=np.float32)

    print(rec_locations)
    print(gt_locations)

    print("Computing matching costs...")
    matching_costs = np.zeros((num_recs, num_gts), dtype=np.float32)

    for i in range(num_recs):
        for j in range(num_gts):

            distance = np.linalg.norm(rec_locations[i] - gt_locations[j])

            # ensure that pairs exceeding the matching threshold are not
            # considered during the matching by giving them a very high cost
            if distance > matching_threshold:
                distance = 100*matching_threshold

            matching_costs[i][j] = distance

    return matching_costs, rec_labels, gt_labels

def evaluate(rec_divisions, gt_divisions):

    costs, rec_labels, gt_labels = create_matching_costs(
        rec_divisions,
        gt_divisions)

    if rec_divisions:

        print("Finding cost-minimal matches...")
        matches = linear_sum_assignment(costs - np.amax(costs) - 1)
        matches = zip(matches[0], matches[1])

    else:

        matches = []

    filtered_matches = [
        (i,j, costs[i][j])
        for (i,j) in matches
        if costs[i][j] <= matching_threshold
    ]

    print("%d matches found"%len(filtered_matches))

    # unmatched in rec = FP
    fp = len(rec_divisions) - len(filtered_matches)

    # unmatched in gt = FN
    fn = len(gt_divisions) - len(filtered_matches)

    # matched = TP
    tp = len(filtered_matches)

    # all positives
    n = len(gt_divisions)
    assert tp + fn == n

    precision = float(tp)/(tp + fp) if tp + fp > 0 else 0.0
    recall = float(tp)/(tp + fn) if tp + fn > 0 else 0.0
    if precision + recall > 0:
        fscore = 2.0*precision*recall/(precision + recall)
    else:
        fscore = 0

    return (precision, recall, fscore, fp, fn, n)

if __name__ == "__main__":

    rec_file = sys.argv[1]
    gt_file = sys.argv[2]
    frame = int(sys.argv[3])

    print("Evaluating frame %d"%frame)

    with open(rec_file, 'r') as f:
        rec_divisions = json.load(f)['divisions']

    print("Read %d rec divisions"%len(rec_divisions))

    gt_divisions = {}
    gt_label = 1
    for line in open(gt_file, 'r'):

        tokens = line.split()
        if int(round(float(tokens[0]))) == frame:
            center = tuple(float(x) for x in tokens[1:])
            gt_divisions[gt_label] = {
                'center': center
            }
            gt_label += 1

    print("Read %d GT divisions"%len(gt_divisions))

    precision, recall, fscore, fp, fn, n = evaluate(rec_divisions, gt_divisions)

    result = {
        'divisions': rec_divisions,
        'scores': {
            'precision': precision,
            'recall': recall,
            'f-score': fscore,
            'fp': fp,
            'fn': fn,
            'n': n
        }
    }

    with open(rec_file, 'w') as f:
        json.dump(result, f)

    print("precision: %f"%precision)
    print("recall   : %f"%recall)
    print("f-score  : %f"%fscore)
    print("FPs      : %d"%fp)
    print("FNs      : %d"%fn)

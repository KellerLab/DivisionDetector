from __future__ import print_function
from scipy.optimize import linear_sum_assignment
import json
import sys
import numpy as np

# maximal distance in world units to consider a detection to match with a
# ground-truth annotation
#
# setting it too high results in non-divisions being counted as false positives
# setting it too low results in divisions being counted as false negatives
#
# 15 voxels (3 in z) seems to be a good radius that captures the area of a
# single cell
matching_threshold = 15

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

def evaluate(rec_divisions, gt_divisions, gt_nondivisions):

    # create a dictionary of all annotations
    gt_annotations = dict(gt_divisions)
    gt_annotations.update(gt_nondivisions)

    assert len(gt_annotations) == len(gt_divisions) + len(gt_nondivisions), (
        "divisions and non-divisions should not share label IDs")

    # create matching costs to all annotations
    costs, rec_labels, gt_labels = create_matching_costs(
        rec_divisions,
        gt_annotations)

    if rec_divisions:

        print("Finding cost-minimal matches...")
        matches = linear_sum_assignment(costs - np.amax(costs) - 1)
        matches = zip(matches[0], matches[1])

    else:

        matches = []

    filtered_matches = [
        (rec_labels[i], gt_labels[j], costs[i][j])
        for (i,j) in matches
        if costs[i][j] <= matching_threshold
    ]
    matched_rec_annotations = [ f[0] for f in filtered_matches ]
    matched_gt_annotations = [ f[1] for f in filtered_matches ]

    print("%d matches found"%len(filtered_matches))
    print(filtered_matches)

    # matched GT non-division = FP
    fps = [ l for l in gt_nondivisions.keys() if l in matched_gt_annotations ]
    fp = len(fps)

    # unmatched GT division = FN
    fns = [ l for l in gt_divisions.keys() if l not in matched_gt_annotations ]
    fn = len(fns)

    # matched GT divisions = TP
    tps = [ l for l in gt_divisions.keys() if l in matched_gt_annotations ]
    tp = len(tps)

    # unmatched GT non-divisions = TN
    tns = [ l for l in gt_nondivisions.keys() if l not in matched_gt_annotations ]
    tn = len(tns)

    # all positives
    n = len(gt_divisions)
    assert tp + fn == n

    # all negatives
    m = len(gt_nondivisions)
    assert tn + fp == m

    precision = float(tp)/(tp + fp) if tp + fp > 0 else 0.0
    recall = float(tp)/(tp + fn) if tp + fn > 0 else 0.0
    if precision + recall > 0:
        fscore = 2.0*precision*recall/(precision + recall)
    else:
        fscore = 0

    return (precision, recall, fscore, fp, fn, tp, tn, fps, fns, tps, tns)

if __name__ == "__main__":

    rec_file = sys.argv[1]
    benchmark_file = sys.argv[2]
    if len(sys.argv) > 3:
        outfile = sys.argv[3]
    else:
        outfile = rec_file

    with open(rec_file, 'r') as f:
        rec = json.load(f)
    rec_divisions = { int(l): div for (l, div) in rec['divisions'].items() }

    print("Read %d rec divisions"%len(rec_divisions))

    benchmark = json.load(open(benchmark_file, 'r'))
    gt_divisions = benchmark['divisions']
    gt_nondivisions = benchmark['non_divisions']

    print("Read %d GT divisions"%len(gt_divisions))
    print("Read %d GT non-divisions"%len(gt_nondivisions))

    precision, recall, fscore, fp, fn, tp, tn, fps, fns, tps, tns = evaluate(
        rec_divisions,
        gt_divisions,
        gt_nondivisions)

    rec.update({
        'scores': {
            'precision': precision,
            'recall': recall,
            'f-score': fscore,
            'fp': fp,
            'fn': fn,
            'tp': tp,
            'tn': tn,
            'fps': fps,
            'fns': fns,
            'tps': tps,
            'tns': tns,
            'num_divs': len(gt_divisions),
            'num_nondivs': len(gt_nondivisions)
        },
        'evaluation_method': 'selected_points',
        'matching_threshold': matching_threshold
    })

    with open(outfile, 'w') as f:
        json.dump(rec, f, indent=2)

    print("precision: %f"%precision)
    print("recall   : %f"%recall)
    print("f-score  : %f"%fscore)
    print("FPs      : %d"%fp)
    print("FNs      : %d"%fn)

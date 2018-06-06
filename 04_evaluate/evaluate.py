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
matching_threshold = 20

#maximal distance in frames to consider a detection to match with a ground-truth annotation
time_matching_threshold = 2

def create_matching_costs(rec_divisions, gt_divisions, prefer_high_scores=False):

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
    rec_scores = np.array(
        [
            rec_divisions[l]['score']
            for l in rec_divisions.keys()
        ], dtype=np.float32)
    gt_locations = np.array(
        [
            gt_divisions[l]['center']
            for l in gt_divisions.keys()
        ], dtype=np.float32)

    print("Computing matching costs...")
    matching_costs = np.zeros((num_recs, num_gts), dtype=np.float32)

    for i in range(num_recs):
        for j in range(num_gts):

            distance = np.linalg.norm(rec_locations[i] - gt_locations[j])

            # ensure that pairs exceeding the matching threshold are not
            # considered during the matching by giving them a very high cost
            if distance > matching_threshold:
                distance = 100*matching_threshold

            if prefer_high_scores:
                matching_costs[i][j] = 1.0 - rec_scores[i]
            else:
                matching_costs[i][j] = distance

    return matching_costs, rec_labels, gt_labels

def match(rec, gt, prefer_high_scores=False):
    '''Match reconstructions to ground-truth. Returns filtered_matches, a list
    of tuples (label_rec, label_gt, cost) and lists of matched rec labels and
    matched gt labels.'''

    costs, rec_labels, gt_labels = create_matching_costs(
        rec,
        gt)

    if rec:

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

    return filtered_matches, matched_rec_annotations, matched_gt_annotations

def evaluate_threshold(
        threshold,
        rec_divisions,
        gt_divisions,
        gt_nondivisions,
        method):

    rec_divisions = {
        l: div
        for l, div in rec_divisions.items()
        if div['score'] >= threshold
    }

    if method == 'selected_points':

        # create a dictionary of all annotations
        gt_annotations = dict(gt_divisions)
        gt_annotations.update(gt_nondivisions)

        assert len(gt_annotations) == len(gt_divisions) + len(gt_nondivisions), (
            "divisions and non-divisions should not share label IDs")

        # match reconstruction to both divisions and non-divisions
        filtered_matches, matched_rec_annotations, matched_gt_annotations = match(
            rec_divisions,
            gt_annotations)

    elif method == 'selected_divisions':

        # match reconstruction only to divisions
        filtered_matches, matched_rec_annotations, matched_gt_annotations = match(
            rec_divisions,
            gt_divisions)

    if method == 'selected_points':
        # matched GT non-division = FP
        fps = [ l for l in gt_nondivisions.keys() if l in matched_gt_annotations ]
    elif method == 'selected_divisions':
        # unmatched REC divisions = FP
        #we only consider FP in frame = 360, not the other frames
        fps = [ l for l in rec_divisions.keys() if (l not in matched_rec_annotations and (str(l)[-3:])=="360")]
    fp = len(fps)

    # unmatched GT division = FN
    fns = [ l for l in gt_divisions.keys() if l not in matched_gt_annotations ]
    fn = len(fns)

    # matched GT divisions = TP
    tps = [ l for l in gt_divisions.keys() if l in matched_gt_annotations ]
    tp = len(tps)

    if method == 'selected_points':
        # unmatched GT non-divisions = TN
        tns = [ l for l in gt_nondivisions.keys() if l not in matched_gt_annotations ]
        tn = len(tns)
    elif method == 'selected_divisions':
        # TNs can not be counted (this is where this method is only an
        # approximation)
        tns = []
        tn = np.nan

    # all positives
    n = len(gt_divisions)
    assert tp + fn == n

    if method == 'selected_points':
        # all negatives
        m = len(gt_nondivisions)
        assert tn + fp == m

    precision = float(tp)/(tp + fp) if tp + fp > 0 else 0.0
    recall = float(tp)/(tp + fn) if tp + fn > 0 else 0.0
    if precision + recall > 0:
        fscore = 2.0*precision*recall/(precision + recall)
    else:
        fscore = 0

    return {
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
        'tns': tns
    }

def evaluate(rec_divisions, gt_divisions, gt_nondivisions, method):

    if method == 'selected_divisions':

        # get high-score matches between rec and gt_divisions to determine thresholds
        _, matched_rec_annotations, _ = match(
            rec_divisions,
            gt_divisions,
            prefer_high_scores=True)

    if method == 'selected_points':

        # also include thresholds for non-divisions
        gt_annotations = dict(gt_divisions)
        gt_annotations.update(gt_nondivisions)

        _, matched_rec_annotations, _ = match(
            rec_divisions,
            gt_annotations)

    thresholds = np.array(sorted([
        rec_divisions[l]['score']
        for l in matched_rec_annotations
    ]))
    print("Evaluating thresholds %s"%thresholds)

    result_rows = []
    for threshold in thresholds:

        result_row = evaluate_threshold(
            threshold,
            rec_divisions,
            gt_divisions,
            gt_nondivisions,
            method)
        result_row['threshold'] = threshold

        result_rows.append(result_row)

    return result_rows


if __name__ == "__main__":
    #set folder with rec_division files
    rec_file_folder= sys.argv[1]    
    benchmark_file = sys.argv[2]
    method = sys.argv[3]
    #in case different contexts are in the same folder
    context = sys.argv[4]
    assert method in ['selected_points', 'selected_divisions']
    if len(sys.argv) > 5:
        outfile = sys.argv[5]
    else:
        
        outfile = rec_file_folder+'/140521_f=360_c='+context+ '_scores.json'

    rec_file_list = []
    for i in range(time_matching_threshold,-(time_matching_threshold+1),-1):
        j = 360 + i
        #print(j)
        part = "/140521_f=%i_c="%j
        name = rec_file_folder+part+context+".json"
        rec_file_list.append(name)
    #to open rec_file for 360, set time_matching_threshold = 0: this would be equivalent to matching_threshold = [20,20,4,1]
    rec_divisions = []
    for i in range(2*t_matching_threshold+1):
        with open(rec_file_list[i], 'r') as f:
            r = json.load(f)
        frame = i+360-t_matching_threshold
        rec_division = {
            int(l): [div, frame] 
                for (l, div) in r['divisions'].items()
        }
        rec_divisions.append(rec_division)
    #now avoid the list and use a single rec_divisions dictionary

    rec_divisions = {
        (int(str(l)+str(c[1]))): c[0]
        for element in rec_divisions
        for l,c in element.items()
    }
    print(rec_divisions.keys())
    #print("Read %d rec divisions"%len(rec_divisions)); does not make much sense right now
    #print(rec)
    #print(total)
    benchmark = json.load(open(benchmark_file, 'r'))
    gt_divisions = benchmark['divisions']
    gt_nondivisions = benchmark.get('non_divisions', {})

    print("Read %d GT divisions"%len(gt_divisions))
    print("Read %d GT non-divisions"%len(gt_nondivisions))

    result_rows = evaluate(
        rec_divisions,
        gt_divisions,
        gt_nondivisions,
        method)

    rec = {
        'scores': {
            key: [ row[key] for row in result_rows ]
            for key in [
                'threshold',
                'precision',
                'recall',
                'f-score',
                'fp',
                'fn',
                'tp',
                'tn',
                'fps',
                'fns',
                'tps',
                'tns',
            ]
        },
        'evaluation': {
            'evaluation_method': method,
            'matching_threshold': matching_threshold,
            'num_divs': len(gt_divisions),
            'num_nondivs': len(gt_nondivisions)
        }
    }
    # don't store divisions, files get too big otherwise
    #del rec['divisions']

    with open(outfile, 'w') as f:
        json.dump(rec, f, indent=2)

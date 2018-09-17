#Script that takes divisions and non-divisions from frame 360 and puts them into a single JSON file
#for use in evaluation

import json

for frame in [360]:

    non_divisions = {
        int(k): {
            'center': v
        }
        for k, v in json.load(
            open('random_points_t=%d_non-divisions.json'%frame, 'r')).items()
    }

    next_id = max(non_divisions.keys()) + 1

    divisions = {}
    for i, line in enumerate(open('full_divisions_t=%d.txt'%frame, 'r')):
        t = line.split()
        divisions[i + next_id] = {
            'center': ((int(t[0])*5, int(t[1]), int(t[2])))
        }

    benchmark = {
        'divisions': divisions,
        'non_divisions': non_divisions
    }

    with open('test_benchmark_t=%d.json'%frame, 'w') as f:
        json.dump(benchmark, f, indent=2)

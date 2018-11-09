#Script that takes divisions and non-divisions from frame 360 and puts them into a single JSON file
#for use in evaluation

import json

frames = [50, 150, 250]

for frame in frames:

    next_id = 0

    divisions = {}
    for i, line in enumerate(open('140813_DivisionAnnotations_FullVolume.csv', 'r')):
        t = line.split()
        if int(t[0]) != frame:
            continue
        divisions[next_id] = {
            'center': ((int(t[1])*5, int(t[2]), int(t[3])))
        }
        next_id += 1

    benchmark = {
        'divisions': divisions
    }

    with open('test_benchmark_t=%d.json'%frame, 'w') as f:
        json.dump(benchmark, f, indent=2)

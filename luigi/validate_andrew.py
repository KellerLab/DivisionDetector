import luigi

import os
import numpy as np
from tasks import *

if __name__ == '__main__':

    combinations = {
        'experiment': 'DIV',
        'setups': [
            'andrew',
        ],
        'iterations': [0],
        'sample': '140521',
        'frame': 360,
        'downsample': [2, 4, 4],
        'find_divisions_method': 'max_points',
        'find_divisions_method_args': {
            'radius': [20, 20, 20],
            'sigma': [1.0, 1.0, 1.0],
            'min_score_threshold': 1e-4
        },
        'thresholds': list(np.arange(0.0, 1.0, 0.005)),
    }

    range_keys = [
        'setups',
        'iterations',
    ]

    set_base_dir(os.path.abspath('..'))

    luigi.build(
            [EvaluateCombinations(combinations, range_keys)],
            workers=50,
            scheduler_host='slowpoke1.int.janelia.org',
            logging_conf_file='/groups/saalfeld/home/funkej/.luigi/logging.conf'
    )

import luigi

import os
import numpy as np
from tasks import *

if __name__ == '__main__':

    combinations = {
        'experiment': 'DIV',
        'setups': [
            # 'setup06',
            # 'setup07',
            # 'setup08',
            # 'setup09',
            # 'setup10',
            # 'setup11',
            # 'setup12',
            # 'setup13',
            # 'setup14',
            # 'setup15',
            # 'setup16',
            # 'setup17',
            # 'setup18',
            'setup19',
        ],
        # 'iterations': [4000, 10000, 30000, 60000, 100000, 150000, 200000, 250000, 300000],
        'iterations': [300000],
        'sample': '140521',
        'frame': 360,
        'context': 2,
        'downsample': [2, 4, 4],
        'find_divisions_method': 'max_points',
        'find_divisions_method_args': {
            'radius': [5, 20, 20, 20],
            'sigma': [0.5, 1.0, 1.0, 1.0],
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

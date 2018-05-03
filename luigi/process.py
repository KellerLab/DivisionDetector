import luigi

import os
import numpy as np
from tasks import *

if __name__ == '__main__':

    combinations = {
        'experiment': 'DIV',
        'setups': [
            'setup18',
        ],
        'iterations': [150000],
        'sample': '140521',
        'frame': 200,
        'blob_prediction_thresholds': [0.01],
        'thresholds': list(np.arange(0.0, 1.0, 0.01)),
    }

    range_keys = [
        'setups',
        'iterations',
        'blob_prediction_thresholds',
    ]

    set_base_dir(os.path.abspath('..'))

    luigi.build(
            [EvaluateCombinations(combinations, range_keys)],
            workers=50,
            scheduler_host='slowpoke1.int.janelia.org',
            logging_conf_file='/groups/saalfeld/home/funkej/.luigi/logging.conf'
    )

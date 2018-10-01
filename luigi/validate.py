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
            # 'setup19',
            'setup20',
            'setup34',
            #'setup35'
        ],
        'iterations': [100000, 150000, 200000, 250000, 300000],
        'sample': '140521',
        'frame': 360,
        'downsample': [2, 4, 4],
        'radiuss': [[context, 20, 20, 20] for context in range(3)],
        'sigma': [0.5, 1.0, 1.0, 1.0],
        'min_score_threshold': 1e-4,
        'thresholds': list(np.arange(0.0, 1.0, 0.005)),
        'evaluation_method': 'selected_divisions'
    }

    range_keys = [
        'setups',
        'iterations',
        'radiuss'
    ]

    set_base_dir(os.path.abspath('..'))

    luigi.build(
            [EvaluateCombinations(combinations, range_keys)],
            workers=50,
            scheduler_host='slowpoke1.int.janelia.org',
            logging_conf_file='/groups/scicompsoft/home/malinmayorc/.luigi/logging.conf'
    )

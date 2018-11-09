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
            # 'setup20',
	    'setup20checkout',
	        # 'setup20_0404',
	        # 'setup20copy',
	        # 'setup20orig',
            #'setup34',
            #'setup34copy1',
            #'setup34copy2',
            #'setup34copy3',
            #'setup34copy4',
            # 'setup35'
        ],
        'iterations': [100000, 150000, 200000, 250000, 300000],
        'sample': '140813',
        'frames': [50, 150, 250],
        'downsample': [2, 4, 4],
        'radius': [0, 20, 20, 20],
        'sigma': [0.5, 1.0, 1.0, 1.0],
        'min_score_threshold': 1e-4,
        # 'thresholds': list(np.arange(0.0, 1.0, 0.005)),
        'matching_method': 'hungarian'
    }

    range_keys = [
        'setups',
        'iterations',
        'frames'
    ]

    set_base_dir(os.path.abspath('..'))

    luigi.build(
            [EvaluateCombinations(combinations, range_keys)],
            workers=8,
            scheduler_host='slowpoke1.int.janelia.org',
            logging_conf_file='/groups/scicompsoft/home/malinmayorc/.luigi/logging.conf'
    )

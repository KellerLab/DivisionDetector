import luigi

import os
from tasks import *

if __name__ == '__main__':

    combinations = {
        'experiment': 'DIV',
        'setups': [
            'setup06',
            'setup07',
            'setup08',
            'setup09',
            'setup10',
            'setup11',
            'setup12',
            'setup13',
            'setup14',
            'setup15',
            'setup16',
            'setup17',
        ],
        'iterations': [4000, 10000, 30000, 60000, 100000],
        'sample': '140521',
        'frame': 360,
        'thresholds': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
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

import luigi

import os
import numpy as np
from tasks import *

if __name__ == '__main__':

    jobs = [
        ProcessTask(
            experiment='DIV',
            setup='setup18',
            iteration=150000,
            sample='140521',
            frame=frame)
        for frame in range(355, 365 + 1)]

    set_base_dir(os.path.abspath('..'))

    luigi.build(
            jobs,
            workers=50,
            scheduler_host='slowpoke1.int.janelia.org',
            logging_conf_file='/groups/saalfeld/home/funkej/.luigi/logging.conf'
    )

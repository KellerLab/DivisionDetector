import luigi

import os
import sys
import numpy as np
from tasks import *

if __name__ == '__main__':

    setup = sys.argv[1]
    iterations = int(sys.argv[2])

    jobs =[
        TrainTask(
            experiment='DIV',
            setup=setup,
            iteration=iterations)]

    set_base_dir(os.path.abspath('..'))

    luigi.build(
            jobs,
            workers=3,
            scheduler_host='slowpoke1.int.janelia.org',
            logging_conf_file='/groups/scicompsoft/home/malinmayorc/.luigi/logging.conf'
    )
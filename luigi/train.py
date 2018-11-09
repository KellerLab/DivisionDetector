import luigi

import os
import sys
import numpy as np
from tasks import *

if __name__ == '__main__':
    
    if len(sys.argv) < 3:
        print("usage: python train.py <setupA>,<setupB>,... <iterations>")
        sys.exit(1)

    setups = sys.argv[1].split(",")
    iterations = int(sys.argv[2])

    jobs =[
        TrainTask(
            experiment='DIV',
            setup=setup,
            iteration=iterations) for setup in setups]

    set_base_dir(os.path.abspath('..'))

    luigi.build(
            jobs,
            workers=5,
            scheduler_host='slowpoke1.int.janelia.org',
            logging_conf_file='/groups/scicompsoft/home/malinmayorc/.luigi/logging.conf'
    )

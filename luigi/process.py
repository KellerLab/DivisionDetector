import luigi

import os
import sys
from tasks import *

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python process.py <setups> <iterations> <sample> <frames>")
        exit(1)

    setups = sys.argv[1].split(",")
    iterations = sys.argv[2].split(",")
    sample = sys.argv[3]
    frames = sys.argv[4].split(",")

    jobs = [
        ProcessTask(
            experiment='DIV',
            setup=setup,
            iteration=int(iteration),
            sample=sample,
            frame=int(frame))
        for iteration in iterations for setup in setups for frame in frames]

    set_base_dir(os.path.abspath('..'))

    luigi.build(
            jobs,
            workers=8,
            scheduler_host='slowpoke1.int.janelia.org',
            logging_conf_file='/groups/funke/home/funkej/.luigi/logging.conf'
    )

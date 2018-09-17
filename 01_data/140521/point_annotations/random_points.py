#Script for sampling random foreground points from a given file

import pyklb
import json
import random
import sys

foreground_threshold = 196

def sample(filename, n):

    header = pyklb.readheader(filename)
    size = header['imagesize_tczyx'][-3:]

    points = []

    print("Sampling points...")
    while len(points) < n:

        z = random.randint(0, size[0] - 1)
        y = random.randint(0, size[1] - 1)
        x = random.randint(0, size[2] - 1)

        if pyklb.readroi(filename, (z, y, x), (z, y, x)) > foreground_threshold:
            points.append((z, y, x))

    print("Done.")

    return points

if __name__ == "__main__":

    filename = sys.argv[1]
    samples = int(sys.argv[2])
    outfile = sys.argv[3]

    points = sample(filename, samples)

    points = {
        i: [z*5, y, x]
        for i, (z, y, x) in enumerate(points)
    }

    with open(outfile, 'w') as f:
        json.dump(points, f, indent=2)

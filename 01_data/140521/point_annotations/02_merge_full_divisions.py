#Script that consolidates the point annotations for all fully-annotated frames into a single file

frames = [100, 120, 240, 250, 360, 400]

with open('full_divisions.txt', 'w') as f:
    for frame in frames:
        for line in open('full_divisions_t=%d.txt'%frame, 'r'):
            z, y, x = [ int(float(t)) for t in line.split() ]
            f.write('%d\t%d\t%d\t%d\n'%(frame, z, y, x))

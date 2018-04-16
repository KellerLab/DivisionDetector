frames = [100, 250, 400]

for frame in frames:
    divisions = []
    for line in open('full_divisions_t=%d.txt'%frame, 'r'):
        z, y, x = [ float(t) for t in line.split() ]
        divisions.append((z/5, y, x))
    with open('full_divisions_t=%d.txt'%frame, 'w') as f:
        for (z, y, x) in divisions:
            f.write('%d\t%d\t%d\n'%(z, y, x))

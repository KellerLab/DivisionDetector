
divisions = []
non_divisions = []

for line in open('point_annotations.txt', 'r'):

    tokens = line.split()
    t, typ, x, y, z, _ = [ float(a) for a in tokens ]

    if typ in [1, 2, 3, 4, 5, 103]:

        divisions.append((t, z, y, x))

    elif typ in [0, 100]:

        non_divisions.append((t, z, y, x))

    else:
        print("Unknown annotation type %d"%typ)

with open('sparse_divisions_2.txt', 'w') as f:
    for division in sorted(divisions):
        f.write('%d\t %d\t %d\t %d\n'%division)
with open('sparse_non-divisions_2.txt', 'w') as f:
    for non_division in sorted(non_divisions):
        f.write('%d\t %d\t %d\t %d\n'%non_division)

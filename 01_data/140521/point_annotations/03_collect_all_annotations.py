#Script that consolidates all the annotated divisions into one file,
# and all the annotated non-divisions into another file

division_files = ['full_divisions.txt', 'sparse_divisions.txt', 'sparse_divisions_2.txt' ]
non_division_files = ['sparse_non-divisions.txt', 'sparse_non-divisions_2.txt', 'sparse_non-divisions_3.txt']

def collect(filenames):

    print("Collecting annotations from %s"%str(filenames))

    annotations = set()

    for filename in filenames:
        i = 0
        l = len(annotations)
        for line in open(filename, 'r'):
            annotation = tuple( int(float(a)) for a in line.split() )
            annotations.add(annotation)
            i += 1
        print("Collected %d annotations from %s, %d of them new"%(i, filename, len(annotations) - l))

    return annotations

def store(filename, collection):

    collection = sorted(list(collection))
    with open(filename, 'w') as f:
        for c in collection:
            f.write('%d\t%d\t%d\t%d\n'%c)

store('all_divisions.txt', collect(division_files))
store('all_non-divisions.txt', collect(non_division_files))

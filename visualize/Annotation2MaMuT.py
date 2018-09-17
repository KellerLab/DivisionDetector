#!/usr/bin/env/python

import json
import mamut_xml_templates as mamut_temp
import argparse
import os

current_id = 0;

#point: [id, positive, z, x y]

def main():
    parser = argparse.ArgumentParser(description='Create a MaMuT xml file with annotations from multiple json or text files. ' +
                                                 'Must specify time and annotation type for each input file.')
    parser.add_argument('-i', '--input', help='input file (json or txt)', required=True, action='append')
    parser.add_argument('-t', '--time', help='time frame represented in the corresponding input file ' +
                                             '(-1 if txt file contains points from multiple frames)', type = int, required=True, action='append')
    parser.add_argument('-a', '--annotation-type', help='if the annotations in the corresponding input file are positive, negative, or both.' +
                        'Meaningless for json files because information is contained within.',
                        choices=['pos', 'neg', 'both'], required=True, action='append')
    parser.add_argument('-o', '--output', help='output file name. If in the working directory, ./ must be prepended.', required=True)
    parser.add_argument('-bdv', '--path-BDV', help=('path to the xml of the raw data created by loading into big data viewer and saving as xml.'), required=True)
    #parser.add_argument('-f', '--format', help='data format for the ground truth', required=True)
    parser.add_argument('-ani', '--anisotropy', help='scales ground truth points, format z, y, x', nargs=3, required=False, type = float, default=[1.0,1.0,1.0], metavar=('Z', 'Y', 'X'))

    args = parser.parse_args()

    if len(args.input) != len(args.time) != len(args.annotation_type):
        print("Must provide time and annotation type for each input file")
        return

    point_time_dict = {}
    total_points = 0

    for index, file in enumerate(args.input):
        time = args.time[index]
        ann_type = args.annotation_type[index]
        if file.endswith('.json'):
            if time < 0:
                print("Must specify single time frame for json file " + file)
                continue
            if ann_type != 'both':
                print('Ignoring annotation type argument ' + ann_type + "for json file " + file)

            total_points, point_time_dict = read_json(file, point_time_dict, time, total_points)
        elif file.endswith('.txt'):
            if time < 0:
                time = None
            if ann_type == 'both':
                print("Must specify single annotation type for text file " + file)
                continue
            elif ann_type == 'pos':
                total_points, point_time_dict = read_txt(file, point_time_dict, total_points, time=time)
            elif ann_type == 'neg':
                total_points, point_time_dict = read_txt(file, point_time_dict, total_points, positive=False, time=time)

    if total_points == 0:
        print("Error: no points to plot")
        return

    # Make a single artificial spot in consecutive frame so that MaMuT doesn't complain about tracks being null
    extra_time_start = int(point_time_dict.keys()[0])
    extra_time_end = extra_time_start + 1
    extra_point_start = point_time_dict[extra_time_start][0]
    extra_point_end = list(extra_point_start)
    extra_point_end[0] = total_points
    if extra_time_end not in point_time_dict:
        point_time_dict[extra_time_end] = []
    point_time_dict[extra_time_end].append(extra_point_end)
    total_points += 1

    #Write XML Output
    if not os.path.exists(os.path.dirname(args.output)):
        os.makedirs(os.path.dirname(args.output))
    with open(args.output, 'w') as output:

        output.write(mamut_temp.begin_template)

        # Begin AllSpots.
        output.write(mamut_temp.allspots_template.format(nspots=total_points))

        for time in point_time_dict.keys():

            # Begin SpotsInFrame
            output.write(mamut_temp.inframe_template.format(frame=time))
            points = point_time_dict[time]
            for point in points:
                write_annotation(output, time, point, args.anisotropy)

            #End SpotsInFrame
            output.write(mamut_temp.inframe_end_template)

        # End AllSpots.
        output.write(mamut_temp.allspots_end_template)

        # Begin AllTracks.
        output.write(mamut_temp.alltracks_template)

        #Make a single artificial track with a single edge so that MaMuT doesn't complain about null tracks
        output.write(mamut_temp.track_template.format(id=1, duration=2,
                                               start=extra_time_start, stop=extra_time_end, nspots=2,
                                               displacement=0))
        output.write(mamut_temp.edge_template.format(source_id=extra_point_start[0],
                                                     target_id=extra_point_end[0],
                                                     velocity=0, displacement=0,
                                                     t_id=0, time=extra_time_start))
        output.write(mamut_temp.track_end_template)

        # End AllTracks.
        output.write(mamut_temp.alltracks_end_template)

        # Filtered tracks (also just to avoid null error).
        output.write(mamut_temp.filteredtracks_start_template)
        output.write(mamut_temp.filteredtracks_template.format(t_id=1))
        output.write(mamut_temp.filteredtracks_end_template)

        # End XML file.
        filename = os.path.basename(args.path_BDV)
        folder = os.path.dirname(args.path_BDV)
        output.write(mamut_temp.end_template.format(image_data=mamut_temp.im_data_template.format(filename=filename, folder=folder)))


def write_annotation(output, time, point, ani):
    """ Writes an annotation point to xml

    :param output: file to write to
    :param time: the time frame the point is in
    :param point: [id, positive, z, y, x] where positive is true or false
    """
    positive = point[1]
    name = "DIV" if positive else "NEG"
    output.write(mamut_temp.spot_template.format(id=int(point[0]), name=name, frame=time,
                                                 x= float(point[4]) * ani[2],
                                                 y= float(point[3]) * ani[1],
                                                 z= float(point[2]) * ani[0],
                                                 quality=1 if positive else -1))


def read_txt(file, point_time_dict, current_id, positive=True, time=None):
    """ Reads a text file with one point per line into the point_time_dict

    :param file: one point per line, either in format t,z,y,x if time=None, or z,y,x if time is specifiied
    :param point_time_dict: a dictionary of time frame -> point list
    :param current_id:
    :param positive: true if the text file contains positive division annotations, false otherwise
    :param time: if not None, specifies the time frame for all points in file
    :return:
    """
    with open(file, 'r') as gt:
        for line in gt.readlines():
            if time == None:
                point = line.split()
                point_time = int(point[0])
                point[0] = positive
                point.insert(0, current_id)
            else:
                point = [current_id, positive]
                point.extend(line.split())
            assert(len(point) == 5)
            if point_time not in point_time_dict:
                point_time_dict[point_time] = []
            point_time_dict[point_time].append(point)
            current_id += 1
    return (current_id, point_time_dict)

def read_json(file, point_time_dict, time, current_id):
    """ Reads a json file of the format generated by create_test_benchmark.py into point_time_dict

    :param file: the name of the json file
    :param point_time_dict: a dictionary of time frame -> point list
    :param time: the time frame that the json contains points in (not supporting json files with multiple time frames)
    :param total_points: the total number of points before this file is read
    :return: the total number of points in this file
    """

    with open(file) as json_data:
        d = json.load(json_data)
    pos_annotations = d['divisions']
    neg_annotations = d['non_divisions']

    if time not in point_time_dict:
        point_time_dict[time] = []

    positive = True
    for _, cell_dict in pos_annotations.items():
        point = [current_id, positive]
        point.extend(cell_dict['center'])
        assert(len(point) == 5)
        point_time_dict[time].append(point)
        current_id += 1

    positive = False
    for _, cell_dict in neg_annotations.items():
        point = [current_id, positive]
        point.extend(cell_dict['center'])
        assert(len(point) == 5)
        point_time_dict[time].append(point)
        current_id += 1

    return (current_id, point_time_dict)


if __name__ == '__main__':
    main()

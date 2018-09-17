#!/usr/bin/env/python
#takes the output of find_divisions.py in 03_process and visualizes the points in MaMuT
#can also visualize the ground truth for that frame

import json
import mamut_xml_templates as mamut_temp
import argparse
import os

current_id = 0;

#point: [id, score, z, x y]

def main():
    parser = argparse.ArgumentParser(description='Create a MaMuT xml file with annotations from multiple json or text files. ' +
                                                 'Must specify time and annotation type for each input file.')
    parser.add_argument('-p', '--predictions', help='output of find_divisions.py', required=True)
    parser.add_argument('-gt', '--ground-truth', help='ground truth json for the frame (optional)')
    parser.add_argument('-t', '--time', help='time frame of predictions', type = int, required=True)
    parser.add_argument('-o', '--output', help='output file name. If in the working directory, ./ must be prepended.', required=True)
    parser.add_argument('-bdv', '--path-BDV', help=('path to the xml of the raw data created by loading into big data viewer and saving as xml.'), required=True)
    parser.add_argument('-ani_p', '--anisotropy-prediction', help='scales prediction points, format z, y, x', nargs=3, type=float, default=[1.0,1.0,1.0], metavar=('Z', 'Y', 'X'))
    parser.add_argument('-ani_gt', '--anisotropy-ground-truth', help='scales ground truth points, format z, y, x', nargs=3, type=float, default=[1.0,1.0,1.0], metavar=('Z', 'Y', 'X'))

    args = parser.parse_args()

    point_time_dict = {}
    total_points = 0

    total_points, predictions = read_json_predictions(args.predictions, total_points)
    point_time_dict[args.time] = predictions
    if args.ground_truth:
        total_points, ground_truth = read_json_gt(args.ground_truth, total_points)
        point_time_dict[args.time].extend(ground_truth)

    if total_points == 0:
        print("Error: no points to plot")
        return

    # Make a single artificial spot in consecutive frame so that MaMuT doesn't complain about tracks being null
    extra_time_start = args.time
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
                #this process only works under the assumption that predictions will not have scores of 1 or -1
                score = point[1]
                if score == 1 or score == -1:
                    write_annotation(output, time, point, args.anisotropy_ground_truth)
                else:
                    write_annotation(output, time, point, args.anisotropy_prediction)


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


def write_annotation(output, time, point, ani,):
    """ Writes an annotation point to xml

    :param output: file to write to
    :param time: the time frame the point is in
    :param point: [id, positive, z, y, x] where positive is true or false
    """
    #this process only works under the assumption that predictions will not have scores of 1 or -1
    score = point[1]
    name = "PRED"
    if score == 1:
        name = "DIV"
    elif score == -1:
        name = "NEG"
    output.write(mamut_temp.spot_template.format(id=int(point[0]), name=name, frame=time,
                                                 x= float(point[4]) * ani[2],
                                                 y= float(point[3]) * ani[1],
                                                 z= float(point[2]) * ani[0],
                                                 quality=score))

def read_json_gt(file, current_id):
    """ Reads a json file of the format generated by create_test_benchmark.py into point_time_dict
    :param file: the name of the json file
    :param current_id: the total number of points before this file is read
    :return: the total number of points after this file is read and the list of points in the file
    """

    with open(file) as json_data:
        d = json.load(json_data)
    pos_annotations = d['divisions']
    neg_annotations = d['non_divisions']

    points = []

    score = 1
    for _, cell_dict in pos_annotations.items():
        point = [current_id, score]
        point.extend(cell_dict['center'])
        assert(len(point) == 5)
        points.append(point)
        current_id += 1

    score = -1
    for _, cell_dict in neg_annotations.items():
        point = [current_id, score]
        point.extend(cell_dict['center'])
        assert(len(point) == 5)
        points.append(point)
        current_id += 1

    return (current_id, points)

def read_json_predictions(file, current_id):
    """ Reads a json file of the format generated by find_divisions.py and returns a list of points

    :param file: the name of the json file
    :param current_id: the total number of points before this file is read
    :return: the total number of points after this file is read and the list of points in the file
    """

    with open(file) as json_data:
        d = json.load(json_data)
    pos_predictions = d['divisions']

    points = []

    for _, cell_dict in pos_predictions.items():
        score = cell_dict['score']
        point = [current_id, score]
        point.extend(cell_dict['center'])
        assert(len(point) == 5)
        points.append(point)
        current_id += 1

    return (current_id, points)

if __name__ == '__main__':
    main()

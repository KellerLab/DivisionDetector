from IPython.core.display import HTML, display
# import bokeh.charts
import bokeh.layouts
import bokeh.models
import bokeh.palettes
import bokeh.plotting
import copy
import fnmatch
import json
import numpy as np
import os
import pandas
import time

verbose = False

def set_verbose(v):
    global verbose
    verbose = v

eps = 0.000001

configuration_keywords = ['color', 'label', 'style', 'title', 'show_legend', 'legend_columns', 'columns']

colors_tikz = [
    'red!75!black',
    'blue!50!white',
    'black',
    'green!75!black',
    'brown',
    'blue!50!green!75!black',
    'purple!75!black',
]

colors_rgb = [
[1, 0, 103],
[213, 255, 0],
[255, 0, 86],
[158, 0, 142],
[14, 76, 161],
[255, 229, 2],
[0, 95, 57],
[0, 255, 0],
[149, 0, 58],
[255, 147, 126],
[164, 36, 0],
[0, 21, 68],
[145, 208, 203],
[98, 14, 0],
[107, 104, 130],
[0, 0, 255],
[0, 125, 181],
[106, 130, 108],
[0, 174, 126],
[194, 140, 159],
[190, 153, 112],
[0, 143, 156],
[95, 173, 78],
[255, 0, 0],
[255, 0, 246],
[255, 2, 157],
[104, 61, 59],
[255, 116, 163],
[150, 138, 232],
[152, 255, 82],
[167, 87, 64],
[1, 255, 254],
[255, 238, 232],
[254, 137, 0],
[189, 198, 255],
[1, 208, 255],
[187, 136, 0],
[117, 68, 177],
[165, 255, 210],
[255, 166, 254],
[119, 77, 0],
[122, 71, 130],
[38, 52, 0],
[0, 71, 84],
[67, 0, 44],
[181, 0, 255],
[255, 177, 103],
[255, 219, 102],
[144, 251, 146],
[126, 45, 210],
[189, 211, 147],
[229, 111, 254],
[222, 255, 116],
[0, 255, 120],
[0, 155, 255],
[0, 100, 1],
[0, 118, 255],
[133, 169, 0],
[0, 185, 23],
[120, 130, 49],
[0, 255, 198],
[255, 110, 65],
[232, 94, 190]
]

# decorator for filter configurations that allow multiple values
class Any:
    def __init__(self, values):
        self.values = values

def fetch_data_frame_from_json(rows_generator, data_dir='processed'):

    start = time.time()

    print("Collecting record files...")
    record_files = []
    for root, _, filenames in os.walk(data_dir):
        for filename in fnmatch.filter(filenames, '*scores*.json'):
                record_files.append(os.path.join(root, filename))
    print("Found %d record files"%len(record_files))
    if verbose:
        print("Found record files:")
        for f in record_files:
            print('\t' + f)

    print("Parsing record files...")
    all_records = []
    for i, f in enumerate(record_files):

        if i%10 == 0:
            print("%d/%d"%(i+1, len(record_files)))

        try:
            record = json.load(open(f, 'r'))
        except Exception:
            print("record file " + f + " seems broken (maybe in process?)")
            record = {}
            pass

        if rows_generator is not None:
            records = rows_generator(record, f)
            if records is None:
                continue
        else:
            records = [record]
        all_records += records

    print("Creating data frame...")
    columns = set()
    for r in all_records:
        for k in r:
            columns.add(k)

    data_frame = pandas.DataFrame(index=range(len(all_records)), columns=list(columns))
    for i in range(len(all_records)):
        data_frame.loc[i] = all_records[i]

    print("Read all results in " + str(time.time() - start) + "s")

    return data_frame

def read_all_results(data_backend = 'json_files', rows_generator = None, **kwargs):
    '''Reads all results provided by the given data backend, and returns them
    as a panda data frame with at least the following columns:

        voi_split, voi_merge, voi_sum, rand_split, rand_merge, arand, cremi_score

    More columns might be present to describe the configuration that produced
    the results. There are no assumptions about the number, names, or datatypes
    of these configuration columns. Their presence depends entirely on the data
    backend from where the results are read.

    The columns

        voi_sum, arand, cremi_score

    will be computed from {voi,rand}_{split,merge} and should not be provided
    by the data backend.

    Available data backends:

        'json_files'

            Parses all '*.json' files under the 'processed' directory
            (including subdirectories). Each file is supposed to contain one
            row of the data frame as a dictionary, with column names as
            dictionary keys.

    You can pass an optional 'rows_generator', which you can use to overwrite 
    the way rows are created. The column generator returns a dictionary or None 
    if the row should be skipped. The keys and values will be added as one row 
    to the record. For the 'json_files' backend, the arguments passed to the 
    rows_generator are the orginal record (as a dict) and the filename of the 
    json file.
    '''

    if data_backend == 'json_files':
        data_frame = fetch_data_frame_from_json(rows_generator, **kwargs)
    else:
        raise RuntimeError("no such data backend '" + str(data_backend) + "'")

    try:
        ts = data_frame['ted_split']
        tm = data_frame['ted_merge']
        tp = data_frame['ted_fp']
        tn = data_frame['ted_fn']
        data_frame['ted_sum'] = np.array(ts+tm+tp+tn, dtype=np.float32)
    except:
        pass

    try:
        vs = data_frame['voi_split']
        vm = data_frame['voi_merge']
        data_frame['voi_sum'] = np.array(vs + vm, dtype=np.float32)
    except:
        pass

    try:
        vs = data_frame['syn_voi_split']
        vm = data_frame['syn_voi_merge']
        data_frame['syn_voi_sum'] = np.array(vs + vm, dtype=np.float32)
    except:
        pass

    try:
        rs = data_frame['rand_split']
        rm = data_frame['rand_merge']
        data_frame['arand'] = np.array(1.0 - (2.0*rs*rm)/(rs+rm), dtype=np.float32)
        data_frame['cremi_score'] = np.sqrt(np.array(data_frame['voi_sum']*data_frame['arand'], dtype=np.float32))
    except:
        pass

    try:
        rs = data_frame['syn_rand_split']
        rm = data_frame['syn_rand_merge']
        data_frame['syn_arand'] = np.array(1.0 - (2.0*rs*rm)/(rs+rm), dtype=np.float32)
    except:
        pass

    return data_frame

def matches(record, configuration):

    # symbolic group 'average' matches all
    if isinstance(configuration, str) and configuration == 'average':
        return True

    for key in configuration.keys():

        # skip non-record fields
        if key in configuration_keywords:
            continue

        if key not in record:
            return False

        allowed_values = configuration[key]
        if not isinstance(allowed_values, list):
            allowed_values = [allowed_values]

        if record[key] not in allowed_values:
            return False

    return True

def key_value_to_str(d, k):
    if isinstance(d[k], str):
        return d[k]
    return str(k)[:2] + ":" + str(d[k])

def get_configuration_label(configuration):

    if 'label' in configuration:
        return configuration['label']

    label = ""

    keys = sorted(list(configuration.keys()))
    keys = [ k for k in keys if not k in configuration_keywords ]

    return ", ".join([key_value_to_str(configuration, k) for k in keys])

def get_title(group):

    if 'title' in group:
        return str(group['title'])
    return str(group)

def filter(records, configurations):

    match_mask = None

    values_queries = []
    for c in configurations:
        for k in c:

            if k in configuration_keywords:
                continue

            possible_values = c[k].values if isinstance(c[k], Any) else [c[k]]

            value_mask = None

            for value in possible_values:
                if value is None or value is np.nan:
                    expression = (pandas.isnull(records[k]))
                else:
                    if isinstance(value, float):
                        expression = ((records[k] >= value - eps) & (records[k] <= value + eps))
                    else:
                        expression = (records[k] == value)
                if value_mask is None:
                    value_mask = expression
                else:
                    value_mask |= expression

            values_queries.append("(" + " or ".join([str(k) + " == " + str(v) for v in possible_values]) + ")")

            if match_mask is None:
                match_mask = value_mask
            else:
                match_mask &= value_mask

    if len(values_queries) == 0:
        return records

    query_string = " and ".join(values_queries)

    global verbose
    if verbose:
        print("Filtering with: " + query_string)

    filtered = records[match_mask]
    return filtered

def smooth(dataframe, columns, factor, downsample=True):

    result = dataframe.copy()
    for column in columns:
        result[column] = np.convolve(result[column], np.ones((factor,))/factor, mode='same')

    if downsample:
        result = result.iloc[factor/2:-factor/2:factor,:]

    return result

def create_figure(title, figure_spec, curves, backend):

    if figure_spec['x_axis'] == 'label':
        if backend == 'bokeh':
            return create_bokeh_ybar_figure(title, figure_spec, curves)
        else:
            return create_tikz_ybar_figure(title, figure_spec, curves)

    if backend == 'bokeh':
        return create_bokeh_xy_figure(title, figure_spec, curves)
    elif backend == 'tikz':
        return create_tikz_xy_figure(title, figure_spec, curves)
    else:
        raise RuntimeError("No such backend " + backend)

def create_bokeh_xy_figure(title, figure_spec, curves):

    # configure the tool-tip to show all keys common to all curves
    all_keys = None
    for curve in curves:
        if all_keys is None:
            all_keys = set(curve['records'].keys())
        else:
            all_keys &= set(curve['records'].keys())
    all_keys = list(all_keys)
    all_keys.sort()
    tooltips="".join([
        "<div><span>%s: @%s</span></div>"%(key,key) for key in all_keys if len(key) < 50
    ])
    hover = bokeh.models.HoverTool(tooltips=tooltips)
    tools = ['save','pan','wheel_zoom','box_zoom','reset']

    # create the bokeh figure
    figure = bokeh.plotting.figure(
            title=title,
            tools=[hover] + tools,
            active_scroll='wheel_zoom',
            x_axis_label=figure_spec['x_label'] if 'x_label' in figure_spec else figure_spec['x_axis'],
            y_axis_label=figure_spec['y_label'] if 'y_label' in figure_spec else figure_spec['y_axis'],
            x_range=figure_spec['x_range'] if 'x_range' in figure_spec else None,
            y_range=figure_spec['y_range'] if 'y_range' in figure_spec else None)

    num_visible_curves = len([ c for c in curves if len(c['records'] > 0) ])

    if num_visible_curves == 0:
        return None

    for curve in curves:

        records = curve['records']
        if len(records) == 0:
            continue

        # bokeh does not handle nan in python notebooks correctly, we 
        # replace them with a string here
        records = records.fillna('nan')

        if 'smooth' in figure_spec and figure_spec['smooth'] > 0:
            k = figure_spec['smooth']
            records = smooth(records, [figure_spec['y_axis']], k)

        source = bokeh.models.ColumnDataSource(bokeh.models.ColumnDataSource.from_df(records))

        draw_function = figure.circle
        draw_args = {}

        color = curve['color']
        if isinstance(color, int):
            color = '#%02x%02x%02x'%tuple(colors_rgb[color%len(colors_rgb)])

        if curve['style'] in ['line', 'dashed', 'dotted']:

            draw_function = figure.line
            draw_args['line_color'] = color
            draw_args['line_width'] = 2
            if curve['style'] in ['dashed', 'dotted']:
                draw_args['line_dash'] = curve['style']

        elif curve['style'] == 'square':

            draw_function = figure.square
            draw_args['color'] = color
            draw_args['size'] = 10

        else:

            draw_function = figure.circle
            draw_args['color'] = color
            draw_args['size'] = 10

        if curve['show_legend'] is not None:
            show_legend = curve['show_legend']
        else:
            show_legend = num_visible_curves < 5

        if show_legend:
            draw_args['legend'] = curve['label']

        draw_function(
                figure_spec['x_axis'],
                figure_spec['y_axis'],
                source=source,
                **draw_args)

    if 'legend_position' in figure_spec and len(figure.legend) > 0:
        figure.legend[0].location = figure_spec['legend_position']

    return figure

def create_bokeh_ybar_figure(title, figure_spec, curves):

    # put all curves into one data frame
    df = None
    for curve in curves:

        if len(curve['records']) != 1:
            if len(curve['records']) > 1:
                print("Ignoring curve " + curve['label'] + " for bar plot, as it has more than one entry (%d)"%len(curve['records']))
            continue

        label = curve['label']
        append = curve['records'].copy()
        append['label'] = [label]
        if df is None:
            df = append
        else:
            df = df.append(append)

    if df is None or len(df) == 0:
        return None

    raise RuntimeError("Currently broken, charts not in bokeh anymore...")
    # figure = bokeh.charts.Bar(
            # df,
            # values=figure_spec['y_axis'],
            # label='label',
            # title=title,
            # active_scroll='wheel_zoom',
            # xlabel="",
            # ylabel=figure_spec['y_label'] if 'y_label' in figure_spec else figure_spec['y_axis'],
            # y_range=figure_spec['y_range'] if 'y_range' in figure_spec else None)

    return figure

def create_tikz_xy_figure(title, figure_spec, curves):

    num_visible_curves = len([ c for c in curves if len(c['records'] > 0) ])

    if num_visible_curves == 0:
        return None

    figure = """
\\begin{{tikzpicture}}
    \\begin{{axis}}[
        ymajorgrids=true,
        xmajorgrids=true,
        width=\\plotwidth,
        height=\\plotheight,
        legend style={{font=\\tiny}},
        legend image post style={{scale=0.5}},
        legend columns={legend_columns},
        xlabel={xlabel},
        ylabel={ylabel},
        xticklabel style={{font=\\tiny,overlay}},
        yticklabel style={{font=\\tiny,overlay}},
        ylabel style={{font=\\tiny,overlay}},
        xlabel style={{font=\\tiny,overlay}},{style}
    ]
        {plots}
    \\end{{axis}}
\\end{{tikzpicture}}
"""

    figure_data = {
        'style': '',
        'xlabel': figure_spec['x_label'] if 'x_label' in figure_spec else figure_spec['x_axis'],
        'ylabel': figure_spec['y_label'] if 'y_label' in figure_spec else figure_spec['y_axis'],
        'plots': '',
        'legend_columns': figure_spec['legend_columns'] if 'legend_columns' in figure_spec else 1
    }

    for a in ['x', 'y']:
        if a+'_range' in figure_spec:

            amin, amax = figure_spec[a+'_range']

            if amin is not None and amax is not None and amin > amax:
                amin, amax = amax, amin
                figure_data['style'] += ','+a+' dir=reverse'
            if amin is not None:
                figure_data['style'] += ','+a+'min=%f'%amin
            if amax is not None:
                figure_data['style'] += ','+a+'max=%f'%amax

        if a+'_mode' in figure_spec:
            figure_data['style'] += ','+a+'mode=%s'%figure_spec[a+'_mode']

    for curve in curves:

        records = curve['records']
        if len(records) == 0:
            continue

        if 'smooth' in figure_spec and figure_spec['smooth'] > 0:
            x, y = smooth(records[figure_spec['x_axis']], records[figure_spec['y_axis']], figure_spec['smooth'])
            source = pandas.DataFrame({figure_spec['x_axis']: x, figure_spec['y_axis']: y})
        else:
            source = records

        plot = """
    \\addplot[{style},{color}]
        coordinates {{
            {coordinates}
        }};
    \\addlegendentry{{ {label} }}
"""

        color = curve['color']
        if isinstance(color, int):
            color = colors_tikz[color%len(colors_tikz)]
        plot_data = {
            'style': '',
            'color': color,
            'coordinates': '',
            'label': curve['label']
        }

        if curve['style'] == 'square':
            plot_data['style'] += ',only marks,mark=square'
        elif curve['style'] == 'circle':
            plot_data['style'] += ',only marks,mark=*'

        x = figure_spec['x_axis']
        y = figure_spec['y_axis']

        for row in source.iterrows():
            plot_data['coordinates'] += '(%f,%f)'%(row[1][x],row[1][y])

        figure_data['plots'] += plot.format(**plot_data)

    return figure.format(**figure_data)

def create_tikz_ybar_figure(title, figure_spec, curves):

    # put all curves into one data frame
    df = None
    for curve in curves:

        if len(curve['records']) != 1:
            if len(curve['records']) > 1:
                print("Ignoring curve " + curve['label'] + " for bar plot, as it has more than one entry (%d)"%len(curve['records']))
            continue

        label = curve['label']
        append = curve['records'].copy()
        append['label'] = [label]
        if df is None:
            df = append
        else:
            df = df.append(append)

    if df is None or len(df) == 0:
        return None

    figure = """
\\begin{{tikzpicture}}
    \\begin{{axis}}[
        symbolic x coords={{{labels}}},
        xtick=data,
        ymajorgrids=true,
        width=\\plotwidth,
        height=\\plotheight,
        legend style={{font=\\tiny}},
        legend image post style={{scale=0.5}},
        legend columns=2,
        ylabel={ylabel},{style},
        xticklabel style={{font=\\tiny,overlay}},
        yticklabel style={{font=\\tiny,overlay}},
        ylabel style={{font=\\tiny,overlay}},
        xlabel style={{font=\\tiny,overlay}},
    ]

        \\addplot[ybar,fill=blue!50!white] coordinates {{
            {coordinates}
        }};

    \\end{{axis}}
\\end{{tikzpicture}}
"""

    figure_data = {
        'style': '',
        'ylabel': figure_spec['y_label'] if 'y_label' in figure_spec else figure_spec['y_axis'],
        'labels': '',
        'coordinates': ''
    }

    if 'y_range' in figure_spec:
        ymin = figure_spec['y_range'][0]
        ymax = figure_spec['y_range'][1]
        if ymin is not None and ymax is not None:
            if ymin > ymax:
                ymin, ymax = ymax, ymin
                figure_data['style'] += ',y dir=reverse'
        if ymin is not None:
            figure_data['style'] += ',ymin=%f'%ymin
        if ymax is not None:
            figure_data['style'] += ',ymax=%f'%ymax

    if 'x_mode' in figure_spec:
        figure_data['style'] += ',xmode=%s'%figure_spec['x_mode']
    if 'y_mode' in figure_spec:
        figure_data['style'] += ',ymode=%s'%figure_spec['y_mode']

    figure_data['labels'] = ','.join(df['label'])
    for label, value in zip(df['label'],df[figure_spec['y_axis']]):
        figure_data['coordinates'] += '(%s,%f)'%(label, value)

    return figure.format(**figure_data)

def create_html_table(title, data, highlight_min=None, highlight_max=None, label_name=''):

    display(HTML('<h3>' + title + '</h3>'))

    if len(data) == 0:
        display(HTML('[empty]'))
        return

    styler = data.style
    if highlight_min is not None:
        styler = styler.apply(pandas_highlight_min, subset=highlight_min)
    if highlight_max is not None:
        styler = styler.apply(pandas_highlight_max, subset=highlight_max)

    display(HTML(styler.render()))

def create_tex_table(title, data, highlight_min=None, highlight_max=None, label_name='', precisions = {}):

    table = """\\rowcolors{{2}}{{gray!2!white}}{{gray!12!white}}
\\begin{{tabular}}{{{format}}}
    {head}
    \\hline
{rows}
\\end{{tabular}}
"""

    num_rows = len(data)
    num_columns = len(data.columns)

    column_indices = { name: index for (index, name) in enumerate(data.columns) }

    values = np.zeros((num_rows, num_columns))
    highlight = np.zeros((num_rows, num_columns))
    i = 0
    for (label, row) in data.iterrows():
        j = 0
        for x in row:
            values[i,j] = x
            j += 1
        i += 1

    if highlight_min is not None:
        for name in highlight_min:
            i = np.argmin(values[:,column_indices[name]])
            highlight[i,column_indices[name]] = 1
    if highlight_max is not None:
        for name in highlight_max:
            i = np.argmax(values[:,column_indices[name]])
            highlight[i,column_indices[name]] = 1

    for column in data.columns:
        if column not in precisions:
            precisions[column] = 3

    format = 'l|' + ''.join([
        'D{.}{.}{'+str(precisions[column])+'}'
        for column in data.columns
    ])

    head = label_name + '&'+'&'.join([format_tabular_cell(x, numeric=False) for x in data.columns])+'\\\\\n'
    rows = '\\\\\n'.join(
            [
                data.index[i] + '&' + \
                    '&'.join([
                        format_tabular_cell(
                            values[i,j],
                            highlight[i,j]==1,
                            precision=precisions[data.columns[j]]
                        )
                        for j in range(num_columns)
                    ])
                for i in range(num_rows)
            ]
    )

    return table.format(**{
        'format':format,
        'num_columns':num_columns+1,
        'title':title,
        'head':head,
        'rows':rows,
    })

def format_tabular_cell(s, highlight=False, numeric=True, precision=None):

    if highlight and not numeric:
        s = '{\\bf ' + s + '}'

    if numeric:
        if precision is None:
            s = '%.3f'%s
        else:
            s = ('%.'+str(precision)+'f')%s
    else:
        s = '\\multicolumn{1}{c}{'+s+'}'

    if highlight and numeric:
        s = '\\multicolumn{1}{B{.}{.}{'+str(precision)+'} }{'+s+'}'

    return s

def escape(s):
    s = s.replace(' ','_')
    s = s.replace('/','_')
    return s

def plot(groups, figures, configurations, all_records, backend='bokeh', output_dir='plots'):
    '''Creates figures from the given records.

    Figures are created for each given group.

    Curves are drawn in the figures for each given configuration.
    '''

    global verbose
    print("Verbose set to: " + str(verbose))

    # configurations are dictionaries with keys like
    #
    #  sample, setup, iteration, threshold
    #
    # (basically everything that can occur in a record)
    #
    # Values are one or more values, e.g.,
    #
    #   'setup': 'setup26',
    #   'threshold': [0,100]
    #
    # We create a curve for every configuration out of all records that have the 
    # given keys set to the same value(s).

    print("Preparing plot data")
    start = time.time()

    # list of curves per group, identified by title
    curves = { get_title(group): [] for group in groups }

    configuration_labels = [ get_configuration_label(c) for c in configurations ]
    configuration_indices = {}
    i = 0
    for l in configuration_labels:
        if l not in configuration_indices:
            configuration_indices[l] = i
            i += 1

    # prepare panda data frames for each curve in each group
    for group in groups:

        if verbose:
            print("processing group " + str(group))

        show_legend_group_default = group['show_legend'] if 'show_legend' in group else None
        style_group_default = group['style'] if 'style' in group else 'circle'

        configuration_num = -1
        for configuration in configurations:
            configuration_num += 1

            if verbose:
                print("adding configuration " + str(configuration))

            filtered_records = filter(all_records, [configuration, group])

            if verbose:
                print("filtered records for " + str(configuration) + ", group " + str(group) + ":")
                print(filtered_records)

            # create curve with meta-data
            curve = {
                'records': filtered_records,
                'label': get_configuration_label(configuration),
                'color': configuration_indices[get_configuration_label(configuration)] if 'color' not in configuration else configuration['color'],
                'style': configuration['style'] if 'style' in configuration else style_group_default,
                'show_legend': configuration['show_legend'] if 'show_legend' in configuration else show_legend_group_default
            }

            curves[get_title(group)].append(curve)

    print("Prepared data in " + str(time.time() - start) + "s")

    # plotting
    start = time.time()

    # create the figures for each group
    all_figures = {}
    for group in groups:

        group_curves = curves[get_title(group)]

        if len(group_curves) == 0:
            print("No record matches group " + get_title(group))
            continue

        group_figures = []
        for figure_spec in figures:

            title = get_title(group) + " " + get_title(figure_spec)
            group_figure = create_figure(title, figure_spec, group_curves, backend)

            if group_figure is not None:
                group_figures.append(group_figure)
                all_figures[title] = group_figure
            else:
                print("Skipping empty figure " + get_title(figure_spec))

        if len(group_figures) > 0:

            if backend == 'bokeh':
                group_grid = bokeh.layouts.gridplot([group_figures], title=get_title(group))
                bokeh.plotting.show(group_grid)
        else:
            print("Skipping empty group " + get_title(group))

    print("Plotted in " + str(time.time() - start) + "s")

    if backend == 'tikz':

        try:
            os.mkdir(output_dir)
        except:
            pass

        for title in all_figures:
            filename = escape(title)
            with open(os.path.join(output_dir, filename + '.tikz.tex'), 'w') as f:
                f.write(all_figures[title])

    return all_figures

def table(tables, configurations, results, groups=None, collapse_configurations=None, backend='html', output_dir='tables', filename=None, **kwargs):
    '''
    If groups is set, multiple tables are rendered, filtered for each group.
    '''

    if groups is None:
        groups = [None]

    for group in groups:
        for table in tables:

            table_data = pandas.DataFrame()
            title = get_title(table)
            if group is not None:
                title += ' ' + get_title(group)

            for configuration in configurations:

                if group is not None:
                    rows = filter(results, [configuration, group])
                else:
                    rows = filter(results, [configuration])
                if len(rows) == 0:
                    continue

                if 'columns' in table:
                    rows = rows[table['columns']]

                    if 'column_names' in table:
                        rows.rename(columns = {old: new for old, new in zip(table['columns'], table['column_names']) }, inplace=True)

                if collapse_configurations is not None:
                    method, column = collapse_configurations
                    rows = rows.sort_values(by=column)
                    if method == 'min':
                        rows = rows.head(1)
                    elif method == 'max':
                        rows = rows.tail(1)
                    else:
                        raise RuntimeError("method for collapsing has to be 'min' or 'max'")

                # this is madness...
                if backend == 'tex':

                    rows.loc[:,'label'] = pandas.Series([get_configuration_label(configuration)]*len(rows), index=rows.index)
                    rows = rows.set_index('label')
                    rows.index.name = None # oh, of course, because otherwise we get an empyt row! m(

                    table_data = table_data.append(rows)

                else:

                    rows.insert(0, 'label', pandas.Series([get_configuration_label(configuration)]*len(rows), index=rows.index))
                    table_data = table_data.append(rows, ignore_index=True)

            if backend == 'tex':

                try:
                    os.mkdir(output_dir)
                except:
                    pass

                if filename is None:
                    filename = escape(title) + '.tex'
                with open(os.path.join(output_dir, filename), 'w') as f:
                    f.write(create_tex_table(title, table_data, **kwargs))
            else:
                create_html_table(title, table_data, **kwargs)


def render_table(data):
    create_html_table("", data)

def pandas_highlight_min(s):
    is_min = s == s.min()
    return ['background-color: yellow' if v else '' for v in is_min]

def pandas_highlight_max(s):
    is_max = s == s.max()
    return ['background-color: yellow' if v else '' for v in is_max]

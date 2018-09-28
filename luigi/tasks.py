import luigi
import os
import itertools
import json
from targets import *
from subprocess import Popen, check_output, CalledProcessError, STDOUT

base_dir = '.'
def set_base_dir(d):
    global base_dir
    base_dir = d

def mkdirs(path):
    try:
        os.makedirs(path)
    except:
        pass

def call(command, log_out, log_err):

    # using Popen directly (seems to sometimes respond to ^C? what a mess!)
    ##############################

    print("Calling %s, logging to %s"%(' '.join(command), log_out))

    with open(log_out, 'w') as stdout:
        with open(log_err, 'w') as stderr:
            process = Popen(command, stdout=stdout, stderr=stderr)
            # wait for output to finish
            try:
                process.communicate()
            except KeyboardInterrupt:
                try:
                    print("Killing process...")
                    process.kill()
                except OSError:
                    pass
                process.wait()
                raise
            if process.returncode != 0:
                raise Exception(
                    "Calling %s failed with code %s, see log in %s, %s"%(
                        ' '.join(command),
                        process.returncode,
                        log_out,
                        log_err))

    # using check_output, output written only at end (^C not working,
    # staircasing)
    ##############################

    # output = ""
    # try:
        # output = check_output(command)
    # except CalledProcessError as e:
        # output = e.output
        # raise Exception("Calling %s failed with recode %s, log in %s"%(
            # ' '.join(command),
            # e.returncode,
            # log_out))
    # finally:
        # with open(log_out, 'w') as stdout:
            # stdout.write(output)

    # using check_call, seems to cause trouble (^C not working, staircasing)
    ##############################

    # try:
        # output = check_call(command, stdout=stdout, stderr=stderr)
    # except CalledProcessError as exc:
        # raise Exception("Calling %s failed with recode %s, stderr in %s"%(
            # ' '.join(command),
            # exc.returncode,
            # stderr.name))

    # return output

class RunTasks(luigi.WrapperTask):
    '''Top-level task to run several tasks.'''

    tasks = luigi.Parameter()

    def requires(self):
        return self.tasks

class MakeNetworkTask(luigi.Task):
    experiment = luigi.Parameter()
    setup = luigi.Parameter()
    name = luigi.Parameter()

    def output_filename(self):
        return os.path.join(
            base_dir,
            '02_train',
            str(self.setup),
            '%s_config.json' % self.name)

    def requires(self):
        return []

    def output(self):
        return FileTarget(self.output_filename())

    def run(self):
        log_base = os.path.join(base_dir, '02_train', str(self.setup), 'mknet_%s' % self.name)
        log_out = log_base + '.out'
        log_err = log_base + '.err'
        os.chdir(os.path.join(base_dir, '02_train', self.setup))
        call([
            'run_docker',
            '-d', 'funkey/division_detection:v0.3',
            'python -u mknet.py ' + self.name
        ], log_out, log_err)

class TrainTask(luigi.Task):

    experiment = luigi.Parameter()
    setup = luigi.Parameter()
    iteration = luigi.IntParameter()

    def output_filename(self):
        return os.path.join(
            base_dir,
            '02_train',
            str(self.setup),
            'train_net_checkpoint_%d.meta'%self.iteration)

    def requires(self):
        if self.iteration == 10000:
            return [MakeNetworkTask(self.experiment, self.setup, "train_net"),
                MakeNetworkTask(self.experiment, self.setup, "test_net") ]
        return TrainTask(self.experiment, self.setup, self.iteration - 10000)

    def output(self):
        return FileTarget(self.output_filename())

    def run(self):
        log_base = os.path.join(base_dir, '02_train', str(self.setup), 'train_%d'%self.iteration)
        log_out = log_base + '.out'
        log_err = log_base + '.err'
        os.chdir(os.path.join(base_dir, '02_train', self.setup))
        call([
            'run_lsf',
            '-c', '10',
            '-g', '1',
            '-d', 'funkey/division_detection:v0.3',
            'python -u train.py ' + str(self.iteration)
        ], log_out, log_err)

class ProcessTask(luigi.Task):

    experiment = luigi.Parameter()
    setup = luigi.Parameter()
    iteration = luigi.IntParameter()
    sample = luigi.Parameter()
    frame = luigi.IntParameter()

    def output_filename(self):
        return os.path.join(
            base_dir,
            '03_process',
            'processed',
            self.setup,
            str(self.iteration),
            '%s_%d.hdf'%(self.sample, self.frame))

    def requires(self):
        return TrainTask(self.experiment, self.setup, self.iteration)

    def output(self):
        return HdfDatasetTarget(self.output_filename(), 'volumes/divisions')

    def run(self):
        mkdirs(os.path.join(
            base_dir,
            '03_process',
            'processed',
            self.setup,
            str(self.iteration)))
        log_base = os.path.join(
            base_dir,
            '03_process',
            'processed',
            self.setup,
            str(self.iteration),
            '%s'%self.sample)
        log_out = log_base + '.out'
        log_err = log_base + '.err'
        os.chdir(os.path.join(base_dir, '03_process'))
        call([
            'run_lsf',
            '-c', '2',
            '-g', '1',
            '-m', '10000',
            '-d', 'funkey/division_detection:v0.3',
            'python -u predict.py ' + \
                    self.setup + ' ' + \
                    str(self.iteration) + ' ' + \
                    self.sample + ' ' + \
                    str(self.frame)
        ], log_out, log_err)

class ConfigTask(luigi.Task):
    '''Base class for FindDivisions and Evaluate.'''

    parameters = luigi.DictParameter()

    def get_setup(self):
        if isinstance(self.parameters['setup'], int):
            return 'setup%02d'%self.parameters['setup']
        return self.parameters['setup']

    def get_iteration(self):
        return self.parameters['iteration']

    def tag(self):

        parameters = dict(self.parameters)

        dict_str = json.dumps(dict(parameters), sort_keys=True)
        tag = str(hash(dict_str))

        return tag

    def output_basename(self):

        basename = self.tag()

        return os.path.join(
                base_dir,
                '03_process',
                'processed',
                self.get_setup(),
                str(self.get_iteration()),
                basename)

class FindDivisions(ConfigTask):

    def requires(self):

        context = self.parameters.get('context', 0)

        return [
            ProcessTask(
                self.parameters['experiment'],
                self.get_setup(),
                self.get_iteration(),
                self.parameters['sample'],
                self.parameters['frame'] + c)
            for c in range(-context, context + 1)]

    def output(self):

        return JsonTarget(
            self.output_basename() + '.json',
            'divisions')

        return targets

    def run(self):

        log_out = self.output_basename() + '.out'
        log_err = self.output_basename() + '.err'

        args = dict(self.parameters)
        args['output_filename'] = self.output_basename() + '.json'
        args['method'] = self.parameters['find_divisions_method']
        args['method_args'] = dict(self.parameters['find_divisions_method_args'])
        del args['find_divisions_method']
        del args['find_divisions_method_args']

        with open(self.output_basename() + '.config', 'w') as f:
            json.dump(args, f)

        os.chdir(os.path.join(base_dir, '03_process'))
        call([
            'run_lsf',
            '-c', '2',
            '-g', '1', # can't ask for 0
            '-m', '100000',
            'python', '-u', 'find_divisions.py',
            self.output_basename() + '.config'
        ], log_out, log_err)

class Evaluate(ConfigTask):

    evaluation_method = luigi.Parameter()

    def requires(self):
        return FindDivisions(self.parameters)

    def outfile(self):
        return self.output_basename() + '_scores_%s.json'%self.evaluation_method

    def output(self):

        return JsonTarget(
            self.outfile(),
            'scores')

    def run(self):

        benchmark_file = os.path.join(
            '../01_data/',
            self.parameters['sample'],
            'point_annotations',
            'test_benchmark_t=%d.json'%self.parameters['frame'])

        log_out = self.output_basename() + '_%s.out'%self.evaluation_method
        log_err = self.output_basename() + '_%s.err'%self.evaluation_method
        res_file = self.output_basename() + '.json'

        os.chdir(os.path.join(base_dir, '04_evaluate'))
        call([
            # 'run_lsf',
            # '-c', '2',
            # '-g', '0',
            # '-m', '10000',
            'python',
            '-u', 'evaluate.py',
            res_file,
            benchmark_file,
            self.evaluation_method,
            self.outfile()
        ], log_out, log_err)

class EvaluateCombinations(luigi.task.WrapperTask):

    # a dictionary containing lists of parameters to evaluate
    parameters = luigi.DictParameter()
    range_keys = luigi.ListParameter()

    def requires(self):

        for k in self.range_keys:
            assert len(k) > 0 and k[-1] == 's', ("Explode keys have to end in "
                                                 "a plural 's'")

        # get all the values to explode
        range_values = {
            k[:-1]: v
            for k, v in self.parameters.items()
            if k in self.range_keys }

        other_values = {
            k: v
            for k, v in self.parameters.items()
            if k not in self.range_keys }

        range_keys = range_values.keys()
        tasks = []
        for concrete_values in itertools.product(*list(range_values.values())):

            parameters = { k: v for k, v in zip(range_keys, concrete_values) }
            parameters.update(other_values)

            evaluation_method = parameters['evaluation_method']
            del parameters['evaluation_method']

            tasks.append(Evaluate(parameters, evaluation_method))

        print("EvaluateCombinations: require %d configurations"%len(tasks))

        return tasks
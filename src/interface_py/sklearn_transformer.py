"""
:copyright: 2018 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
import os
import shutil

BASEPATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(BASEPATH, '../../'))


class SkCache(object):
    '''Cache for timestamp of scikit-learn files. Prevents re-copying file to
    spare some re-builds.'''
    def __init__(self, build_dir):
        self.build_dir = build_dir

        self._cache_path = os.path.join(self.build_dir, 'skcache')
        if os.path.exists(self.build_dir):
            if os.path.exists(self._cache_path):
                with open(self._cache_path, 'r') as fd:
                    self.cache = eval(fd.read())
            else:
                self.cache = {}
        else:
            self.cache = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        with open(self._cache_path, 'w') as f:
            print(self.cache, file=f)

    def changed(self, path):
        '''Whether is the timestamp read from path changed.'''
        if not os.path.exists(path):
            raise ValueError(path, 'doesn\'t exist')
        timestamp = os.path.getmtime(path)
        if path in self.cache:
            cached_timestamp = self.cache[path]
            return timestamp != cached_timestamp
        else:
            self.cache[path] = timestamp
            return True


class SkTransformer(object):
    '''Merge scikit-learn into h2o4gpu.'''
    def __init__(self, cache):
        self.sklearn_path = os.path.join(PROJECT_ROOT, 'scikit-learn')
        self.cache = cache
        sk_files = []
        for root, subdirs, files in os.walk(self.sklearn_path):
            paths = [os.path.join(root, f) for f in files]
            for p in paths:
                if p.find('.git') == -1:
                    sk_files.append(p)
            paths = [os.path.join(root, d) for d in subdirs]
            for p in paths:
                if p.find('.git') == -1:
                    sk_files.append(p)
        self.sk_files = sk_files

        def s(old):
            return old + 'Sklearn'

        # names subsituted by <name>Sklearn
        features = ['KMeans', 'Ridge', 'Lasso', 'LogisticRegression',
                    'LinearRegression', 'ElasticNet',
                    'GradientBoostingRegressor',
                    'GradientBoostingClassifier', 'RandomForestRegressor',
                    'RandomForestClassifier', 'TruncatedSVD', 'PCA']

        # Initialize other subsitutions.
        self.mapping = {'sklearn': 'h2o4gpu', 'scikit-learn': 'h2o4gpu',
                        # avoid duplicate conversions
                        'Sklearn_Sklearn': 'Sklearn',
                        'from .. import get_config as _get_config':
                        # FIXME: Find out what this does
                        r'''import os
_ASSUME_FINITE = bool(os.environ.get('SKLEARN_ASSUME_FINITE', False))
def _get_config():
    return {'assume_finite': _ASSUME_FINITE}'''}

        for feat in features:
            self.mapping[feat] = s(feat)

        # Don't read these files, just copy them
        self.just_copy = ['jpg', 'jpeg', 'bmp', 'svg', 'ico', 'gz', 'png',
                          'pdf', 'pyc']

    @staticmethod
    def merge_sklearn_init(target_dir):
        '''Merge the __init__.py files of scikit-learn and h2o4gpu.'''
        # __init__ moved here by transformer
        sklearn_init_path = os.path.join(target_dir,
                                         'h2o4gpu', '__init__.py')
        h2o4gpu_base_path = os.path.join(BASEPATH,
                                         './h2o4gpu/__init__.base.py')
        final_init_path = os.path.join(target_dir,
                                       './h2o4gpu/__init__.py')

        with open(h2o4gpu_base_path, 'r') as sklearn_fd:
            with open(sklearn_init_path, 'r') as base_fd:
                with open(final_init_path, 'w') as merged_fd:
                    sklearn_init = sklearn_fd.read()
                    h2o4gpu_base = base_fd.read()

                    merged_init = h2o4gpu_base + sklearn_init
                    merged_fd.write(merged_init)

    @staticmethod
    def append_imports(project_dir):
        project_dir = os.path.join(project_dir, 'h2o4gpu')
        with open(os.path.join(project_dir, 'cluster', '__init__.py'),
                  'a') as f:
            f.write('\n# Inject h2o4gpu\n')
            f.write('import h2o4gpu.solvers.kmeans\n')
        with open(os.path.join(project_dir, 'linear_model', '__init__.py'),
                  'a') as f:
            f.write('\n# Inject h2o4gpu\n')
            f.write('import h2o4gpu.solvers.ridge\n')
            f.write('import h2o4gpu.solvers.lasso\n')
            f.write('import h2o4gpu.solvers.logistic\n')
            f.write('import h2o4gpu.solvers.linear_regression\n')
            f.write('import h2o4gpu.solvers.elastic_net\n')

    def _plaintext(self, path):
        '''Can be read as plain text.'''
        if os.path.isdir(path):
            return False

        ext_idx = path.rfind('.')
        if ext_idx != -1:       # has extension
            return path[ext_idx+1:] not in self.just_copy
        else:
            # File in git repo without extension should not be binary files?
            return True

    @staticmethod
    def merge_dir(src, dst):
        src = os.path.abspath(src)
        dst = os.path.abspath(dst)
        src_files = []
        dst_files = []

        for root, subdirs, files in os.walk(src):
            for f in files:
                src_path = os.path.join(root, f)
                commonpath = os.path.commonpath([src, src_path])
                relpath = src_path[len(commonpath)+1:]
                dst_path = os.path.join(dst, relpath)
                src_files.append(os.path.abspath(src_path))
                dst_files.append(os.path.abspath(dst_path))

        for i in range(len(src_files)):
            if not os.path.exists(os.path.dirname(dst_files[i])):
                os.makedirs(os.path.dirname(dst_files[i]))
            shutil.copyfile(src_files[i], dst_files[i])

    def transform(self, cache_dir):
        '''Convert scikit-learn into part of h2o4gpu.

        Parameters
        -------
        cache_dir: str
            Directory to host converted files, which will be used to merge
            with source code from h2o4gpu.
        '''
        cache_dir = os.path.abspath(cache_dir)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        for old_file_path in self.sk_files:

            new_file_path = old_file_path
            for k, v in self.mapping.items():
                new_file_path = new_file_path.replace(k, v)
            new_file_path = new_file_path[
                len(os.path.commonpath([PROJECT_ROOT, new_file_path]))+1:]

            target_path = os.path.join(cache_dir, new_file_path)
            dirname = os.path.dirname(target_path)
            if not os.path.exists(dirname):
                os.makedirs(dirname)

            if (os.path.exists(target_path) and
                    not self.cache.changed(old_file_path)):
                continue        # No need to process again

            # Subsitute file content
            if self._plaintext(old_file_path):
                with open(old_file_path) as f:
                    content = f.read()
                    for k, v in self.mapping.items():
                        content = content.replace(k, v)

                with open(target_path, 'w') as f:
                    f.write(content)
            elif os.path.isdir(old_file_path):
                if not os.path.exists(target_path):
                    os.makedirs(target_path)
            else:
                with open(old_file_path, 'rb') as f:
                    content = f.read()
                with open(target_path, 'wb') as f:
                    f.write(content)

# -*- encoding: utf-8 -*-
"""
:copyright: 2017-2020 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""


def load_pkl(name):
    """Load xgboost model from pickle and perform conversion from version
    0.90 if necessary.

    :return:
        XGBoost model
    """
    import pickle
    import xgboost
    with open(name, 'rb') as f:
        try:
            model = pickle.load(f)
            return model
        except xgboost.core.XGBoostError as e:
            if "Check failed: header == serialisation_header_" in str(e):
                import xgboost_prev  # pylint: disable=unused-import
                import tempfile

                class Unpickler(pickle.Unpickler):
                    def find_class(self, module, name):
                        if module.startswith("xgboost"):
                            return pickle.Unpickler.find_class(
                                self, module.replace(
                                    "xgboost", "xgboost_prev"),
                                name)
                        return pickle.Unpickler.find_class(self, module, name)
                f.seek(0)
                model = Unpickler(f).load()
                temp_file = tempfile.NamedTemporaryFile(
                    prefix='xgboost_migration', suffix='.model')
                model.save_model(temp_file.name)
                migrated_model = xgboost.XGBModel()
                migrated_model.load_model(temp_file.name)
                return migrated_model
            raise

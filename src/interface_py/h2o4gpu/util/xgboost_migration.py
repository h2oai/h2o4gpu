def load_pkl(name):
    """Load xgboost model from pickle and perform conversion from version 0.90 if require.

    :return:
        Total number of GPUs and total available memory
    """
    import pickle
    import xgboost
    with open(name, 'rb') as f:
        try:
            model = pickle.load(f)
            return model
        except xgboost.core.XGBoostError as e:
            if "Check failed: header == serialisation_header_" in str(e):
                import xgboost_prev
                import tempfile

                class Unpickler(pickle.Unpickler):
                    def find_class(self, module, name):
                        if module.startswith("xgboost"):
                            return pickle.Unpickler.find_class(self, module.replace("xgboost", "xgboost_prev"), name)
                        else:
                            return pickle.Unpickler.find_class(self, module, name)
                f.seek(0)
                model = Unpickler(f).load()
                temp_file = tempfile.NamedTemporaryFile(
                    prefix='xgboost_migration', suffix='.model')
                model.save_model(temp_file.name)
                import xgboost
                migrated_model = xgboost.XGBModel()
                migrated_model.load_model(temp_file.name)
                return migrated_model
            else:
                raise

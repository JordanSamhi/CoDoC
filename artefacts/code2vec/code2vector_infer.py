from extractor import Extractor
import tempfile
import os
import numpy as np


EXTRACTOR_JAR = "JavaExtractor/JPredict/target/" \
                "JavaExtractor-0.0.1-SNAPSHOT.jar"
MAX_CONTEXTS = 200
MAX_PATH_LENGTH = 8
MAX_PATH_WIDTH = 2


class Code2vector:
    def __init__(self, model, config):
        self.model = model
        self.vector = None
        self.config = config

    def convert(self, function):
        f = tempfile.NamedTemporaryFile(mode='w+', dir='/tmp', delete=True)
        f.write(function)
        file_path = f.name
        f.seek(0)
        extractor = Extractor(self.config, EXTRACTOR_JAR,
                              MAX_PATH_LENGTH, MAX_PATH_WIDTH)
        paths, _ = extractor.extract_paths(file_path)
        f.close()
        result = self.model.predict(paths)

        if result:
            self.vector = result[0].code_vector

        return self.vector


if __name__ == '__main__':
    from config import Config
    from model_base import Code2VecModelBase

    def load_model_dynamically(config: Config) -> Code2VecModelBase:
        assert config.DL_FRAMEWORK in {'tensorflow', 'keras'}
        if config.DL_FRAMEWORK == 'tensorflow':
            from tensorflow_model import Code2VecModel
        elif config.DL_FRAMEWORK == 'keras':
            from keras_model import Code2VecModel
        return Code2VecModel(config)

    MODEL_MODEL_LOAD_PATH = 'models/android_source/saved_model'
    # Init and Load the model
    config = Config(set_defaults=True, load_from_args=True, verify=False)
    config.MODEL_LOAD_PATH = MODEL_MODEL_LOAD_PATH
    config.EXPORT_CODE_VECTORS = True
    config.MAX_CONTEXTS = MAX_CONTEXTS

    model = load_model_dynamically(config)
    config.log('Done creating code2vec model')

    c2v = Code2vector(model, config)
    for (root, _, files) in os.walk('./android_source/', topdown=False):
        for f in files:
            filename = os.path.join(root, f)
            filename_without_extension = f.split(".")[0]
            code = open(filename, 'r').read()
            empty_body = False
            try:
                vector = c2v.convert(code)
            except Exception:
                empty_body = True
            with open(f"{config.OUTPUT_PATH}/source_code_vectors.txt", 'a') as output:
                output.write(f"{filename_without_extension},{' '.join([str(elt) for elt in vector])},{empty_body}\n")

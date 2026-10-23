"""The predictor script of the KServe deployment, set in jobs.yaml and created by deploy.apply.

Hopsworks copies this file into the pod, where HopsFS is not mounted, so src comes from the zip deploy.apply uploads
for the deployed commit. Predict's constructor may only take project, deployment, model, and async_logger.
"""

import sys
import tempfile

SRC_ZIP = "Resources/deploy/src.zip"


class Predict:
    def __init__(self, project=None, deployment=None, model=None, async_logger=None):
        local = project.get_dataset_api().download(SRC_ZIP, tempfile.mkdtemp(), overwrite=True)
        sys.path.insert(0, local)
        from src.inference_pipeline.predictor import Server

        self.server = Server(project)

    def predict(self, inputs):
        return self.server(inputs if isinstance(inputs, list) else [inputs])

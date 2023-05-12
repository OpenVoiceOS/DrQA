from ovos_plugin_manager.templates.solvers import EvidenceSolver
from drqa import DrQA


class DrQASolver(EvidenceSolver):
    def __init__(self, config=None):
        config = config or {}
        config["lang"] = "en"  # only supports english
        super().__init__(name="DrQA", priority=50, config=config,
                         enable_cache=False, enable_tx=True)
        # TODO - auto download from huggingface
        model = self.config["model_path"]
        meta = self.config["meta_path"]
        self.dr = DrQA(model_path=model, meta_path=meta)

    def get_best_passage(self, evidence, question, context):
        """
        evidence and question assured to be in self.default_lang
         returns summary of provided document
        """
        return self.dr.predict(evidence, question)

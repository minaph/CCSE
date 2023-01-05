from collections import OrderedDict
from detectron2.engine.hooks import EvalHook

import optuna


class PruningHook(EvalHook):
    def __init__(self, period: int, trainer, study: optuna.study.Study, trial: optuna.trial.Trial):
        super(PruningHook, self).__init__(period, lambda: None, True)
        self.trainer = trainer
        self.study = study
        self.trial = trial
        self.last_value = None

    def _do_eval(self):
        if (
            not hasattr(self.trainer, "_last_eval_results")
            or "segm" not in self.trainer._last_eval_results
            or "AP" not in self.trainer._last_eval_results["segm"]
        ):
            return
        ap = self.trainer._last_eval_results["segm"]["AP"]
        self.trial.report(ap, self.trainer.iter)
        self.last_value = ap

    def after_step(self):
        super(PruningHook, self).after_step()
        
        if self.trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    def after_train(self):
        if self.last_value:
            self.study.tell(self.trial, self.last_value, optuna.trial.TrialState.COMPLETE)

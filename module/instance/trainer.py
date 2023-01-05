from detectron2.data import build_detection_train_loader, build_detection_test_loader, DatasetMapper
import detectron2.data.transforms as T
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator

from common.utils import join

# from .loss_eval_hook import LossEvalHook
from .optuna_hook import PruningHook

_study = None
_trial = None

class TrainerWithoutHorizontalFlip(DefaultTrainer):
    def __init__(self, cfg, study=None, trial=None):
        global _study, _trial
        _study = study
        _trial = trial
        super(TrainerWithoutHorizontalFlip, self).__init__(cfg)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name=dataset_name,
                             tasks=cfg.TASKS,
                             distributed=False,
                             output_dir=output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        dataset_mapper = DatasetMapper(cfg, is_train=True,
                                       augmentations=[
                                           T.ResizeShortestEdge(short_edge_length=cfg.INPUT.MIN_SIZE_TRAIN,
                                                                max_size=cfg.INPUT.MAX_SIZE_TRAIN,
                                                                sample_style=cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING)
                                       ])
        dataloader = build_detection_train_loader(cfg,
                                                  mapper=dataset_mapper)

        return dataloader

    def build_hooks(self):
        global _study, _trial
        hooks = super(TrainerWithoutHorizontalFlip, self).build_hooks()
        cfg = self.cfg
        
        if _study is not None:
            hooks.append(PruningHook(cfg.TEST.EVAL_PERIOD, self, _study, _trial))
            del _study, _trial
        
        return hooks

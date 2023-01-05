import logging
import os
import cv2
import random

from detectron2.engine import DefaultPredictor, launch
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog
from detectron2.evaluation import COCOEvaluator

from common.cmd_parser import parse_cmd_arg
from pre_process.pre_process import read_to_gray_scale
from module.instance.trainer import TrainerWithoutHorizontalFlip
from common.utils import plt_show, join

from initializer.instance_initializer import InstanceInitializer

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner

import copy


def main(init: InstanceInitializer, study: optuna.study.Study = None, trial: optuna.trial.Trial = None):
    config = init.config
    logger = logging.getLogger('detectron2')

    # launch again to support multi-process!
    init.launch_calling()

    # visualize the dataset
    visualize_dataset(init.train_set_name,
                      init.dataset_metadata,
                      config.OUTPUT_DIR,
                      logger,
                      num_vis=config.NUM_VIS)

    # train and evaluate the model
    train_and_evaluate(init, config, study, trial)

    # visualize the prediction
    # predictor = DefaultPredictor(config)
    # visualize_prediction(predictor, init.val_set_name, dataset_metadata=init.dataset_metadata,
    #                      OUTPUT_DIR=config.OUTPUT_DIR, logger=logger, num_vis=config.NUM_VIS)
    visualize(config, init.val_set_name, dataset_metadata=init.dataset_metadata, logger=logger, num_vis=config.NUM_VIS)


def visualize_dataset(dataset_name, dataset_metadata, OUTPUT_DIR, logger, num_vis=10):
    visualize_dataset_path = join(OUTPUT_DIR, 'visualize_dataset')
    logger.info("Saving dataset visualization results in {}".format(visualize_dataset_path))
    if not os.path.exists(visualize_dataset_path):
        os.makedirs(visualize_dataset_path, exist_ok=True)
    dataset_dicts = DatasetCatalog.get(dataset_name)
    for d in random.sample(dataset_dicts, num_vis):
        print('===> d["file_name"]: {}'.format(d["file_name"]))
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=dataset_metadata, scale=5.0)
        out = visualizer.draw_dataset_dict(d)
        plt_show(out.get_image()[:, :, ::-1], join(visualize_dataset_path, os.path.basename(d['file_name'])))


def visualize_prediction(predictor, dataset_name, dataset_metadata, OUTPUT_DIR, logger, num_vis=10):
    visualize_prediction_path = join(OUTPUT_DIR, 'visualize_prediction')
    logger.info("Saving prediction visualization results in {}".format(visualize_prediction_path))
    if not os.path.exists(visualize_prediction_path):
        os.makedirs(visualize_prediction_path, exist_ok=True)
    dataset_dicts = DatasetCatalog.get(dataset_name)
    for d in random.sample(dataset_dicts, num_vis):
        im = cv2.imread(d["file_name"])
        # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                       metadata=dataset_metadata,
                       scale=5.0,
                       # instance_mode=ColorMode.IMAGE_BW
                       # remove the colors of unsegmented pixels. This option is only available for segmentation models
                       )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        plt_show(out.get_image()[:, :, ::-1], join(visualize_prediction_path, os.path.basename(d['file_name'])))


def train_and_evaluate(init, config, study: optuna.study.Study = None, trial: optuna.trial.Trial = None):
    evaluator = COCOEvaluator(init.val_set_name, config.TASKS, False, output_dir=config.OUTPUT_DIR)

    trainer = TrainerWithoutHorizontalFlip(config, study, trial)
    trainer.resume_or_load(resume=False)
    trainer.train()
    trainer.test(config, model=trainer.model, evaluators=[evaluator])


def visualize(config, dataset_name, dataset_metadata, logger, num_vis=10):
    # Inference should use the config with parameters that are used in training
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    config.MODEL.WEIGHTS = join(config.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
    predictor = DefaultPredictor(config)
    visualize_prediction(predictor, dataset_name, dataset_metadata, config.OUTPUT_DIR, logger, num_vis=num_vis)


if __name__ == '__main__':
    args = parse_cmd_arg()

    initializer = InstanceInitializer(args.config)
    initializer.logger = None
    num_gpu = len(initializer.config.GPU_IDS)

    train_set_name = initializer.train_set_name
    val_set_name = initializer.val_set_name
    dataset_metadata = initializer.dataset_metadata

    n_trials = initializer.config.OPTUNA.N_TRIALS

    if n_trials > 0:
        study = optuna.study.create_study(
            storage=initializer.config.OPTUNA.STORAGE, 
            sampler=TPESampler(), 
            pruner=HyperbandPruner(),
            load_if_exists=True,
            direction="maximize",
            study_name=initializer.config.OPTUNA.STUDY_NAME
        )

        for i in range(n_trials):
            initializer = InstanceInitializer(args.config)
            initializer.logger = None

            initializer.train_set_name = train_set_name
            initializer.val_set_name = val_set_name
            initializer.dataset_metadata = dataset_metadata

            trial = study.ask()
            lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
            initializer.config.SOLVER.BASE_LR = lr
            try:
                launch(main_func=main, num_gpus_per_machine=num_gpu, dist_url='auto', args=(initializer, study, trial))
            except optuna.exceptions.TrialPruned:
                print("Trial is pruned.")
                study.tell(trial, optuna.trial.TrialState.PRUNED)
            except KeyboardInterrupt as e:
                study.tell(trial, optuna.trial.TrialState.FAIL)
                raise e
            except Exception as e:
                study.tell(trial, optuna.trial.TrialState.FAIL)
                raise e
            # else:
            #     study.tell(trial, optuna.trial.TrialState.COMPLETE)
        
        print("Best trial:")
        print("  Value: ", study.best_trial.value)
        print("  Params: ", study.best_trial.params)
    
    else:
        decay_late = initializer.config.SOLVER.WEIGHT_DECAY
        while True:
            initializer = InstanceInitializer(args.config)
            initializer.logger = None

            initializer.train_set_name = train_set_name
            initializer.val_set_name = val_set_name
            initializer.dataset_metadata = dataset_metadata

            initializer.config.SOLVER.WEIGHT_DECAY = decay_late
            try:
                launch(main_func=main, num_gpus_per_machine=num_gpu, dist_url='auto', args=(initializer,))
            except FloatingPointError as e:
                if initializer.config.SOLVER.RESTART_IF_NAN:
                    decay_late += 0.0001
                    print(f"\n=== Restarting training due to NaN loss. decay_late: {decay_late:.4} ===\n")
                    continue
                else:
                    raise e


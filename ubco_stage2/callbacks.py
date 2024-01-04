import lightning.pytorch as pl

def get_logger(metrics, project, no_wandb):
    """
    Function to load wandb logger.
    """
    if no_wandb == True:
       return None, None

    # Get logger
    logger = pl.loggers.WandbLogger(
        project = project, 
        save_dir = None,
        )
    id_ = logger.experiment.id
    
    # Wandb metric summary options (min,max,mean,best,last,none): https://docs.wandb.ai/ref/python/run#define_metric
    for metric, summary in metrics.items():
        logger.experiment.define_metric(metric, summary=summary)
    
    return logger, id_


def load_logger_and_callbacks(
    fast_dev_run,
    metrics,
    overfit_batches,
    no_wandb,
    project,
):
    callbacks = []

    # Test Runs.
    if fast_dev_run or overfit_batches > 0:
        return None, None

    # Other Callbacks
    callbacks.extend([
        pl.callbacks.LearningRateMonitor(),
        pl.callbacks.StochasticWeightAveraging(1e-7, swa_epoch_start=0.8, annealing_epochs=4)
    ])

    # Logger
    logger, id_ = get_logger(
        metrics = metrics, 
        project = project,
        no_wandb = no_wandb,
        )

    return logger, callbacks
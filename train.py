import models
import data_reader
import utils
import logging
import datetime

logging.basicConfig(filename=f"logs/{datetime.datetime.now().strftime('logs_%Y_%m_%d.txt')}",
                    format='%(levelname)s : %(asctime)s : %(message)s',
                    datefmt='%d/%m/%Y %I:%M:%S %p',
                    level=logging.CRITICAL)
logger = logging.getLogger(__name__)


def train(batch_size=32, input_size=224, model_name='inception', epochs=10):
    logger.critical(f'Start training {model_name} for {epochs} epochs.')
    train_generator, validation_generator, _ = data_reader.get_generators(batch_size, input_size)
    model = models.get_model(model_name)
    callbacks = utils.get_callbacks(model_name)
    # train the model on the new data for a few epochs
    history = model.fit(train_generator,
                      validation_data=validation_generator,
                      epochs=epochs,
                      shuffle=True,
                      use_multiprocessing=False,
                      verbose=1,
                      callbacks=callbacks)
    logger.critical(f'Finish training {model_name}.')
    return history
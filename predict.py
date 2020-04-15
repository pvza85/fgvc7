import pandas as pd
import models
import data_reader
import os
import logging
import datetime

logging.basicConfig(filename=f"logs/{datetime.datetime.now().strftime('logs_%Y_%m_%d.txt')}",
                    format='%(levelname)s : %(asctime)s : %(message)s',
                    datefmt='%d/%m/%Y %I:%M:%S %p',
                    level=logging.CRITICAL)
logger = logging.getLogger(__name__)


def save_results(pred, name='test'):
    pred = pd.DataFrame(pred,columns=['healthy', 'multiple_diseases', 'rust', 'scab'])
    pred['image_id'] = pd.Series(list(pred.index)).apply(lambda x: 'Test_' + str(x))
    pred = pred[['image_id', 'healthy', 'multiple_diseases', 'rust', 'scab']]
    pred.to_csv(f'results/{name}.csv', index=False)


def get_best_model(model_name):
    best = 0
    for f in os.listdir(f'models/{model_name}/'):
        if f.endswith('.hdf5'):
            acc = int(f.split('-')[-1].split('.')[1])
            if acc > best:
                best = acc
    best_model = [f for f in os.listdir(f'models/{model_name}/') if f.endswith(f'{best}.hdf5')][-1]
    return f'models/{model_name}/{best_model}'


def predict(batch_size=32, input_size=224, model_name='inception'):
    _, _, test_generator = data_reader.get_generators(batch_size, input_size)
    model_file = get_best_model(model_name)
    logger.critical(f'Start predicting with {model_name} from model: {model_file}.')
    print(model_file)
    model = models.get_model(model_name)
    model.load_weights(model_file)
    predictions = model.predict(test_generator)
    save_results(predictions, name=model_name)
    logger.critical(f'Finished predicting with {model_name}.')

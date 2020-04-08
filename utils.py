from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
import os

def get_callbacks(model_name):
    directory = f'models/{model_name}/'
    if not os.path.exists(directory):
        os.mkdir(directory)

    model_checkpoint = ModelCheckpoint(directory+"weights-{epoch:02d}.hdf5", monitor='val_loss', save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    csv_logger = CSVLogger(directory+'log.csv', separator=',', append=False)

    return [model_checkpoint, early_stopping, csv_logger]
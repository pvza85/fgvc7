from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

def get_callbacks():
    model_checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto',
                    period=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
    csv_logger = CSVLogger('log.csv', separator=',', append=False)

    return [model_checkpoint, early_stopping, csv_logger]
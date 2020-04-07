import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_class(row):
    for c in row.iteritems():
        if c[1] == 1:
            return c[0]


def get_generators():
    image_folder = 'data/images/'
    train_data = pd.read_csv('data/train.csv')
    test_data = pd.read_csv('data/test.csv')
    sample_submission = pd.read_csv('data/sample_submission.csv')

    validation_ratio = 0.1
    train_df = train_data.sample(frac=1.0).copy()
    train_df['x_col'] = train_df['image_id'].apply(lambda x: f"data/images/{x}.jpg")
    train_df['y_col'] = train_df.apply(get_class, axis=1)
    train_df = train_df[['x_col', 'y_col']]
    validation_size = int(train_data.shape[0] * validation_ratio)
    training_size = train_data.shape[0] - int(train_data.shape[0] * validation_ratio)
    validation_df = train_df.tail(validation_size).copy()
    train_df = train_df.head(training_size).copy()

    test_df = test_data.copy()
    test_df['x_col'] = test_df['image_id'].apply(lambda x: f"data/images/{x}.jpg")
    test_df = test_df[['x_col']]

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col="x_col",
        y_col="y_col",
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_dataframe(
        dataframe=validation_df,
        x_col="x_col",
        y_col="y_col",
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')

    return train_generator, validation_generator

    
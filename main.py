import train
import predict
import sys


# ToDo: write train plus returning params
# ToDo: write main function
# ToDo: add layer fine-tuning
# ToDo: improve saving models
# ToDo: add more logs + telegram
# ToDo: find best model + submit
# ToDo: find best input shape + submit
# ToDo: test if extra layer + submit
# ToDo: fine tune more layers + submit


def run(model_name, epochs):
    train.train(model_name=model_name, epochs=epochs)
    predict.predict(model_name=model_name)


if __name__ == '__main__':
    run(sys.argv[1], int(sys.argv[2]))
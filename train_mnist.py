import ml
import pickle
import mnisttk
import logging


logging.basicConfig(level=logging.INFO)


data = mnisttk.load_train_data('/home/stderr/.mnist')
validator = ml.CrossValidator(ml.ForwardFeedNetwork, (28*28, 100, 10))

try:
    models = validator.run(data, 5, 0.4, ml.max_epochs(100))
finally:
    with open('results/mnist.ann', 'wb') as fp:
        pickle.dump(models, fp)

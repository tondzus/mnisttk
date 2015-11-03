import ml


def test_mlp_ann_train_and_logic_function(and_dataset):
    ann = ml.ForwardFeedNetwork((2, 1))
    ann.train(and_dataset, None, 0.2, ml.max_epochs(100))
    results = ann.forward_feed(and_dataset[:, :2])
    assert results.shape == (4, 1)
    results = results.reshape(4)
    assert results[0] < 0.5
    assert results[1] < 0.5
    assert results[2] < 0.5
    assert results[3] > 0.5


def test_mlp_ann_train_xor_logic_function(xor_dataset):
    ann = ml.ForwardFeedNetwork((2, 2, 1))
    trainer = ann.train(xor_dataset, None, 0.2,
                        ml.max_error(0.1, max_epoch=20000))
    assert trainer.epoch < 20000
    results = ann.forward_feed(xor_dataset[:, :2])
    assert results.shape == (4, 1)
    results = results.reshape(4)
    assert results[0] < 0.5
    assert results[1] > 0.5
    assert results[2] > 0.5
    assert results[3] < 0.5


def test_cross_validator(xor_dataset):
    ann_args = {'alpha': 0.1, 'stopping_criteria': ml.max_epochs(5000)}
    validator = ml.CrossValidator(ml.ForwardFeedNetwork, (2, 2, 1))
    validator.run(xor_dataset, 4, **ann_args)

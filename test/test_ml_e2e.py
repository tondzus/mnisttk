import ml


def test_mlp_ann_train_and_logic_function(and_dataset):
    ann = ml.ForwardFeedNetwork((2, 1))
    ann.train(and_dataset, 0.2, ml.max_epochs(100))
    results = ann.forward_feed(and_dataset[:, :2])
    assert results.shape == (4, 1)
    results = results.reshape(4)
    assert results[0] < 0.5
    assert results[1] < 0.5
    assert results[2] < 0.5
    assert results[3] > 0.5


def test_mlp_ann_train_xor_logic_function(xor_dataset):
    ann = ml.ForwardFeedNetwork((2, 2, 1))
    trainer = ann.train(xor_dataset, 0.2, ml.max_error(0.1, max_epoch=20000))
    assert trainer.epoch < 20000
    results = ann.forward_feed(xor_dataset[:, :2])
    assert results.shape == (4, 1)
    results = results.reshape(4)
    assert results[0] < 0.5
    assert results[1] > 0.5
    assert results[2] > 0.5
    assert results[3] < 0.5

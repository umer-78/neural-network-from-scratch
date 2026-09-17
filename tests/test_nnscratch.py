import numpy as np
import pytest

from nnscratch import Adam, CrossEntropy, Dense, Dropout, MeanSquaredError, ReLU, SGD, Sigmoid, Softmax, Tanh
from nnscratch.datasets import digits, moons, one_hot
from nnscratch.gradcheck import check_gradients, relative_error
from nnscratch.network import mlp

rng = np.random.default_rng(0)


# --------------------------------------------------------------------- layers
def test_dense_shapes_and_forward():
    layer = Dense(3, 2, rng=np.random.default_rng(1))
    layer.W[...] = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    layer.b[...] = np.array([0.5, -0.5])
    out = layer.forward(np.array([[1.0, 2.0, 3.0]]))
    assert out.tolist() == [[4.5, 4.5]]


def test_dense_backward_averages_over_the_batch():
    layer = Dense(2, 1)
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    layer.forward(x)
    layer.backward(np.array([[1.0], [1.0]]))
    assert layer.dW.flatten().tolist() == [2.0, 3.0]   # (1+3)/2, (2+4)/2
    assert layer.db.tolist() == [1.0]


def test_relu_and_its_gradient():
    r = ReLU()
    assert r.forward(np.array([[-1.0, 0.0, 2.0]])).tolist() == [[0.0, 0.0, 2.0]]
    assert r.backward(np.ones((1, 3))).tolist() == [[0.0, 0.0, 1.0]]


def test_sigmoid_and_tanh_ranges():
    s = Sigmoid().forward(np.array([[-50.0, 0.0, 50.0]]))
    assert s[0][0] == pytest.approx(0, abs=1e-9) and s[0][1] == 0.5 and s[0][2] == pytest.approx(1)
    t = Tanh().forward(np.array([[-50.0, 0.0, 50.0]]))
    assert t.tolist() == [[-1.0, 0.0, 1.0]]


def test_softmax_is_stable_and_sums_to_one():
    out = Softmax().forward(np.array([[1000.0, 1001.0, 1002.0]]))
    assert np.isfinite(out).all()
    assert out.sum() == pytest.approx(1.0)
    assert out[0].argmax() == 2


def test_dropout_only_drops_during_training():
    d = Dropout(0.5, rng=np.random.default_rng(3))
    x = np.ones((200, 20))
    train = d.forward(x, training=True)
    assert (train == 0).any() and train.mean() == pytest.approx(1.0, abs=0.15)  # inverted dropout keeps the scale
    assert (d.forward(x, training=False) == x).all()
    with pytest.raises(ValueError):
        Dropout(1.5)


# --------------------------------------------------------------------- losses
def test_cross_entropy_values_and_gradient():
    loss = CrossEntropy()
    y = np.array([[1.0, 0.0]])
    assert loss.forward(np.array([[1.0, 0.0]]), y) == pytest.approx(0.0, abs=1e-9)
    assert loss.forward(np.array([[0.5, 0.5]]), y) == pytest.approx(np.log(2))
    assert loss.backward(np.array([[0.7, 0.3]]), y).tolist() == [[pytest.approx(-0.3), pytest.approx(0.3)]]


def test_mse():
    loss = MeanSquaredError()
    assert loss.forward(np.array([[1.0, 2.0]]), np.array([[1.0, 4.0]])) == pytest.approx(2.0)


# ---------------------------------------------------------------- correctness
@pytest.mark.parametrize("sizes", [[4, 6, 3], [3, 8, 8, 2], [5, 4, 4]])
def test_backprop_matches_numerical_gradients(sizes):
    net = mlp(sizes, seed=1)
    x = rng.normal(size=(6, sizes[0]))
    y = one_hot(rng.integers(0, sizes[-1], 6), sizes[-1])
    assert check_gradients(net, x, y) < 1e-5


def test_relative_error_helper():
    assert relative_error(np.array([1.0]), np.array([1.0])) == 0
    assert relative_error(np.array([1.0]), np.array([-1.0])) == pytest.approx(1.0)


# ------------------------------------------------------------------- training
def test_learns_xor_which_needs_a_hidden_layer():
    x = np.array([[0.0, 0], [0, 1], [1, 0], [1, 1]])
    y = one_hot(np.array([0, 1, 1, 0]), 2)
    net = mlp([2, 8, 2], seed=4, optimiser=SGD(0.5, momentum=0.9))
    net.fit(x, y, epochs=600, batch_size=4, shuffle=False)
    assert net.score(x, y) == 1.0
    assert net.history.loss[-1] < net.history.loss[0]


def test_moons_and_digits_reach_useful_accuracy():
    xtr, xte, ytr, yte = moons(n=600)
    net = mlp([2, 16, 16, 2], seed=0, optimiser=SGD(0.1, momentum=0.9))
    net.fit(xtr, ytr, epochs=60, batch_size=16)
    assert net.score(xte, yte) > 0.9

    xtr, xte, ytr, yte, images = digits()
    assert images.shape == (len(xte), 8, 8)
    net = mlp([64, 64, 10], seed=0, optimiser=Adam(0.005))
    net.fit(xtr, ytr, epochs=15, batch_size=32)
    assert net.score(xte, yte) > 0.93


def test_early_stopping_restores_the_best_weights():
    xtr, xte, ytr, yte = moons(n=400)
    net = mlp([2, 32, 2], seed=1, optimiser=SGD(0.3, momentum=0.9))
    net.fit(xtr, ytr, epochs=200, batch_size=8, validation=(xte, yte), early_stopping=3)
    assert len(net.history.loss) < 200, "should have stopped before the last epoch"
    best = min(net.history.val_loss)
    final = net.loss.forward(net.forward(xte, training=False), yte)
    assert final == pytest.approx(best, abs=1e-6)


def test_adam_beats_plain_sgd_in_few_epochs():
    xtr, xte, ytr, yte, _ = digits()
    scores = {}
    for name, opt in (("sgd", SGD(0.05)), ("adam", Adam(0.005))):
        net = mlp([64, 64, 10], seed=0, optimiser=opt)
        net.fit(xtr, ytr, epochs=5, batch_size=32)
        scores[name] = net.score(xte, yte)
    assert scores["adam"] > scores["sgd"]


# ---------------------------------------------------------------- persistence
def test_weights_round_trip(tmp_path):
    xtr, xte, ytr, yte, _ = digits()
    net = mlp([64, 32, 10], seed=2, optimiser=Adam(0.01))
    net.fit(xtr, ytr, epochs=5, batch_size=32)
    before = net.predict(xte)
    path = tmp_path / "w.npz"
    net.save(path)
    fresh = mlp([64, 32, 10], seed=99)
    assert (fresh.predict(xte) != before).any()
    fresh.load(path)
    assert (fresh.predict(xte) == before).all()


def test_set_weights_rejects_wrong_shape_count():
    net = mlp([2, 3, 2], seed=0)
    with pytest.raises(ValueError):
        net.set_weights([np.zeros((2, 3))])


def test_network_rejects_backward_before_forward():
    layer = Dense(2, 2)
    with pytest.raises(AssertionError):
        layer.backward(np.ones((1, 2)))


def test_cli_commands(capsys):
    from nnscratch.cli import main
    assert main(["gradcheck"]) == 0
    assert "matches numerical gradients" in capsys.readouterr().out
    assert main(["train", "--epochs", "3", "--hidden", "32"]) == 0
    assert "test accuracy" in capsys.readouterr().out

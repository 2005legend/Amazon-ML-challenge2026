import numpy as np
import polars as pl
from ber.folds import fold_expr
from ber.model import predict, train_lgb


def test_folds_are_deterministic_and_balanced():
    ids = pl.DataFrame({"s1_id": [f"S1-{i}" for i in range(1, 100_001)]})
    f1 = ids.select(fold_expr().alias("f"))["f"]
    f2 = ids.select(fold_expr().alias("f"))["f"]
    assert f1.to_list() == f2.to_list()
    share = f1.value_counts(normalize=True).sort("f")["proportion"].to_list()
    assert np.allclose(share, [0.4, 0.4, 0.2], atol=0.01)


def test_train_lgb_learns_separable_rule():
    rng = np.random.default_rng(0)
    X = rng.random((4000, 3)).astype(np.float32)
    y = (X[:, 0] > 0.5).astype(np.int8)
    params = {"objective": "binary", "learning_rate": 0.1, "num_leaves": 7, "min_data_in_leaf": 20,
              "verbose": -1, "seed": 42}
    b = train_lgb(X, y, params, rounds=50)
    p = predict(b, np.array([[0.9, 0.1, 0.1], [0.1, 0.9, 0.9]], np.float32))
    assert p[0] > 0.9 and p[1] < 0.1

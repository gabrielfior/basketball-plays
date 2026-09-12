import numpy as np

from basketball_plays import plays as P


def blobs(n_per=30, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.zeros((3, 118))
    centers[0, :22] = 1; centers[1, 22:44] = 1; centers[2, 44:66] = 1
    X = np.vstack([c + 0.05 * rng.standard_normal((n_per, 118)) for c in centers])
    y = np.repeat(np.arange(3), n_per)
    return X, y


def test_fit_clusters_recovers_three_blobs_and_is_stable():
    X, y = blobs()
    m = P.fit_clusters(X, k_range=range(2, 7), n_components=5)
    assert m.k == 3
    # labels match the blobs up to permutation
    from sklearn.metrics import adjusted_rand_score
    assert adjusted_rand_score(y, m.labels_train) > 0.95
    assert m.stability_ari > 0.9


def test_assign_puts_new_points_on_their_blob():
    X, _y = blobs()
    m = P.fit_clusters(X, k_range=range(2, 7), n_components=5)
    Xn, yn = blobs(n_per=5, seed=1)
    ln = P.assign(m, Xn)
    from sklearn.metrics import adjusted_rand_score
    assert adjusted_rand_score(yn, ln) > 0.95


def test_small_corpus_caps_k():
    X, _ = blobs(n_per=20)     # 60 rows
    m = P.fit_clusters(X, k_range=range(8, 31), n_components=5)
    assert m.small_corpus and m.k <= 6

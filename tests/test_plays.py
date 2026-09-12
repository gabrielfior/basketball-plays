import numpy as np

from basketball_plays import plays as P
from basketball_plays.features import FeatureRow


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


def test_summarize_reports_n_fit_from_mask():
    X, _y = blobs()
    m = P.fit_clusters(X, k_range=range(2, 7), n_components=5)
    rows = [
        FeatureRow(game_id="g", period=1, index=i, bucket="ato", split="train",
                   no_setup=False, vector=X[i], handler=np.zeros(5, dtype=int))
        for i in range(len(X))
    ]

    fit_mask = np.arange(len(X)) % 3 != 0  # two thirds of rows count as "fit"
    summary = P.summarize(m, rows, m.labels_train, ["z"] * 22, fit_mask=fit_mask)
    assert summary  # sanity: some clusters were found
    for c in summary:
        expected = int(np.sum(fit_mask[m.labels_train == c["cluster"]]))
        assert c["n_fit"] == expected
        assert c["n_fit"] <= c["n"]

    # fit_mask=None (the default) means every row is treated as fit.
    summary_all = P.summarize(m, rows, m.labels_train, ["z"] * 22)
    assert all(c["n_fit"] == c["n"] for c in summary_all)

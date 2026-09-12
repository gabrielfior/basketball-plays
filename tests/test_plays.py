import numpy as np

from basketball_plays import plays as P
from basketball_plays.features import FeatureRow

ZONES = [f"z{i}" for i in range(22)]


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


def test_sweep_window_is_continuous_in_n():
    # k_min is always 4; k_max = clip(n // 12, 6, 30). 60 rows: 60 // 12 = 5 -> clipped up to 6.
    X, _ = blobs(n_per=20)     # 60 rows
    m = P.fit_clusters(X, k_range=range(8, 31), n_components=5)
    assert sorted(m.sweep) == [4, 5, 6]
    assert m.small_corpus     # graded advisory: under ADVISORY_ROWS fit rows

    # No cliff: 148 and 155 rows of the same distribution sweep the same window
    # (148 // 12 == 155 // 12 == 12), unlike the old hard SMALL_CORPUS gate at 150.
    rng = np.random.default_rng(7)
    big, _ = blobs(n_per=60)   # 180 rows to draw from
    a = P.fit_clusters(big[rng.permutation(len(big))[:148]], k_range=range(8, 31), n_components=5)
    b = P.fit_clusters(big[rng.permutation(len(big))[:155]], k_range=range(8, 31), n_components=5)
    assert sorted(a.sweep) == sorted(b.sweep) == list(range(4, 13))


def test_sweep_entries_carry_mean_and_se_and_one_se_rule_picks_smallest_k():
    X, _y = blobs()
    m = P.fit_clusters(X, k_range=range(2, 7), n_components=5)
    assert all(set(v) == {"mean", "se"} and v["se"] >= 0 for v in m.sweep.values())
    best = max(m.sweep, key=lambda k: m.sweep[k]["mean"])
    assert m.k_best_raw == best
    # the chosen k is the smallest whose mean is within one SE of the best mean
    thresh = m.sweep[best]["mean"] - m.sweep[best]["se"]
    assert m.k == min(k for k, v in m.sweep.items() if v["mean"] >= thresh)
    assert m.silhouette == m.sweep[m.k]["mean"]


def test_fit_clusters_raises_when_the_corpus_cannot_be_clustered():
    X, _ = blobs(n_per=1)      # 3 rows: k=2 is still possible
    m = P.fit_clusters(X[:3], k_range=range(8, 31), n_components=2)
    assert m.k == 2 and m.sweep == {}

    try:
        P.fit_clusters(X[:2], k_range=range(8, 31), n_components=2)
    except ValueError as e:
        assert "2" in str(e)
    else:
        raise AssertionError("expected a ValueError naming n")


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


def test_summarize_ignores_non_fit_rows_in_the_centroid_occupancy():
    """`centroid_occupancy`, `top_zones` and `handler_path` describe the fit members only, so
    they match the centroid and the montage tiles rather than the wider assigned membership."""
    X, _y = blobs()
    m = P.fit_clusters(X, k_range=range(2, 7), n_components=5)
    rows = [
        FeatureRow(game_id="g", period=1, index=i, bucket="ato", split="train",
                   no_setup=False, vector=X[i].copy(), handler=np.zeros(5, dtype=int))
        for i in range(len(X))
    ]
    # Take one whole cluster and make the majority of it distinctive, then exclude that majority.
    members = np.where(m.labels_train == m.labels_train[0])[0]
    assert len(members) > 4
    odd, fit = members[:-4], members[-4:]
    fit_mask = np.ones(len(rows), dtype=bool)
    for i in odd:
        rows[i].vector[: 5 * 22] = 0.0
        rows[i].vector[22 + 21] = 50.0          # bin 1, last zone: would dominate top_zones
        rows[i].handler = np.full(5, 21, dtype=int)
        fit_mask[i] = False

    def summary_for(mask):
        return next(c for c in P.summarize(m, rows, m.labels_train, ZONES, fit_mask=mask)
                    if c["cluster"] == int(m.labels_train[0]))

    c = summary_for(fit_mask)
    assert c["n_fit"] == len(fit) and c["n"] == len(members)
    assert c["centroid_occupancy"][1][21] < 0.1          # the excluded rows contributed nothing
    assert ZONES[21] not in c["top_zones"]
    assert ZONES[21] not in c["handler_path"]

    # counted as fit, the very same rows dominate all three
    c_all = summary_for(np.ones(len(rows), dtype=bool))
    assert c_all["centroid_occupancy"][1][21] > 1.0
    assert c_all["top_zones"][0] == ZONES[21]
    assert c_all["handler_path"] == [ZONES[21]] * 5

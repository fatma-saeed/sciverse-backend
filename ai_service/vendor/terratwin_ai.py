"""TerraTwin AI layer, built on terratwin_model.py.

  python terratwin_ai.py evaluate                      blind test of every method (cross-validation)
  python terratwin_ai.py train                         fit final models, write outputs/ai_scores.csv
  python terratwin_ai.py recommend mars_jezero rover_mobility [n]
  python terratwin_ai.py explain ladakh_tsokar mars_jezero [purpose]

Parts
  1. Similarity engine with two body-level aggregations: "mean" over all targets of a body, or
     "best3" = mean of the three closest targets (a site is an analog if it matches SOME terrain).
  2. Purpose presets: theme weights per test objective (expert-set starting points, not learned).
  3. Analog classifier: learns "does this place look like sites scientists already use?" from the
     known analog sites (positives) against control cities and the unlabeled Earth grid.
  4. Explanation: a report built only from numbers in the tables, with an optional LLM rewrite hook.

Honesty rules baked in
  - Egyptian candidate sites are never used as labels.
  - Every reported number comes from held-out sites the method did not see while fitting.
"""
import csv
import json
import pathlib
import sys

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold

import terratwin_model as m

HERE = pathlib.Path(__file__).resolve().parent
BODIES = ("mars", "moon")
PURPOSES = {
    "general":         dict(thermal=1, dryness=1, pressure=1, terrain=1, surface=1),
    "rover_mobility":  dict(thermal=0.1, dryness=0.2, pressure=0.0, terrain=0.5, surface=0.2),
    "thermal_systems": dict(thermal=0.7, dryness=0.1, pressure=0.2, terrain=0.0, surface=0.0),
    "ice_drilling":    dict(thermal=0.6, dryness=0.2, pressure=0.0, terrain=0.2, surface=0.0),
    "crew_operations": dict(thermal=0.3, dryness=0.2, pressure=0.2, terrain=0.3, surface=0.0),
    "life_detection":  dict(thermal=0.3, dryness=0.5, pressure=0.0, terrain=0.0, surface=0.2),
}
# Which score is shipped per body, decided from the blind test. Both use similarity + classifier;
# the Moon needed 34 labelled sites and the geology flags before the classifier helped.
FINAL = {"mars": "combined", "moon": "combined"}
GEO = ["geo_volcanic", "geo_sedimentary", "geo_unconsolidated", "geo_crystalline", "geo_ice", "geo_young"]
UNITS = {"t_mean_c": "C", "t_max_c": "C", "t_min_c": "C", "t_diurnal_range_c": "C", "t_seasonal_range_c": "C",
         "precip_mm_year": "mm/yr", "pressure_kpa": "kPa", "relief_m": "m", "slope_deg": "deg", "albedo": ""}
NICE = {"t_mean_c": "mean temperature", "t_max_c": "hottest temperature", "t_min_c": "coldest temperature",
        "t_diurnal_range_c": "day-night swing", "t_seasonal_range_c": "seasonal swing",
        "precip_mm_year": "precipitation", "pressure_kpa": "air pressure", "relief_m": "local relief",
        "slope_deg": "slope", "albedo": "surface albedo"}


# --------------------------------------------------------------------------- data
class Data:
    def __init__(self):
        self.earth, self.targets = m.load()
        self.Xe, self.Xt, self.Ze, self.Zt, self.mu, self.sd = m.matrices(self.earth, self.targets)
        self.D = m.theme_distances(self.Ze, self.Zt)
        self.tb = np.array([t["body"] for t in self.targets])
        self.kind = np.array([e["kind"] for e in self.earth])
        self.grid = np.where(self.kind == "grid")[0]
        self.eid = {e["id"]: i for i, e in enumerate(self.earth)}
        self.tid = {t["target_id"]: j for j, t in enumerate(self.targets)}
        env = np.nan_to_num(self.Ze, nan=0.0)                # climate/terrain features (missing -> Earth mean)
        geo = np.array([[m.num(e.get(g)) for g in GEO] for e in self.earth])     # rock-type flags from Macrostrat
        self.Fs = {"env": env, "geo": np.column_stack([env, np.nan_to_num(geo, nan=0.0), np.isnan(geo[:, 0])])}
        self.F = self.Fs["geo"]

    def labels(self, body):
        """1 = known analog for this body, 0 = control city, -1 = unlabeled (grid, candidates, other body)."""
        y = np.full(len(self.earth), -1)
        for i, e in enumerate(self.earth):
            if e["kind"] != "site":
                continue
            if e["label"] == "analog" and body in e["analog_for"]:
                y[i] = 1
            elif e["label"] == "control":
                y[i] = 0
        return y

    def similarity(self, body, weights=None, agg="mean", targets=None):
        """Distance of every Earth point to a body (or to given target columns). Smaller = more alike."""
        cols = np.where(self.tb == body)[0] if targets is None else np.array(targets)
        w = weights or PURPOSES["general"]
        O = np.column_stack([m.overall({th: d[:, [j]] for th, d in self.D.items()}, w)[:, 0] for j in cols])
        if agg == "mean" or O.shape[1] == 1:
            return O.mean(axis=1)
        return np.sort(O, axis=1)[:, :3].mean(axis=1)        # best3


def pct(d):
    """Percentile where 100 = smallest distance."""
    return m.percentile(np.asarray(d, float)[:, None])[:, 0]


# --------------------------------------------------------------------- classifier
def make_classifier(kind):
    if kind == "logistic":
        return LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000)
    return GradientBoostingClassifier(n_estimators=120, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0)


def fit_classifier(data, body, kind, train_idx, feats="geo"):
    """Positives: known analogs. Negatives: control cities + unlabeled grid points (down-weighted,
    because some grid points really are analog-like)."""
    y = data.labels(body)
    tr = np.asarray(train_idx)
    lab = tr[y[tr] >= 0]
    bg = tr[(y[tr] == -1) & (data.kind[tr] == "grid")]
    idx = np.concatenate([lab, bg])
    yy = np.concatenate([y[lab], np.zeros(len(bg), int)])
    n_pos = max(int((yy == 1).sum()), 1)
    w = np.where(yy == 1, 1.0, np.concatenate([np.ones(len(lab)), np.full(len(bg), 0.3)]))
    w = np.where(yy == 1, w * (len(yy) - n_pos) / n_pos * 0.5, w)      # balance classes
    clf = make_classifier(kind)
    clf.fit(data.Fs[feats][idx], yy, sample_weight=w)
    clf.feats = feats
    return clf


def classifier_distance(clf, data):
    return -clf.predict_proba(data.Fs[clf.feats])[:, 1]                   # as a "distance": smaller = more analog-like


# ---------------------------------------------------------------------- evaluation
SIM_VARIANTS = {"similarity_mean": dict(agg="mean"), "similarity_best3": dict(agg="best3"),
                "similarity_best3_no_terrain": dict(agg="best3", weights=dict(PURPOSES["general"], terrain=0))}


def evaluate(data, repeats=10, folds=5, seed=0):
    """Repeated stratified K-fold over the labelled sites; the grid is split into the same folds.
    Everything that is fitted sees only the training folds. Returns mean held-out metrics."""
    rng = np.random.default_rng(seed)
    report = {}
    for body in BODIES:
        y = data.labels(body)
        lab = np.where(y >= 0)[0]
        res = {}
        sims = {k: data.similarity(body, **v) for k, v in SIM_VARIANTS.items()}     # nothing fitted
        cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
        for tr, te in cv.split(lab, y[lab]):
            gfold = rng.integers(0, folds, len(data.grid))
            k = rng.integers(0, folds)
            g_tr, g_te = data.grid[gfold != k], data.grid[gfold == k]
            train = np.concatenate([lab[tr], g_tr])
            pos, neg = lab[te][y[lab[te]] == 1], lab[te][y[lab[te]] == 0]
            scores = dict(sims)
            scores["classifier_logistic"] = classifier_distance(fit_classifier(data, body, "logistic", train, "geo"), data)
            scores["classifier_boosting_no_geology"] = classifier_distance(fit_classifier(data, body, "boosting", train, "env"), data)
            scores["classifier_boosting"] = classifier_distance(fit_classifier(data, body, "boosting", train, "geo"), data)
            # choose the best similarity variant on the TRAINING labels only, then combine
            ptr, ntr = lab[tr][y[lab[tr]] == 1], lab[tr][y[lab[tr]] == 0]
            best = max(SIM_VARIANTS, key=lambda v: m.auc(sims[v][ptr], sims[v][ntr]) + m.auc(sims[v][ptr], sims[v][g_tr]))
            scores["combined"] = -(pct(sims[best]) + pct(scores["classifier_boosting"])) / 2
            for name, d in scores.items():
                p_all = pct(d)
                res.setdefault(name, []).append([
                    m.auc(d[pos], d[neg]), m.auc(d[pos], d[g_te]),
                    float(np.mean(p_all[pos])), float(np.mean(p_all[pos] >= 75)), float(np.mean(p_all[neg]))])
        report[body] = {
            "n_analogs": int((y == 1).sum()), "n_controls": int((y == 0).sum()),
            "methods": {name: dict(zip(["auc_vs_controls", "auc_vs_earth_grid", "heldout_analog_mean_percentile",
                                        "heldout_analogs_in_top_quarter", "heldout_control_mean_percentile"],
                                       np.round(np.mean(v, axis=0), 3).tolist()))
                        for name, v in res.items()}}
    return report


def evaluate_spatial(data, folds=5):
    """Harder blind test: whole regions (20 x 30 degree blocks) are held out together, so a site is
    never scored by a model that saw its neighbours. Scores are out-of-fold percentiles."""
    from sklearn.model_selection import GroupKFold
    lat = np.array([float(e["lat"]) for e in data.earth])
    lon = np.array([float(e["lon"]) for e in data.earth])
    block = (np.floor(lat / 20) * 100 + np.floor(lon / 30)).astype(int)
    report = {}
    for body in BODIES:
        y = data.labels(body)
        sim = data.similarity(body, agg="best3")
        oof = {k: np.full(len(y), np.nan) for k in ("similarity_best3", "classifier_boosting", "combined")}
        allidx = np.arange(len(y))
        for tr, te in GroupKFold(n_splits=folds).split(allidx, groups=block):
            cd = classifier_distance(fit_classifier(data, body, "boosting", tr, "geo"), data)
            oof["similarity_best3"][te] = pct(sim)[te]
            oof["classifier_boosting"][te] = pct(cd)[te]
            oof["combined"][te] = pct(-(pct(sim) + pct(cd)) / 2)[te]
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        report[body] = {k: {"auc_vs_controls": round(m.auc(-v[pos], -v[neg]), 3),
                            "auc_vs_earth_grid": round(m.auc(-v[pos], -v[data.grid]), 3),
                            "heldout_analogs_in_top_quarter": round(float(np.mean(v[pos] >= 75)), 2)}
                        for k, v in oof.items()}
    return report


# -------------------------------------------------------------------------- train
def train(data):
    """Fit final models on all labels and write scores for every Earth point."""
    out = HERE / "outputs"
    out.mkdir(exist_ok=True)
    cols, table = ["earth_id", "kind", "name", "lat", "lon", "label"], {}
    imp = {}
    for body in BODIES:
        sim = data.similarity(body, agg="best3")
        clf = fit_classifier(data, body, "boosting", np.arange(len(data.earth)))
        cd = classifier_distance(clf, data)
        comb = pct(-(pct(sim) + pct(cd)) / 2)
        final = comb if FINAL[body] == "combined" else pct(sim)
        table[body] = (pct(sim), pct(cd), comb, final)
        imp[body] = dict(zip(m.FEATS + GEO + ["geo_missing"], np.round(clf.feature_importances_, 3).tolist()))
        cols += [f"{body}_similarity_pct", f"{body}_classifier_pct", f"{body}_combined_pct", f"{body}_final_pct"]
    with open(out / "ai_scores.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i, e in enumerate(data.earth):
            w.writerow([e["id"], e["kind"], e.get("name", ""), e["lat"], e["lon"], e.get("label", "")]
                       + [round(float(table[b][k][i]), 1) for b in BODIES for k in range(4)])
    # places nobody has labelled that both methods agree on
    disc = []
    for body in BODIES:
        y = data.labels(body)
        cand = [i for i in np.argsort(-table[body][3]) if y[i] == -1][:15]
        for r, i in enumerate(cand, 1):
            e = data.earth[i]
            disc.append([body, r, e["id"], e.get("name", ""), e["lat"], e["lon"]]
                        + [round(float(table[body][k][i]), 1) for k in range(4)])
    with open(out / "ai_new_candidates.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["body", "rank", "earth_id", "name", "lat", "lon", "similarity_pct", "classifier_pct", "combined_pct", "final_pct"])
        w.writerows(disc)
    (out / "ai_classifier_feature_importance.json").write_text(json.dumps(imp, indent=1))
    return table


# ---------------------------------------------------------- recommend and explain
def recommend(data, target_id, purpose="general", n=10, kind=None):
    j = data.tid[target_id]
    d = data.similarity(data.targets[j]["body"], weights=PURPOSES[purpose], targets=[j])
    p = pct(d)
    order = [i for i in np.argsort(d) if kind is None or data.earth[i]["kind"] == kind][:n]
    return [dict(rank=r, earth_id=data.earth[i]["id"], name=data.earth[i].get("name", ""),
                 lat=float(data.earth[i]["lat"]), lon=float(data.earth[i]["lon"]), score_pct=round(float(p[i]), 1))
            for r, i in enumerate(order, 1)]


def facts(data, earth_id, target_id, purpose="general"):
    """Everything the explanation is allowed to say, as plain numbers."""
    i, j = data.eid[earth_id], data.tid[target_id]
    e, t = data.earth[i], data.targets[j]
    body = t["body"]
    themes = {}
    for th in m.THEMES:
        col = data.D[th][:, j]
        if not np.isnan(col).all() and not np.isnan(col[i]):
            themes[th] = round(float(pct(col)[i]), 1)
    feats = []
    for k, f in enumerate(m.FEATS):
        a, b = data.Xe[i, k], data.Xt[j, k]
        if not (np.isnan(a) or np.isnan(b)):
            feats.append(dict(feature=f, label=NICE[f], unit=UNITS[f], earth=round(float(a), 2), target=round(float(b), 2),
                              gap_in_earth_std=round(float(abs(data.Ze[i, k] - data.Zt[j, k])), 2)))
    feats.sort(key=lambda x: x["gap_in_earth_std"])
    overall = pct(data.similarity(body, weights=PURPOSES[purpose], targets=[j]))[i]
    eb, bb = m.BODY["earth"], m.BODY[body]
    return dict(
        earth_site=dict(id=e["id"], name=e.get("name") or f"grid point {e['lat']}, {e['lon']}", lat=float(e["lat"]),
                        lon=float(e["lon"]), known_status=e.get("label") or "unlabeled grid point"),
        target=dict(id=t["target_id"], name=t["name"], body=body), purpose=purpose,
        overall_percentile=round(float(overall), 1), theme_percentiles=themes,
        closest_features=feats[:3], widest_gaps=feats[-3:][::-1],
        cannot_be_reproduced=dict(
            gravity="Earth and target gravity differ; numerical literature constants are not independently verified",
            pressure="Compare available pressure values; Earth cannot reproduce the target atmosphere",
            radiation="Radiation environments differ; no validated site-specific radiation comparison"))


def num0(v):
    return 0.0 if v in ("", None) else float(v)


def explain(data, earth_id, target_id, purpose="general", llm=None):
    """Plain-language report built only from facts(). Pass llm=callable(prompt)->text to have a language
    model rewrite it; the prompt forbids adding numbers that are not in the facts."""
    f = facts(data, earth_id, target_id, purpose)
    th = sorted(f["theme_percentiles"].items(), key=lambda x: -x[1])
    fmt = lambda x: f"{x['label']} ({x['earth']} {x['unit']} here vs {x['target']} {x['unit']} on the target)".replace("  ", " ")
    text = (
        f"{f['earth_site']['name']} as an analog for {f['target']['name']} ({f['target']['body'].title()}), "
        f"purpose: {purpose.replace('_', ' ')}.\n"
        f"Overall it is closer to the target than {f['overall_percentile']:.0f}% of the Earth locations we scored.\n"
        f"Strongest theme: {th[0][0]} (better than {th[0][1]:.0f}% of Earth). "
        f"Weakest theme: {th[-1][0]} (better than {th[-1][1]:.0f}%).\n"
        f"Closest matches: " + "; ".join(fmt(x) for x in f["closest_features"]) + ".\n"
        f"Widest gaps: " + "; ".join(fmt(x) for x in f["widest_gaps"]) + ".\n"
        f"What no Earth site can reproduce: " + "; ".join(f["cannot_be_reproduced"].values()) + ".")
    if llm is not None:
        prompt = ("Rewrite this analog-site report for a non-specialist in under 150 words. Use ONLY the numbers "
                  "in the JSON; do not add facts, places or numbers that are not there.\n\n"
                  + json.dumps(f, indent=1) + "\n\nDraft:\n" + text)
        text = llm(prompt)
    return text, f


# ---------------------------------------------------------------------------- CLI
def main(argv):
    data = Data()
    cmd = argv[0] if argv else "evaluate"
    if cmd == "evaluate":
        rep = evaluate(data)
        (HERE / "outputs").mkdir(exist_ok=True)
        (HERE / "outputs/ai_evaluation.json").write_text(json.dumps(rep, indent=1))
        rep["spatial_blocks"] = evaluate_spatial(data)
        (HERE / "outputs/ai_evaluation.json").write_text(json.dumps(rep, indent=1))
        for body, r in rep.items():
            if body == "spatial_blocks":
                continue
            print(f"\n{body.upper()}  ({r['n_analogs']} analogs, {r['n_controls']} controls) - held-out results")
            print(f"  {'method':<30}{'vs controls':>12}{'vs Earth':>10}{'analog pct':>12}{'top 25%':>9}{'ctrl pct':>10}")
            for name, v in r["methods"].items():
                print(f"  {name:<30}{v['auc_vs_controls']:>12.3f}{v['auc_vs_earth_grid']:>10.3f}"
                      f"{v['heldout_analog_mean_percentile']:>12.1f}{v['heldout_analogs_in_top_quarter']:>9.2f}"
                      f"{v['heldout_control_mean_percentile']:>10.1f}")
        print("\nRegion-blocked test (whole 20x30 degree regions held out):")
        for body, r in rep["spatial_blocks"].items():
            for name, v in r.items():
                print(f"  {body:<5}{name:<24}vs controls {v['auc_vs_controls']:.3f}   vs Earth {v['auc_vs_earth_grid']:.3f}   top 25% {v['heldout_analogs_in_top_quarter']:.2f}")
    elif cmd == "train":
        train(data)
        print("wrote outputs/ai_scores.csv, ai_new_candidates.csv, ai_classifier_feature_importance.json")
    elif cmd == "recommend":
        for r in recommend(data, argv[1], argv[2] if len(argv) > 2 else "general", int(argv[3]) if len(argv) > 3 else 10):
            print(f"{r['rank']:>3}. {r['earth_id']:<26}{r['name'][:34]:<36}{r['score_pct']:>6}")
    elif cmd == "explain":
        print(explain(data, argv[1], argv[2], argv[3] if len(argv) > 3 else "general")[0])
    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])

"""TerraTwin analog model: themed similarity between Earth locations and Moon/Mars targets.

Inputs  (data/):   earth_grid_features.csv, earth_features.csv, targets_features.csv
Outputs (outputs/): theme_scores.csv, rankings.csv, fidelity_gaps.csv, validation.json,
                    frontend_scores.json
        (training/): earth_X.csv, earth_X_scaled.csv, targets_X_scaled.csv,
                    theme_distances.csv, feature_spec.json

Method
  1. Each feature is transformed (log1p for skewed ones) and z-scored with the Earth spread.
  2. A theme distance is the RMS z-gap over that theme's features (only features both sides have).
  3. The overall distance is the weighted RMS of theme distances. Weights are equal by default;
     run with --learned to use training/learned_weights.json from train_weights.py.
  4. Scores are percentiles among all Earth points: 100 = the closest place on Earth.
     Absolute similarity is always low, so percentiles are what should be shown to users.
"""
import csv
import json
import math
import pathlib
import sys

import warnings

import numpy as np

warnings.filterwarnings("ignore", message="Mean of empty slice")

HERE = pathlib.Path(__file__).resolve().parent
THEMES = {
    "thermal": ["t_mean_c", "t_max_c", "t_min_c", "t_diurnal_range_c", "t_seasonal_range_c"],
    "dryness": ["precip_mm_year"],
    "pressure": ["pressure_kpa"],
    "terrain": ["relief_m", "slope_deg"],
    "surface": ["albedo"],
}
LOG = {"precip_mm_year", "relief_m", "slope_deg"}
FEATS = [f for fs in THEMES.values() for f in fs]
# Things no Earth site reproduces. Literature values - check before quoting in the write-up.
BODY = {
    "earth": dict(gravity_m_s2=9.81, day_length_h=24.0, surface_radiation_msv_day=0.007),
    "mars": dict(gravity_m_s2=3.71, day_length_h=24.66, surface_radiation_msv_day=0.67),
    "moon": dict(gravity_m_s2=1.62, day_length_h=708.7, surface_radiation_msv_day=1.37),
}


def num(v):
    return np.nan if v in ("", None) else float(v)


def load():
    grid = list(csv.DictReader(open(HERE / "data/earth_grid_features.csv", encoding="utf-8")))
    sites = list(csv.DictReader(open(HERE / "data/earth_features.csv", encoding="utf-8")))
    targets = list(csv.DictReader(open(HERE / "data/targets_features.csv", encoding="utf-8")))
    extra = HERE / "data/targets_extra.csv"                 # written by: python scrape_all.py targets
    if extra.exists():
        targets += list(csv.DictReader(open(extra, encoding="utf-8")))
    earth = []
    for g in grid:
        earth.append(dict(g, id=f"grid_{g['lat']}_{g['lon']}", kind="grid", name="", label="", analog_for=""))
    for s in sites:
        if s.get("t_mean_c"):                              # skip sites not scraped yet
            earth.append(dict(s, id=s["site_id"], kind="site"))
    return earth, targets


def transform(name, x):
    return np.log1p(np.maximum(x, 0)) if name in LOG else x


def matrices(earth, targets):
    Xe = np.array([[num(e.get(f)) for f in FEATS] for e in earth])
    Xt = np.array([[num(t.get(f)) for f in FEATS] for t in targets])
    Te = np.column_stack([transform(f, Xe[:, j]) for j, f in enumerate(FEATS)])
    Tt = np.column_stack([transform(f, Xt[:, j]) for j, f in enumerate(FEATS)])
    mu, sd = np.nanmean(Te, axis=0), np.nanstd(Te, axis=0)
    return Xe, Xt, (Te - mu) / sd, (Tt - mu) / sd, mu, sd


def theme_distances(Ze, Zt):
    """D[theme] = array (n_earth, n_targets); NaN where a theme has no shared feature."""
    D = {}
    for th, fs in THEMES.items():
        idx = [FEATS.index(f) for f in fs]
        gap = Ze[:, None, idx] - Zt[None, :, idx]              # NaN where either side is missing
        n = np.sum(~np.isnan(gap), axis=2)
        with np.errstate(invalid="ignore", divide="ignore"):
            D[th] = np.sqrt(np.nansum(gap ** 2, axis=2) / n)
        D[th][n == 0] = np.nan
    return D


def overall(D, weights):
    num_, den = 0, 0
    for th, d in D.items():
        w = weights.get(th, 0)
        ok = ~np.isnan(d)
        num_ = num_ + np.where(ok, w * d ** 2, 0)
        den = den + np.where(ok, w, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sqrt(num_ / den)


def percentile(d):
    """100 = smallest distance in the column. NaN stays NaN."""
    out = np.full(d.shape, np.nan)
    for j in range(d.shape[1]):
        ok = ~np.isnan(d[:, j])
        r = d[ok, j].argsort().argsort()
        out[ok, j] = 100 * (1 - r / max(ok.sum() - 1, 1))
    return out


def auc(pos, neg):
    """P(a positive is closer than a negative); 1.0 = perfect, 0.5 = chance."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(np.mean((pos[:, None] < neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])))


def load_weights():
    p = HERE / "training/learned_weights.json"
    if "--learned" in sys.argv and p.exists():       # default is equal weights
        return json.loads(p.read_text())["weights"], "learned"
    return {b: {th: 1.0 for th in THEMES} for b in ("mars", "moon")}, "equal"


def main():
    earth, targets = load()
    Xe, Xt, Ze, Zt, mu, sd = matrices(earth, targets)
    D = theme_distances(Ze, Zt)
    weights, wsrc = load_weights()
    tb = [t["body"] for t in targets]
    O = np.column_stack([overall({th: d[:, [j]] for th, d in D.items()}, weights[tb[j]])[:, 0]
                         for j in range(len(targets))])
    P = {th: percentile(d) for th, d in D.items()}
    PO = percentile(O)
    out, trn = HERE / "outputs", HERE / "training"
    out.mkdir(exist_ok=True)
    trn.mkdir(exist_ok=True)
    r1 = lambda v: "" if np.isnan(v) else round(float(v), 1)
    r3 = lambda v: "" if np.isnan(v) else round(float(v), 3)

    # ---- per-theme scores for every Earth point x target
    with open(out / "theme_scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["earth_id", "kind", "name", "lat", "lon", "target_id", "body", "overall_pct"]
                   + [f"{th}_pct" for th in THEMES] + ["overall_distance"] + [f"{th}_distance" for th in THEMES])
        for i, e in enumerate(earth):
            for j, t in enumerate(targets):
                w.writerow([e["id"], e["kind"], e.get("name", ""), e["lat"], e["lon"], t["target_id"], t["body"],
                            r1(PO[i, j])] + [r1(P[th][i, j]) for th in THEMES]
                           + [r3(O[i, j])] + [r3(D[th][i, j]) for th in THEMES])

    # ---- rankings + fidelity gaps for the top matches
    rank, gaps = [], []
    for j, t in enumerate(targets):
        order = np.argsort(O[:, j])
        for kind, n in (("grid", 15), ("site", 10)):
            for r, i in enumerate([i for i in order if earth[i]["kind"] == kind][:n], 1):
                e = earth[i]
                best = max(THEMES, key=lambda th: -1 if np.isnan(P[th][i, j]) else P[th][i, j])
                worst = min(THEMES, key=lambda th: 101 if np.isnan(P[th][i, j]) else P[th][i, j])
                rank.append([t["target_id"], t["body"], kind, r, e["id"], e.get("name", ""), e["lat"], e["lon"],
                             r1(PO[i, j]), best, worst] + [r1(P[th][i, j]) for th in THEMES])
                if r <= 5:
                    b, eb = BODY[t["body"]], BODY["earth"]
                    pe, pt = num(e.get("pressure_kpa")), num(t.get("pressure_kpa"))
                    gaps.append([t["target_id"], kind, r, e["id"], e.get("name", ""),
                                 round(eb["gravity_m_s2"] / b["gravity_m_s2"], 2),
                                 "vacuum on target" if pt == 0 else round(pe / pt, 1),
                                 r1(num(e.get("t_mean_c")) - num(t.get("t_mean_c"))),
                                 r1(num(e.get("t_min_c")) - num(t.get("t_min_c"))),
                                 r1(num(e.get("precip_mm_year"))),
                                 round(b["surface_radiation_msv_day"] / eb["surface_radiation_msv_day"]),
                                 b["day_length_h"]])
    with open(out / "rankings.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["target_id", "body", "earth_kind", "rank", "earth_id", "earth_name", "lat", "lon", "overall_pct",
                    "strongest_theme", "weakest_theme"] + [f"{th}_pct" for th in THEMES])
        w.writerows(rank)
    with open(out / "fidelity_gaps.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["target_id", "earth_kind", "rank", "earth_id", "earth_name", "gravity_times_too_strong",
                    "pressure_times_too_high", "mean_temp_gap_c", "min_temp_gap_c", "earth_precip_mm_year",
                    "target_radiation_times_earth", "target_day_length_h"])
        w.writerows(gaps)

    # ---- validation: known analog sites should rank above the control cities
    val = {"weights": wsrc}
    sites = [i for i, e in enumerate(earth) if e["kind"] == "site"]
    for body in ("mars", "moon"):
        cols = [j for j, b in enumerate(tb) if b == body]
        d = O[:, cols].mean(axis=1)
        pos = [i for i in sites if earth[i]["label"] == "analog" and body in earth[i]["analog_for"]]
        neg = [i for i in sites if earth[i]["label"] == "control"]
        bg = [i for i, e in enumerate(earth) if e["kind"] == "grid"]
        pct = percentile(d[:, None])[:, 0]
        val[body] = {
            "n_analog_sites": len(pos), "n_controls": len(neg),
            "auc_analog_vs_control": round(auc(d[pos], d[neg]), 3),
            "auc_analog_vs_earth_grid": round(auc(d[pos], d[bg]), 3),
            "median_percentile_analog_sites": round(float(np.median(pct[pos])), 1),
            "median_percentile_controls": round(float(np.median(pct[neg])), 1),
            "per_theme_auc_vs_control": {th: round(auc(np.nanmean(D[th][:, cols], axis=1)[pos],
                                                       np.nanmean(D[th][:, cols], axis=1)[neg]), 3)
                                         for th in THEMES if not np.isnan(D[th][:, cols]).all()},
            "site_percentiles": {earth[i]["id"]: round(float(pct[i]), 1) for i in sorted(sites, key=lambda i: -pct[i])},
        }
    (out / "validation.json").write_text(json.dumps(val, indent=1))

    # ---- frontend: one compact JSON for the heat map
    fe = {"themes": ["overall"] + list(THEMES),
          "targets": [dict(id=t["target_id"], name=t["name"], body=t["body"], lat=float(t["lat"]),
                           lon_east=float(t["lon_east"])) for t in targets],
          "points": []}
    ri = lambda v: None if np.isnan(v) else int(round(float(v)))
    for i, e in enumerate(earth):
        fe["points"].append(dict(id=e["id"], kind=e["kind"], name=e.get("name", ""), lat=float(e["lat"]),
                                 lon=float(e["lon"]),
                                 scores={t["target_id"]: [ri(PO[i, j])] + [ri(P[th][i, j]) for th in THEMES]
                                         for j, t in enumerate(targets)}))
    (out / "frontend_scores.json").write_text(json.dumps(fe, separators=(",", ":")))

    # ---- training tables
    meta = ["id", "kind", "name", "lat", "lon"]
    lab = lambda e, b: 1 if e["label"] == "analog" and b in e["analog_for"] else (0 if e["label"] == "control" else "")
    with open(trn / "earth_X.csv", "w", newline="") as f1, open(trn / "earth_X_scaled.csv", "w", newline="") as f2:
        w1, w2 = csv.writer(f1), csv.writer(f2)
        hdr = meta + ["label_mars_analog", "label_moon_analog"] + FEATS
        w1.writerow(hdr)
        w2.writerow(hdr)
        for i, e in enumerate(earth):
            m = [e["id"], e["kind"], e.get("name", ""), e["lat"], e["lon"], lab(e, "mars"), lab(e, "moon")]
            w1.writerow(m + ["" if np.isnan(v) else v for v in Xe[i]])
            w2.writerow(m + [r3(v) for v in Ze[i]])
    with open(trn / "targets_X_scaled.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["target_id", "body"] + FEATS)
        for j, t in enumerate(targets):
            w.writerow([t["target_id"], t["body"]] + [r3(v) for v in Zt[j]])
    with open(trn / "theme_distances.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["earth_id", "target_id", "body"] + list(THEMES))
        for i, e in enumerate(earth):
            for j, t in enumerate(targets):
                w.writerow([e["id"], t["target_id"], t["body"]] + [r3(D[th][i, j]) for th in THEMES])
    (trn / "feature_spec.json").write_text(json.dumps(
        {"features": FEATS, "themes": THEMES, "log1p": sorted(LOG),
         "earth_mean": dict(zip(FEATS, mu.round(4).tolist())), "earth_std": dict(zip(FEATS, sd.round(4).tolist())),
         "scaling": "z = (transform(x) - earth_mean) / earth_std"}, indent=1))

    print(f"{len(earth)} Earth points x {len(targets)} targets, weights: {wsrc}")
    for body in ("mars", "moon"):
        v = val[body]
        print(f"  {body}: AUC analog vs control {v['auc_analog_vs_control']}, vs Earth grid {v['auc_analog_vs_earth_grid']}, "
              f"median percentile analogs {v['median_percentile_analog_sites']} / controls {v['median_percentile_controls']}")
        print("     per theme:", v["per_theme_auc_vs_control"])
    return earth, targets, D, tb


if __name__ == "__main__":
    main()

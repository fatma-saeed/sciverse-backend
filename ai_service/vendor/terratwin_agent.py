"""TerraTwin mission agent: turns a plain-language request into a checked analog-site recommendation.

  python terratwin_agent.py "Where in Egypt can we test a Mars rover for Jezero crater?"
  python terratwin_agent.py "Best place to rehearse Artemis crew operations, not in Antarctica" --json

What makes it an agent rather than a lookup: it plans, calls tools, checks its own picks, and says how
sure it is.
  1. understand   - target or body, test purpose, region limits, how many options
  2. search       - themed similarity under the purpose weights, for each target
  3. cross-check  - compare with the trained classifier; disagreement lowers confidence
  4. audit        - data-quality flags: unverified coordinates, missing terrain, no geology, grid cell
  5. diversify    - drop picks that sit next to a better one (400 km apart, 100 km inside a named region)
  6. critique     - confidence per pick, what Earth cannot reproduce, what to verify before going
  7. answer       - written from the collected facts only (optional LLM rewrite)

Two ways to run it
  Agent(...).run(text)                       built-in planner, no LLM needed (what the CLI uses)
  Agent(...).run_with_llm(text, chat)        an LLM chooses the tool calls; `chat(messages) -> str`
                                             must return one JSON object per turn (see TOOL_SPEC)
"""
import json
import math
import re
import sys

from ai_service.guardrails import POLICY, validate_call, tool_message, validate_question

import numpy as np

import terratwin_ai as ai
import terratwin_model as m

REGIONS = {  # lat_min, lat_max, lon_min, lon_max
    "egypt": (22, 31.7, 24.7, 34.9), "africa": (-35, 37.5, -18, 52), "europe": (35, 72, -25, 45),
    "asia": (0, 78, 45, 180), "north america": (7, 84, -170, -52), "south america": (-56, 13, -82, -34),
    "australia": (-45, -10, 110, 155), "antarctica": (-90, -60, -180, 180), "middle east": (12, 42, 25, 63),
    "usa": (24, 50, -125, -66), "iceland": (63, 67, -25, -13), "chile": (-56, -17, -76, -66),
    "china": (18, 54, 73, 135), "canada": (42, 84, -141, -52),
}
PURPOSE_WORDS = {
    "rover_mobility": r"rover|mobility|driv|wheel|traverse|navigation",
    "thermal_systems": r"thermal|cold|heat|temperature|insulation|battery",
    "ice_drilling": r"\bice\b|drill|isru|water|resource|permafrost|volatile",
    "crew_operations": r"crew|astronaut|eva|habitat|spacesuit|human|rehears|operations",
    "life_detection": r"life|biosignature|astrobiolog|microb|organic|habitab",
}
TOOL_SPEC = {
    "list_targets": {"args": {"body": "mars|moon|any"}, "returns": "targets with id, name, body"},
    "search_sites": {"args": {"target_id": "str", "purpose": "one of PURPOSES", "n": "int", "region": "name or null",
                              "exclude_polar": "bool"}, "returns": "ranked Earth candidates for one target"},
    "cross_check": {"args": {"earth_id": "str", "body": "mars|moon"}, "returns": "similarity vs classifier percentiles"},
    "audit_site": {"args": {"earth_id": "str"}, "returns": "data-quality flags"},
    "site_facts": {"args": {"earth_id": "str", "target_id": "str", "purpose": "str"}, "returns": "numbers for the report"},
    "final_answer": {"args": {"text": "str"}, "returns": "ends the run"},
}


def km(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(h))


class Agent:
    def __init__(self):
        self.d = ai.Data()
        self.scores = ai.train(self.d)                       # body -> (similarity, classifier, combined, final)
        self.trace = []

    # ------------------------------------------------------------------ tools
    def list_targets(self, body="any"):
        return [dict(id=t["target_id"], name=t["name"], body=t["body"], status=t.get("site_status", ""))
                for t in self.d.targets if body in ("any", t["body"])]

    def search_sites(self, target_id, purpose="general", n=15, region=None, exclude_polar=False):
        if region is not None:
            region = region.strip().casefold()
            if region not in REGIONS:
                raise ValueError("Unsupported region")
        if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= len(self.d.earth):
            raise ValueError("Invalid result count")
        out = []
        for r in ai.recommend(self.d, target_id, purpose, n=len(self.d.earth)):
            if region and not self._in(region, r["lat"], r["lon"]):
                continue
            if exclude_polar and abs(r["lat"]) >= 60:
                continue
            if self.d.earth[self.d.eid[r["earth_id"]]].get("label") == "control":
                continue                                     # control cities are test negatives, never advice
            out.append(r)
            if len(out) == n:
                break
        return out

    def cross_check(self, earth_id, body):
        i = self.d.eid[earth_id]
        s, c, _, f = (float(x[i]) for x in self.scores[body])
        return dict(similarity_pct=round(s, 1), classifier_pct=round(c, 1), final_pct=round(f, 1),
                    agree=abs(s - c) <= 25, gap=round(abs(s - c), 1))

    def audit_site(self, earth_id):
        e = self.d.earth[self.d.eid[earth_id]]
        flags = []
        if earth_id == "kamil_crater_eg":
            flags.append("45 m crater: kilometre-scale terrain samples do not characterize crater geometry; dataset coordinate is rounded. Source: https://arpi.unipi.it/handle/11568/841212")
        if e["kind"] == "grid":
            flags.append("5-degree grid cell, not a specific site: treat as a region to survey")
        elif e.get("coords_verified", "no") == "no":
            flags.append("coordinates not verified")
        if e.get("relief_m", "") == "":
            flags.append("no terrain data")
        if e.get("geo_volcanic", "") == "":
            flags.append("no rock-type data")
        status = {"analog": "already used as an analog site", "candidate": "candidate proposed by this project",
                  "control": "control city"}.get(e.get("label", ""), "not a known analog site")
        return dict(status=status, flags=flags)

    def site_facts(self, earth_id, target_id, purpose="general"):
        return ai.facts(self.d, earth_id, target_id, purpose)

    def _in(self, region, lat, lon):
        a, b, c, d2 = REGIONS[region]
        return a <= lat <= b and c <= lon <= d2

    def _log(self, step, action, result):
        self.trace.append(dict(step=step, action=action, result=result))

    # --------------------------------------------------------------- planner
    def request_text(self, text):
        # UI selection is a fallback, never an override of an explicit question.
        question = re.split(r'\n(?:Reference|Earth site):', text, maxsplit=1)[0]
        explicit = re.search(r'\b(?:moon|lunar|artemis|mars|martian)\b|القمر|المريخ', question, re.I)
        if explicit:
            return question
        named_target = any(self.target_mentioned(tg, question.lower()) for tg in self.d.targets)
        return question if explicit or named_target else text

    @staticmethod
    def target_mentioned(target, text):
        words = [word for word in re.split(r"[^a-z]+", target['name'].lower())
                 if len(word) > 4 and word not in
                 ("crater", "planitia", "floor", "permanently", "shadowed", "planum", "vallis", "mensae", "summit")]
        return any(re.search(r'\b' + re.escape(word) + r'\b', text) for word in words)

    def understand(self, text):
        t = self.request_text(text).lower()
        body = "moon" if re.search(r"\b(?:moon|lunar|artemis)\b", t) else "mars" if re.search(r"\b(?:mars|martian)\b", t) else None
        target = None
        for tg in self.d.targets:
            if body and tg['body'] != body:
                continue
            if self.target_mentioned(tg, t):
                target = tg['target_id']
                break
        if target:
            body = self.d.targets[self.d.tid[target]]["body"]
        purpose = next((p for p, rx in PURPOSE_WORDS.items() if re.search(rx, t)), "general")
        region = next((r for r in sorted(REGIONS, key=len, reverse=True) if r in t), None)
        no_polar = bool(re.search(r"not in antarctica|no antarctica|accessible|easy to reach|avoid (the )?pol", t))
        if no_polar and region == "antarctica":
            region = None
        mnum = re.search(r"\b(?:top|best)\s+(\d+)|\b(\d+)\s+(?:[a-z]+\s+){0,2}?(?:sites|places|options|locations|analogs?|candidates|picks)\b", t)
        n = int(next(g for g in mnum.groups() if g)) if mnum else 3
        return dict(body=body, target=target, purpose=purpose, region=region, exclude_polar=no_polar,
                    n=max(1, min(n, 10)), assumed_body=body is None)

    # ---------------------------------------------------- request checks
    _NOT_PLACES = {"mars", "moon", "earth", "the", "a", "an", "my", "our", "this", "that", "space", "nasa", "artemis",
                   "january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
                   "november", "december", "summer", "winter", "spring", "autumn", "fall", "english", "arabic"}

    def unsupported_region(self, question):
        """A capitalised place after in/within/inside/near that we cannot filter on -> name it, never ignore it."""
        known = set(REGIONS)
        site_words = {w for e in self.d.earth if e.get("kind") != "grid" for w in re.split(r"[^a-z]+", (e.get("name") or "").lower()) if w}
        target_words = {w for t in self.d.targets for w in re.split(r"[^a-z]+", t["name"].lower()) if w}
        for m_ in re.finditer(r"\b(?:in|within|inside|near|across|around)\s+((?:the\s+)?[A-Z][A-Za-z\-]+(?:\s+[A-Z][A-Za-z\-]+){0,2})", question):
            place = re.sub(r"^the\s+", "", m_.group(1), flags=re.I).strip()
            low = place.lower()
            if low in known or low in self._NOT_PLACES:
                continue
            words = set(low.split())
            if words & (target_words | site_words) or words <= self._NOT_PLACES:
                continue
            return place
        return None

    def mentioned_site(self, question):
        """A named Earth site (analog, candidate or control city) the user asks about."""
        q = question.lower()
        best = None
        for e in self.d.earth:
            if e.get("kind") == "grid" or not e.get("name"):
                continue
            name = e["name"].lower()
            inner = [p for p in re.findall(r"\((.*?)\)", name)            # e.g. (Yungay), not (Apollo training)
                     if not re.search(r"training|tests?|area|core|rats|amadee|mission", p)]
            phrases = [name, re.sub(r"\s*\(.*?\)", "", name).strip()] + inner
            for ph in phrases:
                ph = ph.strip()
                if len(ph) >= 4 and re.search(r"\b" + re.escape(ph) + r"\b", q):
                    if best is None or len(ph) > best[1]:
                        best = (e["id"], len(ph))
        return best[0] if best else None

    def evaluate_site(self, earth_id, req):
        """Answer 'is X a good analog?' with the numbers for X itself."""
        body = req["body"]
        e = self.d.earth[self.d.eid[earth_id]]
        tid = req["target"] or next((t["id"] for t in self.list_targets(body)
                                     if "2024 list" in t["status"] or "lander site" in t["status"]),
                                    self.list_targets(body)[0]["id"])
        f = self.site_facts(earth_id, tid, req["purpose"])
        ck = self.cross_check(earth_id, body)
        au = self.audit_site(earth_id)
        final = np.asarray(self.scores[body][3], dtype=float)
        i = self.d.eid[earth_id]
        rank = int(np.sum(final > final[i]) + 1)
        total = int(np.sum(~np.isnan(final)))
        self._log("evaluate", f"evaluate_site({earth_id}, {tid}, {req['purpose']})",
                  dict(rank=rank, of=total, check=ck, flags=au["flags"]))
        name = f["earth_site"]["name"]
        tname = self.d.targets[self.d.tid[tid]]["name"]
        th = sorted(f["theme_percentiles"].items(), key=lambda x: -x[1])
        w = f["widest_gaps"][0] if f["widest_gaps"] else None
        verdict = ("This is a control city: it is used as a negative test, not as a recommended analog."
                   if e.get("label") == "control" else
                   "It is a known analog site." if e.get("label") == "analog" else
                   "It is a candidate proposed by this project." if e.get("label") == "candidate" else "")
        lines = [f"Evaluation of {name} as a {body.title()} analog (reference: {tname}, purpose {req['purpose'].replace('_', ' ')}).",
                 f"Overall: rank {rank} of {total} scored Earth locations (dataset percentile {ck['final_pct']}). {verdict}".strip(),
                 f"Purpose-specific percentile against {tname}: {f['overall_percentile']}."]
        if th:
            lines.append(f"Strongest theme: {th[0][0]} ({th[0][1]}); weakest: {th[-1][0]} ({th[-1][1]}).")
        lines.append(f"Second opinion: classifier {ck['classifier_pct']} vs similarity {ck['similarity_pct']}"
                     f" ({'they agree' if ck['agree'] else 'they disagree, treat with care'}).")
        if w:
            lines.append(f"Biggest gap: {w['label']} ({w['earth']} {w['unit']} here vs {w['target']} {w['unit']} on the target).")
        if au["flags"]:
            lines.append("Check first: " + "; ".join(au["flags"]) + ".")
        lines.append("Percentiles are relative rankings among Earth locations, not physical similarity percentages.")
        pick = dict(earth_id=earth_id, name=name, lat=float(e["lat"]), lon=float(e["lon"]), target_id=tid,
                    score_pct=f["overall_percentile"], rank=1, check=ck, audit=au, facts=f,
                    confidence="low" if e.get("label") == "control" else "medium", evaluation=True)
        return dict(request=dict(req, evaluate_site=earth_id), picks=[pick], answer="\n".join(lines),
                    trace=self.trace, status="complete", mode="site_evaluation")

    def run(self, text):
        text = self.request_text(validate_question(text))
        self.trace = []
        req = self.understand(text)
        if req['body'] is None:
            return dict(answer='Please specify Moon or Mars, or name a planetary target, before searching for Earth analogs.', picks=[], trace=[], status='clarification_required')
        question = re.split(r'\n(?:Reference|Earth site):', text, maxsplit=1)[0]
        place = None if req["region"] else self.unsupported_region(question)
        if place:
            supported = ", ".join(sorted(REGIONS))
            return dict(request=req, picks=[], trace=[], status="clarification_required",
                        answer=f"Region '{place}' is not supported yet, so no picks were returned instead of ignoring it. "
                               f"Supported regions: {supported}.")
        self._log("understand", "parse request", req)
        site = self.mentioned_site(question)
        if site:
            return self.evaluate_site(site, req)
        if req["target"]:
            tids = [req["target"]]
        else:                                                # no target named: use the body's listed landing sites
            tl = self.list_targets(req["body"])
            pref = [t["id"] for t in tl if "2024 list" in t["status"] or "lander site" in t["status"]]
            tids = (pref or [t["id"] for t in tl])[:4]
            self._log("understand", "no target named, choose representative targets", tids)
        # search every target, keep each site's best showing
        pool = {}
        for tid in tids:
            hits = self.search_sites(tid, req["purpose"], 25, req["region"], req["exclude_polar"])
            self._log("search", f"search_sites({tid}, {req['purpose']})", [h["earth_id"] for h in hits[:5]])
            for h in hits:
                if h["earth_id"] not in pool or h["score_pct"] > pool[h["earth_id"]]["score_pct"]:
                    pool[h["earth_id"]] = dict(h, target_id=tid)
        cands = sorted(pool.values(), key=lambda h: -h["score_pct"])[:20]
        if not cands:
            self._log("search", "nothing found in that region", None)
            return dict(request=req, picks=[], answer="No scored Earth location falls inside that region.", trace=self.trace)
        for c in cands:
            c["check"] = self.cross_check(c["earth_id"], req["body"])
            c["audit"] = self.audit_site(c["earth_id"])
            c["rank_score"] = 0.6 * c["score_pct"] + 0.4 * c["check"]["classifier_pct"]
        self._log("cross-check", "compare similarity with the trained classifier",
                  {c["earth_id"]: c["check"] for c in cands[:6]})
        self._log("audit", "data-quality flags", {c["earth_id"]: c["audit"]["flags"] for c in cands[:6]})
        cands.sort(key=lambda c: -c["rank_score"])
        picks = []
        gap_km = 100 if req["region"] else 400            # inside a named region, allow closer picks
        for c in cands:                                      # diversify: no two picks side by side
            if all(km((c["lat"], c["lon"]), (p["lat"], p["lon"])) > gap_km for p in picks):
                picks.append(c)
            if len(picks) == req["n"]:
                break
        self._log("diversify", f"keep one pick per {gap_km} km", [p["earth_id"] for p in picks])
        for p in picks:
            f = self.site_facts(p["earth_id"], p["target_id"], req["purpose"])
            conf = 0.5 + (0.2 if p["check"]["agree"] else -0.15) + (0.15 if "already used" in p["audit"]["status"] else 0) \
                - 0.07 * len(p["audit"]["flags"]) + (0.1 if p["rank_score"] >= 90 else 0)
            p["confidence"] = "high" if conf >= 0.75 else "medium" if conf >= 0.5 else "low"
            p["facts"] = f
        self._log("critique", "confidence per pick", {p["earth_id"]: p["confidence"] for p in picks})
        return dict(request=req, picks=picks, answer=self.compose(req, picks), trace=self.trace)

    def compose(self, req, picks):
        tname = lambda tid: self.d.targets[self.d.tid[tid]]["name"]
        lines = [f"Request understood as: {req['body'].title()} analog, purpose {req['purpose'].replace('_', ' ')}"
                 + (f", region {req['region']}" if req["region"] else "")
                 + (", polar sites excluded" if req["exclude_polar"] else "")
                 + (" (no body named, assumed Mars)" if req["assumed_body"] else "") + "."]
        for k, p in enumerate(picks, 1):
            f, ck = p["facts"], p["check"]
            th = sorted(f["theme_percentiles"].items(), key=lambda x: -x[1])
            w = f["widest_gaps"][0]
            lines.append(
                f"\n{k}. {f['earth_site']['name']} ({p['lat']}, {p['lon']}) - confidence {p['confidence']}\n"
                f"   Best match for {tname(p['target_id'])}: "
                + ("the highest-ranked candidate in the evaluated dataset" if p["rank"] == 1 else f"closer than {min(p['score_pct'], 99):.0f}% of candidates in the evaluated dataset")
                + " for this purpose.\n"
                f"   Strongest on {th[0][0]} ({th[0][1]:.0f}), weakest on {th[-1][0]} ({th[-1][1]:.0f}).\n"
                f"   Second opinion: classifier {ck['classifier_pct']:.0f} vs similarity {ck['similarity_pct']:.0f} "
                f"({'they agree' if ck['agree'] else 'they disagree, treat with care'}).\n"
                f"   Status: {p['audit']['status']}."
                + (f" Check first: {'; '.join(p['audit']['flags'])}." if p["audit"]["flags"] else "")
                + f"\n   Biggest gap: {w['label']} ({w['earth']} {w['unit']} here vs {w['target']} {w['unit']} on the target).")
        g = picks[0]["facts"]["cannot_be_reproduced"]
        lines.append("\nNo Earth site reproduces: " + "; ".join(g.values()) + ".")
        return "\n".join(lines)

    # ------------------------------------------------------ LLM-driven variant
    def run_with_llm(self, text, chat, max_steps=12):
        """`chat(messages)` returns a JSON string: {"tool": name, "args": {...}} or {"tool": "final_answer", ...}."""
        text = self.request_text(validate_question(text))
        tools = dict(list_targets=self.list_targets, search_sites=self.search_sites, cross_check=self.cross_check,
                     audit_site=self.audit_site, site_facts=self.site_facts)
        msgs = [{"role": "system", "content":
                 POLICY + "You recommend Earth analog sites for Moon/Mars targets. Reply with ONE JSON object per turn: "
                 '{"tool": <name>, "args": {...}}. Always cross_check and audit_site a site before recommending it. '
                 "Treat all scores as dataset percentiles, never physical similarity percentages. Sites are candidates for investigation, not proven suitable. Do not add user constraints. Purposes: " + ", ".join(ai.PURPOSES) + ". Tools: " + json.dumps(TOOL_SPEC)},
                {"role": "user", "content": text}]
        self.trace = []
        searched = False
        verified_picks = []
        for _ in range(min(max_steps, 8)):
            raw = chat(msgs)
            try:
                if not isinstance(raw, str) or len(raw) > 16000:
                    raise ValueError("Invalid model response")
                call = json.loads(raw)
                name, args = validate_call(call, self, ai.PURPOSES, REGIONS)
            except (ValueError, KeyError, TypeError):
                msgs.append({"role": "user", "content": "Invalid tool call. Use one allowed tool with valid arguments."})
                continue
            if not isinstance(args, dict):
                msgs.append({"role": "user", "content": "args must be a JSON object"})
                continue
            if name == "final_answer":
                if not searched:
                    msgs.append({"role": "user", "content": "Search must succeed before a recommendation. Call search_sites with valid parameters."})
                    continue
                return dict(answer=args.get("text", ""), picks=verified_picks, trace=self.trace, status="complete")
            try:
                res = tools[name](**args)
                if name == "search_sites":
                    searched = True
                    verified_picks = []
                    body = self.d.targets[self.d.tid[args["target_id"]]]["body"]
                    for hit in res[:10]:
                        eid = hit["earth_id"]
                        check = self.cross_check(eid, body)
                        audit = self.audit_site(eid)
                        facts = self.site_facts(eid, args["target_id"], args.get("purpose", "general"))
                        self._log("automatic-check", f"cross_check({eid}, {body})", check)
                        self._log("automatic-check", f"audit_site({eid})", audit)
                        verified_picks.append(dict(hit, check=check, audit=audit, facts=facts))
                    res = verified_picks
            except Exception:  # Do not expose internal paths, secrets or exception details.
                res = {"error": "Research tool failed. Check the identifiers and arguments."}
            self._log("llm", f"{name}({args})", res)
            msgs += [{"role": "assistant", "content": raw},
                     {"role": "user", "content": tool_message(res)}]
        return dict(answer="Stopped: step limit reached.", trace=self.trace, picks=verified_picks, status="incomplete")


def main(argv):
    if not argv:
        print(__doc__)
        return
    res = Agent().run(argv[0])
    if "--json" in argv:
        print(json.dumps(res, indent=1, default=float))
        return
    print("AGENT STEPS")
    for s in res["trace"]:
        print(f"  [{s['step']}] {s['action']}: {json.dumps(s['result'], default=float)[:150]}")
    print("\nANSWER\n" + res["answer"])


if __name__ == "__main__":
    main(sys.argv[1:])

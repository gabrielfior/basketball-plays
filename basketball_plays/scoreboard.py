"""Scoreboard OCR: game clock and both scores from the broadcast's bottom bar (tesseract)."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from basketball_plays.rosters import DUKE, MICHIGAN

# pixel regions (x1, y1, x2, y2) in the 1280x720 broadcast frame
REGIONS = {"away": (530, 598, 602, 648), "clock": (600, 598, 680, 628), "home": (678, 598, 758, 648)}
AWAY_TEAM, HOME_TEAM = MICHIGAN, DUKE


@dataclass
class ScoreboardRead:
    t: float
    clock: float | None  # seconds remaining, None when unreadable
    clock_text: str | None
    away: int | None
    home: int | None

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))


# Height, in pixels, that a *single glyph* is normalised to before it is OCR'd on its own (see
# `_glyph_text`). Tesseract segments a line of digits badly when the glyphs are huge — the 4x
# upscale below makes a 42px score box 168px tall, and tesseract then answers "1" for a clear
# "51" — but it reads those same digits correctly one at a time at this height. Normalising the
# whole crop to it instead was measured and is much worse: it shrinks the crops that already
# read, so the good cases break faster than the bad ones heal.
OCR_TARGET_HEIGHT = 48


def preprocess(crop: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if th.mean() < 127:  # light text on dark: make it dark on light
        th = 255 - th
    return cv2.copyMakeBorder(th, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)


def tesseract(img: np.ndarray, psm: int, whitelist: str = "0123456789:.") -> str:
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        cv2.imwrite(path, img)
        res = subprocess.run(
            ["tesseract", path, "-", "--psm", str(psm), "-c", f"tessedit_char_whitelist={whitelist}"],
            capture_output=True, text=True,
        )
        return res.stdout.strip()
    finally:
        os.unlink(path)


OCR_MODES = (7, 13, 8, 10)  # line, raw line (works for a lone digit), word, single character
# The glyph-at-a-time fallback samples two heights so that a disagreement between the two modes
# at one height can be outvoted rather than broken by mode order.
GLYPH_HEIGHTS = (OCR_TARGET_HEIGHT, 32)
GLYPH_MODES = (10, 13)  # single character, raw line: the two that suit one isolated digit
MAX_GLYPHS = 4  # "12:04" is the longest real reading; more components than that is noise


def _run_modes(img: np.ndarray, modes: tuple[int, ...] = OCR_MODES) -> list[str]:
    """OCR `img` once per page-segmentation mode, in `modes` order. Separate from `ocr_digits`
    so that tests can stub the tesseract calls out."""
    return [tesseract(img, psm).replace(" ", "") for psm in modes]


def _is_plausible(text: str) -> bool:
    """Whether an OCR string looks like something a scoreboard actually shows: a clock (m:ss, or
    tenths under a minute), a one- or two-digit score, or a clock whose colon tesseract dropped.

    That last case has to be admitted so it can compete on digit count: "1204" and "1:04" are
    both clock-shaped readings of 12:04, and the one with more digits is the one that did not
    lose a character. Only a bare four digits qualify — "1400." keeps its trailing junk and stays
    out, so it cannot outrank a clean "4:00"."""
    if re.fullmatch(r"\d{1,2}:\d{2}", text):
        return True
    if re.fullmatch(r"\d{1,2}\.\d", text):
        return float(text) < 60
    if re.fullmatch(r"\d{4}", text):
        return bool(clock_candidates(text))
    return bool(re.fullmatch(r"\d{1,2}", text))


def _glyphs(img: np.ndarray, height: int = OCR_TARGET_HEIGHT) -> list[np.ndarray]:
    """Split a preprocessed crop into its glyphs, left to right, each normalised to `height`.
    Two filters keep non-digits out, because over-segmenting is the dangerous direction: it makes
    the caller believe a digit went missing and re-read a crop that was already right. Anything
    narrower than a fifth of its height is a panel edge or divider — those hairlines run the full
    height of the crop, so they have to go before the tallest component is chosen, or they would
    become it. Then anything shorter than 60% of the tallest survivor is a speck, a colon dot or
    a clipped neighbouring graphic rather than a digit."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(255 - img, 8)
    boxes = [tuple(stats[i][:4]) for i in range(1, n)
             if stats[i][cv2.CC_STAT_AREA] >= 20
             and stats[i][cv2.CC_STAT_WIDTH] >= 0.2 * stats[i][cv2.CC_STAT_HEIGHT]]
    if not boxes:
        return []
    tallest = max(b[3] for b in boxes)
    out = []
    for x, y, w, h in sorted(b for b in boxes if b[3] >= 0.6 * tallest):
        scale = height / h
        g = cv2.resize(img[y:y + h, x:x + w], (max(1, round(int(w) * scale)), height),
                       interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
        out.append(cv2.copyMakeBorder(g, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255))
    return out


def _glyph_text(img: np.ndarray, n_glyphs: int) -> str:
    """Re-read `img` a glyph at a time, voting per position across modes and glyph heights.

    Used only when whole-crop OCR came back with fewer digits than there are glyphs, which is the
    dropped-digit signature. Reading one digit in isolation takes tesseract's line segmentation —
    the part that drops digits — out of the loop entirely.
    """
    by_height = [gs for gs in (_glyphs(img, h) for h in GLYPH_HEIGHTS) if len(gs) == n_glyphs]
    if not by_height:
        return ""
    out = []
    for i in range(n_glyphs):
        votes: Counter = Counter()
        for glyphs in by_height:
            for psm in GLYPH_MODES:
                text = tesseract(glyphs[i], psm, whitelist="0123456789").replace(" ", "")
                if len(text) == 1:
                    votes[text] += 1
            # The modes usually agree at the first height; only then is a second one worth
            # paying for, to break the tie rather than let mode order decide it.
            if len(votes) == 1 and sum(votes.values()) > 1:
                break
        if not votes:
            return ""
        out.append(votes.most_common(1)[0][0])
    return "".join(out)


def ocr_digits(crop: np.ndarray) -> str:
    """Read the digits in `crop`, running every mode and keeping the best answer.

    This used to return the first non-empty result, which let one bad mode decide the read:
    psm 7 answers "1" for a clear "51" often enough to matter, and being non-empty that answer
    was taken even though later modes disagreed. Instead all modes are run and the result is
    chosen — first preferring readings that look like a scoreboard value, then the reading the
    most modes agree on, then the longest, since the failure mode is a dropped digit and the
    longer reading is the more complete one. Ties fall back to the order in OCR_MODES.

    Some crops defeat every mode: tesseract answers "1" for a clear "51" and "7" for "77" at
    every psm and every scale, because its line segmentation, not its classifier, is what loses
    the digit. Those are caught by comparing the reading against the number of glyphs actually in
    the crop and, when digits are missing, re-reading one glyph at a time (`_glyph_text`).
    """
    img = preprocess(crop)
    texts = _run_modes(img)
    ranked = [(text, i) for i, text in enumerate(texts) if text]
    best = ""
    if ranked:
        # When nothing looks like a scoreboard value, keep every non-empty reading rather than
        # returning "": parse_score and clock_candidates can still salvage a dropped colon.
        pool = [(text, i) for text, i in ranked if _is_plausible(text)] or ranked
        counts = Counter(text for text, _ in pool)
        # Digit count leads: a reading that kept all its digits beats one that dropped a digit
        # but kept a colon ("1204" over "1:04"). Only then does agreement across modes decide,
        # then length, then the order in OCR_MODES.
        best = min(pool, key=lambda ti: (-len(re.sub(r"\D", "", ti[0])), -counts[ti[0]],
                                         -len(ti[0]), ti[1]))[0]
    # Whole-crop OCR saw fewer digits than the crop has glyphs, so tesseract's line segmentation
    # dropped one. Re-read a glyph at a time, which bypasses that segmentation. The glyph count
    # is bounded: a scoreboard box holds a score or a clock, so more than MAX_GLYPHS components
    # means the crop is framing something else, and paying for per-glyph OCR on noise is the one
    # way this fallback could cost real time.
    n_glyphs = len(_glyphs(img))
    if 2 <= n_glyphs <= MAX_GLYPHS and n_glyphs > len(re.sub(r"\D", "", best)):
        return _glyph_text(img, n_glyphs) or best
    return best


def parse_score(text: str) -> int | None:
    digits = re.sub(r"\D", "", text)
    if not digits or len(digits) > 3:
        return None
    return int(digits)


def clock_candidates(text: str) -> list[tuple[float, str]]:
    """Possible readings of an OCR clock string, most likely first.

    Tesseract often drops the colon or the decimal point, so "311" may be 3:11 or 31.1 (the
    clock shows tenths under one minute). All consistent readings are returned and
    `clean_timeline` picks the one that fits the neighbouring reads.
    """
    text = text.strip()
    m = re.fullmatch(r"(\d{1,2}):(\d{2})", text)
    if m:
        mm, ss = int(m.group(1)), int(m.group(2))
        return [(mm * 60 + ss, f"{mm}:{ss:02d}")] if mm <= 20 and ss < 60 else []
    out = []
    if re.fullmatch(r"\d{1,2}\.\d", text):
        # a tenths reading; a saved "31.1" may also be "3:11" whose colon was dropped
        val = float(text)
        if val < 60:
            out.append((val, f"{val:.1f}"))
        digits = re.sub(r"\D", "", text)
        if len(digits) == 3 and int(digits[1:]) < 60:
            out.append((int(digits[0]) * 60 + int(digits[1:]), f"{digits[0]}:{digits[1:]}"))
        return out
    digits = re.sub(r"\D", "", text)
    if len(digits) == 4 and int(digits[:2]) <= 20 and int(digits[2:]) < 60:
        out.append((int(digits[:2]) * 60 + int(digits[2:]), f"{int(digits[:2])}:{digits[2:]}"))
    if len(digits) == 3 and int(digits[1:]) < 60:
        out.append((int(digits[0]) * 60 + int(digits[1:]), f"{digits[0]}:{digits[1:]}"))
    if 2 <= len(digits) <= 3 and int(digits) / 10 < 60:
        val = int(digits) / 10
        out.append((val, f"{val:.1f}"))
    return out


def parse_clock(text: str) -> tuple[float | None, str | None]:
    cands = clock_candidates(text)
    return cands[0] if cands else (None, None)


def _clock_rank(text: str | None) -> tuple[int, int]:
    """Rank a `ScoreboardRead.clock_text` for choosing among competing region-set reads (see
    `read_frame`'s `compete` mode): a `m:ss` read beats a sub-minute tenths read (`m.s`), and
    among equally shaped reads the one with more digits wins (the more complete reading, same
    tie-break `ocr_digits` uses for whole-crop OCR). None loses to any plausible read; higher
    tuples win.
    """
    if text is None:
        return (-1, -1)
    digits = len(re.sub(r"\D", "", text))
    shape = 1 if re.fullmatch(r"\d{1,2}:\d{2}", text) else 0
    return (shape, digits)


def _read_regions(frame: np.ndarray, t: float, regions: dict, invert: bool) -> ScoreboardRead:
    def crop(key):
        x1, y1, x2, y2 = regions[key]
        c = frame[y1:y2, x1:x2]
        return cv2.bitwise_not(c) if invert else c

    clock, clock_text = parse_clock(ocr_digits(crop("clock")))
    return ScoreboardRead(t=round(t, 3), clock=clock, clock_text=clock_text,
                          away=parse_score(ocr_digits(crop("away"))),
                          home=parse_score(ocr_digits(crop("home"))))


def read_frame(frame: np.ndarray, t: float, regions: dict | None = None,
               invert: bool = False,
               alternatives: tuple[dict, ...] = (),
               compete: bool = False) -> ScoreboardRead:
    """OCR one frame's scoreboard. `regions` defaults to the ESPN boxes; `invert` flips the
    crop first, for layouts whose digits sit on a busy light ground.

    When `compete` is False (the default), `regions` is read first and, only when it doesn't
    yield a parseable clock, each of `alternatives` is tried in order (a broadcast that
    alternates between two scoreboard graphics) and the first whose clock parses is kept, with
    the scores read from that same alternative's regions.

    When `compete` is True, `regions` AND every entry in `alternatives` are read unconditionally
    (each alternative merged over `regions`, so it may override only some keys) and the read
    whose clock text ranks best under `_clock_rank` is kept, scores included, from that read's
    region set. Ties keep `regions`' own read (or the earliest-listed alternative among
    themselves). For a broadcast whose clock shifts position depending on another element's
    visibility, so neither a single region nor a fallback order can be trusted.
    """
    regions = REGIONS if regions is None else regions
    if compete:
        reads = [_read_regions(frame, t, regions, invert)]
        for alt in alternatives:
            reads.append(_read_regions(frame, t, {**regions, **alt}, invert))
        return max(reads, key=lambda r: _clock_rank(r.clock_text))
    result = _read_regions(frame, t, regions, invert)
    if result.clock is not None:
        return result
    for alt in alternatives:
        alt_result = _read_regions(frame, t, alt, invert)
        if alt_result.clock is not None:
            return alt_result
    return result


def _read_times(
    args: tuple[str, list[float], dict, bool, tuple[dict, ...], bool],
) -> list[ScoreboardRead]:
    video, times, regions, invert, alternatives, compete = args
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    out = []
    try:
        for t in times:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
            ok, frame = cap.read()
            if ok:
                if frame.shape[1] != 1280 or frame.shape[0] != 720:
                    frame = cv2.resize(frame, (1280, 720))
                out.append(read_frame(frame, float(t), regions=regions, invert=invert,
                                      alternatives=alternatives, compete=compete))
    finally:
        cap.release()
    return out


def read_timeline(video: str, start_s: float, end_s: float, every_s: float = 1.0,
                  workers: int = 8, layout: str = "espn") -> list[ScoreboardRead]:
    """OCR the scoreboard every `every_s` seconds, spreading the timestamps over worker threads.

    `layout` names the broadcaster's scoreboard graphic (see basketball_plays.broadcasts).
    Threads are enough: tesseract runs as a subprocess and OpenCV decoding releases the GIL;
    forking a process pool after importing OpenCV is unreliable on macOS.
    """
    from concurrent.futures import ThreadPoolExecutor

    from basketball_plays.broadcasts import get_layout  # here: broadcasts imports this module

    lay = get_layout(layout)
    times = [float(t) for t in np.arange(start_s, end_s, every_s)]
    chunks = [times[i::workers] for i in range(workers)]
    args = [(video, c, lay.regions, lay.invert, lay.alternatives, lay.compete)
            for c in chunks if c]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_read_times, args))
    return sorted((r for rs in results for r in rs), key=lambda r: r.t)


def _consistent(values: list[int | None], i: int) -> bool:
    """A score read is trusted when a neighbouring read agrees with it."""
    v = values[i]
    if v is None:
        return False
    return (i > 0 and values[i - 1] == v) or (i + 1 < len(values) and values[i + 1] == v)


RELOCK_K = 5
RELOCK_TOL_S = 1.5
MAX_DROP_SLACK_S = 120.0
RELOCK_SCAN_MAX = 60


def _drop_slack(t: float, last_t: float) -> float:
    """How far below `last` a candidate at time `t` may drop and still be an ordinary reading,
    given `last` was accepted at `last_t`: the elapsed time plus MAX_DROP_SLACK_S."""
    return (t - last_t) + MAX_DROP_SLACK_S


def _is_implausible_drop(c: float, t: float, last: float | None, last_t: float | None) -> bool:
    """Whether candidate `c` at time `t` drops further below `last` than `_drop_slack` allows —
    almost certainly a stray graphic or a replay of a much later moment, not a real clock drop."""
    return last is not None and last_t is not None and c < last - _drop_slack(t, last_t)


def _relock(
    out: list[ScoreboardRead],
    cands: list[list[tuple[float, str | None]]],
    i: int,
    last: float | None,
    last_t: float | None,
) -> tuple[float, str | None] | None:
    """Try to re-lock the clock onto read `i`'s best (first) candidate.

    Looks at the next RELOCK_K reads that have any candidate, scanning at most RELOCK_SCAN_MAX
    reads ahead to find them (so the worst case, a long run of unreadable reads, is bounded
    rather than scanning to the end of the timeline). Accepts the candidate if, for each of the
    RELOCK_K reads found, the candidate closest to the expected value is within RELOCK_TOL_S of a
    running clock (ticking down from ours) or, unless our candidate is an implausible drop, a
    stopped clock (frozen at ours). Returns the accepted (value, text) pair, or None.

    This applies to increases over `last` too: recovering from a poisoned anchor (e.g. `last`
    itself was a brief misread that the ordinary rule's neighbour check let through) is the
    normal case this is meant to handle, not an exception. Cleaning each period's reads on their
    own (never a whole game in one pass) is what keeps a genuine period boundary, such as half
    time, from being re-locked onto as if it were a recovery.
    """
    c, text = cands[i][0]
    is_drop = _is_implausible_drop(c, out[i].t, last, last_t)
    lookahead = []
    for j in range(i + 1, min(i + 1 + RELOCK_SCAN_MAX, len(out))):
        if cands[j]:
            lookahead.append(j)
            if len(lookahead) == RELOCK_K:
                break
    if len(lookahead) < RELOCK_K:
        return None

    def confirmed(running: bool) -> bool:
        for j in lookahead:
            dt = out[j].t - out[i].t
            expected = c - dt if running else c
            if min(abs(cc - expected) for cc, _ in cands[j]) > RELOCK_TOL_S:
                return False
        return True

    if confirmed(running=True) or (not is_drop and confirmed(running=False)):
        return (c, text)
    return None


def clean_timeline(
    reads: list[ScoreboardRead], valid_states: list[tuple[int, int]] | None = None
) -> list[ScoreboardRead]:
    """Reject OCR glitches.

    Scores must be confirmed by a neighbour (or equal the last accepted value) and never
    decrease. When `valid_states` is given (the (away, home) score sequence from play-by-play),
    a read is only accepted if the pair is one of those states and does not move backwards in
    that sequence, which catches systematic misreads such as 21 -> 27. The clock must be
    confirmed by a neighbour within 2 s and never increase.

    Re-lock (`_relock`) runs whenever the ordinary rule above accepts nothing at a read — a
    candidate that violates the monotonic drop/increase check, or one that never violates it but
    simply lacks neighbour confirmation. It tries to re-lock onto the read's best candidate by
    checking the next RELOCK_K reads against it (see `_relock`); this is how the cleaner recovers
    once `last` has been thrown off by a bad read, whichever direction it was thrown. A candidate
    that drops more than the elapsed time plus MAX_DROP_SLACK_S below the last accepted clock (a
    stray graphic, or a replay of a much later moment) is never accepted by the ordinary rule,
    only through that re-lock look-ahead, and then only via the running-clock hypothesis: a value
    that stays frozen far below where the clock should be is a graphic, not a genuine stoppage.

    Must be called once per period: this only tracks one running "last" value, and treats a
    period boundary such as half time — a real, large jump in the clock — the same as any other
    candidate to re-lock onto.

    Known limitation: a short run of identical misreads within MAX_DROP_SLACK_S of `last` (so
    the drop guard never flags it) is accepted outright by the ordinary rule's neighbour check,
    as a plausible stopped clock — see
    `test_clean_timeline_accepts_a_short_run_of_identical_misreads_as_a_stopped_clock` for an
    example. This is tolerated rather than fixed: such a run is rare, and because re-lock now
    also handles increases, the real reads that follow it re-lock and recovery is immediate.
    """
    out = [ScoreboardRead(r.t, r.clock, r.clock_text, r.away, r.home) for r in reads]
    if valid_states is not None:
        order = {}
        for i, st in enumerate(valid_states):
            order.setdefault(tuple(st), i)
        pairs = [(r.away, r.home) for r in out]
        best_i = -1
        for i, r in enumerate(out):
            pair = pairs[i]
            idx = order.get(pair)
            neighbour = (i > 0 and pairs[i - 1] == pair) or (i + 1 < len(pairs) and pairs[i + 1] == pair)
            ok = idx is not None and idx >= best_i and (neighbour or idx == best_i)
            if ok:
                best_i = idx
            else:
                r.away, r.home = None, None
    else:
        for key in ("away", "home"):
            vals = [getattr(r, key) for r in out]
            best = None
            for i, r in enumerate(out):
                v = vals[i]
                confirmed = v is not None and (_consistent(vals, i) or v == best)
                ok = confirmed and (best is None or best <= v <= best + 6)
                if ok:
                    best = v
                setattr(r, key, v if ok else None)
    # clock: choose, per read, the candidate reading that fits the last accepted clock and a
    # neighbour; the clock never increases while a period runs
    cands = [
        clock_candidates(r.clock_text) if r.clock_text else ([(r.clock, None)] if r.clock is not None else [])
        for r in out
    ]
    last, last_t = None, None
    for i, r in enumerate(out):
        chosen = None
        # the reading closest to the last accepted clock wins (1 s apart, the clock barely moves)
        ordered = sorted(cands[i], key=lambda ct: abs(ct[0] - last)) if last is not None else cands[i]
        for c, text in ordered:
            if last is not None and c > last + 0.5:
                continue
            if _is_implausible_drop(c, r.t, last, last_t):
                continue  # implausible drop; only `_relock` can accept it
            neighbours = [cc for j in (i - 1, i + 1) if 0 <= j < len(out) for cc, _ in cands[j]]
            if any(abs(n - c) <= 2.0 for n in neighbours):
                chosen = (c, text)
                break
        if chosen is None and cands[i]:
            chosen = _relock(out, cands, i, last, last_t)
        if chosen:
            r.clock, r.clock_text = chosen
            last, last_t = chosen[0], r.t
        else:
            r.clock, r.clock_text = None, None
    return out


def value_at(reads: list[ScoreboardRead], t: float, key: str, max_dt: float = 6.0):
    """Nearest trusted value of `key` to time t within max_dt seconds."""
    best, best_dt = None, None
    for r in reads:
        v = getattr(r, key)
        if v is None:
            continue
        dt = abs(r.t - t)
        if dt <= max_dt and (best_dt is None or dt < best_dt):
            best, best_dt = v, dt
    return best


def write_timeline(path: str | Path, reads: list[ScoreboardRead]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.writelines(r.to_json() + "\n" for r in reads)


def load_timeline(path: str | Path) -> list[ScoreboardRead]:
    with open(path) as f:
        return [ScoreboardRead(**json.loads(line)) for line in f if line.strip()]

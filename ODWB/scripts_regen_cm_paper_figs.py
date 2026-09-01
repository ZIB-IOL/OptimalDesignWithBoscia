#!/usr/bin/env python3
"""Regenerate selected paper figures with Computer Modern Unicode (CMU Serif)."""
from pathlib import Path
import re, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

ROOT = Path('/home/htc/dhendryc/SCRATCH/research_projects/OptimalDesignWithBoscia')
ODWB = ROOT / 'ODWB'
FR = ODWB / 'csv/full_runs_boscia'
BOS = ODWB / 'csv/Boscia'
SCIP = ODWB / 'csv/SCIPSDP'
OUT = ODWB / 'plots/root_lb'
PAPER = ROOT / 'plot'
OUT.mkdir(parents=True, exist_ok=True)
PAPER.mkdir(parents=True, exist_ok=True)

# Computer Modern Unicode (CMU Serif) — not Latin Modern
CMU = ROOT / '.fonts/cmu/cmunrm.otf'
assert CMU.is_file(), CMU
fm.fontManager.addfont(str(CMU))
PROP = fm.FontProperties(fname=str(CMU))
FONT_NAME = PROP.get_name()  # typically "CMU Serif"
plt.rcParams.update({
    'font.family': FONT_NAME,
    'font.serif': [FONT_NAME],
    'mathtext.fontset': 'cm',
    'axes.unicode_minus': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
})
print('font', FONT_NAME, 'from', CMU)

COLORS = {'Boscia': '#32CD32', 'SCIP B&B': '#2E8B57', 'SCIP OA': '#FA8072'}


def apply_cm(ax, legend=None):
    ax.set_xlabel(ax.get_xlabel(), fontproperties=PROP)
    ax.set_ylabel(ax.get_ylabel(), fontproperties=PROP)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontproperties(PROP)
    if legend is not None:
        for t in legend.get_texts():
            t.set_fontproperties(PROP)


def read_summary(path):
    raw = open(path, 'rb').read(80)
    d = ';' if b';' in raw else ','
    return next(csv.DictReader(open(path), delimiter=d))


def downsample(t, a, b, n=800):
    t, a, b = map(lambda x: np.asarray(x, float), (t, a, b))
    if len(t) <= n:
        return t, a, b
    idx = np.unique(np.concatenate([np.linspace(0, len(t) - 1, n).astype(int), [len(t) - 1]]))
    return t[idx], a[idx], b[idx]


def load_boscia_traj_max(path):
    rows = list(csv.DictReader(open(path)))
    t, lb, ub = [], [], []
    for r in rows:
        try:
            t.append(float(r['time']))
            lb.append(float(r['lowerBound']))
            ub.append(float(r['upperBound']))
        except Exception:
            continue
    t = np.asarray(t, float)
    if len(t) and t[-1] > 1e4:
        t = t / 1000.0
    dual = np.minimum.accumulate(-np.maximum.accumulate(lb))
    prim = np.maximum.accumulate(-np.minimum.accumulate(ub))
    return downsample(t, dual, prim)


def parse_scip_display(block):
    ts, duals, prims = [], [], []
    for line in block.splitlines():
        if '|' not in line:
            continue
        s = line.lstrip()
        if s.lower().startswith('time') or 'dualbound' in s.lower():
            continue
        while s and s[0] in '^~pL* ':
            s = s[1:]
        parts = [p.strip() for p in s.split('|')]
        if len(parts) < 15:
            continue
        try:
            tstr = re.sub(r'[^0-9.eE+-]', '', parts[0].split()[0])
            tt = float(tstr)
            db = float(parts[-4])
            pb = float(parts[-3])
        except Exception:
            continue
        if abs(pb) > 1e19:
            continue
        ts.append(tt)
        duals.append(db)
        prims.append(pb)
    if not ts:
        return None
    t = np.asarray(ts, float)
    dual = np.minimum.accumulate(duals)
    prim = np.maximum.accumulate(prims)
    return downsample(t, dual, prim)


def scip_blocks_in_order(log_path):
    text = open(log_path, errors='ignore').read()
    idxs = [m.start() for m in re.finditer(r'SCIP Status\s*:', text)]
    blocks = []
    for i, st in enumerate(idxs):
        prev = idxs[i - 1] if i else 0
        end = min(len(text), st + 800)
        blocks.append(text[prev:end])
    return blocks


def extract_scip(log_path, seed, summary_csv):
    summary = read_summary(summary_csv)
    target = float(summary.get('solution'))
    term = summary['termination']
    blocks = scip_blocks_in_order(log_path)
    candidates = []
    if 1 <= seed <= len(blocks):
        candidates.append(blocks[seed - 1])
    candidates.extend(blocks)
    best = None
    best_score = 1e300
    for block in candidates:
        traj = parse_scip_display(block)
        if traj is None:
            continue
        t, dual, prim = traj
        score = abs(prim[-1] - target)
        finished = ('OPTIMAL' in term) or ('gap limit reached' in block.lower())
        if score < best_score:
            best_score = score
            best = dict(t=t, dual=dual, prim=prim, finished=finished, score=score, term=term)
            if score < 1e-4:
                break
    return best


def find_log(crit, mode, tag, m):
    files = sorted(
        ODWB.glob(f'cb_{crit}_SCIPSDP_{mode}_{m}_{tag}_0_baseline_one_*.txt'),
        key=lambda p: -p.stat().st_size,
    )
    return files[0] if files else None


def plot_progress(series, outfile):
    fig, ax = plt.subplots(figsize=(5.8, 4.1), constrained_layout=True)
    for name in ['Boscia', 'SCIP OA', 'SCIP B&B']:
        if name not in series:
            continue
        s = series[name]
        c = COLORS[name]
        ax.plot(s['t'], s['prim'], color=c, lw=1.8, ls='-', label=f'{name} incumbent')
        ax.plot(s['t'], s['dual'], color=c, lw=1.5, ls='--', label=f'{name} dual')
        if s.get('finished'):
            ax.plot([s['t'][-1]], [s['prim'][-1]], marker='*', color=c, ms=12, ls='None', zorder=5)
            ax.plot([s['t'][-1]], [s['dual'][-1]], marker='*', color=c, ms=12, ls='None', zorder=5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Objective')
    ax.grid(True, alpha=0.25)
    leg = ax.legend(frameon=False, loc='best', fontsize=8)
    apply_cm(ax, leg)
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'{outfile}.{ext}', dpi=150)
        fig.savefig(PAPER / f'{outfile}.{ext}', dpi=150)
    plt.close(fig)
    print('wrote', outfile)


PROGRESS = [
    dict(
        outfile='progress_E_IND_80_12_1',
        crit='E',
        tag='IND',
        m=80,
        seed=1,
        bos=FR / 'boscia_optimized_E_optimality_independent__80_8_12_1.csv',
        bos_csv=BOS / 'boscia_optimized_E_optimality_independent__80_8_12_1.csv',
        bnb_csv=SCIP / 'scip_sdp_bnb_E_optimality_independent__80_8_12_1.csv',
        oa_csv=SCIP / 'scip_sdp_oa_E_optimality_independent__80_8_12_1.csv',
    ),
    dict(
        outfile='progress_E_CORR_80_12_1',
        crit='E',
        tag='CORR',
        m=80,
        seed=1,
        bos=FR / 'boscia_optimized_E_optimality_correlated__80_8_12_1.csv',
        bos_csv=BOS / 'boscia_optimized_E_optimality_correlated__80_8_12_1.csv',
        bnb_csv=SCIP / 'scip_sdp_bnb_E_optimality_correlated__80_8_12_1.csv',
        oa_csv=SCIP / 'scip_sdp_oa_E_optimality_correlated__80_8_12_1.csv',
    ),
    dict(
        outfile='progress_AGC_IND_80_s3',
        crit='AGC',
        tag='IND',
        m=80,
        seed=3,
        bos=FR / 'boscia_optimized_AGC_optimality_independent_disconnected_80_26_40_3.csv',
        bos_csv=BOS / 'boscia_optimized_AGC_optimality_independent_disconnected_80_26_40_3.csv',
        bnb_csv=SCIP / 'scip_sdp_bnb_AGC_optimality_independent_disconnected_80_26_40_3.csv',
        oa_csv=SCIP / 'scip_sdp_oa_AGC_optimality_independent_disconnected_80_26_40_3.csv',
    ),
    dict(
        outfile='progress_AGC_CORR_80_s5',
        crit='AGC',
        tag='CORR',
        m=80,
        seed=5,
        bos=FR / 'boscia_optimized_AGC_optimality_correlated_connected_80_26_40_5.csv',
        bos_csv=BOS / 'boscia_optimized_AGC_optimality_correlated_connected_80_26_40_5.csv',
        bnb_csv=SCIP / 'scip_sdp_bnb_AGC_optimality_correlated_connected_80_26_40_5.csv',
        oa_csv=SCIP / 'scip_sdp_oa_AGC_optimality_correlated_connected_80_26_40_5.csv',
    ),
]

for inst in PROGRESS:
    print('\n===', inst['outfile'])
    t, dual, prim = load_boscia_traj_max(inst['bos'])
    bos_term = read_summary(inst['bos_csv'])['termination']
    series = {'Boscia': dict(t=t, dual=dual, prim=prim, finished=(bos_term == 'OPTIMAL'))}
    print('  Boscia', bos_term, f'prim[{prim[0]:.3g},{prim[-1]:.3g}] dual[{dual[0]:.3g},{dual[-1]:.3g}]')
    for mode, label, csvk in [('bnb', 'SCIP B&B', 'bnb_csv'), ('oa', 'SCIP OA', 'oa_csv')]:
        log = find_log(inst['crit'], mode, inst['tag'], inst['m'])
        traj = extract_scip(log, inst['seed'], inst[csvk])
        if traj is None:
            print('  FAIL', label)
            continue
        print(
            f"  {label} score={traj['score']:.3g} fin={traj['finished']} "
            f"prim_end={traj['prim'][-1]:.4g} t_end={traj['t'][-1]:.1f}"
        )
        series[label] = traj
    plot_progress(series, inst['outfile'])


# ---- ACST / ACSTS pruning rel-gap (n=12, seed 1) ----
cb_blue = (0.0, 109 / 255, 219 / 255)
cb_purple = (73 / 255, 0.0, 146 / 255)
cb_rose = (255 / 255, 109 / 255, 182 / 255)
cb_blue_green = (0.0, 73 / 255, 73 / 255)
cb_salmon_pink = (255 / 255, 182 / 255, 119 / 255)

n, seed = 12, 1
tag = '66_12_11_1'
series_acst = [
    ('ACST default', f'boscia__ACST_optimality_independent__{tag}.csv', '-', cb_blue),
    ('ACST rank-prune', f'boscia_rank_based_pruning_ACST_optimality_independent__{tag}.csv', ':', cb_purple),
    ('ACSTS default', f'boscia__ACSTS_optimality_independent__{tag}.csv', '--', cb_rose),
    ('ACSTS rank-prune', f'boscia_rank_based_pruning_ACSTS_optimality_independent__{tag}.csv', '-.', cb_blue_green),
    ('ACSTS exclusion', f'boscia_exclusion_criterion_ACSTS_optimality_independent__{tag}.csv', ':', cb_salmon_pink),
]


def load_acst_traj(path):
    iters, gaps = [], []
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            it = float(row['LMOcalls'])
            lb = float(row['lowerBound'])
            ub = float(row['upperBound'])
            if not (np.isfinite(it) and it > 0 and np.isfinite(lb) and np.isfinite(ub)):
                continue
            gap = abs(ub - lb) / max(abs(ub), abs(lb), 1e-16)
            if gap <= 0 or not np.isfinite(gap):
                continue
            iters.append(it)
            gaps.append(gap)
    iters = np.asarray(iters)
    gaps = np.asarray(gaps)
    order = np.argsort(iters, kind='mergesort')
    iters, gaps = iters[order], gaps[order]
    gaps = np.minimum.accumulate(gaps)
    uniq_x, uniq_y = [], []
    for x, y in zip(iters, gaps):
        if uniq_x and x == uniq_x[-1]:
            uniq_y[-1] = y
        else:
            uniq_x.append(x)
            uniq_y.append(y)
    return np.asarray(uniq_x), np.asarray(uniq_y)


fig, ax = plt.subplots(figsize=(7.0, 5.4))
for label, fname, ls, col in series_acst:
    path = FR / fname
    x, y = load_acst_traj(path)
    ax.loglog(x, y, linestyle=ls, color=col, linewidth=2.0, alpha=0.9, label=label)
ax.set_xlabel('iteration')
ax.set_ylabel('relative gap')
ax.grid(True, which='both', linestyle=':', linewidth=0.6, alpha=0.7)
leg = ax.legend(loc='lower left', frameon=True)
apply_cm(ax, leg)
fig.tight_layout()
stem = f'acst_acsts_relgap_nodes_n{n}_seed{seed}'
for d in (ODWB / 'plots', PAPER):
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / f'{stem}.pdf')
    fig.savefig(d / f'{stem}.png', dpi=200)
    print('wrote', d / f'{stem}.pdf')
plt.close(fig)
print('\nDONE with Computer Modern Unicode (CMU Serif)')

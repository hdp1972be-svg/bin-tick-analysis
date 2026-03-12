# ----------------------------
# Drawing Utilities
# ----------------------------

def ecdf(arr):
    ''' DOCSTRING PLACEHOLDER '''
    x = np.sort(arr)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y

def stats_dict(arr):
    ''' DOCSTRING PLACEHOLDER '''
    med = np.median(arr)
    return {
        "min": np.min(arr),
        "max": np.max(arr),
        "mean": np.mean(arr),
        "mad": np.median(np.abs(arr - med)),
        "p90": np.percentile(arr, 90),
    }

def draw_regime_background(ax, center_x, width, y0, y1, side, color, alpha=0.25, zorder=0):
    if side == "left":
        x0 = center_x - width
    else:
        x0 = center_x

    rect = Rectangle(
        (x0, y0),
        width,
        y1 - y0,
        facecolor=color,
        edgecolor="none",
        alpha=alpha,
        zorder=zorder
    )
    ax.add_patch(rect)

def clip_violin_half(body, center_x, side):
    verts = body.get_paths()[0].vertices
    if side == "left":
        verts[:, 0] = np.minimum(verts[:, 0], center_x)
    else:
        verts[:, 0] = np.maximum(verts[:, 0], center_x)

def violin_half_width(violin_body, center_x):
    ''' DOCSTRING PLACEHOLDER '''
    verts = violin_body.get_paths()[0].vertices
    xs = verts[:, 0]
    return np.max(np.abs(xs - center_x))

def annotate_stats(ax, x, stats, side, color, violin=None, leader_length=0.05):
    """
    side: 'left' of 'right'
    violin: het violon object (v['bodies'][0]) voor deze data
    leader_length: lengte van de leader line
    """
    if violin is None:
        # fallback: gewoon dx
        dx = -leader_length if side == "left" else leader_length
        ha = "right" if side == "left" else "left"
        start_x = x
        for k, y in stats.items():
            ax.annotate(
                f"{k}: {y:.2f}".rstrip('0').rstrip('.'),
                xy=(start_x, y),
                xytext=(start_x + dx, y),
                arrowprops=dict(arrowstyle='-', lw=0.8),
                fontsize=8,
                ha=ha,
                va="center",
                color=color,
                alpha=0.85
            )
        return

    # haal vertices van de body
    verts = violin.get_paths()[0].vertices
    xs = verts[:, 0]
    ys = verts[:, 1]

    # voor elke stat: vind de x aan de rand bij die y
    for k, y in stats.items():
        # vind vertices dichtbij de y waarde
        idx = np.argmin(np.abs(ys - y))
        if side == "left":
            start_x = min(xs[ys == ys[idx]])
        else:
            start_x = max(xs[ys == ys[idx]])

        dx = -leader_length if side == "left" else leader_length
        ha = "right" if side == "left" else "left"

        ax.annotate(
            f"{k}: {y:.2f}".rstrip('0').rstrip('.'),
            xy=(start_x, y),
            xytext=(start_x + dx, y),
            arrowprops=dict(arrowstyle='-', lw=0.8),
            fontsize=8,
            ha=ha,
            va="center",
            color=color,
            alpha=0.85
        )

# ----------------------------
# Stats Utilities
# ----------------------------

def poisson_lambda(iat):
    iat = np.array(iat)
    return 1.0 / iat.mean()

def poisson_ks_stat(iat):
    ''' Goodness-of-fit against exponential '''
    iat = np.array(iat)
    lam = poisson_lambda(iat)
    sorted_iat = np.sort(iat)
    ecdf = np.arange(1, len(iat)+1) / len(iat)
    model = 1 - np.exp(-lam * sorted_iat)
    return np.max(np.abs(ecdf - model))

def mad(arr):
    """Median absolute deviation"""
    med = np.median(arr)
    return np.median(np.abs(arr - med))

def suggest_global_grid_width(price_moves, streaks):
    gaps = []
    for s in streaks:
        data = np.asarray(price_moves[s])
        if len(data) < 20:
            continue
        med = np.median(data)
        p90 = np.percentile(data, 90)
        gaps.append(p90 - med)
    if not gaps:
        return None
    return np.median(gaps)

def suggested_grid_width(arr):
    """Voorstel voor grid width gebaseerd op p90 - median"""
    arr = np.array(arr)
    med = np.median(arr)
    p90 = np.percentile(arr, 90)
    return float(p90 - med)

def compute_regimes(arr):
    """
    Bepaal regime-grenzen zoals p90 en p99
    """
    arr = np.array(arr)
    return {
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "p99.9": float(np.percentile(arr, 99.9)),
        "max": float(arr.max())
    }

def calculate_mass_ratios_old(arr, p90, p99):
    arr = np.array(arr)
    normal = np.sum(arr <= p90) / len(arr)
    burst = np.sum((arr > p90) & (arr <= p99)) / len(arr)
    extreme = np.sum(arr > p99) / len(arr)
    return {"normal": normal, "burst": burst, "extreme": extreme}

def calculate_mass_ratios(arr, p90, p99):
    arr = np.array(arr)
    total = len(arr)
    normal = np.sum(arr <= p90) / total if total > 0 else 0
    burst = np.sum((arr > p90) & (arr <= p99)) / total if total > 0 else 0
    extreme = np.sum(arr > p99) / total if total > 0 else 0
    return {"normal": normal, "burst": burst, "extreme": extreme}

def directional_asymmetry(up_data, down_data):
    if len(up_data) == 0 or len(down_data) == 0:
        return 0.0
    mean_up = np.mean(up_data)
    mean_down = np.mean(down_data)
    denominator = mean_up + mean_down
    if denominator == 0:
        return 0.0
    return (mean_up - mean_down) / denominator

def stability_score_old(arr):
    med = np.median(arr)
    mad_val = np.median(np.abs(arr - med))
    return mad_val / (med + 1e-10)

def stability_score(arr):
    med = np.median(arr)
    if med == 0:
        return 0.0
    mad_val = np.median(np.abs(arr - med))
    return mad_val / med

def calculate_burstiness(streak_sequence):
    """
    Bereken burstiness index:
    B = (σ - μ) / (σ + μ)
    waar μ en σ de mean en std van streak lengths zijn.
    
    B ≈ -1: regelmatig (periodiek)
    B ≈ 0: willekeurig (Poisson)
    B ≈ 1: bursty
    """
    if len(streak_sequence) < 2:
        return 0.0
    
    arr = np.array(streak_sequence)
    mu = np.mean(arr)
    sigma = np.std(arr)
    
    if sigma + mu == 0:
        return 0.0
    
    return float((sigma - mu) / (sigma + mu))

def entropy_of_deltas_old(arr, bins=20):
    hist, _ = np.histogram(arr, bins=bins, density=True)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))

def entropy_of_deltas(arr, bins=20):
    if len(arr) == 0:
        return 0.0
    hist, _ = np.histogram(arr, bins=bins, density=True)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))

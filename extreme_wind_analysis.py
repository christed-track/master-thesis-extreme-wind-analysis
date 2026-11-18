import numpy as np
import xarray as xr
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from scipy.stats import genextreme
import warnings

warnings.filterwarnings("ignore")


RADIUS_KM = 75.0
T_RETURN = 50


HIST_PATH = "data/TC-NH_TRACK_HadGEM3-GC31-HH_hist-1950_r1i1p1f1_gn_19500101-20141231.nc"

FUT_PATH = "data/TC-NH_TRACK_HadGEM3-GC31-HH_highres-future_r1i1p1f1_gn_20150101-20501231.nc"


SITES = {

    "Greater Changhua 1 SE": (24.1500, 119.9500),

    "Changhua Phase 2": (24.2871, 120.0506),

    "Yunlin": (23.5998, 120.0283),

    "Hai Long 2A": (24.0335, 119.7085),

    "Hai Long 3": (24.0528, 119.6414),

    "Formosa 1": (24.7097, 120.8152),

    "Formosa 2": (24.7720, 120.6472),

    "Formosa 4": (24.6071, 120.4459),

    "Guo Feng": (23.9555, 119.7375),

    "Shantou Nanpeng": (22.5830, 116.3330),

    "Jieyang Shenquan 1": (22.6527, 116.2684),

    "Shanwei Houhu": (22.6957, 115.9960),

    "Shanwei Jiazi 1": (22.5865, 116.0901),

    "Shanwei Jieshi": (22.7704, 116.0457),

    "Jieyang Qianzhan 1": (22.9344, 116.4310),

    "Jieyang Huilai 4": (22.8941, 116.3189),

}


# Beschriftungs-Versatz (dx, dy, Ausrichtung) – Standard: rechts oberhalb des Punktes

LABEL_OFFSET = {

    "Hai Long 3": (-0.06, 0.04, "right"),

    "Hai Long 2A": (0.06, -0.02, "left"),

    "Shanwei Jiazi 1": (0.00, -0.09, "center"),

    "Shantou Nanpeng": (0.07, -0.02, "left"),

    "Jieyang Huilai 4": (-0.06, -0.01, "right"),

}

DEFAULT_OFFSET = (0.05, 0.03, "left")


def haversine_km(lat, lon, lat0, lon0):

    R = 6371.0

    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)

    lat0 = np.deg2rad(lat0)
    lon0 = np.deg2rad(lon0)

    a = (
        np.sin((lat0 - lat) / 2.0)**2
        + np.cos(lat) * np.cos(lat0)
        * np.sin((lon0 - lon) / 2.0)**2
    )

    return 2.0 * R * np.arcsin(np.sqrt(a))


def rlT_from_annual_max(annual_max, T=50):

    x = np.asarray(annual_max, dtype=float)

    x = x[np.isfinite(x)]

    if len(x) < 20:

        return np.nan

    c, loc, scale = genextreme.fit(x)

    return float(
        genextreme.ppf(
            1.0 - 1.0 / T,
            c,
            loc=loc,
            scale=scale
        )
    )


def annual_maxima_for_site(ds, lat0, lon0, radius_km, year_min, year_max):

    wind = ds["sfcWind"].values.astype(float)

    lat = ds["lat_sfcWind"].values.astype(float)

    lon = ds["lon_sfcWind"].values.astype(float)

    time = ds["time"].values

    first = ds["FIRST_PT"].values.astype(int)

    npts = ds["NUM_PTS"].values.astype(int)

    dist = haversine_km(lat, lon, lat0, lon0)

    in_radius = dist <= radius_km

    peaks = np.full(len(first), np.nan)

    years = np.full(len(first), -9999, dtype=int)

    for i in range(len(first)):

        s = first[i]
        e = s + npts[i]

        try:

            y = time[s].year

        except AttributeError:

            y = int(str(time[s])[:4])

        years[i] = y

        if y < year_min or y > year_max:

            continue

        m = in_radius[s:e]

        if np.any(m):

            peaks[i] = np.nanmax(wind[s:e][m])

    ok = (
        np.isfinite(peaks)
        & (years >= year_min)
        & (years <= year_max)
    )

    if not np.any(ok):

        return np.array([])

    peaks_ok = peaks[ok]
    years_ok = years[ok]

    uniq = np.unique(years_ok)

    return np.array(
        [
            np.nanmax(peaks_ok[years_ok == y])
            for y in uniq
        ]
    )


# ---------------------------------------------------------
# Daten laden und ΔRL50 berechnen
# ---------------------------------------------------------

print("Lade Datensätze ...")

ds_h = xr.open_dataset(HIST_PATH)

ds_f = xr.open_dataset(FUT_PATH)

if "plev" in ds_h.dims:

    ds_h = ds_h.isel(plev=0)

    ds_f = ds_f.isel(plev=0)


names = list(SITES.keys())

lats = np.array([SITES[n][0] for n in names])

lons = np.array([SITES[n][1] for n in names])

delta_rl = np.full(len(names), np.nan)


for i, name in enumerate(names):

    lat0, lon0 = SITES[name]

    ann_h = annual_maxima_for_site(
        ds_h,
        lat0,
        lon0,
        RADIUS_KM,
        1950,
        2014
    )

    ann_f = annual_maxima_for_site(
        ds_f,
        lat0,
        lon0,
        RADIUS_KM,
        2015,
        2050
    )

    rl_h = (
        rlT_from_annual_max(ann_h)
        if ann_h.size
        else np.nan
    )

    rl_f = (
        rlT_from_annual_max(ann_f)
        if ann_f.size
        else np.nan
    )

    delta_rl[i] = rl_f - rl_h

    print(
        f" {name:<24} "
        f"RL50_h={rl_h:6.2f} "
        f"RL50_f={rl_f:6.2f} "
        f"Δ={delta_rl[i]:+.2f}"
    )


# ---------------------------------------------------------
# Karte
# ---------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 7))


valid = np.isfinite(delta_rl)

nan_mask = ~valid


sc = ax.scatter(

    lons[valid],
    lats[valid],

    c=delta_rl[valid],

    cmap="RdBu_r",
    vmin=-3,
    vmax=3,

    s=160,
    edgecolors="gray",
    linewidths=0.5,
    zorder=3

)

cbar = plt.colorbar(sc, ax=ax, shrink=0.8)

cbar.set_label(
    "ΔRL50 [m/s] (Future − Historic)",
    fontsize=10
)


if np.any(nan_mask):

    ax.scatter(
        lons[nan_mask],
        lats[nan_mask],
        c="lightgray",

        s=80,
        marker="x",
        label="Kein Fit möglich",
        zorder=3
    )

    ax.legend(fontsize=9)


for i, name in enumerate(names):

    dx, dy, ha = LABEL_OFFSET.get(
        name,
        DEFAULT_OFFSET
    )

    ax.text(
        lons[i] + dx,
        lats[i] + dy,
        name,

        fontsize=8,
        ha=ha,
        va="center",
        zorder=4
    )


ax.set_xlim(115.4, 121.2)

ax.set_ylim(22.4, 24.9)

ax.set_xlabel(
    "Longitude (°E)",
    fontsize=11
)

ax.set_ylabel(
    "Latitude (°N)",
    fontsize=11
)

ax.grid(alpha=0.3)


plt.tight_layout()

plt.savefig(
    "karte_delta_rl50_final.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()


print(
    "\nKarte gespeichert: "
    "karte_delta_rl50_final.png"
)
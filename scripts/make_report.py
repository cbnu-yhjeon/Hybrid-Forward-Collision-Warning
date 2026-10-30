"""results/*/metrics.json → results/summary.md + 그래프.

    uv run python scripts/make_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hfcw.distance.pinhole import V2_MEAN_HEIGHTS
from hfcw.eval.kitti_eval import BIN_NAMES, DIST_BINS

RES = ROOT / "results"
FIG = RES / "figures"
ORDER = ["E0", "E1", "E2", "E2ft", "E3", "E3a", "E3b", "E3bft", "E4", "E4m_raw", "E4m", "E4h",
         "E4hft", "E5"]
COLORS = {"E0": "#9aa0a6", "E1": "#5f6368", "E2": "#1a73e8", "E2ft": "#174ea6",
          "E3": "#81c995", "E3a": "#34a853", "E3b": "#188038", "E4": "#fbbc04",
          "E4m_raw": "#f6aea9", "E4m": "#e37400", "E4h": "#c5221f", "E5": "#a142f4",
          "E3bft": "#0d652d", "E4hft": "#7f0000"}


def fmt(v, p=3, pct=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v * 100:.1f}%" if pct else f"{v:.{p}f}"


def load():
    runs = {}
    for name in ORDER:
        p = RES / name / "metrics.json"
        if p.exists():
            runs[name] = json.loads(p.read_text())
    return runs


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join([":---:"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def rq2_decomposition(pairs_csv: Path) -> pd.DataFrame | None:
    """박스 오차 vs 평균 높이 가정 오차 분해 (상대 오차 기준, 2요인 Shapley).

    z_A = H_mean·fy/h_pred   (실제 E2)
    z_B = H_mean·fy/h_gt     (박스 오차 제거 = E5)
    z_C = H_gt·fy/h_pred     (높이 오차 제거)
    z_D = H_gt·fy/h_gt       (둘 다 제거 → 남는 것은 2D 박스 ≠ 3D 높이 투영 차이)
    """
    if not pairs_csv.exists():
        return None
    df = pd.read_csv(pairs_csv)
    Hm = df["cls"].map(V2_MEAN_HEIGHTS).fillna(1.5)
    zA = Hm * df.fy / df.h_pred
    zB = Hm * df.fy / df.h_gt
    zC = df.H_gt * df.fy / df.h_pred
    zD = df.H_gt * df.fy / df.h_gt
    rows = []
    bins = list(zip(DIST_BINS, BIN_NAMES)) + [((0, 1e9), "all")]
    for (lo, hi), name in bins:
        m = (df.z_gt >= lo) & (df.z_gt < hi)
        if m.sum() == 0:
            continue
        e = {k: float((abs(z[m] - df.z_gt[m]) / df.z_gt[m]).mean())
             for k, z in (("A", zA), ("B", zB), ("C", zC), ("D", zD))}
        box = 0.5 * ((e["A"] - e["B"]) + (e["C"] - e["D"]))
        height = 0.5 * ((e["A"] - e["C"]) + (e["B"] - e["D"]))
        # 로그 비율 가법 분해: log(zA/z) = log(Hm/Hgt) + log(h_gt/h_pred) + log(zD/z)
        lh = np.log(Hm[m] / df.H_gt[m])
        lb = np.log(df.h_gt[m] / df.h_pred[m])
        lr = np.log(zD[m] / df.z_gt[m])
        rows.append({"bin": name, "n": int(m.sum()), "total": e["A"], "no_box_err": e["B"],
                     "no_height_err": e["C"], "residual": e["D"], "box_contrib": box,
                     "height_contrib": height,
                     "log_total_mean": float((lh + lb + lr).mean()),
                     "log_total_std": float((lh + lb + lr).std()),
                     "log_height_mean": float(lh.mean()), "log_height_std": float(lh.std()),
                     "log_box_mean": float(lb.mean()), "log_box_std": float(lb.std()),
                     "log_resid_mean": float(lr.mean()), "log_resid_std": float(lr.std())})
    return pd.DataFrame(rows)


def plot_distance_bins(runs):
    keep = ("E1", "E2", "E2ft", "E3b", "E4", "E4m", "E4h", "E4hft", "E5")
    names = [n for n in runs if runs[n].get("distance") and n in keep]
    if not names:
        return
    x = np.arange(len(BIN_NAMES))
    w = 0.8 / len(names)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for k, n in enumerate(names):
        b = runs[n]["distance"]["bins"]
        mae = [b[bn]["mae_m"] or 0 for bn in BIN_NAMES]
        rel = [(b[bn]["rel_err"] or 0) * 100 for bn in BIN_NAMES]
        axes[0].bar(x + k * w - 0.4 + w / 2, mae, w, label=n, color=COLORS.get(n))
        axes[1].bar(x + k * w - 0.4 + w / 2, rel, w, label=n, color=COLORS.get(n))
    for ax, t in zip(axes, ["MAE (m)", "Mean relative error (%)"]):
        ax.set_xticks(x, [f"{b} m" for b in BIN_NAMES])
        ax.set_title(t)
        ax.grid(axis="y", alpha=0.3)
    axes[1].legend(fontsize=8, ncol=2)
    fig.suptitle("Distance error by GT range (KITTI val, matched detections)")
    fig.tight_layout()
    fig.savefig(FIG / "distance_error_bins.png", dpi=130)
    plt.close(fig)


def plot_scatter(runs):
    names = [n for n in ("E2", "E3b", "E4m", "E4h", "E4hft")
             if (RES / n / "distance_pairs.csv").exists()]
    if not names:
        return
    fig, axes = plt.subplots(1, len(names), figsize=(3.6 * len(names), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, n in zip(axes, names):
        df = pd.read_csv(RES / n / "distance_pairs.csv")
        ax.scatter(df.z_gt, df.z, s=2, alpha=0.25, color=COLORS.get(n))
        ax.plot([0, 80], [0, 80], "k--", lw=0.8)
        ax.set_xlim(0, 80)
        ax.set_ylim(0, 80)
        ax.set_title(n)
        ax.set_xlabel("GT z (m)")
    axes[0].set_ylabel("Estimated z (m)")
    fig.tight_layout()
    fig.savefig(FIG / "distance_scatter.png", dpi=130)
    plt.close(fig)


def plot_confusions(runs, key="risk_v2", unit="frame"):
    names = [n for n in runs if runs[n].get(key)]
    if not names:
        return
    fig, axes = plt.subplots(1, len(names), figsize=(2.6 * len(names), 2.8))
    axes = np.atleast_1d(axes)
    labels = ["SAFE", "CAUT", "DANG"]
    for ax, n in zip(axes, names):
        cm = np.array(runs[n][key][unit]["confusion"], dtype=float)
        norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.5 else "black")
        ax.set_xticks(range(3), labels, fontsize=7)
        ax.set_yticks(range(3), labels, fontsize=7)
        ax.set_title(n, fontsize=9)
        ax.set_xlabel("pred", fontsize=7)
    axes[0].set_ylabel("GT", fontsize=7)
    fig.suptitle(f"Risk confusion ({key}, {unit}-level, row-normalized color)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / f"confusion_{key}_{unit}.png", dpi=130)
    plt.close(fig)


def plot_pr(runs):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, cls in zip(axes, ["Car", "Pedestrian", "Cyclist"]):
        for n, m in runs.items():
            c = m["detection"].get(cls, {}).get("pr_curve")
            if c:
                r, p = np.array(c).T
                ax.plot(r, p, label=n, color=COLORS.get(n), lw=1.4)
        ax.set_title(f"{cls} (IoU 0.5)")
        ax.set_xlabel("Recall")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Precision")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "pr_curves.png", dpi=130)
    plt.close(fig)


def plot_rq2(dec: pd.DataFrame):
    d = dec[dec.bin != "all"]
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.bar(x, d.box_contrib * 100, label="box error", color="#1a73e8")
    ax.bar(x, d.height_contrib * 100, bottom=d.box_contrib * 100, label="mean-height assumption",
           color="#e37400")
    ax.bar(x, d.residual * 100, bottom=(d.box_contrib + d.height_contrib) * 100,
           label="residual (2D box vs 3D height)", color="#9aa0a6")
    ax.set_xticks(x, [f"{b} m" for b in d.bin])
    ax.set_ylabel("Mean relative error (%)")
    ax.set_title("RQ2: pinhole distance error decomposition (YOLO, E2)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "rq2_decomposition.png", dpi=130)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    runs = load()
    md = ["# 실험 결과 요약", "",
          f"KITTI Object val 분할 ({next(iter(runs.values()))['n_frames']}장)."
          " `scripts/make_report.py`가 생성.", ""]

    md += ["## 실험 구성", "", table(["ID", "설명"],
           [[n, m["config"].get("description", "")] for n, m in runs.items()]), ""]

    rows = []
    for n, m in runs.items():
        d = m["detection"]
        rows.append([n, fmt(d["Car"]["AP50_R40"]), fmt(d["Pedestrian"]["AP50_R40"]),
                     fmt(d["Cyclist"]["AP50_R40"]), fmt(d["All"]["precision"]),
                     fmt(d["All"]["recall"]), fmt(d["All"]["fp_per_frame"], 2)])
    md += ["## 검출", "", "IoU 0.5, KITTI moderate 조건, 40점 보간 AP. 고전 검출기는 신뢰도가 없어"
           " AP가 단일 PR 점으로 계산됩니다.", "",
           table(["ID", "AP Car", "AP Ped", "AP Cyc", "Precision", "Recall", "FP/frame"], rows),
           ""]

    rows = []
    for n, m in runs.items():
        dist = m.get("distance")
        if not dist:
            rows.append([n, "—", "—", "—"] + ["—"] * len(BIN_NAMES) + ["—", "—", "—"])
            continue
        o = dist["overall"]
        rows.append([n, fmt(o["mae_m"], 2), fmt(o["rel_err"], pct=True),
                     fmt(o["median_rel_err"], pct=True)]
                    + [fmt(dist["bins"][b]["rel_err"], pct=True) for b in BIN_NAMES]
                    + [fmt(dist["truncation"]["<=0.15"]["rel_err"], pct=True),
                       fmt(dist["truncation"][">0.15"]["rel_err"], pct=True), o["n"]])
    md += ["## 거리", "", "검출과 GT를 IoU ≥ 0.5로 매칭한 쌍에서 계산. 구간별 값은 평균 상대 오차"
           " `|ẑ − z| / z`.", "",
           table(["ID", "MAE (m)", "상대 오차", "중앙값"] + [f"{b} m" for b in BIN_NAMES]
                 + ["잘림 ≤ 0.15", "잘림 > 0.15", "매칭 수"], rows),
           "", "![distance](figures/distance_error_bins.png)", "",
           "![scatter](figures/distance_scatter.png)", ""]

    for key, title in (("risk_v2", "위험 등급 — v2 점수 공식 (주 지표)"),
                       ("risk_geo", "위험 등급 — 추정 (Z, X)에 GT 규칙 적용 (보조 지표)")):
        rows = []
        for n, m in runs.items():
            r = m.get(key)
            if not r:
                continue
            rows.append([n, fmt(r["frame"]["macro_f1"]), fmt(r["frame"]["danger_recall"]),
                         fmt(r["frame"]["danger_precision"]), fmt(r["object"]["macro_f1"]),
                         fmt(r["object"]["danger_recall"]), fmt(r["object"]["danger_precision"])])
        md += [f"## {title}", "",
               "프레임 단위: 프레임의 최고 등급끼리 비교 (FCW 경보 관점). 객체 단위: 매칭 쌍 +"
               " 놓친 GT(예측 SAFE) + 오검출(GT SAFE).", "",
               table(["ID", "프레임 Macro-F1", "프레임 DANGER 재현율", "프레임 DANGER 정밀도",
                      "객체 Macro-F1", "객체 DANGER 재현율", "객체 DANGER 정밀도"], rows), "",
               f"![cm](figures/confusion_{key}_frame.png)", ""]

    dec = rq2_decomposition(RES / "E2" / "distance_pairs.csv")
    if dec is not None:
        plot_rq2(dec)
        md += ["## RQ2 — 거리 오차 분해 (E2 YOLO 박스)", "",
               "같은 매칭 쌍에서 박스 높이(예측/GT)와 실제 높이(평균/GT)를 바꿔 끼워 평균 상대 오차를"
               " 계산하고, 두 요인 기여를 Shapley 방식으로 나눴습니다.", "",
               table(["구간", "n", "전체 (E2)", "박스 오차 제거", "높이 오차 제거", "잔차",
                      "박스 기여", "높이 가정 기여"],
                     [[r.bin, r.n, fmt(r.total, pct=True), fmt(r.no_box_err, pct=True),
                       fmt(r.no_height_err, pct=True), fmt(r.residual, pct=True),
                       fmt(r.box_contrib, pct=True), fmt(r.height_contrib, pct=True)]
                      for r in dec.itertuples()]), "",
               "절댓값 기준 분해는 부호가 반대인 오차끼리 상쇄되면 기여가 음수로 나올 수 있습니다."
               " 그래서 로그 비율 `log(ẑ/z) = log(H_mean/H) + log(h_gt/h_pred) + log(z_D/z)`의 정확한"
               " 가법 분해도 함께 봅니다. 평균은 편향(+면 과대추정), 표준편차는 산포입니다.", "",
               table(["구간", "전체 편향 ± 산포", "높이 가정", "박스", "잔차 (2D 박스 ≠ 3D 높이)"],
                     [[r.bin, f"{r.log_total_mean * 100:+.1f} ± {r.log_total_std * 100:.1f}",
                       f"{r.log_height_mean * 100:+.1f} ± {r.log_height_std * 100:.1f}",
                       f"{r.log_box_mean * 100:+.1f} ± {r.log_box_std * 100:.1f}",
                       f"{r.log_resid_mean * 100:+.1f} ± {r.log_resid_std * 100:.1f}"]
                      for r in dec.itertuples()]), "",
               "(단위: 로그 비율 × 100 ≈ %)", "",
               "![rq2](figures/rq2_decomposition.png)", ""]

    speed_path = RES / "speed.json"
    if speed_path.exists():
        speed = json.loads(speed_path.read_text())
        rows = []
        for n in ORDER:
            for dev in ("cuda", "cpu"):
                s = speed.get(n, {}).get(dev)
                if s:
                    rows.append([n, "GPU" if dev == "cuda" else "CPU", fmt(s["fps"], 1),
                                 fmt(s["preprocess_ms"], 1), fmt(s["detect_ms"], 1),
                                 fmt(s["distance_ms"], 1), fmt(s["track_risk_ms"], 1),
                                 fmt(s["total_ms"], 1)])
        md += ["## 속도", "", "`scripts/benchmark_speed.py`, val 200장, 다른 작업 없이 측정. "
               "RTX 5060 Ti 16GB / Core Ultra 5 245KF (WSL2, 12 스레드). 이미지 디코딩 제외."
               " 고전 단계(전처리·특징)는 항상 CPU.", "",
               table(["ID", "장치", "FPS", "전처리 ms", "검출 ms", "거리 ms", "추적·위험도 ms",
                      "전체 ms"], rows), ""]

    ttc_path = RES / "ttc" / "metrics.json"
    if ttc_path.exists():
        ttc = json.loads(ttc_path.read_text())
        rows = []
        for n, m in ttc.items():
            for meth in ("ttc_v2", "ttc_scale", "ttc_distance"):
                s = m.get(meth, {})
                rows.append([n, meth, m["n_pairs"], fmt(s.get("coverage"), pct=True),
                             fmt(s.get("mae_s"), 2), fmt(s.get("median_ae_s"), 2),
                             fmt(s.get("within_1s"), pct=True)])
        md += ["## TTC (KITTI Tracking, GT TTC < 5 s)", "",
               "GT TTC는 GT 트랙 z를 7프레임 창에서 직선 맞춤한 접근 속도로 계산.", "",
               table(["ID", "방법", "쌍 수", "커버리지", "MAE (s)", "중앙 AE (s)", "|오차| < 1 s"],
                     rows), ""]

    plot_distance_bins(runs)
    plot_scatter(runs)
    plot_confusions(runs, "risk_v2", "frame")
    plot_confusions(runs, "risk_geo", "frame")
    plot_pr(runs)
    (RES / "summary.md").write_text("\n".join(md))
    print((RES / "summary.md").read_text())


if __name__ == "__main__":
    main()

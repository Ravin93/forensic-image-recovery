from __future__ import annotations
import csv, json, time
from collections import defaultdict
from pathlib import Path
from typing import Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from app.core.logger import logger
from app.modules.corruption.simulator import corrupt_image
from app.modules.evaluation.metrics import compute_psnr, compute_ssim
from app.modules.reconstruction.repair_pipeline import run_repair_pipeline

_REPORTS_DIR = Path(__file__).resolve().parents[3] / "data" / "reports"
_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

_BENCHMARK_CONFIG = {
    "scratch_lines":        {"count": 4,  "seed": 42},
    "multiple_bars":        {"count": 3,  "seed": 42},
    "zone_deletion":        {"x": 20, "y": 20, "width": 60, "height": 60},
    "random_holes":         {"count": 5,  "seed": 42},
    "local_noise":          {"x": 20, "y": 20, "width": 60, "height": 60, "sigma": 30.0},
    "large_deleted_square": {"size_ratio": 0.3, "seed": 42},
    "block_dropout":        {"block_size": 16, "drop_ratio": 0.3, "seed": 42},
    "mixed":                {"seed": 42},
}
_COLORS = ["#00e5ff","#a259ff","#ff3d71","#00e096","#ffaa00","#ff6b6b","#4ecdc4","#45b7d1"]

def run_benchmark(image_paths, output_dir=None, corruption_types=None, max_attempts=6, seed=42):
    out_dir = Path(output_dir or _REPORTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    types_to_test = corruption_types or list(_BENCHMARK_CONFIG.keys())
    rows = []
    for img_path in image_paths:
        img_path = Path(img_path)
        if not img_path.exists():
            continue
        for ct in types_to_test:
            params = dict(_BENCHMARK_CONFIG.get(ct, {"seed": seed}))
            try:
                t0 = time.perf_counter()
                # seed est déjà dans params via _BENCHMARK_CONFIG, ne pas le passer en double
                params.setdefault("seed", seed)
                cr = corrupt_image(image_path=img_path, corruption_type=ct, **params)
                pb = float(compute_psnr(img_path, cr["image_path"]))
                sb = float(compute_ssim(img_path, cr["image_path"]))
                rr = run_repair_pipeline(
                    corrupted_image_path=cr["image_path"], mask_path=cr["mask_path"],
                    original_image_path=img_path, corruption_type=ct,
                    detection_confidence=1.0, max_attempts=max_attempts)
                elapsed = time.perf_counter() - t0
                pa = float(compute_psnr(img_path, rr["path"]))
                sa = float(compute_ssim(img_path, rr["path"]))
                rows.append({
                    "image": img_path.name, "corruption_type": ct,
                    "psnr_before": round(pb,3), "psnr_after": round(pa,3), "psnr_gain": round(pa-pb,3),
                    "ssim_before": round(sb,4), "ssim_after": round(sa,4), "ssim_gain": round(sa-sb,4),
                    "score": round(float(rr.get("score",0)),2),
                    "strategy": rr.get("selected_repair_strategy",""),
                    "n_candidates": len(rr.get("candidates",[])),
                    "elapsed_s": round(elapsed,2),
                })
            except Exception as exc:
                logger.warning("Benchmark echec %s %s: %s", img_path.name, ct, exc)
                rows.append({"image":img_path.name,"corruption_type":ct,
                    "psnr_before":None,"psnr_after":None,"psnr_gain":None,
                    "ssim_before":None,"ssim_after":None,"ssim_gain":None,
                    "score":None,"strategy":"error","n_candidates":0,"elapsed_s":None})
    summary = _build_summary(rows)
    ts = int(time.time())
    csv_path  = out_dir / f"benchmark_{ts}.csv"
    json_path = out_dir / f"benchmark_{ts}.json"
    _save_csv(rows, csv_path)
    json_path.write_text(
        __import__("json").dumps({"rows":rows,"summary":summary},indent=2,default=str),
        encoding="utf-8")
    plots = generate_benchmark_plots(rows, summary, out_dir, ts)
    return {"rows":rows,"summary":summary,"paths":{"csv":str(csv_path),"json":str(json_path),**plots}}


def _build_summary(rows):
    groups = defaultdict(list)
    for r in rows:
        if r.get("score") is not None:
            groups[r["corruption_type"]].append(r)
    summary = {}
    for ct, group in groups.items():
        pg=[r["psnr_gain"] for r in group if r.get("psnr_gain") is not None]
        sg=[r["ssim_gain"] for r in group if r.get("ssim_gain") is not None]
        sc=[r["score"]     for r in group if r.get("score")     is not None]
        tm=[r["elapsed_s"] for r in group if r.get("elapsed_s") is not None]
        st=[r["strategy"]  for r in group if r.get("strategy") and r["strategy"]!="error"]
        summary[ct]={
            "n_runs":len(group),
            "avg_psnr_gain":round(float(np.mean(pg)),3) if pg else None,
            "avg_ssim_gain":round(float(np.mean(sg)),4) if sg else None,
            "avg_score":    round(float(np.mean(sc)),2) if sc else None,
            "avg_time_s":   round(float(np.mean(tm)),2) if tm else None,
            "best_strategy":max(set(st),key=st.count) if st else "—",
        }
    return summary


def _save_csv(rows, path):
    if not rows: return
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def generate_benchmark_plots(rows, summary, out_dir, ts):
    paths = {}
    valid=[r for r in rows if r.get("psnr_gain") is not None]
    if not valid: return paths
    types=list(summary.keys()); colors=_COLORS[:len(types)]
    p=out_dir/f"bench_psnr_{ts}.png"
    _bar([t.replace("_","\n") for t in types],
         [summary[t]["avg_psnr_gain"] or 0 for t in types],
         "Gain PSNR moyen par type","Gain PSNR (dB)",colors,p)
    paths["plot_psnr"]=str(p)
    p=out_dir/f"bench_ssim_{ts}.png"
    _bar([t.replace("_","\n") for t in types],
         [summary[t]["avg_ssim_gain"] or 0 for t in types],
         "Gain SSIM moyen par type","Gain SSIM",colors,p)
    paths["plot_ssim"]=str(p)
    ss=defaultdict(list)
    for r in valid:
        s=r.get("strategy","")
        if s and s!="error": ss[s].append(float(r["score"]))
    if ss:
        strats=sorted(ss,key=lambda s:np.mean(ss[s]),reverse=True)[:8]
        p=out_dir/f"bench_strategy_{ts}.png"
        _bar([s.replace("_","\n") for s in strats],
             [round(float(np.mean(ss[s])),2) for s in strats],
             "Score moyen par strategie","Score (0-100)",_COLORS[:len(strats)],p,horizontal=True)
        paths["plot_strategy"]=str(p)
    p=out_dir/f"bench_time_{ts}.png"
    _bar([t.replace("_","\n") for t in types],
         [summary[t]["avg_time_s"] or 0 for t in types],
         "Temps moyen par type","Temps (s)",colors,p)
    paths["plot_time"]=str(p)
    return paths


def _bar(labels,values,title,ylabel,colors,path,horizontal=False):
    fig,ax=plt.subplots(figsize=(10,5))
    fig.patch.set_facecolor("#0a0a0f"); ax.set_facecolor("#111118")
    n=len(labels); bc=(colors*(n//len(colors)+1))[:n]
    mv=max(values) if values else 1
    if horizontal:
        bars=ax.barh(labels,values,color=bc,height=0.55)
        ax.set_xlabel(ylabel,color="#e8e8f0",fontsize=9); ax.invert_yaxis()
        for b,v in zip(bars,values):
            ax.text(v+mv*0.01,b.get_y()+b.get_height()/2,
                    f"{v:.1f}",va="center",color="#e8e8f0",fontsize=8)
    else:
        x=np.arange(n); bars=ax.bar(x,values,color=bc,width=0.6)
        ax.set_xticks(x); ax.set_xticklabels(labels,color="#6b6b80",fontsize=8)
        ax.set_ylabel(ylabel,color="#e8e8f0",fontsize=9)
        for b,v in zip(bars,values):
            ax.text(b.get_x()+b.get_width()/2,b.get_height()+mv*0.01,
                    f"{v:.2f}",ha="center",color="#e8e8f0",fontsize=8)
    ax.set_title(title,color="#00e5ff",fontsize=11,pad=12)
    ax.tick_params(colors="#6b6b80")
    for sp in ax.spines.values(): sp.set_edgecolor("#2a2a3a")
    ax.grid(axis="x" if horizontal else "y",color="#2a2a3a",linewidth=0.5,alpha=0.6)
    plt.tight_layout()
    plt.savefig(str(path),dpi=120,bbox_inches="tight",facecolor=fig.get_facecolor())
    plt.close(fig)
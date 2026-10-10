"""Generate the review slide deck (docs/presentation.pptx) from the REAL results in results/.

Usage:
  python src/make_slides.py --team "Name1 (SRN), Name2 (SRN)" --problem 7 \
         --contrib1 "EDA, features, ablation" --contrib2 "models, tuning, write-up"
Needs: pip install -r requirements-report.txt
"""
import argparse
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

from summary import cfg_sentence, collect, conclusions, FUTURE, LIMITS

NAVY, BLUE, LIGHT, GREY, WHITE = (RGBColor(0x1F, 0x3A, 0x5F), RGBColor(0x2E, 0x75, 0xB6),
                                  RGBColor(0xE8, 0xF0, 0xF8), RGBColor(0x44, 0x44, 0x44), RGBColor(255, 255, 255))
W, H = 13.333, 7.5


def tb(slide, x, y, w, h, text, size=18, bold=False, color=GREY, align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.alignment = align
    r = p.add_run(); r.text = text; r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = color
    return tf


def bullets(slide, x, y, w, h, items, size=18):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame; tf.word_wrap = True
    for i, t in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(8)
        r = p.add_run(); r.text = "\u2022  " + t; r.font.size = Pt(size); r.font.color.rgb = GREY
    return tf


def new_slide(prs, title, notes=""):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(1.0))
    bar.fill.solid(); bar.fill.fore_color.rgb = NAVY; bar.line.fill.background()
    tb(s, 0.5, 0.18, 12.3, 0.7, title, size=30, bold=True, color=WHITE)
    if notes:
        s.notes_slide.notes_text_frame.text = notes
    return s


def pic(slide, path, x, y, w=None, h=None):
    if Path(path).exists():
        kw = {}
        if w: kw["width"] = Inches(w)
        if h: kw["height"] = Inches(h)
        slide.shapes.add_picture(str(path), Inches(x), Inches(y), **kw)
    else:
        tb(slide, x, y, 4, 0.5, f"[missing figure: {Path(path).name}]", size=12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True)
    ap.add_argument("--problem", required=True)
    ap.add_argument("--contrib1", default="[fill in: member 1 contribution]")
    ap.add_argument("--contrib2", default="[fill in: member 2 contribution]")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="docs/presentation.pptx")
    ap.add_argument("--allow-synthetic", action="store_true", help="layout test only; never present")
    a = ap.parse_args()
    f = collect(a.data_dir, a.results, a.allow_synthetic)
    fig = Path(a.results) / "figures"

    prs = Presentation(); prs.slide_width, prs.slide_height = Inches(W), Inches(H)

    # 1 title
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg.fill.solid(); bg.fill.fore_color.rgb = NAVY; bg.line.fill.background()
    tb(s, 0.8, 2.2, 11.5, 1.2, "Forecasting Bike Rental Demand", size=44, bold=True, color=WHITE)
    tb(s, 0.8, 3.5, 11.5, 0.6, "UE24CS352A Machine Learning - Mini-Project", size=24, color=LIGHT)
    tb(s, 0.8, 4.4, 11.5, 0.6, f"Team: {a.team}   |   Problem: {a.problem}", size=20, color=WHITE)
    s.notes_slide.notes_text_frame.text = "Introduce the team, the problem number, and the one-line goal: predict hourly bike rentals."

    # 2 problem
    s = new_slide(prs, "Problem & Motivation",
                  "Why it matters: bike-share operators must rebalance bikes ahead of demand. Why RMSLE: counts are skewed, "
                  "so relative error matters more than absolute error.")
    bullets(s, 0.6, 1.4, 12, 5, [
        "Task: predict the number of bikes rented per hour (Capital Bikeshare, Washington D.C.) from date/time and weather.",
        "Use case: station rebalancing, fleet sizing and maintenance scheduling.",
        "Metric: RMSLE = sqrt(mean((log(p+1) - log(a+1))^2)) - penalises relative, not absolute, error.",
        "Reference: Du, He & Zhechev, Stanford CS229 (2014) - we reproduce their pipeline and compare with scikit-learn tools."], size=22)

    # 3 dataset
    s = new_slide(prs, "Dataset", "Kaggle Bike Sharing Demand. Train = days 1-19 of each month, test = days 20+ (labels hidden).")
    test = f"{f['n_test']:,}" if f["n_test"] else "unlabeled"
    bullets(s, 0.6, 1.4, 6.2, 5, [
        f"{f['n_train']:,} hourly training rows, {f['date_min']} to {f['date_max']}",
        f"Test: {test} rows (later days of each month)",
        "Features: datetime, season, holiday, workingday, weather (1-4), temp, atemp, humidity, windspeed",
        "Target: count (= casual + registered)",
        f"Missing values: {f['n_missing']}"], size=17)
    pic(s, fig / "eda_hourly.png", 7.0, 1.5, w=5.8)

    # 4 EDA
    s = new_slide(prs, "EDA Insights", "Hour of day dominates; temperature and weather matter; target is right-skewed so we use log1p.")
    pic(s, fig / "eda_temp.png", 0.4, 1.3, w=4.1)
    pic(s, fig / "eda_month.png", 4.6, 1.3, w=4.1)
    pic(s, fig / "eda_target.png", 8.8, 1.3, w=4.2)
    bullets(s, 0.6, 4.3, 12.2, 2.8, [
        f"Weekday peaks near hours {f['wd_peaks'][0]} and {f['wd_peaks'][1]}; weekend peak at hour {f['we_peak']}",
        f"count is right-skewed (mean {f['count_mean']:.0f}, median {f['count_median']:.0f}) -> train on log(1+count)",
        f"correlation of count with atemp: {f['corr_atemp']:.2f}"], size=20)

    # 5 features
    s = new_slide(prs, "Feature Engineering & Ablation",
                  "Six toggleable ideas from the paper, all 2^6 subsets tried with a fast tree, ranked by mean(CV, hold-out).")
    bullets(s, 0.6, 1.4, 6.0, 5, [
        "Always: hour and year extracted from datetime",
        "6 optional ideas: month vs season, day-of-week, drop temp, drop holiday, peak-hour flags, discretised weather",
        f"{f['n_subsets']} subsets evaluated (CV + temporal hold-out)",
        f"Chosen: {cfg_sentence(f)}"], size=17)
    pic(s, fig / "ablation.png", 6.9, 1.6, w=6.0)

    # 6 models
    s = new_slide(prs, "Models & Validation Design",
                  "Log-target trick: squared error on log1p(count) equals the RMSLE objective. Temporal hold-out avoids leakage.")
    bullets(s, 0.6, 1.4, 12, 5, [
        "7 models: Linear, Poisson GLM, PCR, SVR (RBF), HistGradientBoosting, Random Forest, Decision Tree",
        "Stacking in log space: tree + RF + GBM -> linear meta-model on out-of-fold predictions",
        "Regressors trained on log(1+count); grid-search tuning with K-fold CV",
        "Random K-fold leaks neighbouring hours -> also a temporal hold-out (train days < 15, validate days 15-19)",
        "Winner refit on all training data and saved for prediction"], size=22)

    # 7 results
    s = new_slide(prs, "Results", "Read the table top to bottom: sorted by hold-out RMSLE. Mention best model and gap to the linear baseline.")
    m = f["models"]
    rows, cols = len(m) + 1, 3
    tbl = s.shapes.add_table(rows, cols, Inches(0.5), Inches(1.4), Inches(6.2), Inches(0.4 * rows)).table
    for j, hd in enumerate(["Model", "CV RMSLE", "Hold-out RMSLE"]):
        tbl.cell(0, j).text = hd
    for i, (_, r) in enumerate(m.iterrows(), start=1):
        tbl.cell(i, 0).text = str(r["model"])
        tbl.cell(i, 1).text = "-" if r["cv_rmsle"] != r["cv_rmsle"] else f"{r['cv_rmsle']:.4f}"
        tbl.cell(i, 2).text = f"{r['holdout_rmsle']:.4f}"
    for row in tbl.rows:
        for c in row.cells:
            for p in c.text_frame.paragraphs:
                for r_ in p.runs: r_.font.size = Pt(14)
    pic(s, fig / "model_comparison.png", 7.0, 1.5, w=5.9)
    tb(s, 0.5, 1.4 + 0.4 * rows + 0.3, 12, 0.6,
       f"Best: {f['best']} - hold-out RMSLE {f['best_holdout']:.4f}", size=20, bold=True, color=BLUE)

    # 8 analysis
    s = new_slide(prs, "Analysis", "Explain why the best model wins and which features matter. Predicted-vs-actual shows the daily shape is captured.")
    pic(s, fig / "pred_vs_actual.png", 0.4, 1.3, w=6.3)
    pic(s, fig / "feature_importance.png", 6.8, 1.3, w=6.0)
    bullets(s, 0.6, 5.0, 12.2, 2.2, conclusions(f)[1:3], size=15)

    # 9 demo
    s = new_slide(prs, "Live Demo", "Run these commands live. Weekday 8am -> high; weekend 3am with rain -> very low.")
    code = ("python src/predict.py\n\n"
            'python src/demo.py --datetime "2012-06-15 08:00:00" --temp 24 --humidity 55 --windspeed 10 --weather 1\n\n'
            'python src/demo.py --datetime "2012-06-16 03:00:00" --temp 20 --weather 3')
    box = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.6), Inches(1.5), Inches(12.1), Inches(3.2))
    box.fill.solid(); box.fill.fore_color.rgb = LIGHT; box.line.fill.background()
    tf = box.text_frame; tf.word_wrap = True
    for i, line in enumerate(code.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        r = p.add_run(); r.text = line; r.font.size = Pt(15); r.font.name = "Courier New"; r.font.color.rgb = NAVY
    tb(s, 0.6, 5.1, 12, 1.2, "Then show results/model_results.csv and results/figures/.", size=18)

    # 10 conclusions
    s = new_slide(prs, "Conclusions, Limitations & Contributions", "Finish with limitations and who did what. Each member must be able to explain every file.")
    bullets(s, 0.6, 1.3, 12.2, 3.6, [conclusions(f)[0], LIMITS.replace("Limitations: ", "Limitations: "), FUTURE], size=15)
    tb(s, 0.6, 5.2, 12, 0.5, "Individual contributions", size=20, bold=True, color=BLUE)
    bullets(s, 0.6, 5.7, 12, 1.5, [a.contrib1, a.contrib2], size=16)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    prs.save(a.out)
    print(f"Wrote {a.out} ({len(prs.slides)} slides)")
    if f["synthetic"]:
        print("WARNING: built from SYNTHETIC data - do NOT present this file.")


if __name__ == "__main__":
    main()

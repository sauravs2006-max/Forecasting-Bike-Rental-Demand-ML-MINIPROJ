"""Generate the 2-page PDF write-up (docs/writeup.pdf) from the REAL results in results/.

Usage:
  python src/make_report.py --team "Name1 (PES1UG..), Name2 (PES1UG..)" --problem 7 \
         --repo https://github.com/<user>/<repo>
Needs: pip install -r requirements-report.txt
"""
import argparse
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.pdfgen.canvas import Canvas

from summary import cfg_sentence, collect, conclusions, FUTURE, LIMITS


def build(f, a, size, out):
    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=size, leading=size * 1.25, alignment=TA_JUSTIFY, spaceAfter=2)
    h = ParagraphStyle("h", parent=ss["Heading3"], fontSize=size + 2, leading=size + 4, spaceBefore=5, spaceAfter=2,
                       textColor=colors.HexColor("#1F3A5F"))
    title = ParagraphStyle("t", parent=ss["Title"], fontSize=15, leading=18, spaceAfter=2)
    meta = ParagraphStyle("m", parent=body, fontSize=size - 0.5, alignment=0, textColor=colors.HexColor("#444444"))
    bullet = ParagraphStyle("bl", parent=body, leftIndent=10, bulletIndent=0)

    fig = Path(a.results) / "figures"
    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm,
                            topMargin=1.3 * cm, bottomMargin=1.3 * cm,
                            title="Forecasting Bike Rental Demand", author=a.team)
    S = []
    S.append(Paragraph("Forecasting Bike Rental Demand", title))
    S.append(Paragraph(f"<b>UE24CS352A Machine Learning - Mini-Project</b> | Team: {a.team} | Problem: {a.problem} | "
                       f"Repo: {a.repo} (private, shared with faculty/TAs)", meta))

    S.append(Paragraph("1. Problem Statement", h))
    S.append(Paragraph(
        "Predict the total number of bikes rented <b>per hour</b> in Washington D.C.'s Capital Bikeshare system from date/time "
        "and weather information (Kaggle <i>Bike Sharing Demand</i>; reference: Du, He &amp; Zhechev, Stanford CS229, 2014). "
        "The metric is RMSLE = sqrt(mean((log(p+1) - log(a+1))<super>2</super>)), which penalises relative rather than absolute error.", body))

    S.append(Paragraph("2. Dataset", h))
    test = f"{f['n_test']:,} test rows (later days of each month, no labels)" if f["n_test"] else "a separate unlabeled test set"
    yr = f["year_means"]; ys = sorted(yr)
    growth = ""
    if len(ys) >= 2 and yr[ys[0]] > 0:
        growth = f" Mean hourly demand was {yr[ys[-1]] / yr[ys[0]]:.2f}x higher in {ys[-1]} than in {ys[0]}."
    S.append(Paragraph(
        f"{f['n_train']:,} hourly training rows from {f['date_min']} to {f['date_max']} (days 1-19 of each month) and {test}. "
        f"Features: datetime, season, holiday, workingday, weather (1-4), temp, atemp, humidity, windspeed. Targets: casual, "
        f"registered, count (= casual + registered; <b>count</b> is predicted). Missing values: {f['n_missing']}. "
        f"<b>EDA findings:</b> weekday demand peaks around hours {f['wd_peaks'][0]} and {f['wd_peaks'][1]}, weekend demand peaks at "
        f"hour {f['we_peak']}; count is right-skewed (mean {f['count_mean']:.0f}, median {f['count_median']:.0f}, skew "
        f"{f['count_skew']:.2f}), so models are trained on log(1+count); correlation of count with atemp is "
        f"{f['corr_atemp']:.2f}.{growth}", body))

    S.append(Paragraph("3. Approach", h))
    S.append(Paragraph(
        f"<b>Feature engineering.</b> Hour and year are extracted from datetime; six optional ideas from the paper (month instead of "
        f"season, day-of-week, drop temp, drop holiday, peak-hour flags, discretised weather) were searched over all "
        f"{f['n_subsets']} subsets with a fast decision tree, scoring each by the mean of 5-fold CV and temporal hold-out RMSLE. "
        f"Best subset: <b>{cfg_sentence(f)}</b>. "
        "<b>Models.</b> Linear regression, Poisson GLM, PCR, SVR (RBF), gradient boosting (HistGradientBoosting), Random Forest, "
        "Decision Tree, and a log-space stack (tree + RF + GBM -> linear meta-model). Regressors are trained on log(1+count) so "
        "squared error equals the RMSLE objective; hyper-parameters come from grid search with K-fold CV. "
        "<b>Validation.</b> Random K-fold leaks neighbouring hours, so we also report a temporal hold-out "
        "(train: days &lt; 15, validate: days 15-19) that mimics the real split (test = days 20+). The winner is refit on all training data.", body))

    S.append(Paragraph("4. Implementation Overview", h))
    S.append(Paragraph(
        "Python 3 / scikit-learn. <font face='Courier'>features.py</font> (feature flags) -> <font face='Courier'>models.py</font> "
        "(model zoo, LogStack) -> <font face='Courier'>ablation.py</font> / <font face='Courier'>train.py</font> (search, tuning, "
        "comparison, saving) -> <font face='Courier'>predict.py</font> (Kaggle-format submission) and <font face='Courier'>demo.py</font> "
        "(single-hour prediction). Everything is reproducible with <font face='Courier'>bash run_all.sh</font>; seeds are fixed.", body))

    S.append(Paragraph("5. Results", h))
    rows = [["Model", "CV RMSLE", "Hold-out RMSLE"]]
    for _, r in f["models"].iterrows():
        rows.append([r["model"], "-" if r["cv_rmsle"] != r["cv_rmsle"] else f"{r['cv_rmsle']:.4f}", f"{r['holdout_rmsle']:.4f}"])
    t = Table(rows, colWidths=[4.2 * cm, 3 * cm, 3.4 * cm], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), size - 0.5), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F3A5F")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#E3EEF9")), ("TOPPADDING", (0, 0), (-1, -1), 1.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]))
    imgs = []
    for name in ("model_comparison.png", "feature_importance.png"):
        p = fig / name
        if p.exists():
            w, h_ = ImageReader(str(p)).getSize()
            imgs.append(Image(str(p), width=6.1 * cm, height=6.1 * cm * h_ / w))
    side = Table([[t, imgs]], colWidths=[10.8 * cm, 7 * cm], hAlign="LEFT")
    side.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    S.append(side)
    S.append(Paragraph("Rows are sorted by hold-out RMSLE (lower is better); the shaded row is the selected model.", meta))

    S.append(Paragraph("6. Conclusions", h))
    for c in conclusions(f):
        S.append(Paragraph(c, bullet, bulletText="\u2022"))
    S.append(Paragraph(LIMITS, bullet, bulletText="\u2022"))
    S.append(Paragraph(FUTURE, bullet, bulletText="\u2022"))

    def foot(c: Canvas, d):
        c.setFont("Helvetica", 7); c.setFillColor(colors.grey)
        c.drawRightString(A4[0] - 1.6 * cm, 0.7 * cm, f"Page {d.page}")

    doc.build(S, onFirstPage=foot, onLaterPages=foot)
    return doc.page


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True, help='"Name1 (SRN), Name2 (SRN)"')
    ap.add_argument("--problem", required=True, help="your problem statement serial number")
    ap.add_argument("--repo", required=True, help="private GitHub repo URL")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="docs/writeup.pdf")
    ap.add_argument("--allow-synthetic", action="store_true", help="layout test only; never submit")
    a = ap.parse_args()

    f = collect(a.data_dir, a.results, a.allow_synthetic)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    for size in (9.5, 9, 8.5, 8, 7.5):
        pages = build(f, a, size, a.out)
        if pages <= 2:
            print(f"Wrote {a.out}: {pages} page(s), font {size}pt")
            break
    else:
        print(f"WARNING: still {pages} pages at the smallest font; shorten the text.")
    if f["synthetic"]:
        print("WARNING: built from SYNTHETIC data - do NOT submit this file.")


if __name__ == "__main__":
    main()

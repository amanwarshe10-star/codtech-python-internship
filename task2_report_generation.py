"""
================================================================================
CODTECH INTERNSHIP - TASK 2
Automated Report Generation
--------------------------------------------------------------------------------
Description : Reads a CSV sales dataset, analyzes it, and generates a
              professionally formatted PDF report using ReportLab.
Libraries   : pandas, matplotlib, reportlab
================================================================================
"""

import io
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, Image, HRFlowable)
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from datetime import datetime


def create_sample_csv(path="sales_data.csv"):
    """Creates a realistic sample sales CSV. Replace with your own file."""
    np.random.seed(42)
    months   = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    products = ["Laptop","Mobile","Tablet","Monitor","Keyboard"]
    rows = []
    for month in months:
        for product in products:
            rows.append({
                "Month":   month,
                "Product": product,
                "Units":   np.random.randint(20, 200),
                "Price":   np.random.randint(500, 80000),
                "Region":  np.random.choice(["North","South","East","West"]),
            })
    df = pd.DataFrame(rows)
    df["Revenue"] = df["Units"] * df["Price"]
    df.to_csv(path, index=False)
    print(f"  Sample CSV created -> {path}")
    return df


def analyze_data(df):
    month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    stats = {
        "total_revenue":   df["Revenue"].sum(),
        "total_units":     df["Units"].sum(),
        "avg_price":       df["Price"].mean(),
        "best_product":    df.groupby("Product")["Revenue"].sum().idxmax(),
        "best_region":     df.groupby("Region")["Revenue"].sum().idxmax(),
        "monthly_revenue": df.groupby("Month")["Revenue"].sum().reindex(month_order),
        "product_revenue": df.groupby("Product")["Revenue"].sum().sort_values(ascending=False),
        "top_table":       df.groupby("Product").agg(Units=("Units","sum"),Revenue=("Revenue","sum"))
                             .sort_values("Revenue",ascending=False).reset_index(),
    }
    return stats


def make_chart_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf


def create_charts(stats):
    # Monthly Revenue line chart
    fig1, ax1 = plt.subplots(figsize=(7, 3.2))
    ax1.plot(stats["monthly_revenue"].index, stats["monthly_revenue"].values/1e6,
             marker="o", color="#E63946", linewidth=2.5)
    ax1.fill_between(range(12), stats["monthly_revenue"].values/1e6, alpha=0.15, color="#E63946")
    ax1.set_title("Monthly Revenue (Rs. Millions)", fontweight="bold", fontsize=11)
    ax1.set_ylabel("Revenue (Rs. M)")
    ax1.set_xticks(range(12))
    ax1.set_xticklabels(stats["monthly_revenue"].index, rotation=45, fontsize=8)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    fig1.tight_layout()
    c1 = make_chart_bytes(fig1)

    # Product Revenue bar chart
    fig2, ax2 = plt.subplots(figsize=(6, 3.2))
    colors_list = ["#E63946","#457B9D","#1D3557","#A8DADC","#F1FAEE"]
    bars = ax2.barh(stats["product_revenue"].index, stats["product_revenue"].values/1e6, color=colors_list)
    ax2.set_title("Revenue by Product (Rs. Millions)", fontweight="bold", fontsize=11)
    ax2.set_xlabel("Revenue (Rs. M)")
    for bar in bars:
        ax2.text(bar.get_width()+0.1, bar.get_y()+bar.get_height()/2,
                 f"Rs.{bar.get_width():.1f}M", va="center", fontsize=8)
    ax2.grid(axis="x", linestyle="--", alpha=0.5)
    fig2.tight_layout()
    c2 = make_chart_bytes(fig2)
    return c1, c2


def generate_pdf(stats, chart1, chart2, out_path="task2_sales_report.pdf"):
    doc = SimpleDocTemplate(out_path, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm, topMargin=2.5*cm, bottomMargin=2.5*cm)

    title_s   = ParagraphStyle("T", fontSize=22, textColor=colors.HexColor("#1D3557"),
                               alignment=TA_CENTER, fontName="Helvetica-Bold", spaceAfter=4)
    sub_s     = ParagraphStyle("S", fontSize=11, textColor=colors.HexColor("#457B9D"),
                               alignment=TA_CENTER, spaceAfter=2)
    section_s = ParagraphStyle("H", fontSize=13, textColor=colors.HexColor("#E63946"),
                               fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=6)
    body_s    = ParagraphStyle("B", fontSize=10, leading=14, textColor=colors.HexColor("#333333"))

    story = [Spacer(1,1.5*cm)]
    story.append(Paragraph("Annual Sales Analysis Report", title_s))
    story.append(Paragraph("CodTech Internship -- Task 2: Automated Report Generation", sub_s))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}", sub_s))
    story.append(Spacer(1,0.5*cm))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#E63946")))

    story.append(Paragraph("1. Executive Summary", section_s))
    story.append(Paragraph(
        f"Total revenue for the year: <b>Rs. {stats['total_revenue']:,.0f}</b>. "
        f"Total units sold: <b>{stats['total_units']:,}</b>. "
        f"Average price: <b>Rs. {stats['avg_price']:,.0f}</b>. "
        f"Top product: <b>{stats['best_product']}</b>. "
        f"Top region: <b>{stats['best_region']}</b>.", body_s))

    story.append(Paragraph("2. Key Performance Indicators", section_s))
    kpi = [["Metric","Value"],
           ["Total Revenue",    f"Rs. {stats['total_revenue']:,.0f}"],
           ["Total Units Sold", f"{stats['total_units']:,}"],
           ["Average Price",    f"Rs. {stats['avg_price']:,.0f}"],
           ["Top Product",      stats["best_product"]],
           ["Top Region",       stats["best_region"]]]
    t = Table(kpi, colWidths=[8*cm,8*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#1D3557")),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("FONTSIZE",(0,0),(-1,-1),10),
        ("ALIGN",(0,0),(-1,-1),"CENTER"),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#F1FAEE"),colors.white]),
        ("GRID",(0,0),(-1,-1),0.5,colors.HexColor("#A8DADC")),
        ("BOTTOMPADDING",(0,0),(-1,-1),7),("TOPPADDING",(0,0),(-1,-1),7),
    ]))
    story.append(t)

    story.append(Paragraph("3. Monthly Revenue Trend", section_s))
    story.append(Image(chart1, width=15*cm, height=7*cm))

    story.append(Paragraph("4. Product-Wise Revenue", section_s))
    story.append(Image(chart2, width=13*cm, height=7*cm))

    story.append(Paragraph("5. Detailed Product Breakdown", section_s))
    tbl = [["#","Product","Units Sold","Revenue (Rs.)"]]
    for i, row in stats["top_table"].iterrows():
        tbl.append([str(i+1), row["Product"], f"{row['Units']:,}", f"Rs. {row['Revenue']:,.0f}"])
    dt = Table(tbl, colWidths=[1.5*cm,6*cm,4*cm,5*cm])
    dt.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#457B9D")),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("FONTSIZE",(0,0),(-1,-1),10),
        ("ALIGN",(0,0),(-1,-1),"CENTER"),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#F1FAEE"),colors.white]),
        ("GRID",(0,0),(-1,-1),0.5,colors.HexColor("#A8DADC")),
        ("BOTTOMPADDING",(0,0),(-1,-1),7),("TOPPADDING",(0,0),(-1,-1),7),
    ]))
    story.append(dt)

    story.append(Paragraph("6. Conclusion", section_s))
    story.append(Paragraph(
        f"<b>{stats['best_product']}</b> is the top revenue driver. "
        f"The <b>{stats['best_region']}</b> region leads in sales. "
        "Management should focus on mid-year promotions and inventory planning "
        "based on observed seasonal trends.", body_s))
    story.append(Spacer(1,0.5*cm))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#A8DADC")))
    story.append(Spacer(1,0.3*cm))
    story.append(Paragraph("CodTech Python Internship - Task 2", sub_s))

    doc.build(story)
    print(f"  PDF Report saved -> {out_path}")


if __name__ == "__main__":
    print("="*60)
    print("  TASK 2 -- Automated Report Generation")
    print("="*60)

    print("\n[1/4] Creating sample CSV...")
    df = create_sample_csv("sales_data.csv")

    print("\n[2/4] Analyzing data...")
    stats = analyze_data(df)
    print(f"  Total Revenue : Rs. {stats['total_revenue']:,.0f}")
    print(f"  Best Product  : {stats['best_product']}")
    print(f"  Best Region   : {stats['best_region']}")

    print("\n[3/4] Creating charts...")
    c1, c2 = create_charts(stats)

    print("\n[4/4] Generating PDF...")
    generate_pdf(stats, c1, c2, "task2_sales_report.pdf")

    print("\nTask 2 Complete!")

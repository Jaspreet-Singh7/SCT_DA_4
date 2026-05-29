# ============================================================
#  SkillCraft Technology — Data Analyst Internship
#  Task 04: Business Insights Report (EDA)
#  Dataset: Marketing Campaign Data
#  Author : Jaspreet Singh  |  Track Code: DA
#  GitHub Repo: SCT_DA_4
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── Style ──────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#1a1d2e",
    "axes.edgecolor":   "#2d3250",
    "axes.labelcolor":  "#c9d1d9",
    "axes.titlecolor":  "#ffffff",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "text.color":       "#c9d1d9",
    "grid.color":       "#21262d",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "font.family":      "DejaVu Sans",
    "axes.titlesize":   12,
    "axes.labelsize":   10,
})

COLORS = ["#00d4ff","#7c3aed","#10b981","#f59e0b","#ef4444",
          "#ec4899","#8b5cf6","#14b8a6","#f97316","#6366f1"]
ACCENT = "#00d4ff"

# ─────────────────────────────────────────────────────────────
# STEP 1 — Load Data
# ─────────────────────────────────────────────────────────────
print("=" * 65)
print("  SkillCraft Technology | Task 04 — Business Insights Report")
print("  Dataset: Marketing Campaign Data")
print("=" * 65)

df = pd.read_csv("marketing_campaign_raw.csv")
print(f"\n✅ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\n📋 Columns: {list(df.columns)}")
print(f"\n🔍 Sample Data:\n{df.head(3).to_string()}")

# ─────────────────────────────────────────────────────────────
# STEP 2 — Data Cleaning
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 2: Data Cleaning")
print("=" * 65)

print(f"\n📌 Missing values: {df.isnull().sum().sum()}")
print(f"📌 Duplicates    : {df.duplicated().sum()}")

# Remove rows with 0 spend or 0 impressions
before = len(df)
df = df[(df["Spend"] > 0) & (df["Impressions"] > 0)]
print(f"📌 Removed {before - len(df)} rows with zero spend/impressions")

# Cap extreme ROI outliers
p1  = df["ROI"].quantile(0.01)
p99 = df["ROI"].quantile(0.99)
df["ROI"] = df["ROI"].clip(lower=p1, upper=p99)

print(f"✅ Clean dataset: {len(df)} rows ready for analysis")

# ─────────────────────────────────────────────────────────────
# STEP 3 — EDA: Summary Statistics
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 3: Exploratory Data Analysis — Summary Statistics")
print("=" * 65)

total_spend    = df["Spend"].sum()
total_revenue  = df["Revenue"].sum()
total_conv     = df["Conversions"].sum()
total_clicks   = df["Clicks"].sum()
total_impr     = df["Impressions"].sum()
overall_roi    = (total_revenue - total_spend) / total_spend * 100
avg_ctr        = df["CTR"].mean()
avg_conv_rate  = df["Conversion_Rate"].mean()

print(f"""
📊 OVERALL CAMPAIGN PERFORMANCE
{'─'*40}
  Total Impressions  : {total_impr:>12,.0f}
  Total Clicks       : {total_clicks:>12,.0f}
  Total Spend        : ₹{total_spend:>11,.2f}
  Total Revenue      : ₹{total_revenue:>11,.2f}
  Total Conversions  : {total_conv:>12,.0f}
  Overall ROI        : {overall_roi:>11.1f}%
  Avg CTR            : {avg_ctr:>11.2f}%
  Avg Conversion Rate: {avg_conv_rate:>11.2f}%
""")

# ─────────────────────────────────────────────────────────────
# STEP 4 — Channel Analysis
# ─────────────────────────────────────────────────────────────
print("=" * 65)
print("STEP 4: Channel Performance Analysis")
print("=" * 65)

ch = df.groupby("Channel").agg(
    Total_Spend    =("Spend","sum"),
    Total_Revenue  =("Revenue","sum"),
    Total_Conv     =("Conversions","sum"),
    Avg_ROI        =("ROI","mean"),
    Avg_CTR        =("CTR","mean"),
    Avg_Conv_Rate  =("Conversion_Rate","mean"),
    Campaigns      =("Campaign_ID","count")
).round(2).reset_index()

ch["ROI_Rank"] = ch["Avg_ROI"].rank(ascending=False).astype(int)
ch_sorted = ch.sort_values("Avg_ROI", ascending=False)

print(f"\n📊 Channel Summary:\n{ch_sorted.to_string(index=False)}")

best_channel  = ch_sorted.iloc[0]["Channel"]
worst_channel = ch_sorted.iloc[-1]["Channel"]
print(f"\n🏆 Best ROI Channel  : {best_channel} ({ch_sorted.iloc[0]['Avg_ROI']:.1f}% avg ROI)")
print(f"⚠️  Worst ROI Channel : {worst_channel} ({ch_sorted.iloc[-1]['Avg_ROI']:.1f}% avg ROI)")

# ─────────────────────────────────────────────────────────────
# STEP 5 — Campaign Type Analysis
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 5: Campaign Type Analysis")
print("=" * 65)

ct = df.groupby("Campaign_Type").agg(
    Total_Spend   =("Spend","sum"),
    Total_Revenue =("Revenue","sum"),
    Avg_ROI       =("ROI","mean"),
    Total_Conv    =("Conversions","sum"),
    Campaigns     =("Campaign_ID","count")
).round(2).reset_index().sort_values("Avg_ROI", ascending=False)

print(f"\n{ct.to_string(index=False)}")

# ─────────────────────────────────────────────────────────────
# STEP 6 — Region & Age Group Analysis
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 6: Region & Age Group Analysis")
print("=" * 65)

rg = df.groupby("Region").agg(
    Total_Revenue=("Revenue","sum"),
    Avg_ROI=("ROI","mean"),
    Total_Spend=("Spend","sum")
).round(2).reset_index().sort_values("Avg_ROI", ascending=False)
print(f"\n🌍 By Region:\n{rg.to_string(index=False)}")

ag = df.groupby("Age_Group").agg(
    Total_Revenue=("Revenue","sum"),
    Avg_ROI=("ROI","mean"),
    Avg_Conv_Rate=("Conversion_Rate","mean"),
    Total_Conv=("Conversions","sum")
).round(2).reset_index().sort_values("Avg_Conv_Rate", ascending=False)
print(f"\n👥 By Age Group:\n{ag.to_string(index=False)}")

# ─────────────────────────────────────────────────────────────
# STEP 7 — Business Insights & Recommendations
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 7: KEY BUSINESS INSIGHTS & RECOMMENDATIONS")
print("=" * 65)

top_ch = ch_sorted.head(2)["Channel"].tolist()
bot_ch = ch_sorted.tail(1)["Channel"].tolist()
top_age = ag.iloc[0]["Age_Group"]
top_region = rg.iloc[0]["Region"]
top_camp = ct.iloc[0]["Campaign_Type"]

print(f"""
💡 INSIGHT 1 — Channel Budget Allocation
   → {top_ch[0]} and {top_ch[1]} deliver the highest ROI.
   RECOMMENDATION: Increase budget allocation to these channels
   by 20-30%. Reduce spending on {bot_ch[0]} which shows lowest ROI.

💡 INSIGHT 2 — Age Group Targeting
   → Age group {top_age} shows the highest conversion rate ({ag.iloc[0]['Avg_Conv_Rate']:.1f}%).
   RECOMMENDATION: Focus ad creatives and targeting on {top_age}
   demographic for maximum conversion efficiency.

💡 INSIGHT 3 — Regional Performance
   → {top_region} region generates the highest ROI ({rg.iloc[0]['Avg_ROI']:.1f}%).
   RECOMMENDATION: Prioritize {top_region} region campaigns and
   investigate why other regions underperform to replicate success.

💡 INSIGHT 4 — Campaign Type Strategy
   → {top_camp} campaigns show highest average ROI.
   RECOMMENDATION: Run more {top_camp} campaigns especially
   during peak seasons to maximize revenue.

💡 INSIGHT 5 — Overall ROI Health
   → Overall campaign ROI is {overall_roi:.1f}% which is
   {'HEALTHY ✅' if overall_roi > 100 else 'NEEDS IMPROVEMENT ⚠️'}.
   RECOMMENDATION: {'Maintain current strategy while optimizing' if overall_roi > 100 else 'Review low-performing campaigns and reallocate budget'}.
""")

# ─────────────────────────────────────────────────────────────
# STEP 8 — Visualizations (1-page report)
# ─────────────────────────────────────────────────────────────
print("=" * 65)
print("STEP 8: Generating Visual Report...")
print("=" * 65)

fig = plt.figure(figsize=(20, 24), facecolor="#0f1117")
fig.suptitle(
    "MARKETING CAMPAIGN — BUSINESS INSIGHTS REPORT\nSkillCraft Technology | Data Analyst Internship | Task 04 | Jaspreet Singh",
    fontsize=16, fontweight="bold", color="white", y=0.98, linespacing=1.6
)

gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35,
                        top=0.94, bottom=0.04, left=0.06, right=0.97)

# ── KPI Banner ────────────────────────────────────────────
ax_kpi = fig.add_subplot(gs[0, :])
ax_kpi.set_facecolor("#0f1117")
ax_kpi.axis("off")

kpis = [
    ("Total Spend",       f"₹{total_spend/1e6:.2f}M",  "#00d4ff"),
    ("Total Revenue",     f"₹{total_revenue/1e6:.2f}M","#10b981"),
    ("Total Conversions", f"{total_conv:,}",            "#7c3aed"),
    ("Overall ROI",       f"{overall_roi:.1f}%",        "#f59e0b"),
    ("Avg CTR",           f"{avg_ctr:.2f}%",            "#ef4444"),
]
for i, (label, val, color) in enumerate(kpis):
    x = 0.1 + i * 0.2
    rect = mpatches.FancyBboxPatch((x-0.085, 0.05), 0.17, 0.88,
        boxstyle="round,pad=0.02", facecolor="#1a1d2e",
        edgecolor=color, linewidth=2, transform=ax_kpi.transAxes)
    ax_kpi.add_patch(rect)
    ax_kpi.text(x, 0.72, label, ha="center", va="center",
                fontsize=9, color="#8b949e", transform=ax_kpi.transAxes,
                fontweight="bold")
    ax_kpi.text(x, 0.38, val, ha="center", va="center",
                fontsize=18, color=color, transform=ax_kpi.transAxes,
                fontweight="bold")

# ── Chart 1: ROI by Channel ───────────────────────────────
ax1 = fig.add_subplot(gs[1, 0])
bars = ax1.barh(ch_sorted["Channel"], ch_sorted["Avg_ROI"],
                color=COLORS[:len(ch_sorted)], edgecolor="none", height=0.6)
ax1.set_title("Avg ROI by Channel", fontweight="bold")
ax1.set_xlabel("ROI (%)")
ax1.axvline(ch_sorted["Avg_ROI"].mean(), color="white", linestyle="--", alpha=0.4, linewidth=1)
for bar, val in zip(bars, ch_sorted["Avg_ROI"]):
    ax1.text(val+1, bar.get_y()+bar.get_height()/2,
             f"{val:.1f}%", va="center", fontsize=9, color="white")
ax1.grid(axis="x")

# ── Chart 2: Spend vs Revenue by Channel ─────────────────
ax2 = fig.add_subplot(gs[1, 1])
x = np.arange(len(ch_sorted))
w = 0.35
ax2.bar(x-w/2, ch_sorted["Total_Spend"]/1000,   width=w, label="Spend (K)",   color="#00d4ff", alpha=0.85)
ax2.bar(x+w/2, ch_sorted["Total_Revenue"]/1000, width=w, label="Revenue (K)", color="#10b981", alpha=0.85)
ax2.set_xticks(x)
ax2.set_xticklabels(ch_sorted["Channel"], rotation=15, ha="right", fontsize=8)
ax2.set_title("Spend vs Revenue by Channel", fontweight="bold")
ax2.set_ylabel("Amount (₹K)")
ax2.legend(fontsize=8)
ax2.grid(axis="y")

# ── Chart 3: Campaign Type ROI ────────────────────────────
ax3 = fig.add_subplot(gs[1, 2])
wedges, texts, autotexts = ax3.pie(
    ct["Total_Revenue"], labels=ct["Campaign_Type"],
    colors=COLORS[:len(ct)], autopct="%1.1f%%",
    startangle=90, pctdistance=0.75,
    wedgeprops=dict(edgecolor="#0f1117", linewidth=2)
)
for t in texts: t.set_color("#c9d1d9"); t.set_fontsize(8)
for t in autotexts: t.set_color("white"); t.set_fontsize(8); t.set_fontweight("bold")
ax3.set_title("Revenue Share by Campaign Type", fontweight="bold")

# ── Chart 4: Conversion Rate by Age Group ────────────────
ax4 = fig.add_subplot(gs[2, 0])
ag_sorted = ag.sort_values("Avg_Conv_Rate", ascending=True)
colors4 = [COLORS[i % len(COLORS)] for i in range(len(ag_sorted))]
bars4 = ax4.barh(ag_sorted["Age_Group"], ag_sorted["Avg_Conv_Rate"],
                 color=colors4, height=0.55)
ax4.set_title("Conversion Rate by Age Group", fontweight="bold")
ax4.set_xlabel("Conversion Rate (%)")
for bar, val in zip(bars4, ag_sorted["Avg_Conv_Rate"]):
    ax4.text(val+0.1, bar.get_y()+bar.get_height()/2,
             f"{val:.1f}%", va="center", fontsize=9, color="white")
ax4.grid(axis="x")

# ── Chart 5: Revenue by Region ────────────────────────────
ax5 = fig.add_subplot(gs[2, 1])
rg_sorted = rg.sort_values("Total_Revenue", ascending=False)
bars5 = ax5.bar(rg_sorted["Region"], rg_sorted["Total_Revenue"]/1000,
                color=COLORS[4:8], edgecolor="none")
ax5.set_title("Total Revenue by Region (₹K)", fontweight="bold")
ax5.set_ylabel("Revenue (₹K)")
for bar in bars5:
    ax5.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
             f"₹{bar.get_height():.0f}K", ha="center", fontsize=9, color="white")
ax5.grid(axis="y")

# ── Chart 6: CTR Distribution ────────────────────────────
ax6 = fig.add_subplot(gs[2, 2])
ax6.hist(df["CTR"], bins=30, color="#7c3aed", alpha=0.85, edgecolor="none")
ax6.axvline(df["CTR"].mean(), color="#f59e0b", linestyle="--", linewidth=1.5,
            label=f"Mean: {df['CTR'].mean():.2f}%")
ax6.axvline(df["CTR"].median(), color="#10b981", linestyle="--", linewidth=1.5,
            label=f"Median: {df['CTR'].median():.2f}%")
ax6.set_title("CTR Distribution", fontweight="bold")
ax6.set_xlabel("CTR (%)")
ax6.set_ylabel("Count")
ax6.legend(fontsize=8)
ax6.grid(axis="y")

# ── Chart 7: ROI Heatmap (Channel × Campaign Type) ───────
ax7 = fig.add_subplot(gs[3, :2])
pivot = df.pivot_table(values="ROI", index="Channel",
                        columns="Campaign_Type", aggfunc="mean")
sns.heatmap(pivot, ax=ax7, cmap="RdYlGn", annot=True, fmt=".1f",
            linewidths=0.5, linecolor="#0f1117",
            cbar_kws={"label":"Avg ROI (%)", "shrink":0.8},
            annot_kws={"size":10, "weight":"bold"})
ax7.set_title("ROI Heatmap — Channel × Campaign Type\n(Green = High ROI, Red = Low ROI)",
              fontweight="bold")
ax7.set_xlabel("Campaign Type")
ax7.set_ylabel("Channel")
ax7.tick_params(axis="x", rotation=15)
ax7.tick_params(axis="y", rotation=0)

# ── Chart 8: Recommendations Box ─────────────────────────
ax8 = fig.add_subplot(gs[3, 2])
ax8.set_facecolor("#0f1117")
ax8.axis("off")
recs = [
    f"1. Boost {top_ch[0]} & {top_ch[1]}",
    f"   budget by 20-30%",
    f"2. Focus on {top_age} age group",
    f"   (highest conversions)",
    f"3. Prioritize {top_region} region",
    f"   campaigns",
    f"4. Run more {top_camp}",
    f"   campaigns",
    f"5. Review & cut spend on",
    f"   {bot_ch[0]} (lowest ROI)",
]
ax8.text(0.5, 0.97, "📋 RECOMMENDATIONS", ha="center", va="top",
         fontsize=11, fontweight="bold", color=ACCENT,
         transform=ax8.transAxes)
ax8.add_patch(mpatches.FancyBboxPatch((0.02, 0.01), 0.96, 0.92,
    boxstyle="round,pad=0.02", facecolor="#1a1d2e",
    edgecolor=ACCENT, linewidth=1.5, transform=ax8.transAxes))
for i, line in enumerate(recs):
    color = "white" if line.startswith(("1","2","3","4","5")) else "#8b949e"
    ax8.text(0.08, 0.83 - i*0.08, line, ha="left", va="top",
             fontsize=9, color=color, transform=ax8.transAxes)

output = "/mnt/user-data/outputs/SCT_DA_4_Business_Insights_Report.png"
plt.savefig(output, dpi=150, bbox_inches="tight",
            facecolor="#0f1117", edgecolor="none")
plt.close()
print(f"\n✅ Report saved: {output}")
print("\n🎉 Task 04 Complete — Business Insights Report (EDA) Done!")
print("=" * 65)

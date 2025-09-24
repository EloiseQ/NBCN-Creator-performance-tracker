# -*- coding: utf-8 -*-
"""
Creator performance analysis (CSV only; no Excel engine)
- Input: combined_raw_import.csv (must be in the SAME folder as this .py)
- Steps:
  1) Read CSV
  2) Normalize columns + coalesce synonyms (Creator, Impressions, CTR, GMV, etc.)
  3) Build unique key and de-duplicate
  4) Group by Creator: videos_count, GMV, weighted CTR, impressions, orders, etc.
  5) Export: creator_summary_*.csv, creator_totals_*.csv, creator_detail_clean_*.csv (same folder as input)
"""

print("[SENTINEL] VSCode CSV analysis script running (performance-analysis folder)")

import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

# ===== 0) Locate input (same directory as this script) =====
BASE_DIR = Path(__file__).expanduser().resolve().parent
DATA_PATH = BASE_DIR / "combined_raw_import.csv"

print(f"[RUNNING FILE] {__file__}")
print(f"[DEBUG] Python: {sys.executable}")
print(f"[DEBUG] pandas: {pd.__version__}")
print(f"[INPUT] {DATA_PATH}")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"未找到输入文件：{DATA_PATH}\n请将 combined_raw_import.csv 放到与本脚本相同的文件夹。")

# ===== 1) Load =====
raw0 = pd.read_csv(DATA_PATH)
print(f"[OK] 读取 CSV：{raw0.shape}")

# ===== 2) Helpers =====

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def to_numeric_scalar(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip()
    s = s.replace("USD", "").replace("$", "").replace(",", "").strip()
    if s in {"", "--", "—", "–", "-", "N/A", "na", "NaN", "nan"}:
        return np.nan
    if s.endswith("%"):
        try:
            return float(s[:-1])
        except Exception:
            return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan

def to_percent_decimal(series: pd.Series) -> pd.Series:
    return series.apply(to_numeric_scalar) / 100.0

def coalesce_numeric(df: pd.DataFrame, candidates, is_percent=False) -> pd.Series:
    """逐列取第一个非空；百分比可转小数。"""
    out = pd.Series([np.nan] * len(df), index=df.index, dtype="float64")
    for c in candidates:
        if c in df.columns:
            ser = df[c]
            ser_conv = to_percent_decimal(ser) if is_percent else ser.apply(to_numeric_scalar)
            out = out.where(~out.isna(), ser_conv)
    return out

def coalesce_text(df: pd.DataFrame, candidates) -> pd.Series:
    out = pd.Series(pd.NA, index=df.index, dtype="object")
    for c in candidates:
        if c in df.columns:
            out = out.fillna(df[c].astype(str).str.strip())
    out = out.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return out

# ===== 3) Standardize fields =====
raw = normalize_cols(raw0)

# creator (多别名 + 兜底)
creator_candidates = [
    "Creator username","Creator Username","Creator","creator",
    "Creator name","Creator Name","Author","author",
    "Username","Handle","TikTok Handle","Creator handle",
    "Author name","Author Nickname","Display Name","Creator Display Name",
    # CN/local
    "作者昵称","作者","达人昵称","达人名称","用户名","创作者昵称","创作者",
]
raw["_creator"] = coalesce_text(raw, creator_candidates)

# 主页链接兜底（tiktok.com/@handle）
link_candidates = [
    "TikTok Profile Link","Profile Link","Profile","Creator Profile Link",
    "Profile URL","Creator URL","Author URL","作者主页链接","达人主页","主页链接",
]
if raw["_creator"].isna().any():
    for c in link_candidates:
        if c in raw.columns:
            extracted = raw[c].astype(str).str.extract(r"tiktok\\.com/@([^/?\\s]+)", expand=False)
            raw["_creator"] = raw["_creator"].fillna(extracted)
    for hc in ["Handle","TikTok Handle","Username","用户名"]:
        if hc in raw.columns:
            raw["_creator"] = raw["_creator"].fillna(raw[hc].astype(str).str.strip())

# video name & date
video_name_candidates = ["Video name","Video Title","Title","视频标题","标题","Video"]
raw["_video_name"] = coalesce_text(raw, video_name_candidates)

date_candidates = ["Video post date","Post date","Date","post date","video post date","视频发布时间","发布时间","上传时间","发布日","发布日期"]
postd = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")
for c in date_candidates:
    if c in raw.columns:
        postd = postd.fillna(pd.to_datetime(raw[c], errors="coerce"))
raw["_post_date"] = postd

# metrics
raw["_impressions"] = coalesce_numeric(raw, ["Shoppable video impressions","Video impressions","Impressions","曝光量","展示量"])
raw["_ctr"]         = coalesce_numeric(raw, ["Affiliate CTR","CTR","Video CTR","Shoppable video CTR","点击率"], is_percent=True)
raw["_orders"]      = coalesce_numeric(raw, ["Affiliate orders","Orders","Shoppable video orders","订单数"])
raw["_items_sold"]  = coalesce_numeric(raw, ["Affiliate items sold","Items sold","件数","销量"])
raw["_gmv"]         = coalesce_numeric(raw, ["Affiliate GMV","Shoppable video GMV","GMV","Affiliate orders GMV","销售额"])
raw["_likes"]       = coalesce_numeric(raw, ["Shoppable video likes","Likes","点赞数"])
raw["_comments"]    = coalesce_numeric(raw, ["Shoppable video comments","Comments","评论数"])
raw["_aov"]         = coalesce_numeric(raw, ["Shoppable video avg. order value","Average order value","AOV","客单价"])

# 确保源文件列存在（用于去重键）
if "_source_file" not in raw.columns:
    raw["_source_file"] = DATA_PATH.name

# 去掉没有 Creator 的行
before_n = len(raw)
raw = raw[raw["_creator"].notna()].copy()
print(f"[FILTER] 去除缺少 Creator 的行：{before_n} -> {len(raw)}")

# ===== 4) De-duplicate =====
key_cols = [c for c in ["_source_file","_creator","_video_name","_post_date"] if c in raw.columns]
raw["_video_key"] = raw[key_cols].astype(str).agg(" | ".join, axis=1)
raw = raw.drop_duplicates(subset=["_video_key"], keep="first")
print(f"[DEDUP] 去重后行数：{len(raw)}（键：{key_cols}）")

# ===== 5) Group by Creator (vectorized weighted CTR) =====
grp = raw.groupby("_creator", dropna=False)
creator_summary = pd.DataFrame({
    "videos_count": grp.size(),
    "gmv_sum": grp["_gmv"].sum(min_count=1),
    "orders_sum": grp["_orders"].sum(min_count=1),
    "items_sold_sum": grp["_items_sold"].sum(min_count=1),
    "impressions_sum": grp["_impressions"].sum(min_count=1),
    "likes_sum": grp["_likes"].sum(min_count=1),
    "comments_sum": grp["_comments"].sum(min_count=1),
})

weighted_ctr_num = (raw["_ctr"] * raw["_impressions"]).groupby(raw["_creator"], dropna=False).sum(min_count=1)
weighted_ctr_den = creator_summary["impressions_sum"].replace(0, np.nan)
creator_summary["ctr_weighted"] = weighted_ctr_num / weighted_ctr_den

creator_summary["orders_per_1k_impr"] = creator_summary["orders_sum"] / (creator_summary["impressions_sum"] / 1000.0)
creator_summary["gmv_per_1k_impr"]    = creator_summary["gmv_sum"]    / (creator_summary["impressions_sum"] / 1000.0)
creator_summary["avg_order_value"]    = creator_summary["gmv_sum"]    / creator_summary["orders_sum"]

creator_summary = creator_summary.reset_index().rename(columns={"_creator":"Creator username"})
creator_summary = creator_summary.sort_values(["gmv_sum","impressions_sum"], ascending=False)

summary_display = creator_summary.copy()
summary_display["ctr_weighted_pct"] = (summary_display["ctr_weighted"] * 100).round(2)
summary_display = summary_display.drop(columns=["ctr_weighted"])
summary_display = summary_display[[
    "Creator username","videos_count","gmv_sum","orders_sum","items_sold_sum",
    "impressions_sum","ctr_weighted_pct","gmv_per_1k_impr","orders_per_1k_impr",
    "avg_order_value","likes_sum","comments_sum"
]]

# ===== 6) Totals =====
_tot_impr = float(np.nansum(raw["_impressions"]))
totals = pd.DataFrame([{
    "videos_count": int(raw.shape[0]),
    "gmv_sum": float(np.nansum(raw["_gmv"])),
    "orders_sum": float(np.nansum(raw["_orders"])),
    "items_sold_sum": float(np.nansum(raw["_items_sold"])),
    "impressions_sum": float(_tot_impr),
    "ctr_weighted_pct": float((np.nansum(raw["_ctr"] * raw["_impressions"]) / _tot_impr) * 100.0) if _tot_impr > 0 else np.nan,
}])

# ===== 7) Save =====
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
summary_path = BASE_DIR / f"creator_summary_{ts}.csv"
totals_path  = BASE_DIR / f"creator_totals_{ts}.csv"
detail_path  = BASE_DIR / f"creator_detail_clean_{ts}.csv"

summary_display.to_csv(summary_path, index=False)
totals.to_csv(totals_path, index=False)
raw[["_source_file","_sheet","_creator","_video_name","_post_date",
     "_impressions","_ctr","_orders","_items_sold","_gmv","_likes","_comments","_aov","_video_key"]
   ].to_csv(detail_path, index=False)

print("\n=== Creator Summary (Top 20) ===")
print(summary_display.head(20).to_string(index=False))

print("\n=== Totals ===")
print(totals.to_string(index=False))

print(f"\n[SAVE] {summary_path}")
print(f"[SAVE] {totals_path}")
print(f"[SAVE] {detail_path}")

# -*- coding: utf-8 -*-
"""
Wrapper script: forwards to the performance-analysis folder CSV analysis
This keeps outputs consistent if you accidentally run the Desktop root script.
"""

print("[SENTINEL] Desktop ROOT wrapper -> using /Users/wenxiqin/Desktop/performance-analysis")

import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

BASE_DIR = Path("/Users/wenxiqin/Desktop/performance-analysis").expanduser().resolve()
DATA_PATH = BASE_DIR / "combined_raw_import.csv"

print(f"[RUNNING FILE] {__file__}")
print(f"[DEBUG] Python: {sys.executable}")
print(f"[DEBUG] pandas: {pd.__version__}")
print(f"[INPUT] {DATA_PATH}")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"未找到输入文件：{DATA_PATH}\n请将 combined_raw_import.csv 放到 /Users/wenxiqin/Desktop/performance-analysis 。")

# == load ==
raw0 = pd.read_csv(DATA_PATH)

# (The rest of the analysis code is identical to the main script)

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def to_numeric_scalar(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip()
    s = s.replace("USD", "").replace("$", "").replace(",", "").strip()
    if s in {"", "--", "—", "–", "-", "N/A", "na", "NaN", "nan"}:
        return np.nan
    if s.endswith("%"):
        try:
            return float(s[:-1])
        except Exception:
            return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan

def to_percent_decimal(series: pd.Series) -> pd.Series:
    return series.apply(to_numeric_scalar) / 100.0

def coalesce_numeric(df: pd.DataFrame, candidates, is_percent=False) -> pd.Series:
    out = pd.Series([np.nan] * len(df), index=df.index, dtype="float64")
    for c in candidates:
        if c in df.columns:
            ser = df[c]
            ser_conv = to_percent_decimal(ser) if is_percent else ser.apply(to_numeric_scalar)
            out = out.where(~out.isna(), ser_conv)
    return out

def coalesce_text(df: pd.DataFrame, candidates) -> pd.Series:
    out = pd.Series(pd.NA, index=df.index, dtype="object")
    for c in df.columns:
        if c in df.columns:
            out = out.fillna(df[c].astype(str).str.strip())
    out = out.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return out

raw = normalize_cols(raw0)

creator_candidates = [
    "Creator username","Creator Username","Creator","creator",
    "Creator name","Creator Name","Author","author",
    "Username","Handle","TikTok Handle","Creator handle",
    "Author name","Author Nickname","Display Name","Creator Display Name",
    "作者昵称","作者","达人昵称","达人名称","用户名","创作者昵称","创作者",
]
raw["_creator"] = coalesce_text(raw, creator_candidates)

link_candidates = [
    "TikTok Profile Link","Profile Link","Profile","Creator Profile Link",
    "Profile URL","Creator URL","Author URL","作者主页链接","达人主页","主页链接",
]
if raw["_creator"].isna().any():
    for c in link_candidates:
        if c in raw.columns:
            extracted = raw[c].astype(str).str.extract(r"tiktok\\.com/@([^/?\\s]+)", expand=False)
            raw["_creator"] = raw["_creator"].fillna(extracted)
    for hc in ["Handle","TikTok Handle","Username","用户名"]:
        if hc in raw.columns:
            raw["_creator"] = raw["_creator"].fillna(raw[hc].astype(str).str.strip())

video_name_candidates = ["Video name","Video Title","Title","视频标题","标题","Video"]
raw["_video_name"] = coalesce_text(raw, video_name_candidates)

from pandas import to_datetime

postd = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")
for c in ["Video post date","Post date","Date","post date","video post date","视频发布时间","发布时间","上传时间","发布日","发布日期"]:
    if c in raw.columns:
        postd = postd.fillna(to_datetime(raw[c], errors="coerce"))
raw["_post_date"] = postd

raw["_impressions"] = coalesce_numeric(raw, ["Shoppable video impressions","Video impressions","Impressions","曝光量","展示量"])
raw["_ctr"]         = coalesce_numeric(raw, ["Affiliate CTR","CTR","Video CTR","Shoppable video CTR","点击率"], is_percent=True)
raw["_orders"]      = coalesce_numeric(raw, ["Affiliate orders","Orders","Shoppable video orders","订单数"])
raw["_items_sold"]  = coalesce_numeric(raw, ["Affiliate items sold","Items sold","件数","销量"])
raw["_gmv"]         = coalesce_numeric(raw, ["Affiliate GMV","Shoppable video GMV","GMV","Affiliate orders GMV","销售额"])
raw["_likes"]       = coalesce_numeric(raw, ["Shoppable video likes","Likes","点赞数"])
raw["_comments"]    = coalesce_numeric(raw, ["Shoppable video comments","Comments","评论数"])
raw["_aov"]         = coalesce_numeric(raw, ["Shoppable video avg. order value","Average order value","AOV","客单价"])

if "_source_file" not in raw.columns:
    raw["_source_file"] = DATA_PATH.name

before_n = len(raw)
raw = raw[raw["_creator"].notna()].copy()

key_cols = [c for c in ["_source_file","_creator","_video_name","_post_date"] if c in raw.columns]
raw["_video_key"] = raw[key_cols].astype(str).agg(" | ".join, axis=1)
raw = raw.drop_duplicates(subset=["_video_key"], keep="first")

# groupby
grp = raw.groupby("_creator", dropna=False)
creator_summary = pd.DataFrame({
    "videos_count": grp.size(),
    "gmv_sum": grp["_gmv"].sum(min_count=1),
    "orders_sum": grp["_orders"].sum(min_count=1),
    "items_sold_sum": grp["_items_sold"].sum(min_count=1),
    "impressions_sum": grp["_impressions"].sum(min_count=1),
    "likes_sum": grp["_likes"].sum(min_count=1),
    "comments_sum": grp["_comments"].sum(min_count=1),
})

weighted_ctr_num = (raw["_ctr"] * raw["_impressions"]).groupby(raw["_creator"], dropna=False).sum(min_count=1)
weighted_ctr_den = creator_summary["impressions_sum"].replace(0, np.nan)
creator_summary["ctr_weighted"] = weighted_ctr_num / weighted_ctr_den

creator_summary["orders_per_1k_impr"] = creator_summary["orders_sum"] / (creator_summary["impressions_sum"] / 1000.0)
creator_summary["gmv_per_1k_impr"]    = creator_summary["gmv_sum"]    / (creator_summary["impressions_sum"] / 1000.0)
creator_summary["avg_order_value"]    = creator_summary["gmv_sum"]    / creator_summary["orders_sum"]

creator_summary = creator_summary.reset_index().rename(columns={"_creator":"Creator username"})
creator_summary = creator_summary.sort_values(["gmv_sum","impressions_sum"], ascending=False)

summary_display = creator_summary.copy()
summary_display["ctr_weighted_pct"] = (summary_display["ctr_weighted"] * 100).round(2)
summary_display = summary_display.drop(columns=["ctr_weighted"])
summary_display = summary_display[[
    "Creator username","videos_count","gmv_sum","orders_sum","items_sold_sum",
    "impressions_sum","ctr_weighted_pct","gmv_per_1k_impr","orders_per_1k_impr",
    "avg_order_value","likes_sum","comments_sum"
]]

_tot_impr = float(np.nansum(raw["_impressions"]))
totals = pd.DataFrame([{
    "videos_count": int(raw.shape[0]),
    "gmv_sum": float(np.nansum(raw["_gmv"])),
    "orders_sum": float(np.nansum(raw["_orders"])),
    "items_sold_sum": float(np.nansum(raw["_items_sold"])),
    "impressions_sum": float(_tot_impr),
    "ctr_weighted_pct": float((np.nansum(raw["_ctr"] * raw["_impressions"]) / _tot_impr) * 100.0) if _tot_impr > 0 else np.nan,
}])

# save to performance-analysis folder
from datetime import datetime as _dt
_ts = _dt.now().strftime("%Y%m%d_%H%M%S")
summary_path = BASE_DIR / f"creator_summary_{_ts}.csv"
totals_path  = BASE_DIR / f"creator_totals_{_ts}.csv"
detail_path  = BASE_DIR / f"creator_detail_clean_{_ts}.csv"

summary_display.to_csv(summary_path, index=False)
totals.to_csv(totals_path, index=False)
raw[["_source_file","_sheet","_creator","_video_name","_post_date",
     "_impressions","_ctr","_orders","_items_sold","_gmv","_likes","_comments","_aov","_video_key"]
   ].to_csv(detail_path, index=False)

print(f"[SAVE] {summary_path}")
print(f"[SAVE] {totals_path}")
print(f"[SAVE] {detail_path}")

# -*- coding: utf-8 -*-
"""
Creator performance analysis → 归一化榜单 & HTML 报告（CSV only）
- 读取同目录 combined_raw_import.csv
- 清洗 → 去重 → Creator 汇总
- 生成归一化榜单（GMV/1k、Orders/1k、加权CTR 综合分）
- 导出：
  1) HTML 报告（不含 raw 明细）
  2) leaderboard_priority.csv / leaderboard_gmv_per_1k.csv / leaderboard_orders_per_1k.csv
  3) creator_summary_plus.csv（含扩展指标）
"""

print("[SENTINEL] Generating normalization leaderboard + HTML report")

import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

# ===== 0) 输入路径（同目录） =====
BASE_DIR = Path(__file__).expanduser().resolve().parent
DATA_PATH = BASE_DIR / "combined_raw_import.csv"

print(f"[RUNNING FILE] {__file__}")
print(f"[DEBUG] Python: {sys.executable}")
print(f"[DEBUG] pandas: {pd.__version__}")
print(f"[INPUT] {DATA_PATH}")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"未找到输入文件：{DATA_PATH}\n请将 combined_raw_import.csv 放到与本脚本相同的文件夹。")

# ===== 1) 读取 =====
raw0 = pd.read_csv(DATA_PATH)
print(f"[OK] 读取 CSV：{raw0.shape}")

# ===== 2) 工具函数 =====

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def to_numeric_scalar(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip()
    s = s.replace("USD", "").replace("$", "").replace(",", "").strip()
    if s in {"", "--", "—", "–", "-", "N/A", "na", "NaN", "nan"}:
        return np.nan
    if s.endswith("%"):
        try:
            return float(s[:-1])
        except Exception:
            return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan

def to_percent_decimal(series: pd.Series) -> pd.Series:
    return series.apply(to_numeric_scalar) / 100.0

def coalesce_numeric(df: pd.DataFrame, candidates, is_percent=False) -> pd.Series:
    out = pd.Series([np.nan] * len(df), index=df.index, dtype="float64")
    for c in candidates:
        if c in df.columns:
            ser = df[c]
            ser_conv = to_percent_decimal(ser) if is_percent else ser.apply(to_numeric_scalar)
            out = out.where(~out.isna(), ser_conv)
    return out

def coalesce_text(df: pd.DataFrame, candidates) -> pd.Series:
    out = pd.Series(pd.NA, index=df.index, dtype="object")
    for c in candidates:
        if c in df.columns:
            out = out.fillna(df[c].astype(str).str.strip())
    out = out.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return out

# ===== 3) 标准化字段 =====
raw = normalize_cols(raw0)

creator_candidates = [
    "Creator username","Creator Username","Creator","creator",
    "Creator name","Creator Name","Author","author",
    "Username","Handle","TikTok Handle","Creator handle",
    "Author name","Author Nickname","Display Name","Creator Display Name",
    "作者昵称","作者","达人昵称","达人名称","用户名","创作者昵称","创作者",
]
raw["_creator"] = coalesce_text(raw, creator_candidates)

# 兜底：从主页链接提取 @handle
link_candidates = [
    "TikTok Profile Link","Profile Link","Profile","Creator Profile Link",
    "Profile URL","Creator URL","Author URL","作者主页链接","达人主页","主页链接",
]
if raw["_creator"].isna().any():
    for c in link_candidates:
        if c in raw.columns:
            extracted = raw[c].astype(str).str.extract(r"tiktok\.com/@([^/?\s]+)", expand=False)
            raw["_creator"] = raw["_creator"].fillna(extracted)
    for hc in ["Handle","TikTok Handle","Username","用户名"]:
        if hc in raw.columns:
            raw["_creator"] = raw["_creator"].fillna(raw[hc].astype(str).str.strip())

video_name_candidates = ["Video name","Video Title","Title","视频标题","标题","Video"]
raw["_video_name"] = coalesce_text(raw, video_name_candidates)

date_candidates = ["Video post date","Post date","Date","post date","video post date","视频发布时间","发布时间","上传时间","发布日","发布日期"]
postd = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")
for c in date_candidates:
    if c in raw.columns:
        postd = postd.fillna(pd.to_datetime(raw[c], errors="coerce"))
raw["_post_date"] = postd

# 指标
raw["_impressions"] = coalesce_numeric(raw, ["Shoppable video impressions","Video impressions","Impressions","曝光量","展示量"])
raw["_ctr"]         = coalesce_numeric(raw, ["Affiliate CTR","CTR","Video CTR","Shoppable video CTR","点击率"], is_percent=True)
raw["_orders"]      = coalesce_numeric(raw, ["Affiliate orders","Orders","Shoppable video orders","订单数"])
raw["_items_sold"]  = coalesce_numeric(raw, ["Affiliate items sold","Items sold","件数","销量"])
raw["_gmv"]         = coalesce_numeric(raw, ["Affiliate GMV","Shoppable video GMV","GMV","Affiliate orders GMV","销售额"])
raw["_likes"]       = coalesce_numeric(raw, ["Shoppable video likes","Likes","点赞数"])
raw["_comments"]    = coalesce_numeric(raw, ["Shoppable video comments","Comments","评论数"])
raw["_aov"]         = coalesce_numeric(raw, ["Shoppable video avg. order value","Average order value","AOV","客单价"])

if "_source_file" not in raw.columns:
    raw["_source_file"] = DATA_PATH.name

# 丢弃缺 Creator 的行
before_n = len(raw)
raw = raw[raw["_creator"].notna()].copy()
print(f"[FILTER] 去除缺少 Creator 的行：{before_n} -> {len(raw)}")

# ===== 4) 去重 =====
key_cols = [c for c in ["_source_file","_creator","_video_name","_post_date"] if c in raw.columns]
raw["_video_key"] = raw[key_cols].astype(str).agg(" | ".join, axis=1)
raw = raw.drop_duplicates(subset=["_video_key"], keep="first")
print(f"[DEDUP] 去重后行数：{len(raw)}（键：{key_cols}）")

# ===== 5) Creator 汇总（含密度指标） =====
grp = raw.groupby("_creator", dropna=False)
creator = pd.DataFrame({
    "videos_count": grp.size(),
    "gmv_sum": grp["_gmv"].sum(min_count=1),
    "orders_sum": grp["_orders"].sum(min_count=1),
    "items_sold_sum": grp["_items_sold"].sum(min_count=1),
    "impressions_sum": grp["_impressions"].sum(min_count=1),
    "likes_sum": grp["_likes"].sum(min_count=1),
    "comments_sum": grp["_comments"].sum(min_count=1),
})

weighted_ctr_num = (raw["_ctr"] * raw["_impressions"]).groupby(raw["_creator"], dropna=False).sum(min_count=1)
weighted_ctr_den = creator["impressions_sum"].replace(0, np.nan)
creator["ctr_weighted"] = weighted_ctr_num / weighted_ctr_den

# 归一化/密度指标
creator["gmv_per_1k_impr"]    = creator["gmv_sum"]   / (creator["impressions_sum"] / 1000.0)
creator["orders_per_1k_impr"] = creator["orders_sum"] / (creator["impressions_sum"] / 1000.0)
creator["avg_order_value"]    = creator["gmv_sum"]   / creator["orders_sum"]
creator["gmv_per_video"]      = creator["gmv_sum"]   / creator["videos_count"].replace(0, np.nan)
creator["orders_per_video"]   = creator["orders_sum"] / creator["videos_count"].replace(0, np.nan)

creator = creator.reset_index().rename(columns={"_creator":"Creator username"})

# ===== 6) 归一化榜单：综合优先级分 =====
# 使用百分位分数（0-100）避免被极端值影响；可按需调整权重

def pct_rank(series: pd.Series) -> pd.Series:
    if series.dropna().empty:
        return pd.Series(np.nan, index=series.index)
    return series.rank(pct=True, method="average") * 100

creator["score_gmv_per_1k"]    = pct_rank(creator["gmv_per_1k_impr"])
creator["score_orders_per_1k"] = pct_rank(creator["orders_per_1k_impr"])
creator["score_ctr"]           = pct_rank(creator["ctr_weighted"])  # ctr 是小数（0-1）

# 综合分权重（可调）：GMV/1k 50%，Orders/1k 30%，CTR 20%
creator["priority_score"] = (
    0.50 * creator["score_gmv_per_1k"] +
    0.30 * creator["score_orders_per_1k"] +
    0.20 * creator["score_ctr"]
)

# 可选：设置门槛，剔除样本过小（默认：>= 2 条视频 & 曝光 >= 10k）
MIN_VIDEOS = 2
MIN_IMPR   = 10000
creator["eligible"] = (creator["videos_count"] >= MIN_VIDEOS) & (creator["impressions_sum"] >= MIN_IMPR)

# 排名表
leader_priority = creator.sort_values("priority_score", ascending=False)
leader_gmv1k    = creator.sort_values("gmv_per_1k_impr", ascending=False)
leader_ord1k    = creator.sort_values("orders_per_1k_impr", ascending=False)

# 显示列（中文友好）
show_cols = [
    "Creator username","videos_count","impressions_sum","gmv_sum","orders_sum",
    "ctr_weighted","gmv_per_1k_impr","orders_per_1k_impr","avg_order_value",
    "gmv_per_video","orders_per_video","priority_score","eligible"
]
for df in (leader_priority, leader_gmv1k, leader_ord1k):
    for c in show_cols:
        if c not in df.columns:
            df[c] = np.nan

# ===== 7) 导出 CSV =====
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
leader_priority_path = BASE_DIR / f"leaderboard_priority_{ts}.csv"
leader_gmv1k_path    = BASE_DIR / f"leaderboard_gmv_per_1k_{ts}.csv"
leader_ord1k_path    = BASE_DIR / f"leaderboard_orders_per_1k_{ts}.csv"
creator_plus_path    = BASE_DIR / f"creator_summary_plus_{ts}.csv"

leader_priority[show_cols].to_csv(leader_priority_path, index=False)
leader_gmv1k[show_cols].to_csv(leader_gmv1k_path, index=False)
leader_ord1k[show_cols].to_csv(leader_ord1k_path, index=False)
creator[show_cols].to_csv(creator_plus_path, index=False)

# ===== 8) 生成 HTML 报告（不包含 raw 明细） =====
html_path = BASE_DIR / f"creator_leaderboard_report_{ts}.html"

# 概览（Totals）
tot_impr = float(np.nansum(creator["impressions_sum"]))
tot_gmv  = float(np.nansum(creator["gmv_sum"]))
tot_orders = float(np.nansum(creator["orders_sum"]))
ctr_w_num = float(np.nansum((raw["_ctr"] * raw["_impressions"]).fillna(0)))
ctr_w_den = float(np.nansum(raw["_impressions"].fillna(0)))
ctr_w_all = (ctr_w_num / ctr_w_den * 100.0) if ctr_w_den > 0 else np.nan

# 友好展示副本
fmt = lambda s: s.to_frame().T.to_html(index=False, border=0)

def fmt_table(df: pd.DataFrame, topn: int = 20) -> str:
    cols = [
        "Creator username","videos_count","impressions_sum","gmv_sum","orders_sum",
        "ctr_weighted","gmv_per_1k_impr","orders_per_1k_impr","avg_order_value",
        "gmv_per_video","orders_per_video","priority_score","eligible"
    ]
    use = [c for c in cols if c in df.columns]
    out = df[use].copy()
    # 美化单位
    out["ctr_weighted"] = (out["ctr_weighted"] * 100).round(2)
    for c in ["gmv_sum","gmv_per_1k_impr","gmv_per_video","avg_order_value"]:
        if c in out.columns:
            out[c] = out[c].round(2)
    for c in ["orders_sum","orders_per_1k_impr","orders_per_video","impressions_sum","videos_count","priority_score"]:
        if c in out.columns:
            out[c] = out[c].round(2)
    return out.head(topn).to_html(index=False, classes="table", border=0)

overview_html = f"""
<ul>
  <li><b>合计曝光</b>：{tot_impr:,.0f}</li>
  <li><b>合计GMV</b>：{tot_gmv:,.2f}</li>
  <li><b>合计订单</b>：{tot_orders:,.0f}</li>
  <li><b>全局加权CTR</b>：{(ctr_w_all if np.isfinite(ctr_w_all) else float('nan')):.2f}%</li>
  <li><b>样本创作者数</b>：{creator.shape[0]}</li>
  <li><b>优先级门槛</b>：视频≥{MIN_VIDEOS} 且 曝光≥{MIN_IMPR:,}</li>
</ul>
"""

html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<title>Creator 归一化榜单报告</title>
  <style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, 'Noto Sans', 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', sans-serif; margin: 24px; }}
  h1, h2, h3 {{ margin: 0.4em 0; }}
  .muted {{ color: #666; font-size: 12px; }}
  .section {{ margin-top: 28px; }}
  .table {{ border-collapse: collapse; width: 100%; }}
  .table th, .table td {{ border-bottom: 1px solid #eee; padding: 8px 10px; text-align: right; }}
  .table th:first-child, .table td:first-child {{ text-align: left; }}
  .badge {{ display: inline-block; background: #f4f4f4; border: 1px solid #ddd; border-radius: 12px; padding: 2px 8px; font-size: 12px; margin-left: 6px; }}
  .note {{ background: #fff7e6; border: 1px solid #ffe58f; padding: 10px 12px; border-radius: 6px; }}
  .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; align-items: start; }}
  .card {{ background: #fff; border: 1px solid #eee; border-radius: 8px; padding: 8px 10px; }}
  @media (max-width: 980px) {{ .grid2 {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <h1>Creator 归一化榜单报告</h1>
  <div class="muted">生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>

  <div class="section">
    <h2>总览 <span class="badge">不含 raw 明细</span></h2>
    {overview_html}
    <div class="note">说明：综合分 = 0.50×GMV/1k + 0.30×Orders/1k + 0.20×加权CTR（均为百分位分）。为避免小样本误导，默认仅将“视频≥{MIN_VIDEOS} & 曝光≥{MIN_IMPR:,}”视为推荐对象，但下表会展示所有创作者并标注 eligible。</div>
  </div>

  <div class="section">
    <h2>综合优先级榜（Top 20）</h2>
    {fmt_table(leader_priority)}
  </div>

  <div class="section">
    <h2>按 GMV / 1k Impr（Top 20）</h2>
    {fmt_table(leader_gmv1k)}
  </div>

  <div class="section">
    <h2>按 Orders / 1k Impr（Top 20）</h2>
    {fmt_table(leader_ord1k)}
  </div>

  <div class="section">
    <h3>文件导出</h3>
    <ul>
      <li>{leader_priority_path.name}</li>
      <li>{leader_gmv1k_path.name}</li>
      <li>{leader_ord1k_path.name}</li>
      <li>{creator_plus_path.name}</li>
    </ul>
    <div class="muted">（已保存到：{BASE_DIR}）</div>
  </div>
</body>
</html>
"""

with open(html_path, "w", encoding="utf-8") as f:
    f.write(html)

print(f"[SAVE] {leader_priority_path}")
print(f"[SAVE] {leader_gmv1k_path}")
print(f"[SAVE] {leader_ord1k_path}")
print(f"[SAVE] {creator_plus_path}")
print(f"[SAVE] {html_path}")
# -*- coding: utf-8 -*-
"""
Creator performance analysis (CSV only; no Excel engine)
- Input: combined_raw_import.csv (same folder as this .py)
- Outputs: CSVs + one HTML report (no raw detail)
- Modules included:
  • Base: Read → Clean/Standardize → De-duplicate
  • Creator-level summary + normalization leaderboard
  • (1) Funnel metrics: clicks_est, CVR, GMV/Click, Items/Order
  • (2) Cohort windows: D1/D3/D7/D14 (only videos that have reached each window)
"""

print("[SENTINEL] Leaderboard + Funnel + Cohort Windows (1 & 2) — CSV→HTML")

import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

# ===== 0) Locate input (same directory as this script) =====
BASE_DIR = Path(__file__).expanduser().resolve().parent
DATA_PATH = BASE_DIR / "combined_raw_import.csv"

print(f"[RUNNING FILE] {__file__}")
print(f"[DEBUG] Python: {sys.executable}")
print(f"[DEBUG] pandas: {pd.__version__}")
print(f"[INPUT] {DATA_PATH}")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"未找到输入文件：{DATA_PATH}\n请将 combined_raw_import.csv 放到与本脚本相同的文件夹。")

# ===== 1) Load =====
raw0 = pd.read_csv(DATA_PATH)
print(f"[OK] 读取 CSV：{raw0.shape}")

# ===== 2) Helpers =====

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def to_numeric_scalar(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip()
    s = s.replace("USD", "").replace("$", "").replace(",", "").strip()
    if s in {"", "--", "—", "–", "-", "N/A", "na", "NaN", "nan"}:
        return np.nan
    if s.endswith("%"):
        try:
            return float(s[:-1])
        except Exception:
            return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan

def to_percent_decimal(series: pd.Series) -> pd.Series:
    return series.apply(to_numeric_scalar) / 100.0

def coalesce_numeric(df: pd.DataFrame, candidates, is_percent=False) -> pd.Series:
    """逐列取第一个非空；百分比转小数。"""
    out = pd.Series([np.nan] * len(df), index=df.index, dtype="float64")
    for c in candidates:
        if c in df.columns:
            ser = df[c]
            ser_conv = to_percent_decimal(ser) if is_percent else ser.apply(to_numeric_scalar)
            out = out.where(~out.isna(), ser_conv)
    return out

def coalesce_text(df: pd.DataFrame, candidates) -> pd.Series:
    out = pd.Series(pd.NA, index=df.index, dtype="object")
    for c in candidates:
        if c in df.columns:
            out = out.fillna(df[c].astype(str).str.strip())
    out = out.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return out

# ===== 3) Standardize fields =====
raw = normalize_cols(raw0)

creator_candidates = [
    "Creator username","Creator Username","Creator","creator",
    "Creator name","Creator Name","Author","author",
    "Username","Handle","TikTok Handle","Creator handle",
    "Author name","Author Nickname","Display Name","Creator Display Name",
    # CN/local
    "作者昵称","作者","达人昵称","达人名称","用户名","创作者昵称","创作者",
]
raw["_creator"] = coalesce_text(raw, creator_candidates)

# 兜底：从主页链接提取 @handle（tiktok.com/@xxx）
link_candidates = [
    "TikTok Profile Link","Profile Link","Profile","Creator Profile Link",
    "Profile URL","Creator URL","Author URL","作者主页链接","达人主页","主页链接",
]
if raw["_creator"].isna().any():
    for c in link_candidates:
        if c in raw.columns:
            extracted = raw[c].astype(str).str.extract(r"tiktok\\.com/@([^/?\\s]+)", expand=False)
            raw["_creator"] = raw["_creator"].fillna(extracted)
    for hc in ["Handle","TikTok Handle","Username","用户名"]:
        if hc in raw.columns:
            raw["_creator"] = raw["_creator"].fillna(raw[hc].astype(str).str.strip())

video_name_candidates = ["Video name","Video Title","Title","视频标题","标题","Video"]
raw["_video_name"] = coalesce_text(raw, video_name_candidates)

date_candidates = ["Video post date","Post date","Date","post date","video post date","视频发布时间","发布时间","上传时间","发布日","发布日期"]
postd = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")
for c in date_candidates:
    if c in raw.columns:
        postd = postd.fillna(pd.to_datetime(raw[c], errors="coerce"))
raw["_post_date"] = postd

# metrics
raw["_impressions"] = coalesce_numeric(raw, ["Shoppable video impressions","Video impressions","Impressions","曝光量","展示量"])
raw["_ctr"]         = coalesce_numeric(raw, ["Affiliate CTR","CTR","Video CTR","Shoppable video CTR","点击率"], is_percent=True)
raw["_orders"]      = coalesce_numeric(raw, ["Affiliate orders","Orders","Shoppable video orders","订单数"])
raw["_items_sold"]  = coalesce_numeric(raw, ["Affiliate items sold","Items sold","件数","销量"])
raw["_gmv"]         = coalesce_numeric(raw, ["Affiliate GMV","Shoppable video GMV","GMV","Affiliate orders GMV","销售额"])
raw["_likes"]       = coalesce_numeric(raw, ["Shoppable video likes","Likes","点赞数"])
raw["_comments"]    = coalesce_numeric(raw, ["Shoppable video comments","Comments","评论数"])
raw["_aov"]         = coalesce_numeric(raw, ["Shoppable video avg. order value","Average order value","AOV","客单价"])

# 源文件列（用于去重）
if "_source_file" not in raw.columns:
    raw["_source_file"] = DATA_PATH.name

# 去掉没有 Creator 的行
before_n = len(raw)
raw = raw[raw["_creator"].notna()].copy()
print(f"[FILTER] 去除缺少 Creator 的行：{before_n} -> {len(raw)}")

# ===== 4) De-duplicate =====
key_cols = [c for c in ["_source_file","_creator","_video_name","_post_date"] if c in raw.columns]
raw["_video_key"] = raw[key_cols].astype(str).agg(" | ".join, axis=1)
raw = raw.drop_duplicates(subset=["_video_key"], keep="first")
print(f"[DEDUP] 去重后行数：{len(raw)}（键：{key_cols}）")

# ===== 5) Creator summary =====
grp = raw.groupby("_creator", dropna=False)
creator = pd.DataFrame({
    "videos_count": grp.size(),
    "gmv_sum": grp["_gmv"].sum(min_count=1),
    "orders_sum": grp["_orders"].sum(min_count=1),
    "items_sold_sum": grp["_items_sold"].sum(min_count=1),
    "impressions_sum": grp["_impressions"].sum(min_count=1),
    "likes_sum": grp["_likes"].sum(min_count=1),
    "comments_sum": grp["_comments"].sum(min_count=1),
})

# 加权CTR = sum(ctr*impr)/sum(impr)
weighted_ctr_num = (raw["_ctr"] * raw["_impressions"]).groupby(raw["_creator"], dropna=False).sum(min_count=1)
weighted_ctr_den = creator["impressions_sum"].replace(0, np.nan)
creator["ctr_weighted"] = weighted_ctr_num / weighted_ctr_den

# 密度指标
creator["gmv_per_1k_impr"]    = creator["gmv_sum"]   / (creator["impressions_sum"] / 1000.0)
creator["orders_per_1k_impr"] = creator["orders_sum"] / (creator["impressions_sum"] / 1000.0)
creator["avg_order_value"]    = creator["gmv_sum"]   / creator["orders_sum"]
creator["gmv_per_video"]      = creator["gmv_sum"]   / creator["videos_count"].replace(0, np.nan)
creator["orders_per_video"]   = creator["orders_sum"] / creator["videos_count"].replace(0, np.nan)

creator = creator.reset_index().rename(columns={"_creator":"Creator username"})

# ===== (1) Funnel metrics =====
# 约算点击：sum(impr*ctr)
creator["clicks_est_sum"] = weighted_ctr_num.values
creator["cvr"] = creator["orders_sum"] / creator["clicks_est_sum"]
creator["gmv_per_click"] = creator["gmv_sum"] / creator["clicks_est_sum"]
creator["items_per_order"] = creator["items_sold_sum"] / creator["orders_sum"]


leader_cvr = creator.sort_values("cvr", ascending=False)
leader_gmv_click = creator.sort_values("gmv_per_click", ascending=False)

# Export funnel CSVs (ensure ts exists)
try:
    ts
except NameError:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

leader_cvr_path       = BASE_DIR / f"leaderboard_cvr_{ts}.csv"
leader_gmv_click_path = BASE_DIR / f"leaderboard_gmv_per_click_{ts}.csv"

leader_cvr[["Creator username","videos_count","impressions_sum","orders_sum","clicks_est_sum","cvr"]].to_csv(leader_cvr_path, index=False)
leader_gmv_click[["Creator username","videos_count","impressions_sum","gmv_sum","clicks_est_sum","gmv_per_click"]].to_csv(leader_gmv_click_path, index=False)

# ===== Normalization leaderboard（综合优先级分） =====

def pct_rank(series: pd.Series) -> pd.Series:
    if series.dropna().empty:
        return pd.Series(np.nan, index=series.index)
    return series.rank(pct=True, method="average") * 100

creator["score_gmv_per_1k"]    = pct_rank(creator["gmv_per_1k_impr"])
creator["score_orders_per_1k"] = pct_rank(creator["orders_per_1k_impr"])
creator["score_ctr"]           = pct_rank(creator["ctr_weighted"])  # ctr 为小数（0-1）

W_GMV1K, W_ORD1K, W_CTR = 0.50, 0.30, 0.20
creator["priority_score"] = (
    W_GMV1K * creator["score_gmv_per_1k"] +
    W_ORD1K * creator["score_orders_per_1k"] +
    W_CTR   * creator["score_ctr"]
)

MIN_VIDEOS = 2
MIN_IMPR   = 10000
creator["eligible"] = (creator["videos_count"] >= MIN_VIDEOS) & (creator["impressions_sum"] >= MIN_IMPR)

leader_priority = creator.sort_values("priority_score", ascending=False)
leader_gmv1k    = creator.sort_values("gmv_per_1k_impr", ascending=False)
leader_ord1k    = creator.sort_values("orders_per_1k_impr", ascending=False)


# ===== (3) Video-wise analysis =====
# Ensure days-since-post exists (used in video-wise tables)
if "_days_since_post" not in raw.columns:
    _today = pd.Timestamp.today().normalize()
    raw["_days_since_post"] = (_today - raw["_post_date"]).dt.days
    # clip negative values (in case of future-dated posts)
    raw["_days_since_post"] = raw["_days_since_post"].clip(lower=0)
# Per-video metrics & rankings
video_metrics = raw[["_creator","_video_name","_post_date","_impressions","_ctr","_orders","_gmv","_items_sold","_likes","_comments","_aov","_days_since_post"]].copy()
video_metrics = video_metrics.rename(columns={
    "_creator": "Creator username",
    "_video_name": "Video name",
    "_post_date": "Post date",
    "_impressions": "impressions",
    "_ctr": "ctr",
    "_orders": "orders",
    "_gmv": "gmv",
    "_items_sold": "items_sold",
    "_likes": "likes",
    "_comments": "comments",
    "_aov": "aov",
    "_days_since_post": "days_since_post",
})
# Derived per-video metrics
video_metrics["clicks_est"]       = video_metrics["impressions"] * video_metrics["ctr"]
video_metrics["cvr"]              = video_metrics["orders"] / video_metrics["clicks_est"]
video_metrics["gmv_per_1k_impr"]  = video_metrics["gmv"] / (video_metrics["impressions"] / 1000.0)
video_metrics["orders_per_1k_impr"] = video_metrics["orders"] / (video_metrics["impressions"] / 1000.0)
video_metrics["like_rate"]        = video_metrics["likes"] / video_metrics["impressions"]
video_metrics["comment_rate"]     = video_metrics["comments"] / video_metrics["impressions"]

# Thresholds for stable ranking (avoid tiny denominators)
MIN_IMPR_VIDEO = 1000
MIN_CLICKS_VIDEO = 30

video_top_gmv1k = video_metrics.loc[(video_metrics["impressions"] >= MIN_IMPR_VIDEO)].sort_values("gmv_per_1k_impr", ascending=False)
video_top_cvr   = video_metrics.loc[(video_metrics["clicks_est"] >= MIN_CLICKS_VIDEO)].sort_values("cvr", ascending=False)

# Exports
video_analysis_path      = BASE_DIR / f"video_analysis_{ts}.csv"
top_videos_gmv1k_path    = BASE_DIR / f"top_videos_gmv_per_1k_{ts}.csv"
top_videos_cvr_path      = BASE_DIR / f"top_videos_cvr_{ts}.csv"
video_metrics.to_csv(video_analysis_path, index=False)
video_top_gmv1k.to_csv(top_videos_gmv1k_path, index=False)
video_top_cvr.to_csv(top_videos_cvr_path, index=False)


# ===== (9) What-if：CTR+10% / CVR+10% =====
creator_base = creator.copy()
creator_base["delta_orders_CTRp10"] = creator_base["orders_sum"] * 0.10
creator_base["delta_gmv_CTRp10"]    = creator_base["gmv_sum"]   * 0.10
creator_base["delta_orders_CVRp10"] = creator_base["orders_sum"] * 0.10
creator_base["delta_gmv_CVRp10"]    = creator_base["gmv_sum"]   * 0.10

what_if = creator_base[[
    "Creator username","videos_count","impressions_sum","ctr_weighted","cvr","gmv_sum","orders_sum",
    "delta_orders_CTRp10","delta_gmv_CTRp10","delta_orders_CVRp10","delta_gmv_CVRp10"
]].sort_values("delta_gmv_CTRp10", ascending=False)

whatif_path = BASE_DIR / f"what_if_ctr_cvr_{ts}.csv"

what_if.to_csv(whatif_path, index=False)

# ===== (0b) Paid creator spending mapping & ROI / Payback (published-only) =====
# Contract list provided by user (TikTok Handle, #Vids, Spending)
spending_rows = [
    {"Creator username": "j.izzyj",            "contract_vids": 15, "spending": 1250.0},
    {"Creator username": "janessgg",           "contract_vids": 20, "spending": 1200.0},
    {"Creator username": "glowwdani",          "contract_vids": 5,  "spending": 1250.0},
    {"Creator username": "lupitatong",         "contract_vids": 5,  "spending": 1000.0},
    {"Creator username": "therealjoyymaria",   "contract_vids": 5,  "spending": 1000.0},
    {"Creator username": "ekndra",             "contract_vids": 5,  "spending": 1000.0},
    {"Creator username": "iamsk8bordb",        "contract_vids": 3,  "spending": 1500.0},
    {"Creator username": "raamirezsteph",      "contract_vids": 5,  "spending": 825.0},
]
spend_df = pd.DataFrame(spending_rows)

# Contract totals for display
contract_total_vids = int(np.nansum(spend_df["contract_vids"]))
contract_total_budget = float(np.nansum(spend_df["spending"]))

# Join with creator-level metrics (include creators not in spend list; missing spend treated as 0)
creator_spend = pd.merge(creator, spend_df, on="Creator username", how="left")
# Unit cost & spent-to-date (cap at total spending)
creator_spend["contract_vids"] = creator_spend["contract_vids"].astype("float64")
creator_spend["spending"] = creator_spend["spending"].astype("float64")
creator_spend["unit_cost"] = creator_spend["spending"] / creator_spend["contract_vids"]
creator_spend["spent_to_date"] = np.where(
    (creator_spend["contract_vids"].notna()) & (creator_spend["spending"].notna()),
    np.minimum(creator_spend["videos_count"].fillna(0), creator_spend["contract_vids"]) * creator_spend["unit_cost"],
    0.0,
)
creator_spend["spent_to_date"] = creator_spend[["spent_to_date","spending"]].min(axis=1).fillna(0.0)

# ROI & payback
creator_spend["roi_gmv_over_spent"] = np.where(creator_spend["spent_to_date"] > 0,
                                                creator_spend["gmv_sum"] / creator_spend["spent_to_date"], np.nan)
creator_spend["payback_abs"] = creator_spend["gmv_sum"] - creator_spend["spent_to_date"]
creator_spend["payback_pct"] = np.where(creator_spend["spent_to_date"] > 0,
                                         creator_spend["gmv_sum"] / creator_spend["spent_to_date"] - 1.0, np.nan)

# Display table (sorted by spent_to_date desc)
spend_table = creator_spend[[
    "Creator username","videos_count","contract_vids","spending","unit_cost","spent_to_date",
    "impressions_sum","gmv_sum","orders_sum","ctr_weighted","gmv_per_1k_impr","orders_per_1k_impr",
    "roi_gmv_over_spent","payback_abs","payback_pct"
]].sort_values("spent_to_date", ascending=False)

# Export CSV
spend_recovery_path = BASE_DIR / f"creator_spend_recovery_{ts}.csv"
spend_table.to_csv(spend_recovery_path, index=False)

# Totals for HTML summary line
total_spent_to_date = float(np.nansum(spend_table["spent_to_date"]))
overall_roi = (tot_gmv / total_spent_to_date) if total_spent_to_date > 0 else np.nan
overall_payback_abs = tot_gmv - total_spent_to_date
overall_payback_pct = ((tot_gmv / total_spent_to_date) - 1.0) * 100.0 if total_spent_to_date > 0 else np.nan

# ===== HTML report (no raw detail) =====
html_path = BASE_DIR / f"creator_leaderboard_report_{ts}.html"

# Overview
TOT_IMPR = float(np.nansum(raw["_impressions"]))
tot_impr = float(np.nansum(creator["impressions_sum"]))
tot_gmv  = float(np.nansum(creator["gmv_sum"]))
tot_orders = float(np.nansum(creator["orders_sum"]))
ctr_w_all = (float(np.nansum(raw["_ctr"] * raw["_impressions"])) / TOT_IMPR * 100.0) if TOT_IMPR > 0 else np.nan

# pretty tables

def fmt_table(df: pd.DataFrame, cols=None, topn: int = 20, pct_cols=None, round2=None):
    d = df.copy()
    if cols is not None:
        cols = [c for c in cols if c in d.columns]
        d = d[cols]
    if pct_cols:
        for c in pct_cols:
            if c in d.columns:
                d[c] = (d[c] * 100).round(2)
    if round2:
        for c in round2:
            if c in d.columns:
                d[c] = d[c].round(2)
    return d.head(topn).to_html(index=False, classes="table", border=0)

priority_cols = [
    "Creator username","videos_count","impressions_sum","gmv_sum","orders_sum",
    "ctr_weighted","gmv_per_1k_impr","orders_per_1k_impr","avg_order_value",
    "gmv_per_video","orders_per_video","priority_score","eligible"
]

funnel_cvr_cols = ["Creator username","videos_count","impressions_sum","orders_sum","clicks_est_sum","cvr"]
funnel_gmv_click_cols = ["Creator username","videos_count","impressions_sum","gmv_sum","clicks_est_sum","gmv_per_click"]


whatif_cols = [
    "Creator username","videos_count","impressions_sum","ctr_weighted","cvr","gmv_sum","orders_sum",
    "delta_orders_CTRp10","delta_gmv_CTRp10","delta_orders_CVRp10","delta_gmv_CVRp10"
]

spend_cols = [
    "Creator username","videos_count","contract_vids","spending","unit_cost","spent_to_date",
    "impressions_sum","gmv_sum","orders_sum","ctr_weighted","gmv_per_1k_impr","orders_per_1k_impr",
    "roi_gmv_over_spent","payback_abs","payback_pct"
]

video_cols_common = [
    "Creator username","Video name","Post date","days_since_post",
    "impressions","ctr","clicks_est","orders","gmv",
    "gmv_per_1k_impr","orders_per_1k_impr","cvr"
]

# ===== Visualizations (Plotly) =====
try:
    import plotly.graph_objects as go
    from plotly.io import to_html as _to_html
    _PLOTLY_OK = True
except Exception:
    _PLOTLY_OK = False

def _div_placeholder(title):
    return f"<div class='muted'>No data: {title}</div>"

def fig_to_div(fig, title):
    if not _PLOTLY_OK or fig is None:
        return _div_placeholder(title)
    return _to_html(fig, include_plotlyjs='cdn', full_html=False)

# Creators — GMV per 1k Impr（全部 creators，非 Top10）
if _PLOTLY_OK and not creator.empty:
    _gmv_df = leader_gmv1k[["Creator username","gmv_per_1k_impr"]].copy()
    fig_gmv = go.Figure([go.Bar(x=_gmv_df["Creator username"], y=_gmv_df["gmv_per_1k_impr"])])
    fig_gmv.update_layout(width=560, height=360, margin=dict(l=20,r=20,t=40,b=80), xaxis_tickangle=-45, title="Creators — GMV per 1k Impr")
else:
    fig_gmv = None
chart_gmv1k_div = fig_to_div(fig_gmv, "Creators — GMV per 1k Impr")

# Creators — Orders per 1k Impr（全部 creators，非 Top10）
if _PLOTLY_OK and not creator.empty:
    _ord_df = leader_ord1k[["Creator username","orders_per_1k_impr"]].copy()
    fig_ord = go.Figure([go.Bar(x=_ord_df["Creator username"], y=_ord_df["orders_per_1k_impr"])])
    fig_ord.update_layout(width=560, height=360, margin=dict(l=20,r=20,t=40,b=80), xaxis_tickangle=-45, title="Creators — Orders per 1k Impr")
else:
    fig_ord = None
chart_ord1k_div = fig_to_div(fig_ord, "Creators — Orders per 1k Impr")

# ROI vs Spent（仅付费，散点）
_roi = spend_table[["Creator username","spent_to_date","roi_gmv_over_spent"]].dropna().copy() if 'spend_table' in globals() else pd.DataFrame()
_roi = _roi[_roi["spent_to_date"] > 0] if not _roi.empty else _roi
if _PLOTLY_OK and not _roi.empty:
    fig_roi = go.Figure([go.Scatter(x=_roi["spent_to_date"], y=_roi["roi_gmv_over_spent"], mode='markers', text=_roi["Creator username"])])
    fig_roi.update_layout(width=560, height=360, margin=dict(l=20,r=20,t=40,b=60), title="ROI vs Spent（Paid Creators）", xaxis_title="Spent to date", yaxis_title="ROI (GMV/Spent)")
else:
    fig_roi = None
chart_roi_div = fig_to_div(fig_roi, "ROI vs Spent（Paid Creators）")

# Video Top5 — GMV/1k（仍保留 Top5）
_v1 = video_top_gmv1k[["Creator username","Video name","gmv_per_1k_impr"]].head(5).copy()
if _PLOTLY_OK and not _v1.empty:
    _v1["label"] = _v1["Creator username"].astype(str) + ": " + _v1["Video name"].astype(str).str.slice(0, 25)
    fig_v_gmv = go.Figure([go.Bar(x=_v1["label"], y=_v1["gmv_per_1k_impr"])])
    fig_v_gmv.update_layout(width=560, height=360, margin=dict(l=20,r=20,t=40,b=100), xaxis_tickangle=-45, title="Top5 Videos — GMV per 1k Impr")
else:
    fig_v_gmv = None
chart_v_gmv1k_div = fig_to_div(fig_v_gmv, "Top5 Videos — GMV per 1k Impr")

# Video Top5 — CVR（仍保留 Top5）
_v2 = video_top_cvr[["Creator username","Video name","cvr"]].head(5).copy()
if _PLOTLY_OK and not _v2.empty:
    _v2["label"] = _v2["Creator username"].astype(str) + ": " + _v2["Video name"].astype(str).str.slice(0, 25)
    _v2["cvr_pct"] = _v2["cvr"] * 100.0
    fig_v_cvr = go.Figure([go.Bar(x=_v2["label"], y=_v2["cvr_pct"])])
    fig_v_cvr.update_layout(width=560, height=360, margin=dict(l=20,r=20,t=40,b=100), xaxis_tickangle=-45, title="Top5 Videos — CVR (%)")
else:
    fig_v_cvr = None
chart_v_cvr_div = fig_to_div(fig_v_cvr, "Top5 Videos — CVR (%)")

# Pie — GMV 分布（按 Creator）
_gmv_pie_df = creator[["Creator username","gmv_sum"]].copy()
_gmv_pie_df = _gmv_pie_df[_gmv_pie_df["gmv_sum"] > 0] if not _gmv_pie_df.empty else _gmv_pie_df
if _PLOTLY_OK and not _gmv_pie_df.empty:
    fig_pie = go.Figure([go.Pie(labels=_gmv_pie_df["Creator username"], values=_gmv_pie_df["gmv_sum"])])
    fig_pie.update_layout(width=520, height=360, margin=dict(l=10,r=10,t=40,b=10), title="GMV 分布（按 Creator）")
else:
    fig_pie = None
chart_pie_gmv_div = fig_to_div(fig_pie, "GMV 分布（按 Creator）")

overview_html = f"""
<ul>
  <li><b>合计曝光</b>：{tot_impr:,.0f}</li>
  <li><b>合计GMV</b>：{tot_gmv:,.2f}</li>
  <li><b>合计订单</b>：{tot_orders:,.0f}</li>
  <li><b>全局加权CTR</b>：{(ctr_w_all if np.isfinite(ctr_w_all) else float('nan')):.2f}%</li>
  <li><b>样本创作者数</b>：{creator.shape[0]}</li>
  <li><b>优先级门槛</b>：视频≥{MIN_VIDEOS} 且 曝光≥{MIN_IMPR:,}</li>
</ul>
"""

html = f"""
<!DOCTYPE html>
<html lang=\"zh-CN\">
<head>
<meta charset=\"utf-8\" />
<title>Creator 归一化榜单报告（含漏斗 & 同龄窗口）</title>
<style>
 body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, 'Noto Sans', 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', sans-serif; margin: 24px; }}
 h1, h2, h3 {{ margin: 0.4em 0; }}
 .muted {{ color: #666; font-size: 12px; }}
 .section {{ margin-top: 28px; }}
 .table {{ border-collapse: collapse; width: 100%; }}
 .table th, .table td {{ border-bottom: 1px solid #eee; padding: 8px 10px; text-align: right; }}
 .table th:first-child, .table td:first-child {{ text-align: left; }}
 .badge {{ display: inline-block; background: #f4f4f4; border: 1px solid #ddd; border-radius: 12px; padding: 2px 8px; font-size: 12px; margin-left: 6px; }}
 .note {{ background: #fff7e6; border: 1px solid #ffe58f; padding: 10px 12px; border-radius: 6px; }}
</style>
</head>
<body>
  <h1>Creator 归一化榜单报告（含漏斗 & 同龄窗口）</h1>
  <div class=\"muted\">生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>

  <div class=\"section\">
    <h2>投放花费 & 回本（已发布口径）</h2>
    <div class=\"note\">#Vids 为合同约定；“已花费”= 单价 × min(已发布, #Vids)，且不超过合同总额。未在清单中的创作者视为花费 0。</div>
    <p class=\"muted\">合同合计：#Vids：{contract_total_vids:,}；预算：{contract_total_budget:,.2f}</p>
    {fmt_table(spend_table, spend_cols, topn=50, pct_cols=['ctr_weighted','payback_pct'], round2=['impressions_sum','gmv_sum','orders_sum','gmv_per_1k_impr','orders_per_1k_impr','unit_cost','spending','spent_to_date','roi_gmv_over_spent','payback_abs'])}
    <p class=\"muted\">合计已花费：{total_spent_to_date:,.2f}；合计GMV：{tot_gmv:,.2f}；整体ROI：{(overall_roi if np.isfinite(overall_roi) else float('nan')):.2f}×；回本差额：{overall_payback_abs:,.2f}（{(overall_payback_pct if np.isfinite(overall_payback_pct) else float('nan')):.2f}%）</p>
  </div>

  <div class=\"section\">
    <h2>总览 <span class=\"badge\">不含 raw 明细</span></h2>
    {overview_html}
    <div class=\"note\">综合分 = {W_GMV1K:.0%}×GMV/1k + {W_ORD1K:.0%}×Orders/1k + {W_CTR:.0%}×加权CTR（均为百分位分）。仅将“视频≥{MIN_VIDEOS} & 曝光≥{MIN_IMPR:,}”视为推荐对象，但下表会展示所有创作者并标注 eligible。</div>
  </div>

  <div class=\"section\">
    <h2>综合优先级榜（Top 20）</h2>
    {fmt_table(leader_priority, priority_cols, topn=20, pct_cols=['ctr_weighted'], round2=['gmv_sum','orders_sum','gmv_per_1k_impr','orders_per_1k_impr','avg_order_value','gmv_per_video','orders_per_video','priority_score'])}
  </div>

  <div class=\"section\">
    <h2>① 转化漏斗（仅 CVR 排行，Top 20）</h2>
    {fmt_table(leader_cvr, funnel_cvr_cols, topn=20, pct_cols=['cvr'], round2=['impressions_sum','orders_sum','clicks_est_sum'])}
  </div>


  <div class=\"section\">
    <h2>③ Video-wise（关键 Top）</h2>
    <div class=\"note\">为避免小样本噪声，默认：GMV/1k 榜仅统计 Impr ≥ {MIN_IMPR_VIDEO:,}；CVR 榜仅统计 Clicks ≥ {MIN_CLICKS_VIDEO:,}。</div>
    <h3>按 GMV / 1k Impr（Impr ≥ {MIN_IMPR_VIDEO:,}）Top 5</h3>
    {fmt_table(video_top_gmv1k, video_cols_common, topn=5, pct_cols=['ctr','cvr'], round2=['impressions','orders','gmv','gmv_per_1k_impr','orders_per_1k_impr','clicks_est'])}
    <h3>按 CVR（Clicks ≥ {MIN_CLICKS_VIDEO:,}）Top 5</h3>
    {fmt_table(video_top_cvr, video_cols_common, topn=5, pct_cols=['ctr','cvr'], round2=['impressions','orders','gmv','gmv_per_1k_impr','orders_per_1k_impr','clicks_est'])}
  </div>

  <div class="section">
    <h2>📈 可视化</h2>
    <div class="note">Plotly 交互图（两列布局；图幅缩小）。</div>

    <div class="grid2">
      <div class="card">
        <h3>Creators — GMV per 1k Impr</h3>
        {chart_gmv1k_div}
      </div>
      <div class="card">
        <h3>Creators — Orders per 1k Impr</h3>
        {chart_ord1k_div}
      </div>
    </div>

    <div class="grid2">
      <div class="card">
        <h3>GMV 分布（按 Creator）</h3>
        {chart_pie_gmv_div}
      </div>
      <div class="card">
        <h3>ROI vs Spent（Paid Creators）</h3>
        {chart_roi_div}
      </div>
    </div>

    <div class="grid2">
      <div class="card">
        <h3>Top5 Videos — GMV per 1k Impr</h3>
        {chart_v_gmv1k_div}
      </div>
      <div class="card">
        <h3>Top5 Videos — CVR</h3>
        {chart_v_cvr_div}
      </div>
    </div>
  </div>

  <div class=\"section\">
    <h2>⑨ What-if（+10% CTR / +10% CVR）</h2>
    <div class=\"note\">线性假设：在曝光与客单不变的前提下，CTR 或 CVR 相对提升 10% 对 GMV/订单的相对影响等同。下表按“GMV 绝对增量（CTR+10%）”排序展示 Top 20。</div>
    {fmt_table(what_if, whatif_cols, topn=20, pct_cols=['ctr_weighted','cvr'], round2=['impressions_sum','gmv_sum','orders_sum','delta_orders_CTRp10','delta_gmv_CTRp10','delta_orders_CVRp10','delta_gmv_CVRp10'])}
  </div>

  <div class=\"section\">
    <h3>文件导出</h3>
    <ul>
      <li>{leader_priority_path.name}</li>
      <li>{leader_gmv1k_path.name}</li>
      <li>{leader_ord1k_path.name}</li>
      <li>{creator_plus_path.name}</li>
      <li>{leader_cvr_path.name}</li>
      <li>{leader_gmv_click_path.name}</li>
      <li>{whatif_path.name}</li>
      <li>{spend_recovery_path.name}</li>
      <li>{video_analysis_path.name}</li>
      <li>{top_videos_gmv1k_path.name}</li>
      <li>{top_videos_cvr_path.name}</li>
    </ul>
    <div class=\"muted\">（以上文件均已保存到：{BASE_DIR}）</div>
  </div>
</body>
</html>
"""

with open(html_path, "w", encoding="utf-8") as f:
    f.write(html)

print(f"[SAVE] {leader_priority_path}")
print(f"[SAVE] {leader_gmv1k_path}")
print(f"[SAVE] {leader_ord1k_path}")
print(f"[SAVE] {creator_plus_path}")
print(f"[SAVE] {leader_cvr_path}")
print(f"[SAVE] {leader_gmv_click_path}")
print(f"[SAVE] {whatif_path}")
print(f"[SAVE] {video_analysis_path}")
print(f"[SAVE] {top_videos_gmv1k_path}")
print(f"[SAVE] {top_videos_cvr_path}")
print(f"[SAVE] {spend_recovery_path}")
print(f"[SAVE] {html_path}")
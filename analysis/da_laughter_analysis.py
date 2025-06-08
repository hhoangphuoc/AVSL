"""
AMI Meeting Corpus - Laughter Analysis
====================================
This script reproduces the quantitative results and figures reported in the
"Laughter Patterns in the AMI Meeting Corpus" analysis.  It:
    • merges word-level annotations into utterance-level dialogue-act segments
    • marks stand-alone vs. embedded laughter events
    • computes global counts / percentages, breakdowns by dialogue-act type,
      adjacency-pair role (source/target) and speaker role
    • generates the three bar-chart figures used in the write-up
    • creates utterance-level CSV files for further analysis

INPUT FILES (expected in the current working directory)
------------------------------------------------------
ami_dialogue_acts.csv
    Word-level transcription + annotation.  Expected columns::
        meeting_id           - unique meeting identifier
        speaker_id           - one-letter speaker code (A/B/C/D)
        dact_id              - ID that groups tokens into a DA segment
        dialogue_act_type    - ISO 24617-2 compliant DA category (e.g. Inform)
        dialogue_act_gloss   - brief description of DA function
        word                - surface word form ("<laugh>" for laughter)
        event_type           - "laughter" for laugh tokens else "speech"
        start_time, end_time - token-level time stamps in seconds (float)

ami_adjacency_pairs.csv
    Adjacency pair annotations.  Expected columns:
        meeting_id, pair_id, source_dact_id, target_dact_id,
        pair_type              - coarse AP label (Support/Positive, Objection/Negative, Uncertain, Partial Agreement, Elaboration)
        pair_type_gloss             - human-readable function description

ami_da_ap_laughter.csv
    (Optional) additional mapping helpers - *not strictly required* here.

The column names above match the official AMI metadata releases.  If your
field names differ, tweak the constants in SECTION 0 below.

USAGE
-----
python ami_laughter_analysis.py  # saves tables + figures to ./out/

Python ≥3.9, pandas, seaborn, matplotlib and scipy must be installed.
"""
from __future__ import annotations

# SECTION 0 - CONFIGURATION ---------------------------------------------------
import os

from dataclasses import dataclass

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

# File paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'meta')
DA_CSV   = os.path.join(DATA_DIR, 'ami_dialogue_acts.csv')
AP_CSV   = os.path.join(DATA_DIR, 'ami_adjacency_pairs.csv')
DA_AP_CSV = os.path.join(DATA_DIR, 'ami_da_ap_laughter.csv')
OUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
os.makedirs(OUT_DIR, exist_ok=True)

# Column name mapping - change here if your CSV uses other headers -------------
@dataclass(frozen=True)
class Cols:
    # dialogue‑acts CSV
    MEETING   : str = 'meeting_id'
    SPEAKER   : str = 'speaker_id'
    # TOKEN_ID  : str = 'token_id'
    DA_ID     : str = 'dact_id'
    DA_TYPE   : str = 'dialogue_act_type'
    DA_GLOSS  : str = 'dialogue_act_gloss'
    TOKEN     : str = 'word'
    EVENT_T   : str = 'event_type'
    TSTART    : str = 'start_time'
    TEND      : str = 'end_time'
    # adjacency‑pairs CSV
    AP_ID     : str = 'pair_id'
    AP_TYPE   : str = 'pair_type'
    AP_GLOSS  : str = 'pair_type_gloss'
    SRC_DA    : str = 'source_dact_id'
    TGT_DA    : str = 'target_dact_id'
    SRC_SPEAKER : str = 'source_speaker_id'
    TGT_SPEAKER : str = 'target_speaker_id'

    
    

C = Cols()  # shorthand object

# SECTION 1 - LOAD DATA --------------------------------------------------------------------------------------
print('Reading CSV files …')
da_df = pd.read_csv(DA_CSV)
ap_df = pd.read_csv(AP_CSV)

# Load combined DA-AP file if it exists
da_ap_df = None
if os.path.exists(DA_AP_CSV):
    da_ap_df = pd.read_csv(DA_AP_CSV)
    print(f"\tDA-AP combined tokens: {len(da_ap_df):,}")

print(f"\tDialogue-act tokens: {len(da_df):,}")
print(f"\tAdjacency pairs   : {len(ap_df):,}\n")

assert {C.DA_ID, C.DA_GLOSS, C.TOKEN}.issubset(da_df.columns), (
    f"Your ami_dialogue_acts.csv is missing mandatory columns."
    f"Expected: '{C.DA_ID}', '{C.DA_GLOSS}', '{C.TOKEN}'")

# Ensure token column is string to prevent join errors with non-string data
da_df[C.TOKEN] = da_df[C.TOKEN].astype(str)

# SECTION 2 - MERGE TOKENS ➜ UTTERANCE-LEVEL SEGMENTS --------------------------------------------------------------------------------------
print('Aggregating tokens into utterance-level dialogue-acts …')

def _agg_utterance(group: pd.DataFrame) -> pd.Series:
    """Helper: combine token rows belonging to one DA segment."""
    tokens = group[C.TOKEN].tolist()
    text   = ' '.join(tokens)
    laughter_mask = group[C.EVENT_T] == 'laughter'
    return pd.Series({
        'text'               : text,
        'num_tokens'         : len(tokens),
        'contains_laughter'  : laughter_mask.any(),
        'num_laughter_tokens': laughter_mask.sum(),
        'standalone_laughter': laughter_mask.all(),
        'embedded_laughter'  : laughter_mask.any() and not laughter_mask.all(),
        'start_time'         : group[C.TSTART].min(),
        'end_time'           : group[C.TEND].max(),
    })

utterances = (
    da_df
    .sort_values([C.MEETING, C.DA_ID])
    .groupby([C.MEETING, C.DA_ID, C.DA_TYPE, C.DA_GLOSS, C.SPEAKER])
    .apply(_agg_utterance)
    .reset_index()
)
print(f"\tUtterances created: {len(utterances):,}\n")

# Save utterance-level CSV
UTTERANCES_CSV = os.path.join(OUT_DIR, 'utterances_dialogue_acts.csv')
utterances.to_csv(UTTERANCES_CSV, index=False)
print(f"\tSaved utterance-level dialogue acts to {UTTERANCES_CSV}")

# Basic global numbers --------------------------------------------------------
TOTAL_UTTS          = len(utterances)
LAUGHTER_UTTS       = utterances['contains_laughter'].sum()
STANDALONE_LAUGHS   = utterances['standalone_laughter'].sum()
EMBEDDED_LAUGHS     = utterances['embedded_laughter'].sum()
TOTAL_LAUGHTER_TOKENS = utterances['num_laughter_tokens'].sum()
LAUGHTER_PCT        = 100 * LAUGHTER_UTTS / TOTAL_UTTS if TOTAL_UTTS > 0 else 0
STANDALONE_PCT      = 100 * STANDALONE_LAUGHS / LAUGHTER_UTTS if LAUGHTER_UTTS > 0 else 0
EMBEDDED_PCT        = 100 * EMBEDDED_LAUGHS / LAUGHTER_UTTS if LAUGHTER_UTTS > 0 else 0

print(f"GLOBAL STATS:\n  total utterances    = {TOTAL_UTTS:,}"
      f"\n  total laughter tokens = {TOTAL_LAUGHTER_TOKENS:,}"
      f"\n  utterances w/ laugh = {LAUGHTER_UTTS:,}  ({LAUGHTER_PCT:.2f} %)"
      f"\n    • stand-alone     = {STANDALONE_LAUGHS:,} ({STANDALONE_PCT:.1f} % of laughs)"
      f"\n    • embedded        = {EMBEDDED_LAUGHS:,} ({EMBEDDED_PCT:.1f} % of laughs)\n")

# SECTION 3 - BY DIALOGUE-ACT TYPE --------------------------------------------------------------------------------------
print('Computing laughter distribution by dialogue-act type …')
type_stats = (
    utterances
    .groupby(C.DA_GLOSS)
    .agg(total=('contains_laughter', 'size'),
         laughter=('contains_laughter', 'sum'))
)

type_stats['laughter_pct'] = 100 * type_stats['laughter'] / type_stats['total']
type_stats_sorted = type_stats.sort_values('laughter_pct', ascending=False)

# ➜ save CSV breakdown
TYPE_STATS_CSV = os.path.join(OUT_DIR, 'da_type_laughter_stats.csv')
type_stats_sorted.to_csv(TYPE_STATS_CSV)
print(f"\tSaved breakdown to {TYPE_STATS_CSV}")

# Print detailed statistics for specific dialogue acts mentioned in the report
print("\nDETAILED DIALOGUE ACT STATISTICS:")
da_types_of_interest = [
    'Inform', 'Elicit-Inform', 'Backchannel', 'Stall', 
    'Comment-About-Understanding', 'Elicit-Assessment', 'Assess'
]

for da_type in da_types_of_interest:
    if da_type in type_stats.index:
        stats_row = type_stats.loc[da_type]
        print(f"  {da_type}: {stats_row['laughter_pct']:.1f}% ({stats_row['laughter']}/{stats_row['total']} utterances)")

# ➜ plot Figure 1
plt.figure(figsize=(9, max(4, 0.28*len(type_stats_sorted))))
sns.barplot(data=type_stats_sorted.reset_index(),
            x='laughter_pct', y=C.DA_GLOSS, palette='viridis')
plt.xlabel('% of utterances containing laughter')
plt.ylabel('Dialogue-act type')
plt.title('Figure 1 - Laughter by Dialogue-Act Type')
plt.tight_layout()
FIG1 = os.path.join(OUT_DIR, 'figure_1_da_laughter.png')
plt.savefig(FIG1, dpi=300)
plt.close()
print(f"\tSaved {FIG1}\n")

# SECTION 4 - ADJACENCY PAIR ANALYSIS --------------------------------------------------------------------------------------
print('Merging utterances with adjacency pairs …')

# rename helper df for merge clarity
ot_base = utterances[[C.MEETING, C.DA_ID, 'contains_laughter', 'text']]

ap_full = (
    ap_df
    .merge(ot_base, left_on=[C.MEETING, C.SRC_DA], right_on=[C.MEETING, C.DA_ID], how='left')
    .rename(columns={'contains_laughter': 'src_laughter', 'text': 'src_text'})
    .drop(columns=[C.DA_ID])
    .merge(ot_base, left_on=[C.MEETING, C.TGT_DA], right_on=[C.MEETING, C.DA_ID], how='left')
    .rename(columns={'contains_laughter': 'tgt_laughter', 'text': 'tgt_text'})
    .drop(columns=[C.DA_ID])
)
ap_full['mutual_laughter'] = ap_full['src_laughter'] & ap_full['tgt_laughter']
print(f"\tPaired rows after merge: {len(ap_full):,}\n")

# Create utterance-level adjacency pairs CSV
ap_utterances = ap_full[[
    C.MEETING, C.AP_ID, C.AP_TYPE, C.AP_GLOSS,
    C.SRC_SPEAKER, C.SRC_DA, 'src_text', 'src_laughter',
    C.TGT_SPEAKER, C.TGT_DA, 'tgt_text', 'tgt_laughter',
    'mutual_laughter'
]].copy()

AP_UTTERANCES_CSV = os.path.join(OUT_DIR, 'utterances_adjacency_pairs.csv')
ap_utterances.to_csv(AP_UTTERANCES_CSV, index=False)
print(f"\tSaved utterance-level adjacency pairs to {AP_UTTERANCES_CSV}")

ap_stats = (
    ap_full
    .groupby(C.AP_GLOSS)
    .agg(
        total_pairs=('pair_id', 'size'),
        src_laugh=('src_laughter', 'sum'),
        tgt_laugh=('tgt_laughter', 'sum'),
        mutual_laugh=('mutual_laughter', 'sum'))
)

ap_stats['src_laugh_pct'] = 100 * ap_stats['src_laugh'] / ap_stats['total_pairs']
ap_stats['tgt_laugh_pct'] = 100 * ap_stats['tgt_laugh'] / ap_stats['total_pairs']
ap_stats['mutual_laugh_pct'] = 100 * ap_stats['mutual_laugh'] / ap_stats['total_pairs']
AP_STATS_CSV = os.path.join(OUT_DIR, 'ap_laughter_stats.csv')
ap_stats.to_csv(AP_STATS_CSV)
print(f"\tSaved AP stats to {AP_STATS_CSV}")

# Print detailed adjacency pair statistics
print("\nDETAILED ADJACENCY PAIR STATISTICS:")
for ap_type in ap_stats.index:
    stats_row = ap_stats.loc[ap_type]
    print(f"  {ap_type}:")
    print(f"    Source: {stats_row['src_laugh_pct']:.1f}% ({stats_row['src_laugh']}/{stats_row['total_pairs']})")
    print(f"    Target: {stats_row['tgt_laugh_pct']:.1f}% ({stats_row['tgt_laugh']}/{stats_row['total_pairs']})")
    print(f"    Mutual: {stats_row['mutual_laugh_pct']:.1f}% ({stats_row['mutual_laugh']}/{stats_row['total_pairs']})")

# ➜ plot Figure 2 (side-by-side bar chart)
fig2_data = (ap_stats
             .reset_index()
             .melt(id_vars=C.AP_GLOSS,
                   value_vars=['src_laugh_pct', 'tgt_laugh_pct'],
                   var_name='position', value_name='laughter_pct'))

plt.figure(figsize=(10,6))
sns.barplot(data=fig2_data, x=C.AP_GLOSS, y='laughter_pct', hue='position', palette='Set2')
plt.xlabel('Adjacency-pair type')
plt.ylabel('% of utterances with laughter')
plt.title('Figure 2 - Laughter in AP Source vs. Target Utterances')
plt.legend(title='Utterance in pair', labels=['Source', 'Target'])
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
FIG2 = os.path.join(OUT_DIR, 'figure_2_ap_laughter.png')
plt.savefig(FIG2, dpi=300)
plt.close()
print(f"\tSaved {FIG2}\n")

# SECTION 5 - SPEAKER ROLE ANALYSIS ------------------------------------------
print('Computing laughter frequency by speaker role …')
role_stats = (
    utterances
    .groupby(C.SPEAKER)
    .agg(total=('contains_laughter', 'size'),
         laughter=('contains_laughter', 'sum'))
)
role_stats['laughter_pct'] = 100 * role_stats['laughter'] / role_stats['total']
ROLE_STATS_CSV = os.path.join(OUT_DIR, 'speaker_laughter_stats.csv')
role_stats.to_csv(ROLE_STATS_CSV)
print(f"\tSaved speaker role stats to {ROLE_STATS_CSV}")

print("\nSPEAKER ROLE STATISTICS:")
for speaker in sorted(role_stats.index):
    stats_row = role_stats.loc[speaker]
    print(f"  Speaker {speaker}: {stats_row['laughter_pct']:.1f}% ({stats_row['laughter']}/{stats_row['total']} utterances)")

# ➜ plot
plt.figure(figsize=(6,4))
a = (role_stats.reset_index()
     .sort_values('speaker_id'))
sns.barplot(data=a, x=C.SPEAKER, y='laughter_pct', palette='coolwarm')
for i, row in a.iterrows():
    plt.text(i, row['laughter_pct'] + 0.3,
             f"{row['laughter_pct']:.1f}%", ha='center', va='bottom', fontsize=9)
plt.xlabel('Speaker label')
plt.ylabel('% of utterances with laughter')
plt.title('Figure 3 - Laughter Frequency by Speaker Role')
plt.tight_layout()
FIG3 = os.path.join(OUT_DIR, 'figure_3_role_laughter.png')
plt.savefig(FIG3, dpi=300)
plt.close()
print(f"\tSaved {FIG3}\n")

# SECTION 6 - STATISTICAL TEST (OPTIONAL) ------------------------------------
print('Running χ² test of independence (DA-type x laughter) …')
cont = pd.concat([
    type_stats['laughter'],
    type_stats['total'] - type_stats['laughter']],
    axis=1,
    keys=['laughter', 'no_laughter']
)
chi2, pval, dof, _ = stats.chi2_contingency(cont)
print(f"χ² = {chi2:.1f} (df = {dof}), p = {pval:.2e}\n")    

# SECTION 7 - GENERATE COMPREHENSIVE REPORT TEXT --------------------------------------------------------------------------------------
print('Generating comprehensive analysis report...')

report_text = f"""
AMI Meeting Corpus - Laughter Analysis Report
============================================

GLOBAL STATISTICS
-----------------
Total utterances: {TOTAL_UTTS:,}
Total laughter tokens: {TOTAL_LAUGHTER_TOKENS:,}
Utterances with laughter: {LAUGHTER_UTTS:,} ({LAUGHTER_PCT:.1f}%)
  • Stand-alone laughter: {STANDALONE_LAUGHS:,} ({STANDALONE_PCT:.1f}% of laughter utterances)
  • Embedded laughter: {EMBEDDED_LAUGHS:,} ({EMBEDDED_PCT:.1f}% of laughter utterances)

DIALOGUE ACT ANALYSIS
--------------------
"""

# Add specific dialogue act statistics
for da_type in da_types_of_interest:
    if da_type in type_stats.index:
        stats_row = type_stats.loc[da_type]
        report_text += f"{da_type}: {stats_row['laughter_pct']:.1f}% ({stats_row['laughter']}/{stats_row['total']} utterances)\n"

report_text += f"""
ADJACENCY PAIR ANALYSIS
----------------------
"""

# Add adjacency pair statistics
for ap_type in ap_stats.index:
    stats_row = ap_stats.loc[ap_type]
    report_text += f"{ap_type}:\n"
    report_text += f"  Source utterances: {stats_row['src_laugh_pct']:.1f}% laughter\n"
    report_text += f"  Target utterances: {stats_row['tgt_laugh_pct']:.1f}% laughter\n"
    report_text += f"  Mutual laughter: {stats_row['mutual_laugh_pct']:.1f}%\n\n"

report_text += f"""
SPEAKER ROLE ANALYSIS
--------------------
"""

for speaker in sorted(role_stats.index):
    stats_row = role_stats.loc[speaker]
    report_text += f"Speaker {speaker}: {stats_row['laughter_pct']:.1f}% ({stats_row['laughter']}/{stats_row['total']} utterances)\n"

report_text += f"""
STATISTICAL SIGNIFICANCE
-----------------------
Chi-square test of independence (DA-type × laughter): χ² = {chi2:.1f} (df = {dof}), p = {pval:.2e}

FILES GENERATED
--------------
- {UTTERANCES_CSV}
- {AP_UTTERANCES_CSV}
- {TYPE_STATS_CSV}
- {AP_STATS_CSV}
- {ROLE_STATS_CSV}
- {FIG1}
- {FIG2}
- {FIG3}
"""

# Save report to file
REPORT_TXT = os.path.join(OUT_DIR, 'laughter_analysis_report.txt')
with open(REPORT_TXT, 'w') as f:
    f.write(report_text)

print(f"Saved comprehensive report to {REPORT_TXT}")
print(report_text)

print('Analysis complete.  All outputs saved to ./out/')

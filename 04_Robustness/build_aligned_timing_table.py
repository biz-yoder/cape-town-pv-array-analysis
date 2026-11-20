import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import TIMING_SENSITIVITY_DIR

#!/usr/bin/env python3
"""Common-calendar PV timing sensitivity table from existing event-study CSV."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

DEFAULT = TIMING_SENSITIVITY_DIR / "event_study_all_timing_shifts.csv"
SHIFTS = [-6, -3, 0, 3, 6]
GROUPS = {0:'<1 kW',1:'1--2 kW',2:'2--3 kW',3:'3--5 kW',4:'>5 kW'}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,default=DEFAULT)
    p.add_argument('--output-dir',type=Path,default=None)
    args=p.parse_args()
    dest=args.output_dir or args.input.parent
    dest.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(args.input)
    required={'dose_group','shift_months','event_time','att'}
    missing=required-set(df.columns)
    if missing: raise ValueError(f'Missing columns: {sorted(missing)}')
    df['original_event_time']=df['event_time']+df['shift_months']
    if df.duplicated(['dose_group','shift_months','original_event_time']).any():
        raise ValueError('Duplicate event-time estimates found; check input.')
    records=[]
    for dose in GROUPS:
        for shift in SHIFTS:
            part=df[(df.dose_group==dose)&(df.shift_months==shift)]
            baseline=part.loc[part.original_event_time==-2,'att']
            window=part[part.original_event_time.between(6,12)].sort_values('original_event_time')
            observed=set(window.original_event_time.astype(int))
            expected=set(range(6,13))
            if len(baseline)!=1 or observed!=expected or window['att'].isna().any():
                raise ValueError(f'Incomplete window: dose={dose}, shift={shift}; reference count={len(baseline)}; missing months={sorted(expected-observed)}. Do NOT average incomplete windows.')
            ref=float(baseline.iloc[0])
            mean_delta=float((window.att-ref).mean())
            records.append({'dose_group':dose,'capacity':GROUPS[dose], 'shift_months':shift,'reference_log_att_at_original_minus2':ref,'mean_normalized_log_change_original_plus6_to_plus12':mean_delta,'mean_normalized_pct':100*np.expm1(mean_delta),'n_months':len(window),'original_months':'6..12'})
    out=pd.DataFrame(records)
    out.to_csv(dest/'timing_shift_common_calendar_normalized_summary.csv',index=False)
    wide=out.pivot(index='dose_group',columns='shift_months',values='mean_normalized_pct').reindex(index=list(GROUPS),columns=SHIFTS)
    print('\nCommon original event months +6..+12, normalized to original -2 (percent):\n')
    print(wide.rename(index=GROUPS).round(2).to_string())
    latex=[r'\begin{table}[htbp]',r'\centering',r'\caption{Installation-timing sensitivity by rooftop PV capacity: common calendar-relative months}',r'\label{tab:timing_common_calendar}',r'\begin{tabular}{lrrrrr}',r'\hline\hline',r'PV capacity & $-6$ mo. & $-3$ mo. & Baseline & $+3$ mo. & $+6$ mo. \\',r'\hline']
    for dose in GROUPS:
        label={0:r'$<1$ kW',1:r'1--2 kW',2:r'2--3 kW',3:r'3--5 kW',4:r'$>5$ kW'}[dose]
        latex.append(label+' & '+' & '.join(f'{wide.loc[dose,s]:.2f}' for s in SHIFTS)+r' \\')
    latex += [r'\hline\hline',r'\end{tabular}',r'\vspace{0.2cm}',r'\begin{minipage}{0.95\textwidth}',r'\footnotesize',r'\textit{Notes:} Each entry equals $100\{\exp[\frac{1}{7}\sum_{k=6}^{12}(\widehat{ATT}_{s,k-s}-\widehat{ATT}_{s,-2-s})]-1\}$, where $s$ is the shift in months relative to the baseline assumed installation date and event times are indexed to each scenario. Thus every column uses the same seven calendar-relative months (original event months +6 to +12) and the same calendar-relative normalization month (original event month $-2$). Values are descriptive normalized changes in estimated log effects, not re-estimated causal ATTs or confidence intervals. Cohort composition and scenario-specific estimation baselines may still differ.',r'\end{minipage}',r'\end{table}']
    (dest/'timing_shift_common_calendar_table.tex').write_text('\n'.join(latex)+'\n')
    print('\nSaved:',dest/'timing_shift_common_calendar_normalized_summary.csv')
    print('Saved:',dest/'timing_shift_common_calendar_table.tex')

if __name__=='__main__': main()

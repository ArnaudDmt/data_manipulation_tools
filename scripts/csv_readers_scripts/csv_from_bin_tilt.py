import mc_log_ui
import pandas as pd
log = mc_log_ui.read_log('mc-control-Passthrough-2025-01-09-16-50-25.bin')
df = pd.DataFrame.from_dict(log)
df = df.reset_index(drop=True)
regex1 = r"Observers_MainObserverPipeline_Tilt_debug_initX_"
regex2 = r"\bt\b|Observers_MainObserverPipeline_Tilt_constants_|Observers_MainObserverPipeline_Tilt_debug_y_"
df1 = df.filter(regex=regex1)
df2 = df.filter(regex=regex2)

for col in df1.columns:
    temp = col.split('_')
    newcol = temp[-2]+temp[-1]
    df1 = df1.rename(columns={col: newcol})

for col in df2.columns:
    temp = col.split('_')
    if temp[-1].isdigit():
        newcol = temp[-2]+temp[-1]    
        df2 = df2.rename(columns={col: newcol})
    else:
        newcol = temp[-1]
        df2 = df2.rename(columns={col: newcol})
cols = list(df2.columns)
cols.insert(0, cols.pop(cols.index('t')))
df2 = df2.loc[:, cols]
firstLine = list(df1.iloc[0])
#df2.t = df2.t.round(decimals=3)
df2.to_csv('data.csv', index=False, header=False, sep=',')#, float_format="%.5f")
print(df2.columns)
with open('data.csv', 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(', '.join(map(str, firstLine)) + '\n' + content)

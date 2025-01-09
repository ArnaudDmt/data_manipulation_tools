import mc_log_ui
import pandas as pd
import plotly.graph_objects as go
import numpy as np


log_mcrtc = mc_log_ui.read_log('mc-control-Passthrough-2025-01-09-16-50-25.bin')
df_mcrtc = pd.DataFrame.from_dict(log_mcrtc)
df_mcrtc = df_mcrtc.reset_index(drop=True)
regex = r"\bt\b|Observers_MainObserverPipeline_Tilt_estimatedState_|Observers_MainObserverPipeline_Tilt_debug_constants_"
df_mcrtc = df_mcrtc.filter(regex=regex)

for col in df_mcrtc.columns:
    if col != 't':
        temp = col.split('_')
        newcol = temp[-2] + "_" + temp[-1]
        df_mcrtc = df_mcrtc.rename(columns={col: newcol})

df_csvReader = pd.read_csv('estimationResults.csv')  
print(df_mcrtc.columns)
print(df_csvReader.columns)

x2prime = df_csvReader[['x2prime_x', 'x2prime_y', 'x2prime_z']].to_numpy()
x2prime_normalized = x2prime / np.linalg.norm(x2prime, axis=1)[:, np.newaxis]

fig = go.Figure()
fig.add_trace(go.Scatter(x=df_mcrtc['t'], y=df_mcrtc["x1_x"], mode='lines', name="mc_rtc_x1_x"))
fig.add_trace(go.Scatter(x=df_mcrtc['t'], y=df_mcrtc["x1_y"], mode='lines', name="mc_rtc_x1_y"))
fig.add_trace(go.Scatter(x=df_mcrtc['t'], y=df_mcrtc["x1_z"], mode='lines', name="mc_rtc_x1_z"))
fig.add_trace(go.Scatter(x=df_mcrtc['t'], y=df_mcrtc["x2_x"], mode='lines', name="mc_rtc_x2_x"))
fig.add_trace(go.Scatter(x=df_mcrtc['t'], y=df_mcrtc["x2_y"], mode='lines', name="mc_rtc_x2_y"))
fig.add_trace(go.Scatter(x=df_mcrtc['t'], y=df_mcrtc["x2_z"], mode='lines', name="mc_rtc_x2_z"))
fig.add_trace(go.Scatter(x=df_mcrtc['t'], y=df_mcrtc["x2prime_x"], mode='lines', name="mc_rtc_x2prime_x"))
fig.add_trace(go.Scatter(x=df_mcrtc['t'], y=df_mcrtc["x2prime_y"], mode='lines', name="mc_rtc_x2prime_y"))
fig.add_trace(go.Scatter(x=df_mcrtc['t'], y=df_mcrtc["x2prime_z"], mode='lines', name="mc_rtc_x2prime_z"))
fig.add_trace(go.Scatter(x=df_mcrtc['t'], y=df_mcrtc["x1_x"], mode='lines', name="mc_rtc_alpha"))
fig.add_trace(go.Scatter(x=df_mcrtc['t'], y=df_mcrtc["x1_y"], mode='lines', name="mc_rtc_beta"))
fig.add_trace(go.Scatter(x=df_mcrtc['t'], y=df_mcrtc["x1_z"], mode='lines', name="mc_rtc_gamma"))

fig.add_trace(go.Scatter(x=df_csvReader['t'], y=df_csvReader["x1_x"], mode='lines', name="csv_x1_x"))
fig.add_trace(go.Scatter(x=df_csvReader['t'], y=df_csvReader["x1_y"], mode='lines', name="csv_x1_y"))
fig.add_trace(go.Scatter(x=df_csvReader['t'], y=df_csvReader["x1_z"], mode='lines', name="csv_x1_z"))
fig.add_trace(go.Scatter(x=df_csvReader['t'], y=df_csvReader["x2_x"], mode='lines', name="csv_x2_x"))
fig.add_trace(go.Scatter(x=df_csvReader['t'], y=df_csvReader["x2_y"], mode='lines', name="csv_x2_y"))
fig.add_trace(go.Scatter(x=df_csvReader['t'], y=df_csvReader["x2_z"], mode='lines', name="csv_x2_z"))
fig.add_trace(go.Scatter(x=df_csvReader['t'], y=x2prime_normalized[:,0], mode='lines', name="csv_x2prime_x"))
fig.add_trace(go.Scatter(x=df_csvReader['t'], y=x2prime_normalized[:,1], mode='lines', name="csv_x2prime_y"))
fig.add_trace(go.Scatter(x=df_csvReader['t'], y=x2prime_normalized[:,2], mode='lines', name="csv_x2prime_z"))
fig.add_trace(go.Scatter(x=df_csvReader['t'], y=df_csvReader["x1_x"], mode='lines', name="csv_alpha"))
fig.add_trace(go.Scatter(x=df_csvReader['t'], y=df_csvReader["x1_y"], mode='lines', name="csv_beta"))
fig.add_trace(go.Scatter(x=df_csvReader['t'], y=df_csvReader["x1_z"], mode='lines', name="csv_gamma"))

fig.show()
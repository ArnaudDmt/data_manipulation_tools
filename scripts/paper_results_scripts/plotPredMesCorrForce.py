from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd

import plotly.io as pio   
pio.kaleido.scope.mathjax = None

observer_data = pd.read_csv(f'/tmp/log.csv',  delimiter=';')

fig = go.Figure()



fig.add_trace(go.Scatter(
            x=observer_data['t'], y=observer_data['Observers_MainObserverPipeline_MCKineticsObserver_MEKF_measurements_contacts_force_LeftFootForceSensor_predicted_x'],
            mode='lines',
            name='Predicted', showlegend=True))

fig.add_trace(go.Scatter(
            x=observer_data['t'], y=observer_data['Observers_MainObserverPipeline_MCKineticsObserver_MEKF_measurements_contacts_force_LeftFootForceSensor_measured_x'],
            mode='lines',
            name='Measured', showlegend=True))

fig.add_trace(go.Scatter(
            x=observer_data['t'], y=observer_data['Observers_MainObserverPipeline_MCKineticsObserver_MEKF_measurements_contacts_force_LeftFootForceSensor_corrected_x'],
            mode='lines',
            name='Corrected', showlegend=True))

fig.update_layout(
    xaxis_title="Time (s)",
    yaxis_title="Force X (N)",
    template="plotly_white",
    legend=dict(
        yanchor="bottom",
        y=1.03,
        xanchor="left",
        x=-0.1,
        orientation='h',
        bgcolor = 'rgba(0,0,0,0)',
        traceorder='reversed',
        font = dict(family = 'Times New Roman', size=22, color="black"),
        ),
    margin=dict(l=0,r=0,b=0,t=0),
    font = dict(family = 'Times New Roman', size=20, color="black")
)

fig.show()

fig.write_image(f'/tmp/test.pdf')
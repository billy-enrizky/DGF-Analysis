import pandas as pd
import plotly.express as px

# Load your DataFrame (replace with your actual data loading method)
df = pd.read_pickle('df.pkl')  # or use pd.read_csv('DGF_Cleaned.csv')

# Ensure 'txdate' is datetime
df['txdate'] = pd.to_datetime(df['txdate'])

# Ensure 'dgf' is numeric (convert non-numeric to NaN)
df['dgf'] = pd.to_numeric(df['dgf'], errors='coerce')

# Extract year
df['year'] = df['txdate'].dt.year

# Filter for years 2000-2022
df_years = df[(df['year'] >= 2000) & (df['year'] <= 2022)]

# Calculate DGF proportion per year
dgf_prop = df_years.groupby('year')['dgf'].mean().reset_index()
dgf_prop['dgf_percent'] = dgf_prop['dgf'] * 100

# Create bar chart
fig = px.bar(
    dgf_prop,
    x='year',
    y='dgf_percent',
    labels={'year': 'Year', 'dgf_percent': 'DGF Proportion (%)'},
    title='DGF Proportion by Year (2000-2022)',
    color='dgf_percent',
    color_continuous_scale='Blues',
    text='dgf_percent'
)
fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
fig.update_layout(
    yaxis=dict(range=[0, dgf_prop['dgf_percent'].max() + 10]),
    plot_bgcolor='white',
    xaxis=dict(dtick=1),
    font=dict(size=14)
)
fig.show()

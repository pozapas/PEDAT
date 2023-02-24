import streamlit as st
import pandas as pd
import plotly.express as px
import pydeck as pdk
from datetime import datetime as dt
import time
import plotly.graph_objs as go
from plotly.subplots import make_subplots



# Load the data
df = pd.read_csv('pedat.csv')

# Convert TIME2 to a datetime format
df['TIME2'] = pd.to_datetime(df['TIME2'])

# Define the list of available signals
signals = df['SIGNAL'].unique().tolist()

# Define the default signal
default_signal = signals[0]

# Define the title
title = 'Pedestrian Activity Data Visualization Dashboard'

# Define the subtitle
subtitle = 'Singleton Transportation Lab'

# Define the x and y axis labels
x_axis_label = 'TIME2'
y_axis_label = 'PED'

def format_metric(value):
    # Check if the value is greater than or equal to 1 billion
    if value >= 1e9:
        return f'{round(value/1e9,1)} B'
    
    # Check if the value is greater than or equal to 1 million
    elif value >= 1e6:
        return f'{round(value/1e6,1)} M'
    
    # Check if the value is greater than or equal to 1 thousand
    elif value >= 1e3:
        return f'{round(value/1e3,1)} K'
    
    # Otherwise, return the value as is
    else:
        return str(value)


def make_chart(df, signals, start_date, end_date, template='plotly'):
    df = df[(df['TIME2'] >= start_date) & (df['TIME2'] <= end_date)]
    fig = px.line(df[df['SIGNAL'].isin(signals)], x=x_axis_label, y=y_axis_label, color='SIGNAL', template=template)
    fig.update_xaxes(title_text='Date')
    fig.update_yaxes(title_text='Pedestrian Estimated')
    fig.update_traces(line=dict(width=3))

    # Set the time slider at the bottom
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )

    return fig

def make_pie_chart(df, signals, start_date, end_date):
    # Filter the dataframe by the selected signals and date range
    df_filtered = df[(df['TIME2'] >= start_date) & (df['TIME2'] < end_date) & (df['SIGNAL'].isin(signals))]

    # Aggregate the data by signal and sum the pedestrian counts
    df_agg = df_filtered.groupby('SIGNAL').agg({'PED': 'sum'}).reset_index()

    # Create the pie chart
    fig = go.Figure(data=[go.Pie(labels=df_agg['SIGNAL'], values=df_agg['PED'])])

    # Update the layout
    fig.update_layout(title='Pedestrian Activity by Signal', showlegend=True)

    return fig


def make_pie_and_bar_chart(df, signals, start_date, end_date):
    # Filter the dataframe by the selected signals and date range
    df_filtered = df[(df['TIME2'] >= start_date) & (df['TIME2'] < end_date) & (df['SIGNAL'].isin(signals))]

    # Aggregate the data by signal and sum the pedestrian counts
    df_agg = df_filtered.groupby('SIGNAL').agg({'PED': 'sum'}).reset_index()
    df_agg1 = df_filtered.groupby('CITY').agg({'PED': 'sum'}).reset_index()

    # Create the pie chart
    fig_pie = go.Figure(data=[go.Pie(labels=df_agg['SIGNAL'], values=df_agg['PED'])])
    fig_pie.update_layout(title='Pedestrian Activity by Signal', showlegend=False)

    # Create the bar chart
    fig_bar = go.Figure(data=[go.Bar(x=df_agg1['CITY'], y=df_agg['PED'] , showlegend=False)])
    fig_bar.update_layout(title='Pedestrian Activity by City', showlegend=False)

    # Combine the pie and bar charts
    fig_combined = make_subplots(rows=1, cols=2, specs=[[{'type': 'domain'}, {'type': 'bar'}]])
    fig_combined.add_trace(fig_pie.data[0], row=1, col=1)
    fig_combined.add_trace(fig_bar.data[0], row=1, col=2)

    return fig_combined




def make_map(df, start_date, end_date, signals):
    # Filter by date and selected signals
    mask = (df['TIME2'] >= start_date) & (df['TIME2'] < end_date) & (df['SIGNAL'].isin(signals))
    df_filtered = df.loc[mask]

    # Aggregate by location
    df_agg = df_filtered.groupby(['LAT', 'LNG']).agg({'SIGNAL': 'sum', 'PED': 'sum'}).reset_index()

    # Create the HeatmapLayer
    heatmap_layer = pdk.Layer(
        'HeatmapLayer',
        data=df_agg,
        get_position='[LNG, LAT]',
        auto_highlight=True,
        radius=50,
        opacity=0.8,
        get_weight='PED'
    )

    # Set the viewport location
    view_state = pdk.ViewState(
        longitude=df['LNG'].mean(),
        latitude=df['LAT'].mean(),
        zoom=11,
        min_zoom=5,
        max_zoom=15,
        pitch=40.5,
        bearing=-27.36
    )
    fig = pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=view_state,
        layers=[heatmap_layer],
        tooltip={'text': 'Pedestrian count: {PED}'}
    )
    return fig


# Define the Streamlit app
def main():
    # Set the app title
    st.set_page_config(page_title='Time Series Dashboard')

    # Add a title to the sidebar
    st.sidebar.title(title)

    # Add a subtitle to the sidebar
    st.sidebar.markdown(f'[Singleton Transportation Lab](https://engineering.usu.edu/cee/research/labs/patrick-singleton/index)')

    # Add a multiselect to choose the signals
    signals = ["All"] + df['SIGNAL'].unique().tolist()
    default_signals = [signals[1]]
    selected_signals = st.sidebar.multiselect('Select signals', signals, default=default_signals)

    # If "All" is selected, show all signals
    if "All" in selected_signals:
        selected_signals = df['SIGNAL'].unique().tolist()
    else:
        selected_signals = selected_signals or default_signals


    # Add a time range slider to select the date range
    start_date = df['TIME2'].min().date()
    end_date = df['TIME2'].max().date()
    start_date_selected, end_date_selected = st.sidebar.select_slider('Select a date range', options=pd.date_range(start=start_date, end=end_date, freq='D'), value=(start_date, end_date))

    # Calculate the total number of pedestrians in the selected date range and signals
    df_filtered = df[(df['TIME2'] >= start_date_selected) & (df['TIME2'] < end_date_selected)]
    if "All" not in selected_signals:
        df_filtered = df_filtered[df_filtered['SIGNAL'].isin(selected_signals)]
    total_pedestrians = df_filtered['PED'].sum()
    num_signals = len(df['SIGNAL'].unique())
    num_cities = len(df['CITY'].unique())

    # Format the metric values
    total_pedestrians_formatted = format_metric(total_pedestrians)
    num_signals_formatted = format_metric(num_signals)
    num_cities_formatted = format_metric(num_cities)

    st.subheader('Metrics')
    # Display the metric boxes
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Pedestrian Predicted", total_pedestrians_formatted)
    col2.metric("Number of Signals", num_signals_formatted)
    col3.metric("Number of Cities", num_cities_formatted)

    st.subheader('Time series')
    # Make the time series plot
    fig = make_chart(df, selected_signals, start_date_selected, end_date_selected, template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True )

    st.subheader('Pedestrian Activity in relation to Signal and City')
    # Add a pie chart to show pedestrian activity by signal
    st.plotly_chart(make_pie_and_bar_chart(df, selected_signals, start_date_selected, end_date_selected))

    st.subheader('Pedestrian Activity by Location')
    # Make the map
    fig = make_map(df, start_date_selected, end_date_selected , selected_signals)
    st.pydeck_chart(fig)

if __name__ == '__main__':
    main()

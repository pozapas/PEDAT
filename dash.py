import streamlit as st
import pandas as pd
import plotly.express as px
import pydeck as pdk
from datetime import datetime as dt
import time
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
import base64
from io import BytesIO
import os


# Load the data
df = pd.read_pickle("pediN" + '.pkl', compression='gzip')

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


def make_chart(df, signals, start_date, end_date, aggregation_method, location, template='plotly'):
    if aggregation_method == 'Hourly':
        groupby = ['SIGNAL', pd.Grouper(key='TIME2', freq='1H')]
    elif aggregation_method == 'Daily':
        groupby = ['SIGNAL', pd.Grouper(key='TIME2', freq='1D')]
    elif aggregation_method == 'Weekly':
        groupby = ['SIGNAL', pd.Grouper(key='TIME2', freq='1W')]
    elif aggregation_method == 'Monthly':
        groupby = ['SIGNAL', pd.Grouper(key='TIME2', freq='1M')]
    elif aggregation_method == 'Yearly':
        groupby = ['SIGNAL', pd.Grouper(key='TIME2', freq='1Y')]

    if location == 'Intersection':
        filter_val = 'all'
        col = 'PED'
    else:
        filter_val = location.split()[-1]
        col = 'P'

    # Filter the dataframe by the selected signals and date range
    df_filtered = df[(df['TIME2'] >= start_date) & (df['TIME2'] <= end_date) & (df['SIGNAL'].isin(signals))]

    # Aggregate the data
    if location == 'Intersection':
        df_agg = df_filtered.groupby(groupby).agg({'PED': 'sum'}).reset_index()
    else:
        df_agg = df_filtered[df_filtered['P'] == int(filter_val)].groupby(groupby).agg({'PED': 'sum'}).reset_index()

    # Create the line chart
    if aggregation_method == 'Hourly':
        x_axis_label = 'Time'
        fig = px.line(df_agg, x='TIME2', y='PED', color='SIGNAL', template=template)
    else:
        x_axis_label = 'Date'
        fig = px.line(df_agg, x='TIME2', y='PED', color='SIGNAL', template=template, line_group='SIGNAL')

    fig.update_xaxes(title_text=x_axis_label)
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
    # Filter the dataframe by the selected signals, date range, and day type
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


def make_map(df, start_date, end_date, signals, aggregation_method):
    # Filter by date and selected signals
    mask = (df['TIME2'] >= start_date) & (df['TIME2'] < end_date) & (df['SIGNAL'].isin(signals))
    df_filtered = df.loc[mask]
    agg_functions = {
    'Hourly': 'sum',
    'Daily': 'sum',
    'Weekly': 'sum',
    'Monthly': 'sum',
    'Yearly': 'sum'
    }

    aggregation_function = agg_functions[aggregation_method]
    # Aggregate by location
    df_agg = df_filtered.groupby(['LAT', 'LNG']).agg({'PED': aggregation_function}).reset_index()

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

import io

@st.cache_data
def save_csv(df, selected_signals, start_date_selected, end_date_selected, location_selected, aggregation_method_selected):
    # Filter the data based on the selected signals, location, and date range
    df_filtered2 = df[(df['TIME2'] >= start_date_selected) & (df['TIME2'] < end_date_selected)]
    if "All" not in selected_signals:
        df_filtered2 = df_filtered2[df_filtered2['SIGNAL'].isin(selected_signals)]
    if location_selected != 'Intersection':
        phase_num = int(location_selected.split()[1])
        df_filtered2 = df_filtered2[df_filtered2['P'] == phase_num]

    # Aggregate the data based on the selected aggregation method
    if aggregation_method_selected == 'Hourly':
        freq = 'H'
    elif aggregation_method_selected == 'Daily':
        freq = 'D'
    elif aggregation_method_selected == 'Weekly':
        freq = 'W-MON'
    elif aggregation_method_selected == 'Monthly':
        freq = 'MS'
    elif aggregation_method_selected == 'Yearly':
        freq = 'YS'

    agg_dict = {col: 'first' for col in df_filtered2.columns if col not in ['PED']}
    agg_dict['PED'] = 'sum'
    df_aggregated = df_filtered2.groupby([pd.Grouper(key='TIME2', freq=freq), 'SIGNAL', 'CITY', 'P']).agg(agg_dict)
    df_aggregated.reset_index(drop=True,inplace=True)

    #filename = f"pedestrian_counts_{location_selected}_{aggregation_method_selected}.csv"
    #st.sidebar.success(f"{filename} successfully saved!")
    return df_aggregated.to_csv(index=False)

# Define the Streamlit app
@st.cache_data
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

    # Add a slider for selecting the location
    locations = ['Intersection'] + ['Phase ' + str(int(i)) for i in sorted(df['P'].dropna().unique().tolist())]
    location_selected = st.sidebar.selectbox('Select approach', options=locations)

    # Add a slider for selecting the aggregation method
    aggregation_methods = ['Hourly', 'Daily', 'Weekly', 'Monthly', 'Yearly']
    aggregation_method_selected = st.sidebar.selectbox('Select aggregation method', options=aggregation_methods)

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
    fig = make_chart(df, selected_signals, start_date_selected, end_date_selected, aggregation_method_selected, location_selected, template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True )

    st.subheader('Pedestrian Activity in relation to Signal and City')
    # Add a pie chart to show pedestrian activity by signal
    st.plotly_chart(make_pie_and_bar_chart(df, selected_signals, start_date_selected, end_date_selected))

    st.subheader('Pedestrian Activity by Location')
    # Make the map
    fig = make_map(df, start_date_selected, end_date_selected , selected_signals , aggregation_method_selected)
    st.pydeck_chart(fig)

    csv = save_csv(df, selected_signals, start_date_selected, end_date_selected, location_selected, aggregation_method_selected)

    st.sidebar.download_button(
        label="Download data as CSV",
        data=csv,
        file_name=f"{selected_signals}_{location_selected}_{aggregation_method_selected}.csv",
        mime='text/csv',
    )

    hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
    st.markdown(hide_menu_style, unsafe_allow_html=True)
   
if __name__ == '__main__':
    main()

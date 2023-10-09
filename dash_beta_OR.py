# PEDAT Dashboard V.0.0.8
#pip install protobuf==3.19.6

import streamlit as st
import pandas as pd
import plotly.express as px
import pydeck as pdk
from datetime import datetime, date, timedelta
import time
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
import base64
from io import BytesIO
import os
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import json
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw
from shapely.geometry import Polygon, Point
from google.cloud import bigquery
from google.oauth2 import service_account
from folium.plugins import MarkerCluster
from fpdf import FPDF
from tempfile import NamedTemporaryFile
import threading
from PIL import Image
import plotly.io as pio
import io
from reportlab.lib.pagesizes import landscape, letter,A4
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
import tempfile
import imgkit
import copy
import kaleido
pio.kaleido.scope.default_scale = 2 
import random
from keplergl import KeplerGl
from streamlit_keplergl import keplergl_static
import matplotlib.colors
from folium.plugins import Search

# Create API client.
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = bigquery.Client(credentials=credentials)

# Define SQL query to retrieve the data from BigQuery
sql_query = """
SELECT *
FROM `decent-digit-387716.pedat_dataset.map_data`
"""
sql_query2 = """
SELECT *
FROM `decent-digit-387716.pedat_dataset.or_pedat`
WHERE ADDRESS IN UNNEST(@selected_signals)
"""
sql_query3 = """
SELECT *
FROM `decent-digit-387716.pedat_dataset.pedatdaily`
WHERE ADDRESS IN UNNEST(@selected_signals)
"""


# Define the title
title = 'Pedestrian Activity Data Visualization Dashboard'
text1 = 'This website provides data and visualizations of pedestrian activity (and changes in pedestrian activity) at various (signalized) intersections throughout Oregon. We are currently showing all locations with available signal data throughout Oregon.'
#text2 = 'Data are derived from pedestrian push-button presses at traffic signals, taken from the Oregon Department of Transportation\'s [Automated Traffic Signal Performance Measures System](https://udottraffic.Oregon.gov/atspm/) website. We hope that this information is useful for public agencies to track changes in walking activity at different locations.'

# Define the x and y axis labels
x_axis_label = 'TIME1'
y_axis_label = 'PED'

def create_color_map(unique_signals):
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_signals)))
    colors = [matplotlib.colors.rgb2hex(c) for c in colors]
    random.shuffle(colors)  # Shuffle colors for variety
    color_map = dict(zip(unique_signals, colors))
    return color_map

@st.cache_resource
def format_metric(value):
    # Check if the value is greater than or equal to 1 billion
    if value >= 1e9:
        return f'{round(value/1e9,0)} B'
    
    # Check if the value is greater than or equal to 1 million
    elif value >= 1e6:
        return f'{round(value/1e6,0)} M'
    
    # Check if the value is greater than or equal to 1 thousand
    elif value >= 1e3:
        return f'{round(value/1e3,0)} K'
    
    # Otherwise, return the value as is
    else:
        return str(value)

@st.cache_resource
def make_chart(df, signals, start_date, end_date, aggregation_method, location, Dash_selected, act_selected, color_map, template='plotly'):
    
    # Map act_selected to its corresponding column
    if act_selected == "All crosswalk users (walk, bicycle, scooter, skateboard, wheelchair, other)":
        col = 'TOTAL'
    elif act_selected == 'Pedestrians (walk, skateboard, wheelchair)':
        col = 'UPED'
    elif act_selected == "People walking":
        col = 'PED'
    else:
        raise ValueError(f"Invalid value for act_selected: {act_selected}")
    
    # Handle the location filter
    if location == 'All':
        filter_val = 'all'
    else:
        filter_val = location.split()[-1]
    
    if aggregation_method == 'Hours':
        groupby = ['SIGNAL','ADDRESS', pd.Grouper(key='TIME1', freq='1H')]
    elif aggregation_method == 'Days':
        groupby = ['SIGNAL','ADDRESS', pd.Grouper(key='TIME1', freq='1D')]
    elif aggregation_method == 'Weeks':
        groupby = ['SIGNAL','ADDRESS', pd.Grouper(key='TIME1', freq='1W')]
    elif aggregation_method == 'Months':
        groupby = ['SIGNAL','ADDRESS', pd.Grouper(key='TIME1', freq='1M')]
    elif aggregation_method == 'Years':
        groupby = ['SIGNAL','ADDRESS', pd.Grouper(key='TIME1', freq='1Y')]

    # Filter the dataframe by the selected signals and date range
    if Dash_selected == 'Recent data (last 1 year)':
        df_filtered = df[(df['TIME1'] >= pd.to_datetime(start_date).tz_localize('UTC')) & (df['TIME1'] <= pd.to_datetime(end_date).tz_localize('UTC')) & (df['ADDRESS'].isin(signals))]
    else:
        df_filtered = df[(df['TIME1'] >= start_date) & (df['TIME1'] <= end_date) & (df['ADDRESS'].isin(signals))]
    
# Aggregate the data
    if location == 'All':
        df_agg = df_filtered.groupby(groupby).agg({col: 'sum'}).reset_index()
        df_agg[col] = df_agg[col].round(0)
        df_agg = df_agg.rename(columns={'SIGNAL': 'Signal ID', 'TIME1':'Timestamp' , col:'Volume' })

    else:
        df_agg = df_filtered[df_filtered['P'] == int(filter_val)].groupby(groupby).agg({col: 'sum'}).reset_index()
        df_agg[col] = df_agg[col].round(0)
        df_agg = df_agg.rename(columns={'SIGNAL': 'Signal ID' , 'TIME1':'Timestamp' , col:'Volume' })

    # Create the line chart
    if aggregation_method == 'Hours':
        x_axis_label = '<b>Time<b>'
        fig = px.line(df_agg, x='Timestamp', y='Volume' ,  color='Signal ID', color_discrete_map=color_map, template=template)
    else:
        x_axis_label = 'Date'
        fig = px.line(df_agg, x='Timestamp', y='Volume', color='Signal ID', color_discrete_map=color_map, template=template)

    fig.update_xaxes(title_text=x_axis_label)
    fig.update_yaxes(title_text='<b>Volume<b>')
    fig.update_traces(line=dict(width=3))
    fig.update_layout(showlegend=True)

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

    fig6 = copy.deepcopy(fig)
    fig6.update_layout(xaxis=dict(rangeselector=dict(visible=False ),rangeslider=dict(visible=False),type="date"))
    fig6.update_layout(title='Time series of crosswalk users activity, by (selected time unit), by location' , showlegend=True)
    fig6.write_image("fig1.png")
    return fig


@st.cache_resource
def make_table(df, signals, start_date, end_date, aggregation_method, location,Dash_selected , act_selected):
    # Map act_selected to its corresponding column
    if act_selected == "All crosswalk users (walk, bicycle, scooter, skateboard, wheelchair, other)":
        col = 'TOTAL'
    elif act_selected == 'Pedestrians (walk, skateboard, wheelchair)':
        col = 'UPED'
    elif act_selected == "People walking":
        col = 'PED'
    else:
        raise ValueError(f"Invalid value for act_selected: {act_selected}")
    
    if aggregation_method == 'Hours':
        groupby = ['ADDRESS', pd.Grouper(key='TIME1', freq='1H')]
    elif aggregation_method == 'Days':
        groupby = ['ADDRESS', pd.Grouper(key='TIME1', freq='1D')]
    elif aggregation_method == 'Weeks':
        groupby = ['ADDRESS', pd.Grouper(key='TIME1', freq='1W')]
    elif aggregation_method == 'Months':
        groupby = ['ADDRESS', pd.Grouper(key='TIME1', freq='1M')]
    elif aggregation_method == 'Years':
        groupby = ['ADDRESS', pd.Grouper(key='TIME1', freq='1Y')]

    # Handle the location filter
    if location == 'All':
        filter_val = 'all'
    else:
        filter_val = location.split()[-1]

    # Filter the dataframe by the selected signals and date range
    if Dash_selected == 'Recent data (last 1 year)':
        df_filtered = df[(df['TIME1'] >= pd.to_datetime(start_date).tz_localize('UTC')) & (df['TIME1'] <= pd.to_datetime(end_date).tz_localize('UTC')) & (df['ADDRESS'].isin(signals))]
    else:
        df_filtered = df[(df['TIME1'] >= start_date) & (df['TIME1'] <= end_date) & (df['ADDRESS'].isin(signals))]
    
    # Aggregate the data
    agg_dict = {col: 'sum', 'CITY': 'first', 'SIGNAL': 'first' , 'LAT': 'first' , 'LON': 'first'}
    if location == 'All':
        df_agg = df_filtered.groupby(groupby).agg(agg_dict).reset_index()
    else:
        df_agg = df_filtered[df_filtered['P'] == int(filter_val)].groupby(groupby).agg(agg_dict).reset_index()
        agg_dict['P'] = 'first'

    df_agg[col] = df_agg[col].apply(lambda x: '{:,.0f}'.format(x))
    renaming_dict = {'SIGNAL': 'Signal ID', 'TIME1': 'Timestamp', col: 'Volume', 'CITY': 'City', 'P': 'Phase', 'LAT': 'Latitude', 'LON': 'Longtitude'}
    df_agg.rename(columns=renaming_dict, inplace=True)

    # Select the columns for display
    if 'Phase' in df_agg.columns:
        df_agg = df_agg[['Signal ID', 'Timestamp', 'Phase', 'Volume', 'City' , 'Latitude' , 'Longtitude']]
    else:
        df_agg = df_agg[['Signal ID', 'Timestamp', 'Volume', 'City' , 'Latitude' , 'Longtitude']]

    df_agg.reset_index(drop=True, inplace=True)  # remove index column
    df_agg['Timestamp'] = df_agg['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    return df_agg

@st.cache_resource
def make_pie_and_bar_chart(df, signals, start_date, end_date, location,Dash_selected , color_map ,  act_selected):
    # Map act_selected to its corresponding column
    if act_selected == "All crosswalk users (walk, bicycle, scooter, skateboard, wheelchair, other)":
        col = 'TOTAL'
    elif act_selected == 'Pedestrians (walk, skateboard, wheelchair)':
        col = 'UPED'
    elif act_selected == "People walking":
        col = 'PED'
    else:
        raise ValueError(f"Invalid value for act_selected: {act_selected}")

    # Handle the location filter
    if location == 'All':
        filter_val = 'all'
    else:
        filter_val = location.split()[-1]

    # Filter the dataframe by the selected signals and date range
    if Dash_selected == 'Recent data (last 1 year)':
        df_filtered = df[(df['TIME1'] >= pd.to_datetime(start_date).tz_localize('UTC')) & (df['TIME1'] <= pd.to_datetime(end_date).tz_localize('UTC')) & (df['ADDRESS'].isin(signals))]
    else:
        df_filtered = df[(df['TIME1'] >= start_date) & (df['TIME1'] <= end_date) & (df['ADDRESS'].isin(signals))]

    # Aggregate the data
    if location == 'All':
        df_agg = df_filtered.groupby(['ADDRESS', 'SIGNAL']).agg({col: 'sum','CITY': 'first'}).reset_index()
    else:
        df_agg = df_filtered[df_filtered['P'] == int(filter_val)].groupby(['ADDRESS', 'SIGNAL']).agg({col: 'sum' , 'CITY': 'first'}).reset_index()

    # Aggregate the data by signal and sum the pedestrian counts
    df_agg1 = df_agg.groupby('SIGNAL').agg({col: 'sum'}).reset_index()
    df_agg1 = df_agg1.rename(columns={'SIGNAL': 'Signal ID'})
    df_agg1['Signal'] = df_agg1['Signal ID'].astype(str)

    # Round the pedestrian count to two decimal places
    df_agg1[col] = df_agg1[col].round(0)

    # Create the pie chart
    fig_pie = go.Figure(data=[go.Pie(labels=df_agg1['Signal ID'], values=df_agg1[col], name='Signal ID', 
                                 marker=dict(colors=[color_map[x] for x in df_agg1['Signal ID']]))])
    fig_pie.update_layout(title='Crosswalk users Activity by Signal', showlegend=True)

    # Create the treemap figure
    fig_treemap = go.Figure(data=go.Treemap(
        labels=df_agg1['Signal ID'],
        parents=['']*len(df_agg1),
        values=df_agg1[col], name='Signal ID',
        marker_colors=df_agg1['Signal ID'].apply(lambda x: color_map[x])
    ))
    fig_treemap.update_layout(title='Crosswalk users Activity by City', showlegend=False)

    # Combine the pie, bar, and treemap charts
    fig_combined = make_subplots(rows=1, cols=2, specs=[[{'type': 'domain'}, {'type': 'treemap'}]])
    fig_combined.add_trace(fig_pie.data[0], row=1, col=1)
    fig_combined.add_trace(fig_treemap.data[0], row=1, col=2)
    fig_combined.update_layout(showlegend=True)
    fig_combined.update_layout(template='plotly')
    fig4 = copy.deepcopy(fig_pie)
    fig4.update_layout(title='Crosswalk users Activity by Signal', showlegend=True , legend=dict(title='Signal ID'))
    fig4.update_layout(template='plotly')
    fig4.write_image("fig2.png")
    
    return fig_combined , df_agg1

@st.cache_resource
def make_bar_chart(df, signals, start_date, end_date, location,Dash_selected , act_selected):
    # Map act_selected to its corresponding column
    if act_selected == "All crosswalk users (walk, bicycle, scooter, skateboard, wheelchair, other)":
        col = 'TOTAL'
    elif act_selected == 'Pedestrians (walk, skateboard, wheelchair)':
        col = 'UPED'
    elif act_selected == "People walking":
        col = 'PED'
    else:
        raise ValueError(f"Invalid value for act_selected: {act_selected}")

    # Handle the location filter
    if location == 'All':
        filter_val = 'all'
    else:
        filter_val = location.split()[-1]

    # Filter the dataframe by the selected signals and date range
    if Dash_selected == 'Recent data (last 1 year)':
        df_filtered = df[(df['TIME1'] >= pd.to_datetime(start_date).tz_localize('UTC')) & (df['TIME1'] <= pd.to_datetime(end_date).tz_localize('UTC')) & (df['ADDRESS'].isin(signals))]
    else:
        df_filtered = df[(df['TIME1'] >= start_date) & (df['TIME1'] <= end_date) & (df['ADDRESS'].isin(signals))]

    # Aggregate the data
    if location == 'All':
        df_agg = df_filtered.groupby('TIME1').agg({col: 'sum'}).reset_index()
    else:
        df_agg = df_filtered[df_filtered['P'] == int(filter_val)].groupby('TIME1').agg({col: 'sum'}).reset_index()

    
    # Convert the "TIME1" column to hour values
    df_agg['TIME1'] = pd.to_datetime(df_agg['TIME1']).dt.hour
    
    # Aggregate the data by hour and sum the pedestrian counts
    df_agg2 = df_agg.groupby('TIME1').agg({col: 'sum'}).reset_index()
    
    # Calculate the hourly average by dividing the sum of PED by number of days/hours
    df_agg2[col] = df_agg2[col] / 24

    # Round the pedestrian count to 0 decimal places
    df_agg2[col] = df_agg2[col].round(0)
    
    # Define the data and layout for the bar chart
    data = [go.Bar(x=df_agg2['TIME1'], y=df_agg2[col], text=df_agg2[col], texttemplate='%{y:,.0f}', textposition='auto')]
    layout = go.Layout(xaxis_title='<b>Hour of the day<b>', yaxis_title='<b>Volume<b>', xaxis=dict(tickmode='linear', dtick=1))

    # Create the bar chart
    fig_bar = go.Figure(data=data, layout=layout)
    fig_bar.update_yaxes(tickformat=".0f")
    fig_bar.update_layout(template='plotly')
    fig2 = copy.deepcopy(fig_bar)
    fig2.update_layout(title='Average hourly crosswalk users activity, by hour-of-day, total of all locations')
    fig2.update_layout(template='plotly')
    fig2.write_image("fig3.png")
    
    return fig_bar , df_agg2

@st.cache_resource
def make_bar_chart2(df, signals, start_date, end_date, location,Dash_selected,act_selected):
    # Map act_selected to its corresponding column
    if act_selected == "All crosswalk users (walk, bicycle, scooter, skateboard, wheelchair, other)":
        col = 'TOTAL'
    elif act_selected == 'Pedestrians (walk, skateboard, wheelchair)':
        col = 'UPED'
    elif act_selected == "People walking":
        col = 'PED'
    else:
        raise ValueError(f"Invalid value for act_selected: {act_selected}")

    # Handle the location filter
    if location == 'All':
        filter_val = 'all'
    else:
        filter_val = location.split()[-1]

    # Filter the dataframe by the selected signals and date range
    if Dash_selected == 'Recent data (last 1 year)':
        df_filtered = df[(df['TIME1'] >= pd.to_datetime(start_date).tz_localize('UTC')) & (df['TIME1'] <= pd.to_datetime(end_date).tz_localize('UTC')) & (df['ADDRESS'].isin(signals))]
    else:
        df_filtered = df[(df['TIME1'] >= start_date) & (df['TIME1'] <= end_date) & (df['ADDRESS'].isin(signals))]
    
    # Aggregate the data
    if location == 'All':
        df_agg = df_filtered.groupby('TIME1').agg({col: 'sum'}).reset_index()
    else:
        df_agg = df_filtered[df_filtered['P'] == int(filter_val)].groupby('TIME1').agg({col: 'sum'}).reset_index()

    # Convert the "TIME1" column to day of the week values (0-6, where Monday=0 and Sunday=6)
    df_agg['TIME1'] = pd.to_datetime(df_agg['TIME1']).dt.dayofweek

    # Aggregate the data by day of the week and calculate the average pedestrian counts
    df_agg2 = df_agg.groupby('TIME1').agg({col: 'sum'}).reset_index()

    # Calculate the daily average by dividing the sum of PED by number of days/hours
    df_agg2[col] = df_agg2[col] / 7

    # Round the pedestrian count to 0 decimal places
    df_agg2[col] = df_agg2[col].round(0)

    # Create the bar chart
    fig_bar = go.Figure(data=[go.Bar(x=df_agg2['TIME1'], y=df_agg2[col], showlegend=False ,  texttemplate='%{y:,.0f}', textposition='auto')])

    # Set the x-axis tick labels to the full names of the days of the week
    fig_bar.update_layout(xaxis_title='<b>Day of the week<b>', yaxis_title='<b>Volume<b>', showlegend=False)
    fig_bar.update_xaxes(tickmode='array', tickvals=[0, 1, 2, 3, 4, 5, 6], ticktext=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    fig_bar.update_yaxes(tickformat=".0f")
    fig_bar.update_layout(template='plotly')
    fig3 = copy.deepcopy(fig_bar)
    fig3.update_layout(title='Average daily crosswalk users activity, by day-of-week, total of all locations')
    fig3.update_layout(template='plotly')
    fig3.write_image("fig4.png")

    return fig_bar, df_agg2

@st.cache_resource
def make_bar_chart3(df, signals, start_date, end_date, location,Dash_selected,act_selected):
    # Map act_selected to its corresponding column
    if act_selected == "All crosswalk users (walk, bicycle, scooter, skateboard, wheelchair, other)":
        col = 'TOTAL'
    elif act_selected == 'Pedestrians (walk, skateboard, wheelchair)':
        col = 'UPED'
    elif act_selected == "People walking":
        col = 'PED'
    else:
        raise ValueError(f"Invalid value for act_selected: {act_selected}")

    # Handle the location filter
    if location == 'All':
        filter_val = 'all'
    else:
        filter_val = location.split()[-1]

    # Filter the dataframe by the selected signals and date range
    if Dash_selected == 'Recent data (last 1 year)':
        df_filtered = df[(df['TIME1'] >= pd.to_datetime(start_date).tz_localize('UTC')) & (df['TIME1'] <= pd.to_datetime(end_date).tz_localize('UTC')) & (df['ADDRESS'].isin(signals))]
    else:
        df_filtered = df[(df['TIME1'] >= start_date) & (df['TIME1'] <= end_date) & (df['ADDRESS'].isin(signals))]
    
    # Aggregate the data
    if location == 'All':
        df_agg = df_filtered.resample('M', on='TIME1').agg({col: 'sum'}).reset_index()
    else:
        df_agg = df_filtered[df_filtered['P'] == int(filter_val)].resample('M', on='TIME1').agg({col: 'sum'}).reset_index()

    # Extract month from the "TIME1" column
    df_agg['TIME1'] = df_agg['TIME1'].dt.month

    # Aggregate the data by month and calculate the sum pedestrian counts
    df_agg2 = df_agg.groupby('TIME1').agg({col: 'sum'}).reset_index()


    # Create the bar chart
    fig_bar = go.Figure(data=[go.Bar(x=df_agg2['TIME1'], y=df_agg2[col], showlegend=False ,  texttemplate='%{y:,.0f}', textposition='auto')])

    # Set the x-axis tick labels to the full names of the days of the week
    fig_bar.update_layout(xaxis_title='<b>Month of the year<b>', yaxis_title='<b>Volume<b>', showlegend=False)
    fig_bar.update_xaxes(tickmode='array', 
                     tickvals=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 
                     ticktext=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
    fig_bar.update_yaxes(tickformat=".0f")
    fig_bar.update_layout(template='plotly')
    fig8 = copy.deepcopy(fig_bar)
    fig8.update_layout(title='Average daily crosswalk users activity, by month-of-year, total of all locations')
    fig8.update_layout(template='plotly')
    fig8.write_image("fig5.png")

    return fig_bar,df_agg2


def make_map5(df, start_date, end_date, signals, aggregation_method, location_selected, Dash_selected, act_selected):

    # Map act_selected to its corresponding column
    if act_selected == "All crosswalk users (walk, bicycle, scooter, skateboard, wheelchair, other)":
        col = 'TOTAL'
    elif act_selected == 'Pedestrians (walk, skateboard, wheelchair)':
        col = 'UPED'
    elif act_selected == "People walking":
        col = 'PED'
    else:
        raise ValueError(f"Invalid value for act_selected: {act_selected}")

    # Handle the location filter
    if location_selected == 'All':
        filter_val = 'all'
    else:
        filter_val = location_selected.split()[-1]
   
    # Convert start_date and end_date to datetime, if not already
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Filter by date, selected signals, and location
    if Dash_selected == 'Recent data (last 1 year)':
        mask = (df['TIME1'] >= start_date.tz_localize('UTC')) & (df['TIME1'] < end_date.tz_localize('UTC')) & (df['ADDRESS'].isin(signals))
        if location_selected == 'All':
            mask &= df['P'] >= 0  # include all values of P for intersections
        else:
            if location_selected.startswith('Phase'):
                phase_num = int(location_selected.split()[1])
                mask &= df['P'] == phase_num
            else:
                mask &= df['ADDRESS'] == location_selected
        df_filtered = df.loc[mask]
    else:
        mask = (df['TIME1'] >= start_date) & (df['TIME1'] <= end_date) & (df['ADDRESS'].isin(signals))
        df_filtered = df.loc[mask]

    # Define aggregation methods
    agg_functions = {
        'Hours': 'sum',
        'Days': 'sum',
        'Weeks': 'sum',
        'Months': 'sum',
        'Years': 'sum'
    }

    aggregation_function = agg_functions[aggregation_method]

    # Aggregate by location
    if location_selected == 'All':
        df_agg = df_filtered.groupby(['LAT', 'LON', 'ADDRESS', 'CITY', 'SIGNAL', 'TIME1']).agg({col: aggregation_function}).reset_index()
        
    else:
        df_agg = df_filtered.groupby(['LAT', 'LON', 'ADDRESS', 'CITY', 'P', 'SIGNAL', 'TIME1']).agg({col: aggregation_function}).reset_index()
        
    
    df_agg['TIME1'] = pd.to_datetime(df_agg['TIME1'])
    df_agg['TIME1'] = df_agg['TIME1'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df_agg = df_agg.rename(columns={'SIGNAL': 'Signal ID', 'TIME1':'Timestamp' , col:'Volume' })

    # Assuming start_date is already a datetime object
    start_date_timestamp_ms = start_date.timestamp() * 1000
    # Convert the timestamp to a Pandas datetime object
    start_date_datetime = pd.to_datetime(start_date_timestamp_ms, unit='ms')
    # Add one month
    next_day_datetime = start_date + timedelta(days=1)
    # Convert to a timestamp in milliseconds
    next_day_timestamp_ms = int(next_day_datetime.timestamp() * 1000)

    map_config = {
        'version': 'v1',
        'config': {
            'mapState': {
                'latitude': df_agg['LAT'].mean(),
                'longitude': df_agg['LON'].mean(),
                'zoom': 13,
            },
            'mapStyle': {
                'styleType': 'light'  
            },
            'visState': {
                'layers': [{
                    'type': 'hexagon',  
                    'config': {
                        'dataId': 'data_1',
                        'label': 'Hexagon',
                        'color': [194, 46, 0],
                        'columns': {
                            'lat': 'LAT',
                            'lng': 'LON',
                        },
                        'isVisible': True,
                        'visConfig': {
                            'opacity': 0.8,
                            'worldUnitSize': 0.05,
                            'radius': 5,
                            'outline': True, 
                            'filled': False,
                            'enable3d': True, 
                            'colorRange': {
                                'name': 'ColorBrewer YlGn-6',
                                'type': 'sequential',
                                'category': 'ColorBrewer',
                                'colors': ['#ffffcc','#d9f0a3','#addd8e','#78c679','#31a354','#006837']
                            },
                                                    },
                        'textLabel': [{
                            'field': None,
                            'color': [255, 255, 255],
                            'size': 18,
                            'offset': [0, 0],
                            'anchor': 'start',
                            'alignment': 'center'
                        }]
                    },
                    'visualChannels': {
                        'colorField': {'name': 'Signal ID', 'type': 'integer'},
                        'colorScale': 'quantile',
                        'sizeField': {'name': 'Volume', 'type': 'integer'},  
                        'sizeScale': 'linear'
                    }
                }],
                'interactionConfig': {'tooltip': {'fieldsToShow': {'data_1': [{'name': 'Signal ID',
                      'format': None}]},
                    'compareMode': False,
                    'compareType': 'absolute',
                    'enabled': True},
                'brush': {'size': 0.5, 'enabled': False},
                'geocoder': {'enabled': False},
                'coordinate': {'enabled': False}},
                'filters': [{
                    'dataId': 'data_1',
                    'name': 'Timestamp',
                    'type': 'timeRange',
                    'value': [start_date_timestamp_ms, next_day_timestamp_ms],
                    'enlarged': True,
                    'plotType': 'lineChart',
                    'yAxis': {'name': 'Volume' , 'type': 'integer'},
                    'animationWindow': 'free',
                    'speed': 0.2
                }],
            }
        }}

    map_1 = KeplerGl(height=600, data={'data_1': df_agg}, config=map_config)
    return map_1


# Define the Streamlit app
def main():
    # Set the app title
    st.set_page_config(page_title='PEDAT Dashboard' , page_icon="📈" , layout="wide"  )
    # Add a title to the sidebar
    st.title("Monitoring crosswalk users activity in Oregon")
    st.markdown(text1)
    #st.markdown(text2)
    
    st.markdown("""
            <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 0rem;
                }
            </style>
            """, unsafe_allow_html=True)
    
    st.markdown("""
    <style>
        .css-6qob1r {
        margin-top: -75px;
        }
    </style>
    """, unsafe_allow_html=True)

    st.sidebar.markdown(f'[**Singleton Transportation Lab**](https://engineering.usu.edu/cee/research/labs/patrick-singleton/index)')
    expander2 = st.sidebar.expander("**How to use**")
    with expander2:
        expander2.write('''
                There are Four tabs available in the above. 
                In these tabs, you can observe pedestrian activity based on selected aggregation methods at one or more signals (on a map or in a figure). 
                You can also download a CSV file containing the data that is creating the map or figure.
        ''')
    expander = st.sidebar.expander("**Notes**")
    with expander:
            expander.write('''
                    As shown on this website in maps, figures, and tables, pedestrian activity is estimated pedestrian volume , 
                    or an estimate of the total number of daily pedestrian crossings at an intersection. These estimated pedestrian 
                    volumes are based on traffic signal pedestrian push-button data, obtained from high-resolution traffic signal controller 
                    log data.
            ''')
    st.sidebar.markdown("[Step 1: Select location(s)](#step-1-select-location-s)")
    st.subheader('Step 1: Select location(s)')

    # Read the data
    df3 = pd.read_json('map_data.json')

    # Create a list of cities and counties
    locations = ["All"] + sorted(df3['CITY'].unique().tolist() + df3['County'].unique().tolist())

    # Create a multiselect box for selecting cities or counties
    selected_location = st.multiselect('**Select 1+ counties and/or cities**', locations, default=["Portland"])

    # Filter the data based on the selected cities or counties
    if "All" in selected_location:
        selected_data = df3
    else:
        selected_data = df3[(df3['CITY'].isin(selected_location)) | (df3['County'].isin(selected_location))]

    # Compute the mean latitude and longitude of the selected cities
    mean_lat = selected_data['LAT'].mean()
    mean_lng = selected_data['LON'].mean()
    df3 = selected_data
    a = ["All"] + df3['ADDRESS'].tolist()
    default_address = [a[1]]
    icon_image = 'ts.png'
    icon_size = (7, 14)
    
    # Create the map object
    m = folium.Map(location=[mean_lat,mean_lng], zoom_start=8 , tiles = 'https://api.mapbox.com/styles/v1/bashasvari/clhgx1yir00h901q1ecbt9165/tiles/256/{z}/{x}/{y}@2x?access_token=pk.eyJ1IjoiYmFzaGFzdmFyaSIsImEiOiJjbGVmaTdtMmIwcXkzM3Jxam9hb2pwZ3BoIn0.JmYank8e3bmQ7RmRiVdTIg' , attr='PEDAT map')



    # Convert your DataFrame to GeoJSON format
    geo_data = df3[["LAT", "LON", "ADDRESS"]].copy()

    def feature_from_row(row):
        return {
            "type": "Feature",
            "properties": {"address": row["ADDRESS"]},
            "geometry": {
                "type": "Point",
                "coordinates": [row["LON"], row["LAT"]]
            }
        }

    geo_json = {
        "type": "FeatureCollection",
        "features": geo_data.apply(feature_from_row, axis=1).tolist()
    }

    # Create the invisible GeoJson layer for search functionality
    geo_layer = folium.GeoJson(geo_json ,
    marker=folium.CircleMarker(radius=0.00000000000000000001, color="white",opacity=7)).add_to(m)
    search = Search(
        layer=geo_layer,
        geom_type="Point",
        placeholder="Search for an address/signals",
        collapsed=True,
        search_label="address"
    ).add_to(m)

    # Add your custom icons
    for index, row in df3.iterrows():
        folium.Marker(
            location=(row['LAT'], row['LON']),
            popup=folium.Popup(row['ADDRESS'], max_width=300, min_width=150),
            tooltip=row['ADDRESS'],
            icon=folium.CustomIcon(icon_image, icon_size)
        ).add_to(m)

    Draw(
        export=False,
        filename="my_data.geojson",
        position="topleft",
        draw_options={
            'circle': False,
            'circlemarker': False,
            'marker': False,
            'polyline': False
        }
    ).add_to(m)

    address = []

    # Fit the map bounds to show all points
    sw = df3[['LAT', 'LON']].min().values.tolist()
    ne = df3[['LAT', 'LON']].max().values.tolist()
    m.fit_bounds([sw, ne]) 

    # Render the map using st_folium
    s = st_folium(m, width='80%', height=400, returned_objects=["last_object_clicked", "last_active_drawing"])

    # Check if the JSON object is not None
    if s is not None and "last_object_clicked" in s and s["last_object_clicked"] is not None:
        json_obj = s["last_object_clicked"]
        lat = json_obj["lat"]
        lng = json_obj["lng"]

        # Filter the dataframe based on the lat and lng values
        filtered_df = df3[(df3['LAT'] == lat) & (df3['LON'] == lng)]

        # Print the 'ADDRESS' value for each row in the filtered dataframe
        for index, row in filtered_df.iterrows():
            address.append(row['ADDRESS'])

        selected_signals = st.multiselect('**Selected location(s)**', a , default = address)
    elif s is not None and "last_active_drawing" in s and s["last_active_drawing"] is not None:
        # A polygon has been drawn on the map
        polygon_coords = s["last_active_drawing"]["geometry"]["coordinates"]
        polygon = Polygon(polygon_coords[0])

        # Create an empty list to hold the selected addresses
        selected_addresses = []

        # Iterate through each row of the dataframe and check if its address falls within the polygon
        for index, row in df3.iterrows():
            point = Point(row['LON'], row['LAT'])
            if polygon.contains(point):
                selected_addresses.append(row['ADDRESS'])

        selected_signals = st.multiselect('**Selected location(s)**', a, default=selected_addresses)
    else:
        selected_signals = st.multiselect('**Selected location(s)**', a , default = address)
 
    if "All" in selected_signals:
        selected_signals = df3['ADDRESS'].unique().tolist()

    font_css = """
    <style>
    button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] > p {
    font-size: 20px;
    }
    </style>
    """
    st.write(font_css, unsafe_allow_html=True)
    
    st.markdown(
        """<style>
    div[class*="stMultiSelect"] > label > div[data-testid="stMarkdownContainer"] > p {
        font-size: 16px;
    }
        </style>
        """, unsafe_allow_html=True)
    
    st.markdown(
        """<style>
    div[class*="stSelectbox"] > label > div[data-testid="stMarkdownContainer"] > p {
        font-size: 16px;
    }
        </style>
        """, unsafe_allow_html=True)
    # Check if selected_signals is empty
    if not selected_signals:
        # If selected_signals is empty, return an empty dataframe
        return pd.DataFrame()

    job_config = bigquery.QueryJobConfig(
    query_parameters=[
        bigquery.ArrayQueryParameter(
            "selected_signals", "STRING", selected_signals
        )
    ]
    )

    st.sidebar.markdown("[Step 2: Select dashboard type](#step-2-select-dashboard-type)")
    st.subheader('Step 2: Select dashboard type')
    dash = ['Recent data (last 1 year)'] #, 'Historical data (last 5 years)']
    Dash_selected = st.selectbox('**Select dashboard type**', options=dash)

    # Add a subtitle to the sidebar
    if Dash_selected == 'Recent data (last 1 year)':
        df = client.query(sql_query2, job_config=job_config).to_dataframe()
        unique_signals = df['SIGNAL'].unique().tolist()
        color_map = create_color_map(unique_signals)
    else:
        df = client.query(sql_query3, job_config=job_config).to_dataframe()
        df['TIME1'] = pd.to_datetime(df['TIME1'])
        unique_signals = df['SIGNAL'].unique().tolist()
        color_map = create_color_map(unique_signals)

    # Create a list of all unique values in the 'ADDRESS' column of the DataFrame
    all_addresses = df3['ADDRESS'].tolist()

    # Check if selected_signals is not empty
    if selected_signals:
        # If selected_signals is not empty, filter the list of all_addresses to include only the selected signals
        addresses_to_keep = set(selected_signals).intersection(set(all_addresses))
        all_addresses = list(addresses_to_keep)

    st.markdown(
        """<style>
    div[class*="stColumn"] > label > div[data-testid="stMarkdownContainer"] > p {
        font-size: 32px;
    }
        </style>
        """, unsafe_allow_html=True)

    st.sidebar.markdown("[Step 3: Select dashboard parameters](#step-3-select-dashboard-parameters)")
    st.subheader('Step 3: Select dashboard parameters')
    form = st.form("sidebar")
    
    # Add a calendar widget to select a date range
    start_date = form.date_input('**Start date**', df['TIME1'].min())
    end_date = form.date_input('**End date**', df['TIME1'].max())

    # Add a slider for selecting the aggregation method
    if Dash_selected == 'Recent data (last 1 year)':
        locations = ['All'] + ['Phase ' + str(int(i)) for i in sorted(df[df['ADDRESS'].isin(all_addresses)]['P'].dropna().unique().tolist())]
        #locations = ['All'] + ['Phase ' + str(int(i)) for i in sorted(df['P'].dropna().unique().tolist())]
        location_selected = form.selectbox('**Select location units**', options=locations)
        aggregation_methods = ['Hours', 'Days', 'Weeks', 'Months', 'Years']
        aggregation_method_selected = form.selectbox('**Select time units**', options=aggregation_methods)
        act = ['All crosswalk users (walk, bicycle, scooter, skateboard, wheelchair, other)', 
               'Pedestrians (walk, skateboard, wheelchair)',
               'People walking']
        act_selected = form.selectbox('**Choose pedestrian activity measure**', options=act) 
    else:
        aggregation_methods = ['Days', 'Weeks', 'Months', 'Years']
        aggregation_method_selected = form.selectbox('**Select time units**', options=aggregation_methods)
        location = ['All']
        location_selected = location[0]

      
    st.markdown(
        """<style>
    div[class*="stSlider"] > label > div[data-testid="stMarkdownContainer"] > p {
        font-size: 16px;
    }
        </style>
        """, unsafe_allow_html=True)

    form.form_submit_button("Submit")
    
    # Convert the date objects to datetime objects
    start_datetime = datetime.combine(start_date, datetime.min.time())
    end_datetime = datetime.combine(end_date, datetime.max.time())
    dt_str = start_date.strftime("%b %d, %Y")
    dt_str2 = end_date.strftime("%b %d, %Y")

    # Map act_selected to its corresponding column
    if act_selected == "All crosswalk users (walk, bicycle, scooter, skateboard, wheelchair, other)":
        col = 'TOTAL'
    elif act_selected == 'Pedestrians (walk, skateboard, wheelchair)':
        col = 'UPED'
    elif act_selected == "People walking":
        col = 'PED'
    else:
        raise ValueError(f"Invalid value for act_selected: {act_selected}")

    total_pedestrians = int(df[col].sum().round(0))
    num_signals = len(df['ADDRESS'].unique())

    # Format the metric values
    total_pedestrians_formatted = format_metric(total_pedestrians)
    num_signals_formatted = format_metric(num_signals)
    
    st.sidebar.markdown("[Metrics](#metrics)")
    st.subheader('**Metrics**')
    col1, col2 , col3 , col4= st.columns(4)
    # Display the metric boxes
    col1.metric("**Total**", total_pedestrians_formatted)
    col2.metric("**Selected Signals**", num_signals_formatted)
    col3.metric("**Start Date**" , dt_str)
    col4.metric("**End Date**" , dt_str2)
    # If "All" is selected, show all signals
    if "All" in selected_signals:
        selected_signals = df3['ADDRESS'].tolist()
    else:
        selected_signals = selected_signals or default_address

    st.sidebar.markdown("[Charts](#charts)")
    st.subheader('**Charts**')
    with st.expander("Expand"):
        if Dash_selected == 'Recent data (last 1 year)':

            st.subheader('Total crosswalk users activity, by location')
            # Add a pie chart to show pedestrian activity by signal
            fig4 , df_agg1 = make_pie_and_bar_chart(df, selected_signals, start_datetime, end_datetime, location_selected, Dash_selected , color_map, act_selected)
            cv2 = df_agg1.to_csv(index=True)
            st.download_button(
                label="📥 Download data",
                data=cv2,
                file_name="piebarchart.csv",
                mime='text/csv',
            )
            st.plotly_chart(fig4, use_container_width=True )

            st.subheader('Average hourly crosswalk users activity, by hour-of-day, total of all locations')
            fig2, df_agg3 = make_bar_chart(df, selected_signals, start_datetime, end_datetime, location_selected, Dash_selected, act_selected)
            cv3 = df_agg3.to_csv(index=True)
            st.download_button(
                label="📥 Download data",
                data=cv3,
                file_name="barcharthourly.csv",
                mime='text/csv',
            )
            st.plotly_chart(fig2, use_container_width=True )

            st.subheader('Average daily crosswalk users activity, by day-of-week, total of all locations')
            fig3 , df_agg4= make_bar_chart2(df, selected_signals, start_datetime, end_datetime, location_selected, Dash_selected, act_selected)
            cv4 = df_agg4.to_csv(index=True)
            st.download_button(
                label="📥 Download data",
                data=cv4,
                file_name="barchartdaily.csv",
                mime='text/csv',
            )
            st.plotly_chart(fig3, use_container_width=True )

            st.subheader('Average daily crosswalk users activity, by month-of-year, total of all locations')
            fig8 , df_agg5= make_bar_chart3(df, selected_signals, start_datetime, end_datetime, location_selected, Dash_selected, act_selected)
            cv5 = df_agg5.to_csv(index=True)
            st.download_button(
                label="📥 Download data",
                data=cv5,
                file_name="barchartmonthly.csv",
                mime='text/csv',
            )
            st.plotly_chart(fig8, use_container_width=True )

            table = make_table(df, selected_signals, start_datetime, end_datetime, aggregation_method_selected, location_selected, Dash_selected, act_selected)
            pivot_table = table.pivot_table(values='Volume', index='Timestamp', columns='Signal ID', aggfunc='sum')
            cv1 = pivot_table.to_csv(index=True)
            st.subheader('Time series of crosswalk users activity, by (selected time unit), by location')
            st.download_button(
                label="📥 Download data",
                data=cv1,
                file_name="TimeSeries.csv",
                mime='text/csv',
            )
        
            # Make the time series plot
            fig1 = make_chart(df, selected_signals, start_datetime, end_datetime, aggregation_method_selected, location_selected, Dash_selected, act_selected, color_map,template='plotly')
            st.plotly_chart(fig1, use_container_width=True )

            table['Signal ID'] = table['Signal ID'].astype(str)
            table['Volume'] = table['Volume'].astype(str)
            table['Signal ID'] = table['Signal ID'].str.replace(',', '.')
            table['Volume'] = table['Volume'].str.replace(',', '.')
            table['Signal ID'] = pd.to_numeric(table['Signal ID'], errors='coerce')
            table['Volume'] = pd.to_numeric(table['Volume'] , errors='coerce')
            grouped = table.groupby('Signal ID')['Volume'].describe()
            missing_counts = table['Volume'].isna().groupby(table['Signal ID']).sum()
            grouped['Missing Count'] = missing_counts
            DS = grouped.to_csv(index=True)

            # Box Plot
            st.subheader('Box plot of crosswalk users activity, by (selected time unit), by location')
            signal_ids = table['Signal ID'].unique() 
            fig = go.Figure()

            for signal_id, group in table.groupby('Signal ID'):
                if signal_id in signal_ids:
                    fig.add_trace(go.Box(y=group['Volume'], name=f'{signal_id}', 
                        marker=dict(color=color_map[signal_id])))

            # Save the box_plot_data DataFrame to a CSV file
            st.download_button(
                label="📥 Download data",
                data=DS,
                file_name="box_plot_data.csv",
                mime='text/csv',
            )

            fig.update_layout(yaxis_title='<b>Volume<b>', xaxis_title='<b>Location ID<b>')
            fig.update_layout(xaxis=dict(title='Signal ID', type='category', tickmode='array', tickvals=signal_ids,
                                        ticktext=[str(signal_id) for signal_id in signal_ids]))
            fig.update_layout(showlegend=True)
            st.plotly_chart(fig, theme='streamlit', use_container_width=True)
            fig7 = copy.deepcopy(fig)
            fig7.update_layout(title='Box plot of crosswalk users activity, by (selected time unit), by location' , showlegend=False)
            fig7.update_layout(template='plotly')
            fig7.write_image("fig6.png")


        else:
            st.subheader('Total crosswalk users activity, by location')
            # Add a pie chart to show pedestrian activity by signal
            fig4 , df_agg1 = make_pie_and_bar_chart(df, selected_signals, start_datetime, end_datetime, location_selected, Dash_selected , act_selected,color_map)
            cv2 = df_agg1.to_csv(index=True)
            st.download_button(
                label="📥 Download data",
                data=cv2,
                file_name="piebarchart.csv",
                mime='text/csv',
            )
            st.plotly_chart(fig4, use_container_width=True )

            st.subheader('Average daily crosswalk users activity, by day-of-week, total of all locations')
            fig3 , df_agg4= make_bar_chart2(df, selected_signals, start_datetime, end_datetime, location_selected, Dash_selected , act_selected)
            cv4 = df_agg4.to_csv(index=True)
            st.download_button(
                label="📥 Download data",
                data=cv4,
                file_name="barchartdaily.csv",
                mime='text/csv',
            )
            st.plotly_chart(fig3, use_container_width=True )

            st.subheader('Average daily crosswalk users activity, by month-of-year, total of all locations')
            fig8 , df_agg5= make_bar_chart3(df, selected_signals, start_datetime, end_datetime, location_selected, Dash_selected, act_selected)
            cv5 = df_agg5.to_csv(index=True)
            st.download_button(
                label="📥 Download data",
                data=cv5,
                file_name="barchartmonthly.csv",
                mime='text/csv',
            )
            st.plotly_chart(fig8, use_container_width=True )
            table = make_table(df, selected_signals, start_datetime, end_datetime, aggregation_method_selected, location_selected, Dash_selected, act_selected)
            pivot_table = table.pivot_table(values='Pedestrian', index='Timestamp', columns='Signal ID', aggfunc='sum')
            cv1 = pivot_table.to_csv(index=True)
            st.subheader('Time series of crosswalk users activity, by (selected time unit), by location')
            st.download_button(
                label="📥 Download data",
                data=cv1,
                file_name="TimeSeries.csv",
                mime='text/csv',
            )
        
           # Make the time series plot
            fig1 = make_chart(df, selected_signals, start_datetime, end_datetime, aggregation_method_selected, location_selected, Dash_selected, act_selected, color_map,template='plotly')
            st.plotly_chart(fig1, use_container_width=True )

            table['Signal ID'] = table['Signal ID'].astype(str)
            table['Volume'] = table['Volume'].astype(str)
            table['Signal ID'] = table['Signal ID'].str.replace(',', '.')
            table['Volume'] = table['Volume'].str.replace(',', '.')
            table['Signal ID'] = pd.to_numeric(table['Signal ID'], errors='coerce')
            table['Volume'] = pd.to_numeric(table['Volume'] , errors='coerce')
            grouped = table.groupby('Signal ID')['Volume'].describe()
            missing_counts = table['Volume'].isna().groupby(table['Signal ID']).sum()
            grouped['Missing Count'] = missing_counts
            DS = grouped.to_csv(index=True)

            # Box Plot
            st.subheader('Box plot of crosswalk users activity, by (selected time unit), by location')
            signal_ids = table['Signal ID'].unique() 
            fig = go.Figure()

            for signal_id, group in table.groupby('Signal ID'):
                if signal_id in signal_ids:
                    fig.add_trace(go.Box(y=group['Volume'], name=f'{signal_id}', 
                        marker=dict(color=color_map[signal_id])))

            # Save the box_plot_data DataFrame to a CSV file
            st.download_button(
                label="📥 Download data",
                data=DS,
                file_name="box_plot_data.csv",
                mime='text/csv',
            )

            fig.update_layout(yaxis_title='<b>Volume<b>', xaxis_title='<b>Location ID<b>')
            fig.update_layout(xaxis=dict(title='<b>Signal ID<b>', type='category', tickmode='array', tickvals=signal_ids,
                                        ticktext=[str(signal_id) for signal_id in signal_ids]))
            fig.update_layout(showlegend=True)
            st.plotly_chart(fig, theme='streamlit', use_container_width=True)
            fig7 = copy.deepcopy(fig)
            fig7.update_layout(title='Box plot of crosswalk users activity, by (selected time unit), by location' , showlegend=False)
            fig7.update_layout(template='plotly')
            fig7.write_image("fig6.png")

    st.sidebar.markdown("[Map](#map)")
    st.subheader('**Map**')  
    with st.expander("Expand"):
        map_2= make_map5(df,start_datetime, end_datetime , selected_signals , aggregation_method_selected , location_selected, Dash_selected, act_selected)
        keplergl_static(map_2)
    
    st.sidebar.markdown("[Downloads](#downloads)")
    st.subheader('**Downloads**')
    with st.expander("Expand"): 
        # Filter your data based on the selected date range
        st.subheader('Raw data')
        
        # Display the filtered data in a table
        
        table = make_table(df, selected_signals, start_datetime, end_datetime, aggregation_method_selected, location_selected, Dash_selected, act_selected)
        cc = table.to_csv(index=False)
        st.download_button(
            label="📥 Download",
            data=cc,
            file_name="RawData.csv",
            mime='text/csv',
        )

        st.dataframe(table , use_container_width=True)
        # CSS to inject contained in a string
        hide_dataframe_row_index = """
                    <style>
                    .row_heading.level0 {display:none}
                    .blank {display:none}
                    </style>
                    """


        # Discriptive statistics
        st.subheader('Discriptive Statistics')
        st.download_button(
        label="📥 Download",
        data=DS,
        file_name="DiscriptiveStat.csv",
        mime='text/csv',)
        st.dataframe(grouped , use_container_width=True)
    
    class PDF(FPDF):

        def __init__(self):
            super().__init__()
            self.page_width = 8.5 * 72  # Letter page width in points (1 inch = 72 points)
            self.page_height = 11 * 72  # Letter page height in points
            self.l_margin = 0.5 * 72    # Left margin in points
            self.r_margin = 0.5 * 72    # Right margin in points

        def header(self):
            # Select Arial bold 15
            self.set_font('Arial', 'I', 8)
            # Move to the right
            self.cell(self.l_margin)
            # Calculate the X position of the center of the available page width
            center_x = self.w / 2
            # Calculate the X position of the center of the title
            title_x = center_x - (self.get_string_width('Pedestrian Activity Data Report') / 2)
            # Framed title at the center
            self.cell(title_x - self.l_margin, 10, 'Pedestrian Activity Data Report', 0, 1, 'C')
            # Line break
            self.ln(20)
            # Check if we are on the first page
            if self.page_no() == 1:
                # Add the logo to the first page
                self.image('logo.png', x=10, y=10, w=33/2)

        def footer(self):
            # Position at 1.5 cm from bottom
            self.set_y(-20)
            # Arial italic 8
            self.set_font('Arial', 'I', 8)
            # Page number
            self.cell(0, 10, 'Page %s' % self.page_no(), 0, 0, 'C')

            # Adjust Y position again to put the text below page number
            self.set_y(-15)
            # Add footer text
            self.cell(0, 10, "This report generated automatically from Singleton's Transportation Lab (STL) Pedestrian Activity Dashboard", 0, 0, 'C')

        def chapter_title(self, title):
            # Arial 12
            self.set_font('Arial', '', 12)
            # Background color
            self.set_fill_color(200, 220, 255)
            # Title
            self.cell(0, 6, 'Chapter %d : %s' % (num, label), 0, 1, 'L', 1)
            # Line break
            self.ln(4)

        def chapter_body(self, body):
            # Read text file
            with open(name, 'rb') as fh:
                txt = fh.read().decode('latin-1')
            # Times 12
            self.set_font('Times', '', 12)
            # Output justified text
            self.multi_cell(0, 5, txt)
            # Line break
            self.ln()
            # Mention in italics
            self.set_font('', 'I')
            self.cell(0, 5, '(end of excerpt)')

    def generate_report():
            # Generate plots
            pdf = PDF()

            # Add a page
            pdf.add_page()

            # Add the logo to the first page after the first add_page() call
            pdf.image('logo.png', x=10, y=10, w=33/2)

            # Set title
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 15, 'Monitoring pedestrian activity in Oregon', ln=True, align='C')

            # Add a sample text
            pdf.set_font('Arial', '', 12)
            pdf.multi_cell(0, 6, "This website provides data and visualizations of pedestrian activity (and changes in pedestrian activity) at various (signalized) intersections throughout Oregon. We are currently showing all locations with available signal data throughout Oregon. We hope that this information is useful for public agencies to track changes in walking activity at different locations.")

            # Add selected signals
            pdf.set_font('Arial', 'B', 14)
            pdf.ln(5) 
            pdf.cell(1, 10, 'Selected Signals:', ln=True, align='L')
            pdf.set_font('Arial', '', 12)
            for signal in selected_signals:
                pdf.cell(1, 6, '- ' + signal, ln=True, align='L')

            # Add duration
            pdf.set_font('Arial', 'B', 14)
            pdf.ln(5)
            pdf.cell(0, 10, 'Duration:', ln=True)
            pdf.set_font('Arial', '', 12)
            start_date = start_datetime.strftime('%Y-%m-%d') 
            end_date = end_datetime.strftime('%Y-%m-%d')
            pdf.cell(0, 10, f'{start_date} to {end_date}', ln=True)

            # Add each figure to the report with subtitles
            if Dash_selected == 'Recent data (last 1 year)':
                subtitles = ['Time series plot', 'Hourly pedestrian activity plot', 'Daily pedestrian activity plot' , 'Pedestrian Activity in relation to Signal and City', 'Pedestrian Activity by Location' , 'Box plot']
                for i, subtitle in zip(range(1, 7), subtitles):
                    pdf.add_page()
                    pdf.set_font('Arial', 'B', 14)
                    pdf.cell(0, 10, subtitle, ln=True)
                    pdf.image(f'fig{i}.png', x=10, y=20, w=190)
            else:
                    subtitles = ['Time series plot', 'Daily pedestrian activity plot', 'Pedestrian Activity in relation to Signal and City', 'Pedestrian Activity by Location' , 'Box plot']
                    figure_indices = [1, 2, 4, 5, 6]  # Indices of the figures to be used

                    for i, subtitle in zip(figure_indices, subtitles):
                        pdf.add_page()
                        pdf.set_font('Arial', 'B', 14)
                        pdf.cell(0, 10, subtitle, ln=True)
                        pdf.image(f'fig{i}.png', x=10, y=20, w=190)

            # Save the PDF to a BytesIO object
            pdf_buffer = BytesIO()
            pdf.output(pdf_buffer, "F")
            pdf_bytes = pdf_buffer.getvalue()

            # Create a download link for the PDF
            b64_pdf = base64.b64encode(pdf_bytes).decode()
            href = f'<a href="data:file/pdf;base64,{b64_pdf}" download="report.pdf">Click here to download the PDF report</a>'
            st.markdown(href, unsafe_allow_html=True)

            files_to_remove = ["fig1.png", "fig3.png", "fig4.png", "fig5.png" ,  "fig6.png"]

            if os.path.exists("fig2.png"):
                os.remove("fig2.png")

            for file in files_to_remove:
                if os.path.exists(file):
                    os.remove(file)

    st.sidebar.markdown("[Report](#report)")
    st.subheader('Report')
    with st.expander("Expand"):
        if st.button('Generate PDF report'):
                generate_report()

    st.sidebar.markdown(
        """<style>
    div[class*="stDate"] > label > div[data-testid="stMarkdownContainer"] > p {
        font-size: 16px;
    }
        </style>
        """, unsafe_allow_html=True)

   
    st.markdown(
        """<style>
    div[class*="stExpander"] > label > div[data-testid="stMarkdownContainer"] > p {
        font-size: 16px;
    }
        </style>
        """, unsafe_allow_html=True)
   

    hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
    st.markdown(hide_menu_style, unsafe_allow_html=True)
   
if __name__ == '__main__':
    main()

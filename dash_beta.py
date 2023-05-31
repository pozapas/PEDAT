# PEDAT Dashboard V.0.0.4

import streamlit as st
import pandas as pd
import plotly.express as px
import pydeck as pdk
from datetime import datetime, date
import time
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
import base64
from io import BytesIO
import os
import datatable as dt
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
FROM `decent-digit-387716.pedat_dataset.pedatutah`
WHERE ADDRESS IN UNNEST(@selected_signals)
"""


# Define the title
title = 'Pedestrian Activity Data Visualization Dashboard'
text1 = 'This website provides data and visualizations of pedestrian activity (and changes in pedestrian activity) at various (signalized) intersections throughout Utah. We are currently showing all locations with available signal data throughout Utah.'
text2 = 'Data are derived from pedestrian push-button presses at traffic signals, taken from the Utah Department of Transportation\'s [Automated Traffic Signal Performance Measures System](https://udottraffic.utah.gov/atspm/) website. We hope that this information is useful for public agencies to track changes in walking activity at different locations.'

# Define the x and y axis labels
x_axis_label = 'TIME1'
y_axis_label = 'PED'

@st.cache_resource
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

@st.cache_resource
def make_chart(df, signals, start_date, end_date, aggregation_method, location, template='plotly'):
    if aggregation_method == 'Hourly':
        groupby = ['ADDRESS', pd.Grouper(key='TIME1', freq='1H')]
    elif aggregation_method == 'Daily':
        groupby = ['ADDRESS', pd.Grouper(key='TIME1', freq='1D')]
    elif aggregation_method == 'Weekly':
        groupby = ['ADDRESS', pd.Grouper(key='TIME1', freq='1W')]
    elif aggregation_method == 'Monthly':
        groupby = ['ADDRESS', pd.Grouper(key='TIME1', freq='1M')]
    elif aggregation_method == 'Yearly':
        groupby = ['ADDRESS', pd.Grouper(key='TIME1', freq='1Y')]

    if location == 'Intersection':
        filter_val = 'all'
        col = 'PED'
    else:
        filter_val = location.split()[-1]
        col = 'P'

    # Filter the dataframe by the selected signals and date range
    df_filtered = df[(df['TIME1'] >= pd.to_datetime(start_date).tz_localize('UTC')) & (df['TIME1'] <= pd.to_datetime(end_date).tz_localize('UTC')) & (df['ADDRESS'].isin(signals))]

    # Aggregate the data
    if location == 'Intersection':
        df_agg = df_filtered.groupby(groupby).agg({'PED': 'sum'}).reset_index()
    else:
        df_agg = df_filtered[df_filtered['P'] == int(filter_val)].groupby(groupby).agg({'PED': 'sum'}).reset_index()

    # Create the line chart
    if aggregation_method == 'Hourly':
        x_axis_label = 'Time'
        fig = px.line(df_agg, x='TIME1', y='PED' ,  color='ADDRESS', template=template)
    else:
        x_axis_label = 'Date'
        fig = px.line(df_agg, x='TIME1', y='PED', color='ADDRESS', template=template)

    fig.update_xaxes(title_text=x_axis_label)
    fig.update_yaxes(title_text='Pedestrian Estimated')
    fig.update_traces(line=dict(width=3))
    fig.update_layout(showlegend=False)

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


@st.cache_resource
def make_table(df, signals, start_date, end_date, aggregation_method, location):
    if aggregation_method == 'Hourly':
        groupby = ['ADDRESS', pd.Grouper(key='TIME1', freq='1H')]
    elif aggregation_method == 'Daily':
        groupby = ['ADDRESS', pd.Grouper(key='TIME1', freq='1D')]
    elif aggregation_method == 'Weekly':
        groupby = ['ADDRESS', pd.Grouper(key='TIME1', freq='1W')]
    elif aggregation_method == 'Monthly':
        groupby = ['ADDRESS', pd.Grouper(key='TIME1', freq='1M')]
    elif aggregation_method == 'Yearly':
        groupby = ['ADDRESS', pd.Grouper(key='TIME1', freq='1Y')]

    if location == 'Intersection':
        filter_val = 'all'
        col = 'PED'
    else:
        filter_val = location.split()[-1]
        col = 'P'

    # Filter the dataframe by the selected signals and date range
    df_filtered = df[(df['TIME1'] >= pd.to_datetime(start_date).tz_localize('UTC')) & (df['TIME1'] <= pd.to_datetime(end_date).tz_localize('UTC')) & (df['ADDRESS'].isin(signals))]

    # Aggregate the data
    if location == 'Intersection':
        df_agg = df_filtered.groupby(groupby).agg({'PED': 'sum', 'CITY': 'first', 'SIGNAL': 'first' , 'LAT': 'first' , 'LNG': 'first'}).reset_index()
    else:
        df_agg = df_filtered[df_filtered['P'] == int(filter_val)].groupby(groupby).agg({'PED': 'sum', 'P': 'first', 'CITY': 'first', 'SIGNAL': 'first' , 'LAT': 'first' , 'LNG': 'first'}).reset_index()

    df_agg['PED'] = df_agg['PED'].apply(lambda x: '{:,.0f}'.format(x))
    df_agg.rename(columns={'SIGNAL': 'Signal ID' , 'TIME1':'Timestamp' , 'PED':'Pedestrian' , 'CITY':'City' , 'P': 'Phase' , 'LAT':'Latitude' , 'LNG': 'Longtitude' }, inplace=True)
    # Select the columns to display in the output table
    if 'Phase' in df_agg.columns:
        df_agg = df_agg[['Signal ID', 'Timestamp', 'Phase', 'Pedestrian', 'City' , 'Latitude' , 'Longtitude']]
    else:
        df_agg = df_agg[['Signal ID', 'Timestamp', 'Pedestrian', 'City' , 'Latitude' , 'Longtitude']]

    df_agg.reset_index(drop=True, inplace=True)  # remove index column
    df_agg['Timestamp'] = df_agg['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    return df_agg

# Using treemap
@st.cache_resource
def make_pie_and_bar_chart(df, signals, start_date, end_date, location):

    if location == 'Intersection':
        filter_val = 'all'
        col = 'PED'
    else:
        filter_val = location.split()[-1]
        col = 'P'

    # Filter the dataframe by the selected signals and date range
    df_filtered = df[(df['TIME1'] >= pd.to_datetime(start_date).tz_localize('UTC')) & (df['TIME1'] <= pd.to_datetime(end_date).tz_localize('UTC')) & (df['ADDRESS'].isin(signals))]

    # Aggregate the data
    if location == 'Intersection':
        df_agg = df_filtered.groupby('ADDRESS').agg({'PED': 'sum','CITY': 'first'}).reset_index()
    else:
        df_agg = df_filtered[df_filtered['P'] == int(filter_val)].groupby('ADDRESS').agg({'PED': 'sum' , 'CITY': 'first'}).reset_index()

    # Aggregate the data by signal and sum the pedestrian counts
    df_agg1 = df_agg.groupby('ADDRESS').agg({'PED': 'sum'}).reset_index()
    df_agg2 = df_agg.groupby('CITY').agg({'PED': 'sum'}).reset_index()

    # Create the pie chart
    fig_pie = go.Figure(data=[go.Pie(labels=df_agg1['ADDRESS'], values=df_agg1['PED'])])
    fig_pie.update_layout(title='Pedestrian Activity by Signal', showlegend=False)

    # Create the bar chart
    fig_bar = go.Figure(data=[go.Bar(x=df_agg2['CITY'], y=df_agg1['PED'], showlegend=False)])
    fig_bar.update_layout(title='Pedestrian Activity by City', showlegend=False)

    # Create the treemap
    fig_treemap = px.treemap(df_agg2, path=['CITY'], values='PED', color='CITY')
    fig_treemap.update_layout(title='Pedestrian Activity by City', showlegend=False)

    # Combine the pie, bar, and treemap charts
    fig_combined = make_subplots(rows=1, cols=3, specs=[[{'type': 'domain'}, {'type': 'bar'}, {'type': 'treemap'}]])
    fig_combined.add_trace(fig_pie.data[0], row=1, col=1)
    fig_combined.add_trace(fig_bar.data[0], row=1, col=2)
    fig_combined.add_trace(fig_treemap.data[0], row=1, col=3)
    fig_combined.update_layout(showlegend=False)
    
    return fig_combined

@st.cache_resource
def make_bar_chart(df, signals, start_date, end_date, location):

    if location == 'Intersection':
        filter_val = 'all'
        col = 'PED'
    else:
        filter_val = location.split()[-1]
        col = 'P'

    # Filter the dataframe by the selected signals and date range
    df_filtered = df[(df['TIME1'] >= pd.to_datetime(start_date).tz_localize('UTC')) & (df['TIME1'] <= pd.to_datetime(end_date).tz_localize('UTC')) & (df['ADDRESS'].isin(signals))]

    # Aggregate the data
    if location == 'Intersection':
        df_agg = df_filtered.groupby('TIME1').agg({'PED': 'sum'}).reset_index()
    else:
        df_agg = df_filtered[df_filtered['P'] == int(filter_val)].groupby('TIME1').agg({'PED': 'sum'}).reset_index()

    
    # Convert the "TIME1" column to hour values
    df_agg['TIME1'] = pd.to_datetime(df_agg['TIME1']).dt.hour
    
    # Aggregate the data by hour and sum the pedestrian counts
    df_agg2 = df_agg.groupby('TIME1').agg({'PED': 'sum'}).reset_index()
    
    # Calculate the hourly average by dividing the sum of PED by number of days/hours
    df_agg2['PED'] = df_agg2['PED'] / 24
    
    # Create the bar chart
    fig_bar = go.Figure(data=[go.Bar(x=df_agg2['TIME1'], y=df_agg2['PED'], showlegend=False)])
    fig_bar.update_layout(xaxis_title='Hour', yaxis_title='Pedestrian Count', showlegend=False, xaxis=dict(tickmode='linear', dtick=1))
    fig_bar.update_yaxes(tickformat=".0f")
    
    return fig_bar

@st.cache_resource
def make_bar_chart2(df, signals, start_date, end_date, location):

    if location == 'Intersection':
        filter_val = 'all'
        col = 'PED'
    else:
        filter_val = location.split()[-1]
        col = 'P'

    # Filter the dataframe by the selected signals and date range
    df_filtered = df[(df['TIME1'] >= pd.to_datetime(start_date).tz_localize('UTC')) & (df['TIME1'] <= pd.to_datetime(end_date).tz_localize('UTC')) & (df['ADDRESS'].isin(signals))]

    # Aggregate the data
    if location == 'Intersection':
        df_agg = df_filtered.groupby('TIME1').agg({'PED': 'sum'}).reset_index()
    else:
        df_agg = df_filtered[df_filtered['P'] == int(filter_val)].groupby('TIME1').agg({'PED': 'sum'}).reset_index()

    # Convert the "TIME1" column to day of the week values (0-6, where Monday=0 and Sunday=6)
    df_agg['TIME1'] = pd.to_datetime(df_agg['TIME1']).dt.dayofweek

    # Aggregate the data by day of the week and calculate the average pedestrian counts
    df_agg2 = df_agg.groupby('TIME1').agg({'PED': 'sum'}).reset_index()

    # Calculate the hourly average by dividing the sum of PED by number of days/hours
    df_agg2['PED'] = df_agg2['PED'] / 7

    # Round the pedestrian count to two decimal places
    df_agg2['PED'] = df_agg2['PED'].round(2)

    # Create the bar chart
    fig_bar = go.Figure(data=[go.Bar(x=df_agg2['TIME1'], y=df_agg2['PED'], showlegend=False)])

    # Set the x-axis tick labels to the full names of the days of the week
    fig_bar.update_layout(xaxis_title='Day of the Week', yaxis_title='Pedestrian Count', showlegend=False)
    fig_bar.update_xaxes(tickmode='array', tickvals=[0, 1, 2, 3, 4, 5, 6], ticktext=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    fig_bar.update_yaxes(tickformat=".0f")

    return fig_bar

@st.cache_resource
def make_map(df, start_date, end_date, signals, aggregation_method, location_selected):
    # Filter by date, selected signals, and location
    mask = (df['TIME1'] >= pd.to_datetime(start_date).tz_localize('UTC')) & (df['TIME1'] < pd.to_datetime(end_date).tz_localize('UTC')) & (df['ADDRESS'].isin(signals))
    if location_selected == 'Intersection':
        mask &= df['P'] >= 0  # include all values of P for intersections
    else:
        if location_selected.startswith('Phase'):
            phase_num = int(location_selected.split()[1])
            mask &= df['P'] == phase_num
        else:
            mask &= df['ADDRESS'] == location_selected
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
    if location_selected == 'Intersection':
        df_agg = df_filtered.groupby(['LAT', 'LNG', 'ADDRESS', 'CITY' , 'SIGNAL']).agg({'PED': aggregation_function}).reset_index()
    else:
        df_agg = df_filtered.groupby(['LAT', 'LNG', 'ADDRESS', 'CITY', 'P' , 'SIGNAL']).agg({'PED': aggregation_function}).reset_index()

    px.set_mapbox_access_token("pk.eyJ1IjoiYmFzaGFzdmFyaSIsImEiOiJjbGVmaTdtMmIwcXkzM3Jxam9hb2pwZ3BoIn0.JmYank8e3bmQ7RmRiVdTIg")
    # Create the HeatmapLayer
    fig = px.density_mapbox(df_agg, lat='LAT', lon='LNG', z='PED', radius=50, zoom=7,
                            mapbox_style='light', opacity=0.8,
                            center=dict(lat=df['LAT'].mean(), lon=df['LNG'].mean()),
                            hover_data={'PED': True, 'SIGNAL':':d'},
                            labels={'PED': 'Pedestrian estimated', 'SIGNAL': 'Signal No.'},
                            color_continuous_scale='YlOrRd' , height = 500)
    fig.update_traces(hovertemplate='Pedestrian estimated: %{z}<br>Signal No.: %{customdata[0]}<extra></extra>')
    return fig



# Define the Streamlit app
def main():
    # Set the app title
    st.set_page_config(page_title='PEDAT Dashboard' , page_icon="📈" , layout="wide"  )
    # Add a title to the sidebar
    st.title("Monitoring pedestrian activity in Utah")
    st.markdown(text1)
    st.markdown(text2)

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
    
    tab1, tab2, tab3, tab4 = st.tabs(["🚦 **Select Signal**" ,"📈 **Chart**", "🗃 **Data**" , "🗺 **Map**"])
    tab1.markdown ("**Please select a Signal ID and location from the Map or the sidebar list**")
    st.sidebar.markdown(f'[**Singleton Transportation Lab**](https://engineering.usu.edu/cee/research/labs/patrick-singleton/index)')
    expander = st.sidebar.expander("**How to use**")
    expander.write('''
            There are Four tabs available in the above. 
            In these tabs, you can observe pedestrian activity based on selected aggregation methods at one or more signals (on a map or in a figure). 
            You can also download a CSV file containing the data that is creating the map or figure.
    ''')
    expander = st.sidebar.expander("**Notes**")
    expander.write('''
            As shown on this website in maps, figures, and tables, pedestrian activity is estimated pedestrian volume , 
            or an estimate of the total number of daily pedestrian crossings at an intersection. These estimated pedestrian 
            volumes are based on traffic signal pedestrian push-button data, obtained from high-resolution traffic signal controller 
            log data, originally obtained from UDOT's Automated Traffic Signal Performance Measures System (ATSPM) system. 
            Recent research conducted by the Singleton Transportation Lab at Utah State University has validated the use of 
            pedestrian traffic signal data as a fairly accurate estimate of pedestrian volumes in Utah.
    ''')
    # Load your DataFrame
    df3 = pd.read_gbq(sql_query, credentials=credentials)
    #df3 = pd.read_csv('mapdata.csv') 
    df3.rename(columns={'LNG': 'LON' }, inplace=True)
    a = ["All"] + df3['ADDRESS'].tolist()
    default_address = [a[1]]
    #icon_image = 'R-min.png'
    #icon_size = (15, 15)
    # Create the map object
    m = folium.Map(location=[39.419220, -111.950684], zoom_start=6 , tiles = 'https://api.mapbox.com/styles/v1/bashasvari/clhgx1yir00h901q1ecbt9165/tiles/256/{z}/{x}/{y}@2x?access_token=pk.eyJ1IjoiYmFzaGFzdmFyaSIsImEiOiJjbGVmaTdtMmIwcXkzM3Jxam9hb2pwZ3BoIn0.JmYank8e3bmQ7RmRiVdTIg' , attr='PEDAT map')
    Draw(
        export=False,
        filename="my_data.geojson",
        position="topleft",
        draw_options={
            'circle': False,
            'circlemarker':False,
            'polyline' : False
            
        }
        
    ).add_to(m)

    with tab1:
        # Create an empty list to hold the selected addresses
        address= []
        # Create a MarkerCluster layer
        marker_cluster = MarkerCluster(name='Markers', control=False)
        # Add a marker for each location in the DataFrame
        for index, row in df3.iterrows():
            folium.Marker(location=(row['LAT'], row['LON']), popup=folium.Popup(row['ADDRESS'], max_width=300,min_width=150),
            tooltip= row['ADDRESS']).add_to(marker_cluster)
            # Add the MarkerCluster layer to the map
            marker_cluster.add_to(m)
        # Render the map using st_folium
        s = st_folium(m, width='80%', height=400 , returned_objects=["last_object_clicked", "last_active_drawing"])
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

            selected_signals = st.sidebar.multiselect('**Signal ID and Location**', a , default = address)
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

            selected_signals = st.sidebar.multiselect('**Signal ID and Location**', a, default=selected_addresses)
        else:
            selected_signals = st.sidebar.multiselect('**Signal ID and Location**', a , default = address)
 

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

    # Add a subtitle to the sidebar
    df = client.query(sql_query2, job_config=job_config).to_dataframe()

    # Create a list of all unique values in the 'ADDRESS' column of the DataFrame
    all_addresses = df3['ADDRESS'].tolist()

    # Check if selected_signals is not empty
    if selected_signals:
        # If selected_signals is not empty, filter the list of all_addresses to include only the selected signals
        addresses_to_keep = set(selected_signals).intersection(set(all_addresses))
        all_addresses = list(addresses_to_keep)

    total_pedestrians = df['PED'].sum()
    num_signals = len(df['ADDRESS'].unique())

    # Format the metric values
    total_pedestrians_formatted = format_metric(total_pedestrians)
    num_signals_formatted = format_metric(num_signals)
    
    st.sidebar.subheader('Metrics')
    col1, col2 = st.sidebar.columns(2)
    # Display the metric boxes
    col1.metric("**Total Pedestrian**", total_pedestrians_formatted)
    col2.metric("**Selected Signals**", num_signals_formatted)

    st.markdown(
        """<style>
    div[class*="stColumn"] > label > div[data-testid="stMarkdownContainer"] > p {
        font-size: 32px;
    }
        </style>
        """, unsafe_allow_html=True)

    st.sidebar.subheader('Dashboard Parameters')
    form = st.sidebar.form("sidebar")

    # Create the locations list based on the modified list of addresses
    locations = ['Intersection'] + ['Phase ' + str(int(i)) for i in sorted(df[df['ADDRESS'].isin(all_addresses)]['P'].dropna().unique().tolist())]

    #locations = ['Intersection'] + ['Phase ' + str(int(i)) for i in sorted(df['P'].dropna().unique().tolist())]
    location_selected = form.selectbox('**Select approach**', options=locations)

    st.markdown(
        """<style>
    div[class*="stSelectbox"] > label > div[data-testid="stMarkdownContainer"] > p {
        font-size: 16px;
    }
        </style>
        """, unsafe_allow_html=True)
    

    # Add a slider for selecting the aggregation method
    aggregation_methods = ['Hourly', 'Daily', 'Weekly', 'Monthly', 'Yearly']
    aggregation_method_selected = form.selectbox('**Select aggregation method**', options=aggregation_methods)

    # Add a calendar widget to select a date range
    start_date = form.date_input('**Start date**', df['TIME1'].min())
    end_date = form.date_input('**End date**', df['TIME1'].max())

    # Convert the date objects to datetime objects
    start_datetime = datetime.combine(start_date, datetime.min.time())
    end_datetime = datetime.combine(end_date, datetime.max.time())
    
    st.markdown(
        """<style>
    div[class*="stSlider"] > label > div[data-testid="stMarkdownContainer"] > p {
        font-size: 16px;
    }
        </style>
        """, unsafe_allow_html=True)

    form.form_submit_button("Submit")
    
    if st.sidebar.button('Create Plot'):
        # Define dataframe for tabular purpose
        df2 = df.groupby([pd.Grouper(key='TIME1', freq='Y'), 'SIGNAL', 'LAT', 'LNG', 'CITY'])['PED'].sum().reset_index()
        df2 = df2.drop('TIME1', axis=1)
        # If "All" is selected, show all signals
        if "All" in selected_signals:
            selected_signals = df3['ADDRESS'].tolist()
        else:
            selected_signals = selected_signals or default_address
    


        tab2.subheader('Time series')
        # Make the time series plot
        fig = make_chart(df, selected_signals, start_datetime, end_datetime, aggregation_method_selected, location_selected, template='plotly_dark')
        tab2.plotly_chart(fig, theme='streamlit', use_container_width=True )

        tab2.subheader('Hourly Pedestrian Activity')
        tab2.plotly_chart(make_bar_chart(df, selected_signals, start_datetime, end_datetime, location_selected),theme='streamlit', use_container_width=True)

        tab2.subheader('Daily Pedestrian Activity')
        tab2.plotly_chart(make_bar_chart2(df, selected_signals, start_datetime, end_datetime, location_selected),theme='streamlit', use_container_width=True)

        tab2.subheader('Pedestrian Activity in relation to Signal and City')
        # Add a pie chart to show pedestrian activity by signal
        tab2.plotly_chart(make_pie_and_bar_chart(df, selected_signals, start_datetime, end_datetime, location_selected),theme='streamlit', use_container_width=True)

        tab4.subheader('Pedestrian Activity by Location')
        # Make the map
        #fig = make_map(df, start_date_selected, end_date_selected , selected_signals , location_selected, aggregation_method_selected)
        fig = make_map(df, start_datetime, end_datetime , selected_signals , aggregation_method_selected , location_selected)
        tab4.plotly_chart(fig , use_container_width=True)


        st.sidebar.markdown(
            """<style>
        div[class*="stDate"] > label > div[data-testid="stMarkdownContainer"] > p {
            font-size: 16px;
        }
            </style>
            """, unsafe_allow_html=True)


        # Filter your data based on the selected date range
        tab3.subheader('Pedestrian Activity Data')
        
        # Display the filtered data in a table
        
        table = make_table(df, selected_signals, start_datetime, end_datetime, aggregation_method_selected, location_selected)
        cc = table.to_csv(index=False)
        tab3.download_button(
            label="📥 Download",
            data=cc,
            file_name="FilteredData.csv",
            mime='text/csv',
        )
        # CSS to inject contained in a string
        hide_dataframe_row_index = """
                    <style>
                    .row_heading.level0 {display:none}
                    .blank {display:none}
                    </style>
                    """

        # Inject CSS with Markdown
        st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)
        
        tab3.dataframe(table , use_container_width=True)

        # Create pivot table
        tab3.subheader('Time series Data')
        pivot_table = table.pivot_table(values='Pedestrian', index='Timestamp', columns='Signal ID', aggfunc='sum')
        cv = pivot_table.to_csv(index=True)
        tab3.download_button(
            label="📥 Download",
            data=cv,
            file_name="TimeSeries.csv",
            mime='text/csv',
        )
        # Display pivot table
        tab3.dataframe(pivot_table , use_container_width=True)


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

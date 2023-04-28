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

# Load the data
df = pd.read_pickle("pediN2" + '.pkl', compression='gzip')

# Convert TIME2 to a datetime format
df['TIME2'] = pd.to_datetime(df['TIME2'])

# Define dataframe for tabular purpose
df2 = df.groupby([pd.Grouper(key='TIME2', freq='Y'), 'SIGNAL', 'LAT', 'LNG', 'CITY'])['PED'].sum().reset_index()
df2 = df2.drop('TIME2', axis=1)


# Define the title
title = 'Pedestrian Activity Data Visualization Dashboard'
text1 = 'This website provides data and visualizations of pedestrian activity (and changes in pedestrian activity) at various (signalized) intersections throughout Utah. We are currently showing all locations with available signal data throughout Utah.'
text2 = 'Data are derived from pedestrian push-button presses at traffic signals, taken from the Utah Department of Transportation\'s [Automated Traffic Signal Performance Measures System](https://udottraffic.utah.gov/atspm/) website. We hope that this information is useful for public agencies to track changes in walking activity at different locations.'

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

#@st.cache_data(experimental_allow_widgets=True)
def make_chart(df, signals, start_date, end_date, aggregation_method, location, template='plotly'):
    if aggregation_method == 'Hourly':
        groupby = ['ADDRESS', pd.Grouper(key='TIME2', freq='1H')]
    elif aggregation_method == 'Daily':
        groupby = ['ADDRESS', pd.Grouper(key='TIME2', freq='1D')]
    elif aggregation_method == 'Weekly':
        groupby = ['ADDRESS', pd.Grouper(key='TIME2', freq='1W')]
    elif aggregation_method == 'Monthly':
        groupby = ['ADDRESS', pd.Grouper(key='TIME2', freq='1M')]
    elif aggregation_method == 'Yearly':
        groupby = ['ADDRESS', pd.Grouper(key='TIME2', freq='1Y')]

    if location == 'Intersection':
        filter_val = 'all'
        col = 'PED'
    else:
        filter_val = location.split()[-1]
        col = 'P'

    # Filter the dataframe by the selected signals and date range
    df_filtered = df[(df['TIME2'] >= start_date) & (df['TIME2'] <= end_date) & (df['ADDRESS'].isin(signals))]

    # Aggregate the data
    if location == 'Intersection':
        df_agg = df_filtered.groupby(groupby).agg({'PED': 'sum'}).reset_index()
    else:
        df_agg = df_filtered[df_filtered['P'] == int(filter_val)].groupby(groupby).agg({'PED': 'sum'}).reset_index()

    # Create the line chart
    if aggregation_method == 'Hourly':
        x_axis_label = 'Time'
        fig = px.line(df_agg, x='TIME2', y='PED' ,  color='ADDRESS', template=template)
    else:
        x_axis_label = 'Date'
        fig = px.line(df_agg, x='TIME2', y='PED', color='ADDRESS', template=template)

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

# Using treemap
def make_pie_and_bar_chart(df, signals, start_date, end_date):
    # Filter the dataframe by the selected signals, date range, and day type
    df_filtered = df[(df['TIME2'] >= start_date) & (df['TIME2'] < end_date) & (df['ADDRESS'].isin(signals))]

    # Aggregate the data by signal and sum the pedestrian counts
    df_agg = df_filtered.groupby('ADDRESS').agg({'PED': 'sum'}).reset_index()
    df_agg1 = df_filtered.groupby('CITY').agg({'PED': 'sum'}).reset_index()

    # Create the pie chart
    fig_pie = go.Figure(data=[go.Pie(labels=df_agg['ADDRESS'], values=df_agg['PED'])])
    fig_pie.update_layout(title='Pedestrian Activity by Signal', showlegend=False)

    # Create the bar chart
    fig_bar = go.Figure(data=[go.Bar(x=df_agg1['CITY'], y=df_agg['PED'], showlegend=False)])
    fig_bar.update_layout(title='Pedestrian Activity by City', showlegend=False)

    # Create the treemap
    fig_treemap = px.treemap(df_agg1, path=['CITY'], values='PED', color='CITY')
    fig_treemap.update_layout(title='Pedestrian Activity by City', showlegend=False)

    # Combine the pie, bar, and treemap charts
    fig_combined = make_subplots(rows=1, cols=3, specs=[[{'type': 'domain'}, {'type': 'bar'}, {'type': 'treemap'}]])
    fig_combined.add_trace(fig_pie.data[0], row=1, col=1)
    fig_combined.add_trace(fig_bar.data[0], row=1, col=2)
    fig_combined.add_trace(fig_treemap.data[0], row=1, col=3)
    fig_combined.update_layout(showlegend=False)
    
    return fig_combined


def make_bar_chart(df, signals, start_date, end_date):
    # Filter the dataframe by the selected signals, date range, and day type
    df_filtered = df[(df['TIME2'] >= start_date) & (df['TIME2'] < end_date) & (df['ADDRESS'].isin(signals))]
    
    # Convert the "TIME1" column to hour values
    df_filtered['TIME2'] = pd.to_datetime(df_filtered['TIME2']).dt.hour
    
    # Aggregate the data by hour and sum the pedestrian counts
    df_agg = df_filtered.groupby('TIME2').agg({'PED': 'sum'}).reset_index()
    
    # Create the bar chart
    fig_bar = go.Figure(data=[go.Bar(x=df_agg['TIME2'], y=df_agg['PED'], showlegend=False)])
    fig_bar.update_layout(xaxis_title='Hour', yaxis_title='Pedestrian Count', showlegend=False , xaxis=dict(tickmode='linear', dtick=1))
    
    return fig_bar

def make_map(df, start_date, end_date, signals, aggregation_method):
    # Filter by date and selected signals
    mask = (df['TIME2'] >= start_date) & (df['TIME2'] < end_date) & (df['ADDRESS'].isin(signals))
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
        zoom=10,
        #min_zoom=5,
        #max_zoom=15,
        #pitch=10.5,
        #bearing=-27.36
    )
    fig = pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v11',
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
        df_filtered2 = df_filtered2[df_filtered2['ADDRESS'].isin(selected_signals)]
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
    df_aggregated = df_filtered2.groupby([pd.Grouper(key='TIME2', freq=freq), 'ADDRESS', 'CITY', 'P']).agg(agg_dict)
    df_aggregated.reset_index(drop=True,inplace=True)

    #filename = f"pedestrian_counts_{location_selected}_{aggregation_method_selected}.csv"
    #st.sidebar.success(f"{filename} successfully saved!")
    return df_aggregated.to_csv(index=False)

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

    address = ["All"] + df['ADDRESS'].unique().tolist()
    default_address = [address[1]]
    #st.multiselect('Address' ,address , default=default_address)
    selected_signals = st.multiselect('**Signal ID and Location**' , address)
    # Add a subtitle to the sidebar
    st.sidebar.markdown(f'[**Singleton Transportation Lab**](https://engineering.usu.edu/cee/research/labs/patrick-singleton/index)')
    font_css = """
    <style>
    button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] > p {
    font-size: 20px;
    }
    </style>
    """
    st.write(font_css, unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["📈 **Chart**", "🗃 **Data**" , "🗺 **Map**"])

    st.markdown(
        """<style>
    div[class*="stMultiSelect"] > label > div[data-testid="stMarkdownContainer"] > p {
        font-size: 16px;
    }
        </style>
        """, unsafe_allow_html=True)
 
    # If "All" is selected, show all signals
    if "All" in selected_signals:
        selected_signals = df['ADDRESS'].unique().tolist()
    else:
        selected_signals = selected_signals or default_address
    

    # Add a slider for selecting the location
    locations = ['Intersection'] + ['Phase ' + str(int(i)) for i in sorted(df['P'].dropna().unique().tolist())]
    location_selected = st.sidebar.selectbox('**Select approach**', options=locations)
  
    st.markdown(
        """<style>
    div[class*="stSelectbox"] > label > div[data-testid="stMarkdownContainer"] > p {
        font-size: 16px;
    }
        </style>
        """, unsafe_allow_html=True)
    

    # Add a slider for selecting the aggregation method
    aggregation_methods = ['Hourly', 'Daily', 'Weekly', 'Monthly', 'Yearly']
    aggregation_method_selected = st.sidebar.selectbox('**Select aggregation method**', options=aggregation_methods)

    # Add a time range slider to select the date range
    start_date = pd.to_datetime('2017-07-01')
    end_date = pd.to_datetime('2018-07-01')
    start_date_selected, end_date_selected = st.sidebar.select_slider('**Select a date range**', options=pd.date_range(start=start_date, end=end_date, freq='D'), value=(start_date, end_date) , format_func=lambda x: x.strftime('%Y-%m-%d'))
    
    st.markdown(
        """<style>
    div[class*="stSlider"] > label > div[data-testid="stMarkdownContainer"] > p {
        font-size: 16px;
    }
        </style>
        """, unsafe_allow_html=True)
    # Calculate the total number of pedestrians in the selected date range and signals
    df_filtered = df[(df['TIME2'] >= start_date_selected) & (df['TIME2'] < end_date_selected)]
    if "All" not in selected_signals:
        df_filtered = df_filtered[df_filtered['ADDRESS'].isin(selected_signals)]
    total_pedestrians = df_filtered['PED'].sum()
    num_signals = len(df['ADDRESS'].unique())
    num_cities = len(df['CITY'].unique())

    # Format the metric values
    total_pedestrians_formatted = format_metric(total_pedestrians)
    num_signals_formatted = format_metric(num_signals)
    num_cities_formatted = format_metric(num_cities)

    tab1.subheader('Metrics')
    # Display the metric boxes
    col1, col2, col3 = tab1.columns(3)
    col1.metric("Total Pedestrian Predicted", total_pedestrians_formatted)
    col2.metric("Number of Signals", num_signals_formatted)
    col3.metric("Number of Cities", num_cities_formatted)

    tab1.subheader('Time series')
    # Make the time series plot
    fig = make_chart(df, selected_signals, start_date_selected, end_date_selected, aggregation_method_selected, location_selected, template='plotly_dark')
    tab1.plotly_chart(fig, theme='streamlit', use_container_width=True )

    tab1.subheader('Hourly Pedestrian Activity')
    tab1.plotly_chart(make_bar_chart(df, selected_signals, start_date_selected, end_date_selected),theme='streamlit', use_container_width=True)

    tab1.subheader('Pedestrian Activity in relation to Signal and City')
    # Add a pie chart to show pedestrian activity by signal
    tab1.plotly_chart(make_pie_and_bar_chart(df, selected_signals, start_date_selected, end_date_selected),theme='streamlit', use_container_width=True)

    tab3.subheader('Pedestrian Activity by Location')
    # Make the map
    #fig = make_map(df, start_date_selected, end_date_selected , selected_signals , location_selected, aggregation_method_selected)
    fig = make_map(df, start_date_selected, end_date_selected , selected_signals , aggregation_method_selected)
    tab3.pydeck_chart(fig)


    # Add a calendar widget to select a date range
    start_date = st.sidebar.date_input('**Start date**', df['TIME2'].min())
    end_date = st.sidebar.date_input('**End date**', df['TIME2'].max())

    st.sidebar.markdown(
        """<style>
    div[class*="stDate"] > label > div[data-testid="stMarkdownContainer"] > p {
        font-size: 16px;
    }
        </style>
        """, unsafe_allow_html=True)

    # Convert the date objects to datetime objects
    start_datetime = datetime.combine(start_date, datetime.min.time())
    end_datetime = datetime.combine(end_date, datetime.max.time())

    # Filter your data based on the selected date range
    tab2.subheader('Pedestrian Activity Data')
    selected_signal_array = np.isin(df['ADDRESS'], selected_signals)
    filtered_data = df.loc[(df['TIME2'] >= start_datetime) & (df['TIME2'] <= end_datetime) & selected_signal_array]

    # Display the filtered data in a table
    tab2.dataframe(filtered_data , use_container_width=True)

    # Create pivot table
    tab2.subheader('Time series Data')
    pivot_table = filtered_data.pivot_table(values='PED', index='TIME2', columns='SIGNAL', aggfunc='sum')

    # Display pivot table
    tab2.dataframe(pivot_table , use_container_width=True)

    expander = st.expander("**How to use**")
    expander.write('''
            There are three tabs available in the above. 
            In these tabs, you can observe pedestrian activity based on selected aggregation methods at one or more signals (on a map or in a figure). 
            You can also download a CSV file containing the data that is creating the map or figure.
    ''')
    expander = st.expander("**Notes**")
    expander.write('''
            As shown on this website in maps, figures, and tables, pedestrian activity is estimated pedestrian volume , 
            or an estimate of the total number of daily pedestrian crossings at an intersection. These estimated pedestrian 
            volumes are based on traffic signal pedestrian push-button data, obtained from high-resolution traffic signal controller 
            log data, originally obtained from UDOT's Automated Traffic Signal Performance Measures System (ATSPM) system. 
            Recent research conducted by the Singleton Transportation Lab at Utah State University has validated the use of 
            pedestrian traffic signal data as a fairly accurate estimate of pedestrian volumes in Utah.
    ''')
    st.markdown(
        """<style>
    div[class*="stExpander"] > label > div[data-testid="stMarkdownContainer"] > p {
        font-size: 16px;
    }
        </style>
        """, unsafe_allow_html=True)
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

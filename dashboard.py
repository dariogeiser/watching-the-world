import streamlit as st
import pandas as pd
import leafmap.foliumap as leafmap
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from pathlib import Path


def get_country_bbox(country_name):
    """
    Function to get bounding box for a given country
    Params:
    - country_name (str): The name of the country for which to get the bounding box
    Returns:
    - list: A list containing the bounding box coordinates [south, north, west, east] or None if not found
    """
    geolocator = Nominatim(user_agent="watching-the-world")
    try:
        if country_name in ["United States"]:
            bbox = [24.396308, 49.384358, -125.0, -66.93457]
        else:
            location = geolocator.geocode(country_name)
            if location:
                bbox = location.raw['boundingbox']
            else:
                return None
        return [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
    except GeocoderTimedOut:
        return [-90.0, 90.0, -180.0, 180.0]


def navigate_to(route):
    """
    Function to navigate to a specified route
    Params:
    - route (str): The route to navigate to
    """
    st.query_params.route = route


def load_data(cams_file_path, results_file_path):
    """
    Function to load camera and results data from CSV files
    Params:
    - cams_file_path (str): Path to the CSV file containing camera data
    - results_file_path (str): Path to the CSV file containing results data
    Returns:
    - tuple: A tuple containing two DataFrames (cams_df, results_df)
    """
    cams_df = pd.read_csv(cams_file_path)
    results_df = pd.read_csv(results_file_path)
    return cams_df, results_df


def filter_data(results_df, cams_df, color_selection, object_selection, country_selection):
    """
    Function to filter data based on user selections
    Params:
    - results_df (DataFrame): DataFrame containing results data
    - cams_df (DataFrame): DataFrame containing camera data
    - color_selection (str): Selected color for filtering
    - object_selection (str): Selected object for filtering
    - country_selection (str): Selected country for filtering
    Returns:
    - DataFrame: Filtered DataFrame containing camera data
    """
    filtered_df = results_df.copy()

    if color_selection != 'All':
        filtered_df = filtered_df[filtered_df[color_selection] > 0]

    if object_selection != 'All':
        filtered_df = filtered_df[filtered_df[object_selection] > 0]

    filtered_cam_ids = filtered_df['camId'].unique()
    filtered_cams_df = cams_df[cams_df['camId'].isin(filtered_cam_ids)]

    if country_selection != 'All':
        filtered_cams_df = filtered_cams_df[filtered_cams_df['country_name'] == country_selection]

    filtered_cams_df = filtered_cams_df.dropna(subset=['latitude', 'longitude'])
    return filtered_cams_df


def prepare_heatmap_data(filtered_cams_df):
    """
    Function to prepare heatmap data
    Params:
    - filtered_cams_df (DataFrame): Filtered DataFrame containing camera data
    Returns:
    - list: A list of coordinates for the heatmap
    """
    return [[row['latitude'], row['longitude']] for index, row in filtered_cams_df.iterrows()]


def create_map(filtered_cams_df, heat_data, country_selection, gradient):
    """
    Function to create the map
    Params:
    - filtered_cams_df (DataFrame): Filtered DataFrame containing camera data
    - heat_data (list): List of coordinates for the heatmap
    - country_selection (str): Selected country for filtering
    - gradient (dict): Custom gradient for heatmap
    Returns:
    - Map: A Folium map object with the specified parameters
    """
    map_center = [20, 0]
    zoom_level = 2

    if country_selection != 'All':
        bbox = get_country_bbox(country_selection)
        if bbox:
            map_center = [(bbox[0] + bbox[1]) / 2, (bbox[2] + bbox[3]) / 2]
            m = leafmap.Map(center=map_center, zoom=2, tiles="openstreetmap",
                            max_bounds=True, max_bounds_viscosity=1.0,
                            attribution="&copy; OpenStreetMap contributors")
            m.fit_bounds([[bbox[0], bbox[2]], [bbox[1], bbox[3]]])
        else:
            m = leafmap.Map(center=map_center, zoom=zoom_level, tiles="openstreetmap",
                            max_bounds=True, max_bounds_viscosity=1.0,
                            attribution="&copy; OpenStreetMap contributors")
    else:
        m = leafmap.Map(center=map_center, zoom=zoom_level, tiles="openstreetmap",
                        max_bounds=True, max_bounds_viscosity=1.0,
                        attribution="&copy; OpenStreetMap contributors")

    marker_enabled = len(filtered_cams_df) <= 5000

    if marker_enabled:
        view_option = st.sidebar.radio(
            "Select View Mode",
            ["Heatmap", "Markers"],
            index=0,
        )
    else:
        view_option = st.sidebar.radio(
            "Select View Mode",
            ["Heatmap", "Markers"],
            index=0,
            help="Markers view is disabled due to the high number of cameras (> 5000)",
            disabled=True
        )

    if view_option == "Heatmap" and heat_data:
        m.add_heatmap(
            heat_data,
            radius=10,
            blur=15,
            gradient=gradient
        )
    elif view_option == "Markers" and heat_data:
        for point in heat_data:
            m.add_marker(location=point)

    return m


def map_idx_to_names():
    """
    Function to map idx columns to more descriptive names
    Returns:
    - dict: A dictionary mapping idx columns to more descriptive names
    """
    return {
        'idxGreen': 'Green',
        'idxBlue': 'Blue',
        'idxRed': 'Red',
        'idxBlack': 'Black',
        'idxWhite': 'White',
        'idxPerson': 'Person',
        'idxTransport': 'Transport',
        'idxCars': 'Cars',
        'idxBikes': 'Bikes',
        'idxMotorcycles': 'Motorcycles',
        'idxTrains': 'Trains',
        'idxBoats': 'Boats',
        'idxPlanes': 'Planes',
        'idxAnimal': 'Animal',
        'idxBeaches': 'Beaches',
        'idxMountains': 'Mountains',
        'idxNoon': 'Noon',
        'idxMidnight': 'Midnight'
    }

st.set_page_config(layout="wide")

st.title("Watching the World")

current_route = st.query_params
navigate_to("home")

st.sidebar.title("Navigation")
if st.sidebar.button("Home"):
    navigate_to("home")
if st.sidebar.button("Report"):
    navigate_to("report")

if current_route.route == "home":

    cams_file_path = 'data/cams.csv'
    results_file_path = 'data/results.csv'

    cams_df, results_df = load_data(cams_file_path, results_file_path)

    st.sidebar.title("Filters")

    idx_to_name = map_idx_to_names()
    color_options = ['All'] + [idx_to_name[col] for col in results_df.columns if
                               col.startswith('idx') and col in ['idxGreen', 'idxBlue', 'idxRed', 'idxBlack',
                                                                 'idxWhite']]
    object_options = ['All'] + [idx_to_name[col] for col in results_df.columns if
                                col.startswith('idx') and col not in ['idxGreen', 'idxBlue', 'idxRed', 'idxBlack',
                                                                      'idxWhite']]

    valid_countries = cams_df[cams_df['country_name'].notna()]['country_name'].unique()

    country_selection = st.sidebar.selectbox("Country", ['All'] + list(valid_countries), key="country_selection")
    color_selection_name = st.sidebar.selectbox("Colors", color_options,
                                                key="color_selection")
    object_selection_name = st.sidebar.selectbox("Objects", object_options,
                                                 key="object_selection")

    color_selection = [key for key, value in idx_to_name.items() if value == color_selection_name][
        0] if color_selection_name != 'All' else 'All'
    object_selection = [key for key, value in idx_to_name.items() if value == object_selection_name][
        0] if object_selection_name != 'All' else 'All'

    filtered_cams_df = filter_data(results_df, cams_df, color_selection, object_selection, country_selection)

    heat_data = prepare_heatmap_data(filtered_cams_df)

    gradient = {
        0.05: '#E8F8F5',
        0.10: '#D1F2EB',
        0.20: '#A3E4D7',
        0.30: '#76D7C4',
        0.40: '#48C9B0',
        0.50: '#1ABC9C',
        0.60: '#17A589',
        0.70: '#148F77',
        0.80: '#117A65',
        0.90: '#0E6655',
        1.00: '#0B5345'
    }

    m = create_map(filtered_cams_df, heat_data, country_selection, gradient)

    col1, col2 = st.columns([3, 1])

    with col1:
        m.to_streamlit(height=700)

    with col2:
        st.markdown("###### Camera Counts by Country")
        country_counts = filtered_cams_df['country_name'].value_counts()
        country_counts = country_counts[:15]

        for country, count in country_counts.items():
            st.write(f"{country}: {count} cameras")

        if len(filtered_cams_df['country_name'].value_counts()) > 15:
            st.write("...")

elif current_route.route == "report":
    html_file_path = Path("notebook/notebook.html")
    if html_file_path.exists():
        with open(html_file_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=9000, width=1200, scrolling=True)
    else:
        st.error("Report not found.")

import streamlit as st
import pandas as pd
import numpy as np
from streamlit_folium import st_folium
import folium
import os
from folium.plugins import Fullscreen

# Get API key from environment variable
try:
    MAPTILER_API_KEY = st.secrets["maptiler"]["MAPTILER_API_KEY"]
except Exception as e:
    MAPTILER_API_KEY = "PLACEHOLDER_KEY"
    st.warning(f"Could not load MapTiler API key from secrets: {e}")

# Map types and overlays
MAP_TYPES = {
    "Standard": f"https://api.maptiler.com/maps/dataviz/256/{{z}}/{{x}}/{{y}}.png?key={MAPTILER_API_KEY}",
    "Landscape": f"https://api.maptiler.com/maps/landscape/256/{{z}}/{{x}}/{{y}}.png?key={MAPTILER_API_KEY}",
    "Satellite": f"https://api.maptiler.com/maps/satellite/256/{{z}}/{{x}}/{{y}}.jpg?key={MAPTILER_API_KEY}",
    "Great Britain, Ordnance Survey, 1900s": {
        "base": f"https://api.maptiler.com/maps/dataviz/256/{{z}}/{{x}}/{{y}}.png?key={MAPTILER_API_KEY}",
        "overlay": f"https://api.maptiler.com/tiles/uk-osgb1888/{{z}}/{{x}}/{{y}}?key={MAPTILER_API_KEY}"
    },
    "Great Britain, Ordnance Survey, 1'' (1885-1903)": {
        "base": f"https://api.maptiler.com/maps/dataviz/256/{{z}}/{{x}}/{{y}}.png?key={MAPTILER_API_KEY}",
        "overlay": f"https://api.maptiler.com/tiles/uk-osgb63k1885/{{z}}/{{x}}/{{y}}.png?key={MAPTILER_API_KEY}"
    },
    "Great Britain, Ordnance Survey, 6'' (1888-1913)": {
        "base": f"https://api.maptiler.com/maps/dataviz/256/{{z}}/{{x}}/{{y}}.png?key={MAPTILER_API_KEY}",
        "overlay": f"https://api.maptiler.com/tiles/uk-osgb10k1888/{{z}}/{{x}}/{{y}}.jpg?key={MAPTILER_API_KEY}"
    }
}

MAPTILER_ATTR = "MapTiler"

# Circle scale config
SCALE_MULTIPLIER = 8  # adjust overall size
MIN_RADIUS = 1  # minimum circle radius
MAX_RADIUS = 20  # maximum circle radius

# Zoom level thresholds for showing location names
# These values determine when location names appear based on circle size and zoom level
ZOOM_THRESHOLDS = {
    6: 15,  # At zoom level 6, only show names for circles with radius > 15
    7: 13,  # At zoom level 7, show names for circles with radius > 10
    8: 10,  # At zoom level 8, show names for circles with radius > 5
    9: 8,  # At zoom level 9, show names for circles with radius > 3
    10: 0  # At zoom level 10 and above, show all names
}

# === PAGE CONFIG ===
st.set_page_config(layout="wide")

# Display menu.png at the top center of the screen
col_left, col_center, col_right = st.columns([0.1, 2, 0.1])
with col_center:
    st.image("menu.png", use_container_width=True)


# === DATA LOAD ===
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data():
    try:
        df = pd.read_excel("output.xlsx")
        # Pre-process to avoid doing this repeatedly
        df = df.fillna({
            "Subject": "no",
            "Recorded": "no",
            "From": "no",
            "Genre": "Unknown",
            "LocationGD": ""  # Ensure LocationGD has empty string instead of NaN
        })
        return df
    except FileNotFoundError:
        st.error("Error: 'output.xlsx' file not found.")
        return pd.DataFrame()


try:
    df = load_data()
except FileNotFoundError:
    st.error("Error: 'output.xlsx' file not found. Please make sure the file is in the same directory as this script.")
    st.stop()

# Check if required columns exist
required_columns = ["Latitude", "Longitude", "Type", "Subject", "Recorded", "From", "Date", "Genre", "LocationEN"]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    st.error(f"Error: The following required columns are missing from the dataset: {', '.join(missing_columns)}")
    st.stop()

# Fill NaN values to avoid filtering errors
df = df.fillna({
    "Subject": "no",
    "Recorded": "no",
    "From": "no",
    "Genre": "Unknown",
    "LocationGD": ""  # Ensure LocationGD has empty string instead of NaN
})

# === SESSION STATE ===
if "show_filters" not in st.session_state:
    st.session_state.show_filters = True
if "show_relevant_items" not in st.session_state:
    st.session_state.show_relevant_items = True
if "map_zoom" not in st.session_state:
    st.session_state.map_zoom = 6  # Default zoom level

# === FILTER PANEL ===
people_types = ["All", "With songs", "With stories", "With information", "With verses", "With music",
                "With radio programmes"]

with st.expander("🔍 Filters", expanded=st.session_state.show_filters):
    filter_col1, filter_col_mid, filter_col2 = st.columns([2,0.5, 2])

    with filter_col1:
        tracks_title_col, show_tracks_col = st.columns([4, 2])  # Adjust ratio as needed
        with tracks_title_col:
            st.markdown("### 🎵 Tracks")
        with show_tracks_col:
            show_tracks = st.toggle("Show Tracks", value=True)

        search_tracks = st.text_input("Full text search in tracks",
                                      help='''Searches in track titles and descriptions/summaries.''')
        genre = st.selectbox("Genre", options=["All"] + sorted(df["Genre"].dropna().unique().tolist()), index=0)
        language = st.selectbox("Language", options=["Any", "English", "Gaelic", "Scots", "Other"], index=0)
        collection = st.selectbox("Collection", options=["All", "SoSS", "BBC", "Canna"], index=0)
        subject = st.toggle("Subject location", value=True, help="Shows/Hides locations mentioned in tracks")
        recorded = st.toggle("Recorded", value=True, help="Shows/Hides locations where tracks have been recorded")
        transcribed = st.toggle("Transcribed only", value=False,
                                help="Shows/Hides locations where tracks have no transcriptions")
        tracks_date_range = st.slider("Date recorded", min_value=1935, max_value=2025,
                                      value=(1935, 2025))
    with filter_col_mid:
        st.markdown(
            """
            <style>
            div[data-testid="stRadio"] {
                display: flex;
                justify-content: center;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        operator = st.radio("", ["OR", "AND"], index=0, help="<TBD>")

    with filter_col2:
        people_title_col, show_people_col = st.columns([4, 2])  # Adjust ratio as needed
        with people_title_col:
            st.markdown("### 👤 People")
        with show_people_col:
            show_people = st.toggle("Show People", value=True)

        search_people = st.text_input("Full text search in people",
                                      help='''Searches in fieldworkers\' or contributors\' names and patronymics.''')
        type2 = st.selectbox("Type", options=people_types, index=0)
        from_toggle = True
        #from_toggle = st.toggle("From", value=True, help="Shows/Hides native areas of fieldworkers and contributors")
        people_date_range = st.slider("Date born", min_value=1850, max_value=2025,
                                      value=(1850, 2025))


    st.divider()

    map_filters_left_col, map_filters_right_col = st.columns([2, 2])
    with map_filters_left_col:
        map_type = st.selectbox("Map type", options=list(MAP_TYPES.keys()), index=0)

    with map_filters_right_col:
        location_names = ["None", "English", "Gàidhlig", "English/Gàidhlig"]
        location_filter = st.selectbox("Location names", options=location_names, index=0)

    st.divider()

    # Add the buttons in a row
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        st.button("🔍 Apply Filters", type="primary")
    with filter_col2:
        st.button("↺ Reset Filters")

# === FILTERING FUNCTION ===
def apply_filters(df):
    df_filtered = df.copy()

    # Track filters
    if not show_tracks:
        df_filtered = df_filtered[df_filtered["Type"] != "track"]
    else:
        # Apply track-specific filters
        tracks_mask = df_filtered["Type"] == "track"

        # Date filter for tracks
        tracks_mask &= (df_filtered["Date"] >= tracks_date_range[0]) & (df_filtered["Date"] <= tracks_date_range[1])

        if genre != "All":
            tracks_mask &= df_filtered["Genre"] == genre

        if language != "Any" and "Language" in df_filtered.columns:
            tracks_mask &= df_filtered["Language"] == language

        if collection != "All":
            if "Collection" in df_filtered.columns:
                tracks_mask &= df_filtered["Collection"] == collection

        if search_tracks:
            # Assuming there's a column like "Title" or "Description" to search in
            # Adjust column names as needed
            search_condition = False
            for col in ["Title", "Description", "Summary"]:
                if col in df_filtered.columns:
                    search_condition |= df_filtered[col].fillna("").str.contains(search_tracks, case=False)
            tracks_mask &= search_condition

        if not subject:
            tracks_mask &= ~((df_filtered["Type"] == "track") & (df_filtered["Subject"].str.lower() == "yes"))

        if not recorded:
            tracks_mask &= ~((df_filtered["Type"] == "track") & (df_filtered["Recorded"].str.lower() == "yes"))

        if transcribed and "Transcribed" in df_filtered.columns:
            tracks_mask &= df_filtered["Transcribed"] == True

        # Keep non-tracks and filtered tracks
        df_filtered = df_filtered[(df_filtered["Type"] != "track") | tracks_mask]

    # People filters
    if not show_people:
        df_filtered = df_filtered[df_filtered["Type"] != "person"]
    else:
        # Apply people-specific filters
        people_mask = df_filtered["Type"] == "person"

        # Date filter for people
        people_mask &= (df_filtered["Date"] >= people_date_range[0]) & (df_filtered["Date"] <= people_date_range[1])

        if type2 != "All":
            # Assuming there's a "PersonType" column or similar
            if "PersonType" in df_filtered.columns:
                people_mask &= df_filtered["PersonType"] == type2.replace("With ", "")

        if search_people:
            # Assuming there are columns like "Name" or "Patronymic" to search in
            # Adjust column names as needed
            search_condition = False
            for col in ["Name", "Patronymic", "FullName"]:
                if col in df_filtered.columns:
                    search_condition |= df_filtered[col].fillna("").str.contains(search_people, case=False)
            people_mask &= search_condition

        if not from_toggle:
            people_mask &= ~((df_filtered["Type"] == "person") & (df_filtered["From"].str.lower() == "yes"))

        # Keep non-people and filtered people
        df_filtered = df_filtered[(df_filtered["Type"] != "person") | people_mask]

    return df_filtered


df_filtered = apply_filters(df)

# === RIGHT-ALIGNED TOGGLE CONTROLLER ===
toggle_col_left, toggle_col_right = st.columns([5, 1])
with toggle_col_right:
    toggle_value = st.toggle("Show/Hide results", value=st.session_state.show_relevant_items,
                             key="toggle_relevant_items")

# === UPDATE SESSION STATE ONLY IF TOGGLE VALUE CHANGED ===
if toggle_value != st.session_state.show_relevant_items:
    st.session_state.show_relevant_items = toggle_value
    st.rerun()

if st.session_state.show_relevant_items:
    col1, col2 = st.columns([4, 1])
else:
    col1, col2 = st.columns([1, 0.0001])


# === COMPUTE SCALED RADIUS FUNCTION ===
def compute_scaled_radius(value):
    raw = np.log10(value + 1) * SCALE_MULTIPLIER
    return min(max(raw, MIN_RADIUS), MAX_RADIUS)


# === GET DISPLAY NAME BASED ON LOCATION FILTER ===
def get_display_name(row, location_filter):
    en_name = row.get("LocationEN", "")
    gd_name = row.get("LocationGD", "")

    if location_filter == "None":
        return ""
    elif location_filter == "English":
        return en_name
    elif location_filter == "Gàidhlig":
        return gd_name if gd_name else en_name  # Fallback to English if Gaelic is empty
    elif location_filter == "English/Gàidhlig":
        if gd_name:
            return f"{en_name}/{gd_name}"
        else:
            return en_name
    return en_name  # Default fallback


# === MAP ===
with col1:
    # Create base map with appropriate base map type
    selected_map_type = map_type

    # Determine the map tile URL based on selection
    if isinstance(MAP_TYPES[selected_map_type], dict):
        # For historical maps with overlay
        base_tiles = MAP_TYPES[selected_map_type]["base"]
    else:
        # For regular maps (Standard, Landscape, Satellite)
        base_tiles = MAP_TYPES[selected_map_type]

    # Get current zoom level from session state or use default
    current_zoom = st.session_state.get("map_zoom", 6)

    m = folium.Map(
        location=[df_filtered["Latitude"].mean() if not df_filtered.empty else 56.4907,
                  df_filtered["Longitude"].mean() if not df_filtered.empty else -4.2026],
        # Default to center of Scotland if empty
        zoom_start=current_zoom,
        tiles=base_tiles,
        attr=MAPTILER_ATTR,
        height_restriction=800
    )

    # Add Fullscreen button to map
    Fullscreen(
        position='topleft',
        title='Expand map to fullscreen',
        title_cancel='Exit fullscreen',
        force_separate_button=True
    ).add_to(m)

    # === DEFINE LAYER SPECS ===
    layer_specs = []

    if show_tracks:
        if subject:
            layer_specs.append({
                "name": "Subject tracks",
                "filter": (df_filtered["Type"] == "track") & (df_filtered["Subject"].str.lower() == "yes"),
                "color": "black"
            })
        if recorded:
            layer_specs.append({
                "name": "Recorded tracks",
                "filter": (df_filtered["Type"] == "track") & (df_filtered["Recorded"].str.lower() == "yes"),
                "color": "green"
            })

    if show_people:
        if from_toggle:
            layer_specs.append({
                "name": "From people",
                "filter": (df_filtered["Type"] == "person") & (df_filtered["From"].str.lower() == "yes"),
                "color": "purple"
            })

    # Add layers to map
    has_data = False

    # Find appropriate threshold for current zoom level
    display_threshold = 100  # Default: hide all
    for zoom_level, min_radius in ZOOM_THRESHOLDS.items():
        if current_zoom >= zoom_level:
            display_threshold = min_radius

    for spec in layer_specs:
        subset = df_filtered[spec["filter"]].copy()
        if not subset.empty:
            has_data = True
            grouped = subset.groupby(["Latitude", "Longitude", "LocationEN", "LocationGD"]).size().reset_index(
                name="Items")
            grouped["Radius"] = grouped["Items"].apply(compute_scaled_radius)

            for _, row in grouped.iterrows():
                # Add CircleMarker for each location
                circle = folium.CircleMarker(
                    location=[row["Latitude"], row["Longitude"]],
                    radius=row["Radius"],
                    color=spec["color"],
                    stroke=False,
                    fill=True,
                    fill_color=spec["color"],
                    fill_opacity=0.5,
                    tooltip=f"{row['LocationEN']}: {row['Items']} items"
                ).add_to(m)

                # Get the display name based on the selected location filter
                display_name = get_display_name(row, location_filter)

                # Only add label if the radius is above the threshold for the current zoom level
                if row["Radius"] >= display_threshold:
                    folium.Marker(
                        location=[row["Latitude"] - 0.015, row["Longitude"]],  # Position below the circle
                        icon=folium.DivIcon(
                            icon_size=(150, 20),
                            icon_anchor=(75, 0),
                            html=f"""
                                <div style="text-align: center; 
                                           white-space: nowrap; 
                                           font-weight: bold;
                                           text-shadow: 1px 1px 1px white, -1px -1px 1px white, 1px -1px 1px white, -1px 1px 1px white;
                                           color: {spec["color"]};">
                                    {display_name}
                                </div>
                            """
                        ),
                        tooltip=display_name
                    ).add_to(m)

    if not has_data:
        st.warning("No data to display based on current filters. Please adjust your filter settings.")

    # Add historical map overlays if selected and if it has an overlay
    if isinstance(MAP_TYPES[selected_map_type], dict) and "overlay" in MAP_TYPES[selected_map_type]:
        folium.TileLayer(
            tiles=MAP_TYPES[selected_map_type]["overlay"],
            attr="Historical Map",
            name=selected_map_type,
            overlay=True,
            opacity=0.7
        ).add_to(m)

    # Display the map and track zoom changes
    st_map = st_folium(m, width="100%", height=800)

    # Update session state with current zoom if available
    if st_map and 'zoom' in st_map:
        if st_map['zoom'] != st.session_state.map_zoom:
            st.session_state.map_zoom = st_map['zoom']
            st.rerun()  # Rerun the app to redraw with new zoom level

    # === LEGEND ===
    legend_html = """
    <div style='padding: 10px; border-radius: 5px; position: relative;'>
    <h4 style='margin-top: 0;'>Legend</h4>
    """

    if show_tracks and subject:
        legend_html += """<div style='margin-bottom: 5px;'><span style='background-color: black; border-radius: 50%; display: inline-block; height: 12px; width: 12px;'></span> Tracks about</div>"""
    if show_tracks and recorded:
        legend_html += """<div style='margin-bottom: 5px;'><span style='background-color: green; border-radius: 50%; display: inline-block; height: 12px; width: 12px;'></span> Tracks recorded</div>"""
    if show_people and from_toggle:
        legend_html += """<div style='margin-bottom: 5px;'><span style='background-color: purple; border-radius: 50%; display: inline-block; height: 12px; width: 12px;'></span> People from</div>"""

    legend_html += "</div>"

    st.markdown(legend_html, unsafe_allow_html=True)

# === RELEVANT ITEMS ===
with col2:
    st.markdown("Search results")

    # Create tabs for Tracks and People
    tracks_tab, people_tab = st.tabs(["🎵 Tracks", "👤 People"])

    with tracks_tab:
        st.image("tracks_list.png", use_container_width=True)

    with people_tab:
        st.image("people_list.png", use_container_width=True)

with st.expander("Advanced searching"):
    advanced_search_html = """
    <div style='padding: 10px; border-radius: 5px; position: relative;'>
    <h4 style='margin-top: 0;'>Advanced searching techniques</h4>
    """

    if show_tracks and subject:
        advanced_search_html += """<div style='margin-bottom: 5px;'>The 'Full text search in tracks' and 'Full text search
         in people' fields offer additional functionality to advanced users.<br><br>        
         <b>Advanced tracks searching:</b><br>
         Use title:, description:, summary:, notes: or classification: keywords to improve search precision. 
        For example:<br>
         - summary:"Highlands"<br>
         - title:"Healing wells"<br><br><br>
         <b>Advanced people searching:</b><br>
         Use name:, patronymics: or bio: keywords to improve search precision. 
        For example:<br>
         - name:"Smith"<br>
         - bio:"singer"<br>
         </div>"""
    advanced_search_html += "</div>"

    st.markdown(advanced_search_html, unsafe_allow_html=True)
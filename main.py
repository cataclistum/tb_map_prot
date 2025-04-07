import streamlit as st
import pandas as pd
import numpy as np
from streamlit_folium import st_folium
import folium
import os

# Get API key from environment variable
try:
    MAPTILER_API_KEY = st.secrets["maptiler"]["MAPTILER_API_KEY"]
except Exception as e:
    MAPTILER_API_KEY = "PLACEHOLDER_KEY"
    st.warning(f"Could not load MapTiler API key from secrets: {e}")

MAPTILER_TILES = f"https://api.maptiler.com/maps/dataviz/256/{{z}}/{{x}}/{{y}}.png?key={MAPTILER_API_KEY}"
MAPTILER_ATTR = "MapTiler"

# Circle scale config
SCALE_MULTIPLIER = 8  # adjust overall size
MIN_RADIUS = 1  # minimum circle radius
MAX_RADIUS = 20  # maximum circle radius

# === PAGE CONFIG ===
st.set_page_config(layout="wide")

# === DATA LOAD ===
try:
    df = pd.read_excel("output.xlsx")
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
    "Genre": "Unknown"
})

# === SESSION STATE ===
if "show_filters" not in st.session_state:
    st.session_state.show_filters = True
if "show_relevant_items" not in st.session_state:
    st.session_state.show_relevant_items = True

# === FILTER PANEL ===
people_types = ["All", "With songs", "With stories", "With information", "With verses", "With music",
                "With radio programmes"]
map_overlay_types = ["None", "Great Britain, Ordnance Survey, 1900s", "Great Britain, Ordnance Survey, 1'' (1885-1903)",
                     "Great Britain, Ordnance Survey, 6'' (1888-1913)"]

with st.expander("🔍 Filters", expanded=st.session_state.show_filters):
    filter_col1, filter_col2 = st.columns(2)

    with filter_col1:
        tracks_title_col, show_tracks_col = st.columns([4, 2])  # Adjust ratio as needed
        with tracks_title_col:
            st.markdown("### 🎵 Tracks")
        with show_tracks_col:
            show_tracks = st.toggle("Show Tracks", value=True)

        search_tracks = st.text_input("Full text search in tracks",
                                      help="Searches in track titles and descriptions/summaries")
        genre = st.selectbox("Genre", options=["All"] + sorted(df["Genre"].dropna().unique().tolist()), index=0)
        subject = st.toggle("Subject", value=True, help="Shows/Hides locations mentioned in tracks")
        recorded = st.toggle("Recorded", value=True, help="Shows/Hides locations where tracks have been recorded")

    with filter_col2:
        people_title_col, show_people_col = st.columns([4, 2])  # Adjust ratio as needed
        with people_title_col:
            st.markdown("### 👤 People")
        with show_people_col:
            show_people = st.toggle("Show People", value=True)

        search_people = st.text_input("Full text search in people",
                                      help="Searches in fieldworkers' or contributors' names and patronymics")
        type2 = st.selectbox("Type", options=people_types, index=0)
        subject2 = st.toggle("Subject ", value=True,
                             help="Shows/Hides locations fieldworkers or contributors mentioned")
        from_toggle = st.toggle("From", value=True, help="Shows/Hides native areas of fieldworkers and contributors")

    st.divider()

    min_year = int(df["Date"].min()) if not pd.isna(df["Date"].min()) else 1935
    max_year = int(df["Date"].max()) if not pd.isna(df["Date"].max()) else 2025

    date_range = st.slider("Date range", min_value=1935, max_value=2025, value=(min_year, max_year))

    map_filters_col, blank_col = st.columns([2, 2])
    with map_filters_col:
        location_names = ["English", "Gàidhlig"]
        location_filter = st.selectbox("Location names", options=location_names, index=0)
        overlays = st.selectbox("Map overlay", options=map_overlay_types, index=0)
        outwith_scotland = st.toggle("Show locations outwith Scotland", value=True)


# === FILTERING FUNCTION ===
def apply_filters(df):
    df_filtered = df.copy()

    # Date filter
    df_filtered = df_filtered[(df_filtered["Date"] >= date_range[0]) & (df_filtered["Date"] <= date_range[1])]

    # Track filters
    if not show_tracks:
        df_filtered = df_filtered[df_filtered["Type"] != "track"]
    else:
        # Apply track-specific filters
        tracks_mask = df_filtered["Type"] == "track"

        if genre != "All":
            tracks_mask &= df_filtered["Genre"] == genre

        if search_tracks:
            # Assuming there's a column like "Title" or "Description" to search in
            # Adjust column names as needed
            for col in ["Title", "Description", "Summary"]:
                if col in df_filtered.columns:
                    tracks_mask &= df_filtered[col].fillna("").str.contains(search_tracks, case=False)

        if not subject:
            tracks_mask &= ~((df_filtered["Type"] == "track") & (df_filtered["Subject"].str.lower() == "yes"))

        if not recorded:
            tracks_mask &= ~((df_filtered["Type"] == "track") & (df_filtered["Recorded"].str.lower() == "yes"))

        # Keep non-tracks and filtered tracks
        df_filtered = df_filtered[(df_filtered["Type"] != "track") | tracks_mask]

    # People filters
    if not show_people:
        df_filtered = df_filtered[df_filtered["Type"] != "person"]
    else:
        # Apply people-specific filters
        people_mask = df_filtered["Type"] == "person"

        if type2 != "All":
            # Assuming there's a "PersonType" column or similar
            if "PersonType" in df_filtered.columns:
                people_mask &= df_filtered["PersonType"] == type2.replace("With ", "")

        if search_people:
            # Assuming there are columns like "Name" or "Patronymic" to search in
            # Adjust column names as needed
            for col in ["Name", "Patronymic", "FullName"]:
                if col in df_filtered.columns:
                    people_mask &= df_filtered[col].fillna("").str.contains(search_people, case=False)

        if not subject2:
            people_mask &= ~((df_filtered["Type"] == "person") & (df_filtered["Subject"].str.lower() == "yes"))

        if not from_toggle:
            people_mask &= ~((df_filtered["Type"] == "person") & (df_filtered["From"].str.lower() == "yes"))

        # Keep non-people and filtered people
        df_filtered = df_filtered[(df_filtered["Type"] != "person") | people_mask]

    # Location filter (Scotland filter)
    if not outwith_scotland:
        # Assuming there's a column indicating if a location is in Scotland
        # If not, you could filter by latitude/longitude bounds
        if "InScotland" in df_filtered.columns:
            df_filtered = df_filtered[df_filtered["InScotland"] == True]
        else:
            # Approximate Scotland bounds
            scotland_lat_min, scotland_lat_max = 54.6, 60.9
            scotland_lon_min, scotland_lon_max = -8.7, -0.7
            df_filtered = df_filtered[
                (df_filtered["Latitude"] >= scotland_lat_min) &
                (df_filtered["Latitude"] <= scotland_lat_max) &
                (df_filtered["Longitude"] >= scotland_lon_min) &
                (df_filtered["Longitude"] <= scotland_lon_max)
                ]

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


# === MAP ===
with col1:
    # Create base map
    m = folium.Map(
        location=[df_filtered["Latitude"].mean() if not df_filtered.empty else 56.4907,
                  df_filtered["Longitude"].mean() if not df_filtered.empty else -4.2026],
        # Default to center of Scotland if empty
        zoom_start=6,
        tiles=MAPTILER_TILES,
        attr=MAPTILER_ATTR
    )

    # === DEFINE LAYER SPECS ===
    layer_specs = []

    if show_tracks:
        if subject:
            layer_specs.append({
                "name": "Subject tracks",
                "filter": (df_filtered["Type"] == "track") & (df_filtered["Subject"].str.lower() == "yes"),
                "color": "yellow"
            })
        if recorded:
            layer_specs.append({
                "name": "Recorded tracks",
                "filter": (df_filtered["Type"] == "track") & (df_filtered["Recorded"].str.lower() == "yes"),
                "color": "green"
            })

    if show_people:
        if subject2:
            layer_specs.append({
                "name": "Subject people",
                "filter": (df_filtered["Type"] == "person") & (df_filtered["Subject"].str.lower() == "yes"),
                "color": "red"
            })
        if from_toggle:
            layer_specs.append({
                "name": "From people",
                "filter": (df_filtered["Type"] == "person") & (df_filtered["From"].str.lower() == "yes"),
                "color": "purple"
            })

    # Add layers to map
    has_data = False
    for spec in layer_specs:
        subset = df_filtered[spec["filter"]].copy()
        if not subset.empty:
            has_data = True
            grouped = subset.groupby(["Latitude", "Longitude", "LocationEN"]).size().reset_index(name="Items")
            grouped["Radius"] = grouped["Items"].apply(compute_scaled_radius)

            for _, row in grouped.iterrows():
                folium.CircleMarker(
                    location=[row["Latitude"], row["Longitude"]],
                    radius=row["Radius"],
                    color=spec["color"],
                    stroke=False,
                    fill=True,
                    fill_color=spec["color"],
                    fill_opacity=0.5,
                    tooltip=f"{row['LocationEN']}: {row['Items']} items"
                ).add_to(m)

    if not has_data:
        st.warning("No data to display based on current filters. Please adjust your filter settings.")

    # Add historical map overlays if selected
    if overlays != "None":
        # You'll need to provide the correct URLs for these historical maps
        overlay_urls = {
            "Great Britain, Ordnance Survey, 1900s": f"https://api.maptiler.com/tiles/uk-osgb1888/{{z}}/{{x}}/{{y}}?key={MAPTILER_API_KEY}",
            "Great Britain, Ordnance Survey, 1'' (1885-1903)": f"https://api.maptiler.com/tiles/uk-osgb63k1885/{{z}}/{{x}}/{{y}}.png?key={MAPTILER_API_KEY}",
            "Great Britain, Ordnance Survey, 6'' (1888-1913)": f"https://api.maptiler.com/tiles/uk-osgb10k1888/{{z}}/{{x}}/{{y}}.jpg?key={MAPTILER_API_KEY}"
        }

        if overlays in overlay_urls:
            folium.TileLayer(
                tiles=overlay_urls[overlays],
                attr="Historical Map",
                name=overlays,
                overlay=True,
                opacity=0.7
            ).add_to(m)

    # Display the map
    st_map = st_folium(m, width="100%", height=700)

    # === LEGEND ===
    legend_html = """
    <div style='padding: 10px; border-radius: 5px; position: relative;'>
    <h4 style='margin-top: 0;'>Legend</h4>
    """

    if show_tracks and subject:
        legend_html += """<div style='margin-bottom: 5px;'><span style='background-color: yellow; border-radius: 50%; display: inline-block; height: 12px; width: 12px;'></span> Tracks about</div>"""
    if show_tracks and recorded:
        legend_html += """<div style='margin-bottom: 5px;'><span style='background-color: green; border-radius: 50%; display: inline-block; height: 12px; width: 12px;'></span> Tracks recorded</div>"""
    if show_people and subject2:
        legend_html += """<div style='margin-bottom: 5px;'><span style='background-color: red; border-radius: 50%; display: inline-block; height: 12px; width: 12px;'></span> People mentioned</div>"""
    if show_people and from_toggle:
        legend_html += """<div style='margin-bottom: 5px;'><span style='background-color: purple; border-radius: 50%; display: inline-block; height: 12px; width: 12px;'></span> People from</div>"""

    legend_html += "</div>"

    st.markdown(legend_html, unsafe_allow_html=True)

# === RELEVANT ITEMS ===
with col2:
    if st.session_state.show_relevant_items:
        results_sort = st.selectbox("Sort", options=["Tracks first", "People first"], index=0)

        # Display results based on map click
        if st_map and 'last_clicked' in st_map and st_map['last_clicked']:
            clicked_lat = st_map['last_clicked']['lat']
            clicked_lng = st_map['last_clicked']['lng']

            # Find locations near the clicked point (with some tolerance)
            tolerance = 0.01  # Adjust based on your needs
            nearby_items = df_filtered[
                (df_filtered['Latitude'] >= clicked_lat - tolerance) &
                (df_filtered['Latitude'] <= clicked_lat + tolerance) &
                (df_filtered['Longitude'] >= clicked_lng - tolerance) &
                (df_filtered['Longitude'] <= clicked_lng + tolerance)
                ]

            if not nearby_items.empty:
                # Sort results based on user preference
                if results_sort == "Tracks first":
                    nearby_items = nearby_items.sort_values(by=["Type", "Date"], key=lambda x: x.map(
                        {"track": 0, "person": 1} if x.name == "Type" else x))
                else:
                    nearby_items = nearby_items.sort_values(by=["Type", "Date"], key=lambda x: x.map(
                        {"person": 0, "track": 1} if x.name == "Type" else x))

                # Display nearby items
                location_name = nearby_items.iloc[0]["LocationEN"]
                st.subheader(f"Items at {location_name}")

                for i, (_, item) in enumerate(nearby_items.iterrows()):
                    with st.expander(f"{item['Type'].title()}: {item.get('Title', item.get('Name', 'Unnamed'))}"):
                        # Display item details based on type
                        for col in item.index:
                            if col not in ["Latitude", "Longitude", "Type"] and not pd.isna(item[col]):
                                st.write(f"**{col}:** {item[col]}")
            else:
                st.markdown("Click on a location on the map to see items from that area.")
        else:
            st.markdown("Click on a location on the map to see items from that area.")
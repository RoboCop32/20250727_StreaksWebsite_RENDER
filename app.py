#1. choose the day streak you want. then it runs the script generating the streak 


#2. make the overall streak table (so days of streak, dates, and country) at top. then you click on the row (instead of enter streak ID) and it makes a map below. 
# problem is line 302 - when we retrieve the streak, it is still retrieivng it form the main tbale hat we haven tupdated. so how do we instead query the dataframe that we have just made with the preivous funciton
#3. and then you click the toggle filtered map (shortest trip)


# nah so change this - make the main table have lots of table ids based on the 1-10 strak gaps. then we just query that table

# 7.3.25 fixed te zoom and also the table underneath map. also made the map container more dyanmic
# NEXT TIME: hide the streak table or make the map near the top. also the filtered table is not populated the first time . better labelling too

#MAKING IT SEUCURE:
'''
# NOTE: other SQL queries in your code should follow the same pattern:
# - define them using `text()`
# - pass parameters via a dictionary
# You can apply this approach to any other raw SQL queries in the file as needed.

When users submit input (e.g., through a form or URL), you must make sure it doesn’t contain malicious content — especially when used in SQL queries.
'''

from flask import Flask, render_template, request, Markup, jsonify, session, render_template_string, g

import os
import geopandas as gpd
import folium
import random
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.sql import text
import sqlalchemy
import psycopg2
from shapely.wkt import loads  # Required for geometry conversion
import branca
import numpy as np
from scipy.spatial.distance import cdist

app = Flask(__name__)

#secure session key
app.secret_key = os.urandom(24)

# Database connection details - for RENDER
local_db = "postgresql://myapp_db_x30v_user:7qClIk24pcyyepGQpvmOg9HV8pVmQVKe@dpg-d2q4ket6ubrc73cv49l0-a/myapp_db_x30v"

# Use DATABASE_URL from Render if available. Comment below out if using locally...
db_url = os.environ.get("DATABASE_URL", local_db)
engine1 = create_engine(db_url)


#engine1 = create_engine(signin_info)


main_table_name = "fixtures_stadium_combined_20260115"

STADIUMS_TABLE = "fixtures_stadium_combined_20260130"  # matches your screenshot

show_lines = True  # Default: Show arrows

# ===== NEW: config for filter page =====
DATA_TABLE = main_table_name  # uses your existing main table
FILTER_COLUMNS = ["country", "league", "home", "away"]  # add more if you want (e.g. "Team Name")
RESULT_COLUMNS = [
    "unique_id", "date", "time", "league", "home", "away", "country",
    "stadium_id", "Team Name", "Stadium Name", "Longitude", "Latitude"
]



def q(col: str) -> str:
    # safe double-quote for mixed‑case / spaces
    return '"' + col.replace('"', '""') + '"'

# --- add this helper near the top of app.py (after q())
COLMAP = {
    "club": "Team Name",   # stadiums page "Club" picker
    # everything else maps to itself by default
}
def key_to_col(key: str) -> str:
    return COLMAP.get(key, key)

def retrieve_sql_table(engine, table_name):
    query = text(f'SELECT * FROM "{table_name}"')
    res = engine.execute(query)
    return pd.DataFrame(res.fetchall(), columns=res.keys())

def retrieve_streaks(my_table,day_interval,engine):
    '''
    This function we work out the streaks from the main view where we have joined fixtures and stadiums
    '''
    
    streak_query = f'''WITH ranked_logins AS (
        SELECT 
            country,
            unique_id,
            CONCAT(home, ' vs ', away) AS fixtures,
            date,
            -- Calculate the gap between consecutive dates
            LAG(date) OVER (PARTITION BY country ORDER BY date) AS previous_date
        FROM {my_table}
        WHERE country IS NOT NULL
    ),

    date_groups AS (
        SELECT 
            country,
            unique_id,
            fixtures,
            date,
            previous_date,
            -- Assign a group based on whether the gap exceeds 2 days
            CASE 
                WHEN previous_date IS NULL OR date - previous_date > INTERVAL '{day_interval}' DAY THEN 1
                ELSE 0
            END AS new_streak
        FROM ranked_logins
    ),

    streak_groups AS (
        SELECT 
            country,
            unique_id,
            fixtures,
            date,
            -- Use SUM() to accumulate streak group IDs
            SUM(new_streak) OVER (PARTITION BY country ORDER BY date) AS streak_id
        FROM date_groups
    ),

    intervals AS (
        SELECT 
            country,
            string_agg(unique_id::character varying, ', ') AS all_ids,
            MIN(date) AS interval_start_date,
            MAX(date) AS interval_end_date
        FROM streak_groups
        GROUP BY country, streak_id
        ORDER BY interval_start_date
    )

    SELECT 
        country AS streak_country,
        interval_start_date,
        interval_end_date,
        all_ids,
        -- Calculate the length of the streak in days
        CAST(EXTRACT(DAY FROM (interval_end_date - interval_start_date)) AS INTEGER) + 1 AS day_interval
    FROM intervals
    WHERE interval_end_date - interval_start_date >= INTERVAL '{day_interval}' DAY -- Filter streaks of 2 days or more
    ORDER BY day_interval DESC;'''

  
    #here we execute our streaks query
    
    
    resoverall = engine.execute(streak_query)
    
    df_result = pd.DataFrame(resoverall.fetchall())
    
    df_result["Streak_ID"] = df_result.index
    
    print("ran streak query")

    return df_result


def retrieve_streak(streak_id, engine, view_name):
    query = text(f'SELECT * FROM public."{view_name}" WHERE "Streak_ID" = :id')
    result = engine.execute(query, {"id": streak_id})
    df_result = pd.DataFrame(result.fetchall(), columns=result.keys())
    return None if df_result.empty else df_result

def convert_sql_to_gdf(df, geom_column):
    """Convert a SQL DataFrame to a GeoDataFrame using the given geometry column."""
    df = df[(df[geom_column] != None) & (df[geom_column] != 'None') & (df[geom_column] != "")]
    df['geometry'] = df[geom_column].str.replace(',', ' ').apply(loads)  
    gdf = gpd.GeoDataFrame(df, geometry=df['geometry'], crs="EPSG:4326")
    gdf["latitude"] = gdf.geometry.y
    gdf["longitude"] = gdf.geometry.x
    return gdf

def find_optimum_stadiums(gdf):

    gdf = gdf[(gdf["geometry"] != None) & (gdf["geometry"] != 'None') & (gdf["geometry"] != "")].sort_values("date")

    categories = gdf['date'].unique()
    
   # categories.sort() #sort the cateogires! hmm perhaps not u get an error
    
    groups = {cat: gdf[gdf['date'] == cat] for cat in categories}

    names = {cat: group['unique_id'].tolist() for cat, group in groups.items()}  #so here we have a dict with date: list of fixtures

    # Extract coordinates for distance calculations
    coords = {cat: group.geometry.apply(lambda x: (x.x, x.y)).tolist() for cat, group in groups.items()} #and as abov but with coords

    route = []
    total_distance = 0

    # Start at the first category
    current_coords = np.array(coords[categories[0]])
    current_names = names[categories[0]]
    
    #so we itrate through the date: coords dicitonaries, calculate distance tables for all of them, then get the minimum distnace
    
    for i in range(len(categories) - 1):
        next_category = categories[i + 1] #so this does go to the next date
        next_coords = np.array(coords[next_category])
        next_names = names[next_category] #and this does get the next names

        # Calculate pairwise distances
        distances = cdist(current_coords, next_coords)

        min_dist_idx = np.unravel_index(distances.argmin(), distances.shape)

        # Append the chosen point's name to the route and update total distance
        route.append((categories[i], current_names[min_dist_idx[0]]))
        total_distance += distances[min_dist_idx]

        #if we are at the last iteration (-2): apend the next category, and get the index in the Next names, using the End value of the min dist index [1]to get the destination from the next list
        if i == (len(categories) - 2):
            route.append((categories[i+1], next_names[min_dist_idx[1]])) # to get the last one too

        # Update current_coords and current_names to the selected next point
        current_coords = next_coords[[min_dist_idx[1]], :]
        current_names = [next_names[min_dist_idx[1]]]

        # Add the final point
    #route.append((categories[-1], current_names[0])) # i removed this line. firstly it was in the for loop when it was supposed to be. secondly, doesnt accurately get the best distace in the last one

        

    # Calculate the shortest route
    
    names_list = [name for _, name in route]
    print(route,total_distance,names_list)
    
def find_optimum_stadiums(gdf):

    gdf = gdf[(gdf["geometry"] != None) & (gdf["geometry"] != 'None') & (gdf["geometry"] != "")].sort_values("date")

    categories = gdf['date'].unique()
    
   # categories.sort() #sort the cateogires! hmm perhaps not u get an error
    
    groups = {cat: gdf[gdf['date'] == cat] for cat in categories}

    names = {cat: group['unique_id'].tolist() for cat, group in groups.items()}  #so here we have a dict with date: list of fixtures

    # Extract coordinates for distance calculations
    coords = {cat: group.geometry.apply(lambda x: (x.x, x.y)).tolist() for cat, group in groups.items()} #and as abov but with coords

    route = []
    total_distance = 0

    # Start at the first category
    current_coords = np.array(coords[categories[0]])
    current_names = names[categories[0]]
    
    #so we itrate through the date: coords dicitonaries, calculate distance tables for all of them, then get the minimum distnace
    
    for i in range(len(categories) - 1):
        next_category = categories[i + 1] #so this does go to the next date
        next_coords = np.array(coords[next_category])
        next_names = names[next_category] #and this does get the next names

        # Calculate pairwise distances
        distances = cdist(current_coords, next_coords)

        min_dist_idx = np.unravel_index(distances.argmin(), distances.shape)

        # Append the chosen point's name to the route and update total distance
        route.append((categories[i], current_names[min_dist_idx[0]]))
        total_distance += distances[min_dist_idx]

        #if we are at the last iteration (-2): apend the next category, and get the index in the Next names, using the End value of the min dist index [1]to get the destination from the next list
        if i == (len(categories) - 2):
            route.append((categories[i+1], next_names[min_dist_idx[1]])) # to get the last one too

        # Update current_coords and current_names to the selected next point
        current_coords = next_coords[[min_dist_idx[1]], :]
        current_names = [next_names[min_dist_idx[1]]]

        # Add the final point
    #route.append((categories[-1], current_names[0])) # i removed this line. firstly it was in the for loop when it was supposed to be. secondly, doesnt accurately get the best distace in the last one

        

    # Calculate the shortest route
    
    names_list = [name for _, name in route]
    print(route,total_distance,names_list)
    return route,total_distance,names_list



def split_streak_table(streak_table):
    df_split = streak_table.assign(all_ids=streak_table['all_ids'].str.split(', ')).explode('all_ids').reset_index(drop=True)
    return df_split

def retrieve_streaks(my_table, day_interval, engine):
    query = text(f"""
        WITH ranked_logins AS (
            SELECT country, unique_id, CONCAT(home, ' vs ', away) AS fixtures, date,
                   LAG(date) OVER (PARTITION BY country ORDER BY date) AS previous_date
            FROM {my_table}
            WHERE country IS NOT NULL
        ),
        date_groups AS (
            SELECT country, unique_id, fixtures, date, previous_date,
                   CASE WHEN previous_date IS NULL OR date - previous_date > INTERVAL :interval DAY THEN 1 ELSE 0 END AS new_streak
            FROM ranked_logins
        ),
        streak_groups AS (
            SELECT country, unique_id, fixtures, date,
                   SUM(new_streak) OVER (PARTITION BY country ORDER BY date) AS streak_id
            FROM date_groups
        ),
        intervals AS (
            SELECT country, string_agg(unique_id::character varying, ', ') AS all_ids,
                   MIN(date) AS interval_start_date, MAX(date) AS interval_end_date
            FROM streak_groups
            GROUP BY country, streak_id
            ORDER BY interval_start_date
        )
        SELECT country AS streak_country, interval_start_date, interval_end_date, all_ids,
               CAST(EXTRACT(DAY FROM (interval_end_date - interval_start_date)) AS INTEGER) + 1 AS day_interval
        FROM intervals
        WHERE interval_end_date - interval_start_date >= INTERVAL :interval DAY
        ORDER BY day_interval DESC;
    """)
    res = engine.execute(query, {"interval": str(day_interval)})
    df_result = pd.DataFrame(res.fetchall(), columns=res.keys())
    df_result["Streak_ID"] = df_result.index
    return df_result




def plot_streak_map(gdf, jitter_amount=0.01, arrows=False, width="1000px", height="600px", long=None, lat=None, myzoom=8):
    """Generate an interactive Folium map where markers can be accessed via JavaScript."""

    if gdf is None or gdf.empty:
        return None

    gdf = gdf.to_crs(epsg=4326).sort_values("date") 
    gdf['date'] = gdf['date'].dt.strftime("%Y-%m-%d").astype(str)

    if long is None or lat is None:
        lat = gdf.geometry.y.mean()
        long = gdf.geometry.x.mean()

    m = folium.Map(location=[lat, long], zoom_start=myzoom, width=width, height=height)

    # Color map
    unique_dates = gdf['date'].unique()
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'black', 'yellow', 'brown', 'olive', 'cyan', 'mediumseagreen', 'white']
    color_map = {date: colors[i % len(colors)] for i, date in enumerate(unique_dates)}

    # Inject leaflet JS and CSS manually
    leaflet_head = """
    <head>
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css"
              integrity="sha512-xodZBNTC5n17Xt2atTPuE1HxjVMSvLVW9ocqUKLsCC5CXdbqCmblAshOMAS6/keqq/sMZMZ19scR4PsZChSR7A=="
              crossorigin=""/>
        <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"
                integrity="sha512-nmmYYl9+jn+1tcpiLAc5RT6oMJc4gLqMXn4euwcJFAfWv1fByNT1rkZcpPj2lmTf5crlWzqGGN6E466+vWXTkA=="
                crossorigin=""></script>
    </head>
    """
    m.get_root().header.add_child(folium.Element(leaflet_head))

    # JS container for marker references
    m.get_root().html.add_child(folium.Element("<script>window.markers = {};</script>"))

    # Add each marker and register in window.markers
    for _, row in gdf.iterrows():
        if row["geometry"] is not None:
            jittered_lat = row.geometry.y + random.uniform(-jitter_amount, jitter_amount)
            jittered_lon = row.geometry.x + random.uniform(-jitter_amount, jitter_amount)
            color = color_map.get(row['date'], 'gray')
            unique_id = row["unique_id"]

            marker_js = f"""
            <script>
                var marker_{unique_id} = L.circleMarker([{jittered_lat}, {jittered_lon}], {{
                    radius: 6,
                    color: 'black',
                    fillColor: '{color}',
                    fillOpacity: 0.9
                }}).addTo(window.map)
                .bindPopup("Date: {row['date']}<br>Home: {row['home']} vs Away: {row['away']}");
                window.markers["{unique_id}"] = marker_{unique_id};
            </script>
            """
            m.get_root().html.add_child(folium.Element(marker_js))
    
    
    
    # Optional: arrows between points
    if arrows:
        print("arrows added!")
        locations = list(zip(gdf.geometry.y, gdf.geometry.x))
        if len(locations) > 1:
            folium.PolyLine(locations, color="red", weight=4, opacity=0.7).add_to(m)

    # Assign map to window.map
    assign_js = """
    <script>
        setTimeout(function() {
            const keys = Object.keys(window);
            for (let k of keys) {
                if (k.startsWith("map_")) {
                    window.map = window[k];
                    console.log("✅ window.map assigned:", k);
                    break;
                }
            }
        }, 500);
    </script>
    """
    m.get_root().html.add_child(folium.Element(assign_js))

    return m.get_root().render()

    return m.get_root().render()
@app.before_request
def load_data():
    if 'df_data' not in g:
        g.df_data = pd.DataFrame(app.config.get("df_data", []))
        
        
@app.route("/", methods=["GET"])
def home():
    return render_template("home.html")
    
@app.route("/streak", methods=["GET", "POST"]) #so this is the Main page. Each flask route is mapped to a specific URL, and executes a python function when accessed
def streak():
    combined_streak_view = "combined_stadium_streaks_view2"
    map_html = None
    error_message = None
    df_data = []
    
    if request.method == "POST": #so if user submits the form (POST request) HTTP Methods: POST is used to send data to a server to create/update a resource. not the same as GET
        streak_gap = request.form.get("streak_gap")
        print(streak_gap)
        
        app.config['streak_gap'] = streak_gap
        
        if streak_gap:
            try:
                streak_gap = int(streak_gap)
                selected_streak = retrieve_streaks(main_table_name, streak_gap, engine1)
                
                if selected_streak is not None:
                
                    df_data = selected_streak.to_dict(orient="records")
                    app.config['original_df_data'] = selected_streak  # Store the original data
                    app.config['df_data'] = selected_streak  # Keep reference to update table
                    print(selected_streak,bool(df_data))
                else:
                    error_message = "No data found for this Streak Gap."
            except ValueError:
                error_message = "Invalid Streak Gap. Please enter a number."

    return render_template("streak.html", error_message=error_message, df_data=df_data,streak_shown=bool(df_data))

@app.get("/api/stadiums/search")
def api_stadiums_search():
    import json
    try:
        raw = json.loads(request.args.get("filters") or "{}")
    except Exception:
        raw = {}

    # We still let Country/League/Club filter via the combined fixtures table:
    filters = {
        "country": (raw.get("country") or []),   # logical -> "country" in combined table
        "league":  (raw.get("league")  or []),   # logical -> "league" in combined table
        "club":    (raw.get("club")    or []),   # logical -> "Team Name" via key_to_col
    }

    where_sql, params = _build_where_and_params(filters)

    # 1) Find clubs that match filters in the combined table
    # 2) Join to stadiums table for coordinates + stadium attributes
    sql = text(f"""
        WITH clubs AS (
          SELECT DISTINCT {q('Team Name')} AS team_name
          FROM {q(DATA_TABLE)}
          {where_sql}
        )
        SELECT
          s.{q('Team Name')}    AS team_name,
          s.{q('Country')}      AS country,
          s.{q('Stadium Name')} AS stadium_name,
          s.{q('Stadium Capacity')} AS capacity,
          s.{q('google_maps')}  AS google_maps,
          s.{q('Longitude')}::float AS lon,
          s.{q('Latitude')}::float  AS lat
        FROM {q(STADIUMS_TABLE)} s
        JOIN clubs c ON c.team_name = s.{q('Team Name')}
        ORDER BY s.{q('Team Name')}, s.{q('Stadium Name')}
    """)
    
    print(sql)

    with engine1.connect() as conn:
        rows = conn.execute(sql, params).mappings().all()

        # table rows
        out_rows = [{
            "team_name":   r["team_name"],
            "country":     r["country"],
            "stadium_name":r["stadium_name"],
            "capacity":         r["capacity"],
            "google_maps":         r["google_maps"],
        } for r in rows]

        # markers for Leaflet
        markers = [{
            "team_name":   r["team_name"],
            "country":     r["country"],
            "stadium_name":r["stadium_name"],
            "capacity":         r["capacity"],
            "google_maps":         r["google_maps"],
            "lon":         r["lon"],
            "lat":         r["lat"],
        } for r in rows if r["lat"] is not None and r["lon"] is not None]

    

    return jsonify({"rows": out_rows, "markers": markers})

# --- homepage -> Stadiums ---
@app.route("/stadiums", methods=["GET"]) #this is called a decorator apparently
def stadiums_home():
    with engine1.connect() as conn:
        def distinct(col):
            rows = conn.execute(text(
                f'SELECT DISTINCT "{col}" AS v FROM "{DATA_TABLE}" '
                f'WHERE "{col}" IS NOT NULL ORDER BY 1'
            )).mappings().all()
            return [r["v"] for r in rows if r["v"] is not None]

        preload = {
            "country": distinct("country"),
            "league":  distinct("league"),
            "club":    distinct("Team Name"),
        }
    return render_template("stadiums.html", preload=preload)

@app.route("/get_streak/<int:streak_id>") # so this streak_id is pulled from the javascript
def get_streak(streak_id):
    """Retrieve and display the selected streak's map."""
    
    #this is how we store df_data outside
    #df_data = g.df_data
    
    print(streak_id)

    full_matches_table = "fixtures_stadium_combined_20260115"
    
    #remember we defined df_data before
    df_data = pd.DataFrame(app.config["df_data"]) 
    
    print(type(df_data))
    
    #so this is where we convert the above to a propoer split df data
    split_streak_table_var = split_streak_table(df_data)
    
    #get the list of ids with the streakid of the one the person clicks on
    filtered_list = split_streak_table_var.loc[split_streak_table_var['Streak_ID'] == streak_id, 'all_ids'].tolist()
    
    full_fixtures = retrieve_sql_table(engine1,main_table_name)
    
    filtered_df = full_fixtures[full_fixtures['unique_id'].isin(filtered_list)]

    
    gdf = convert_sql_to_gdf(filtered_df, 'Stadium Location').sort_values("date")
    
    route, distance, names_list = find_optimum_stadiums(gdf)
    gdf_filtered = gdf[gdf['unique_id'].isin(names_list)]

    app.config['gdf'] = gdf
    app.config['gdf_filtered'] = gdf_filtered
    
    #don't do ipalce = True because this returns NAN. and also, json can't handle nulls, so we need to replace them with None
    df_data_dict=gdf.drop(columns=['geometry']).replace({np.nan: None}).to_dict(orient="records")
    filtered_data_dict = gdf_filtered.drop(columns=['geometry']).replace({np.nan: None}).to_dict(orient="records")
    
    print(type(df_data_dict),'444444',df_data_dict)
    map_html = plot_streak_map(gdf, 0.005)
    
    
    result = {
        'map_html': str(map_html),
        'df_data': df_data_dict,
        'filtered_data':filtered_data_dict
        
    }
    
    return jsonify(result)
    
    
    #(Markup(map_html),df_data,filtered_df_data)  # Send the updated map to the frontend. What is markup?? it is from Flask's Jinja 2 tempalte engine. - basically allows us to send raw html. without markup, it will be rendered as plain text.

@app.route("/filter_streak")
def filter_streak():
    """Returns the filtered gdf as JSON for display in the table."""
    if "gdf_filtered" in app.config:
        gdf_filtered = app.config["gdf_filtered"]
        
        print("HELLO",gdf_filtered.columns)
        
        # Convert to JSON format for the frontend
        map_html = plot_streak_map(gdf_filtered, 0.005,False)
        
        return Markup(map_html)
    
@app.route("/get_original_table")
def get_original_table():
    """Returns the original gdf as JSON to reset the table."""
    if "gdf" in app.config:
        gdf = app.config["gdf"]
        
        # Convert to JSON format
        original_data = gdf.to_dict(orient="records")
        
        return jsonify(original_data)
    return jsonify([])  # Return empty if no data
    
@app.route("/toggle_map_and_table")
def toggle_map_and_table():
    """Toggles between original and filtered map & table using Markup and render_template."""
    
    if "gdf_filtered" in app.config and "gdf" in app.config:
        app.config["show_filtered"] = False
        print(app.config.get("show_filtered"))
        #app.config.get("show_filtered", False):
        # Switch back to original data
        map_html = plot_streak_map(app.config["gdf"], 0.005)
        df_data = app.config["gdf"]  # Retrieve original data - nah so this has to be gdf!
        df_data_dropped=df_data.drop(columns=['geometry']).replace({np.nan: None}).to_dict(orient="records")#.replace({np.nan: None}).to_dict(orient="records")
        app.config["show_filtered"] = True
        
        print(df_data.columns,"34343434")
        result = {
        'map_html': str(map_html),
        'df_data': df_data_dropped,
        
    }
            
        #print("LOLOLOLOO",df_data.to_dict(orient="records"))
        return  jsonify(result)

    return "Error: No data available", 400

@app.route("/get_filtered_table")
def get_filtered_table():
    """Returns filtered table data for the filtered map view."""
    if "gdf_filtered" in app.config:
        gdf_filtered = app.config["gdf_filtered"]
        df_data = gdf_filtered.drop(columns=['geometry']).replace({np.nan: None}).to_dict(orient="records")
        return jsonify(df_data)
    return jsonify([])


@app.route("/toggle_lines/<action>")
def toggle_lines(action):
    global show_lines
    
    show_lines = action == "show"
    print(action,"sdfhjfsdjhfsdhjsfdj") #cant do console ofc
    df_data = app.config["gdf_filtered"]
    if action == 'show':
    
        map_html = plot_streak_map(df_data, 0.001,True)
    else:
        map_html = plot_streak_map(df_data, 0.001,False)
        
    return Markup(map_html) # Re-render map with updated arrow state
     
@app.route("/zoom/<float:lat>/<float:lon>", methods=["GET"])
def zoom(lat,lon):
    #data = request.get_json()
    
    print("WHWHWHW",lat,lon)
    gdf = app.config["gdf"]
    # Call your map function centered on clicked point
    m = plot_streak_map(gdf, lat=lat, long=lon,myzoom=0.5)

    # Save updated map to file
    #m.save("templates/map.html")
    
    return Markup(m)

def q(x):  # helper that quotes identifiers the same way you already do
    return f'"{x}"'
    
    
@app.route("/get_filtered_route")
def get_filtered_route():

    gdf_filtered = app.config['gdf_filtered']
    #don't do ipalce = True because this returns NAN. and also, json can't handle nulls, so we need to replace them with None
    df_data_dict=gdf_filtered.drop(columns=['geometry']).replace({np.nan: None}).to_dict(orient="records")
    
    print(type(df_data_dict),'3333',df_data_dict)
    map_html = plot_streak_map(gdf_filtered, 0.005)
    result = {
        'map_html': str(map_html),
        'df_data': df_data_dict,
        
    }
    
    return jsonify(result)
    
@app.route("/filters", methods=["GET"])
def filter_home():
    # Preload distincts so dropdowns aren’t empty on first load
    with engine1.connect() as conn:
        def distinct(col):
            sql = text(f"SELECT DISTINCT {q(col)} AS v FROM {q(DATA_TABLE)} "
                       f"WHERE {q(col)} IS NOT NULL ORDER BY 1")
            rows = conn.execute(sql).mappings().all()
            return [r["v"] for r in rows if r["v"] is not None]

        preload = {
            "country": distinct("country"),
            "league": distinct("league"),
            "home": distinct("home"),
            "away": distinct("away"),
        }

    return render_template("filters.html", preload=preload)


def _build_where_and_params(filters: dict):
    clauses = []
    params = {}
    for key, val in filters.items():
        if val in (None, "", []):
            continue

        if key in ("date_from", "date_to"):
            if key == "date_from":
                clauses.append(f"{q('date')} >= :date_from")
                params["date_from"] = val
            else:
                clauses.append(f"{q('date')} <= :date_to")
                params["date_to"] = val
            continue

        col = key_to_col(key)  # <-- NEW

        # Multi-select → list: use ANY(CAST(:param AS text[]))
        if isinstance(val, list):
            cleaned = [v for v in val if v not in (None, "")]
            if not cleaned:
                continue
            clauses.append(f"{q(col)} = ANY(CAST(:{key} AS TEXT[]))")
            params[key] = cleaned
        else:
            clauses.append(f"{q(col)} = :{key}")
            params[key] = val

    where_sql = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    return where_sql, params


@app.get("/api/options")
def api_options():
    """
    Returns distinct values for a given column using current filters,
    excluding the column itself and any columns to its right in the page's order.
    """
    import json
    column = request.args.get("column")

    # Two pages use this endpoint:
    # - fixtures page: ["country","league","home","away"]
    # - stadiums page: ["country","league","club"]  (club maps to "Team Name")
    STADIUMS_ORDER = ["country", "league", "club"]
    FIXTURES_ORDER = ["country", "league", "home", "away"]

    # choose which ORDER applies based on requested column
    if column in STADIUMS_ORDER:
        ORDER = STADIUMS_ORDER
    elif column in FIXTURES_ORDER:
        ORDER = FIXTURES_ORDER
    else:
        return jsonify({"values": []}), 400

    try:
        filters = json.loads(request.args.get("filters") or "{}")
    except Exception:
        filters = {}

    # Drop self and anything to the right
    idx = ORDER.index(column)
    for k in ORDER[idx:]:
        filters.pop(k, None)

    # Build WHERE with mapped columns
    where_sql, params = _build_where_and_params(filters)

    # Map requested logical column to actual DB column
    actual_col = key_to_col(column)

    sql = text(
        f"SELECT DISTINCT {q(actual_col)} AS v FROM {q(DATA_TABLE)}{where_sql} "
        f"{' AND ' if where_sql else ' WHERE '}{q(actual_col)} IS NOT NULL "
        f"ORDER BY 1"
    )
    with engine1.connect() as conn:
        rows = conn.execute(sql, params).mappings().all()

    return jsonify({"values": [r['v'] for r in rows if r['v'] is not None]})


@app.get("/api/search")
def api_search():
    # ?filters=<json>&limit=2000
    import json
    try:
        filters = json.loads(request.args.get("filters") or "{}")
    except Exception:
        filters = {}
    limit = max(1, min(int(request.args.get("limit", 1000)), 5000))

    where_sql, params = _build_where_and_params(filters)
    cols_sql = ", ".join(q(c) for c in RESULT_COLUMNS)
    sql = text(
        f"SELECT {cols_sql} FROM {q(DATA_TABLE)}{where_sql} "
        f"ORDER BY {q('date')} ASC LIMIT :lim"
    )
    params["lim"] = limit
    with engine1.connect() as conn:
        rows = conn.execute(sql, params).mappings().all()
        
        # ✅ Sort by date ascending (if not already)
        rows = sorted(rows, key=lambda r: r["date"])

    # craft markers from Lat/Lon if present
    markers, table_rows = [], []
    for r in rows:
        item = dict(r)
        table_rows.append(item)
        lat = item.get("Latitude")
        lon = item.get("Longitude")
        if lat is not None and lon is not None:
            markers.append({
                "lat": float(item["Latitude"]),
                "lng": float(item["Longitude"]),
                "popup": f"{item.get('Stadium Name') or ''} — "
                         f"{item.get('home') or ''} vs {item.get('away') or ''} "
                         f"({item.get('date')})"
            })

    return jsonify({"rows": table_rows, "markers": markers})    
    
    
if __name__ == "__main__":
    app.run(debug=False)

# %%
# Analysing data set

import pandas as pd
import numpy as np
from scipy.interpolate import splev, splprep
import datashader as ds
import colorcet as cc
import matplotlib.pyplot as plt
import seaborn as sns
import openrouteservice as ors
from openrouteservice import directions
import time

# %%
df = pd.read_csv('Daily_Data_EV.csv')

# %%
# split departure time into separate columns (2) for hours and minutes
df_time_split = df['departure_time'].str.split(pat=':',expand=True).astype(int)

# converting hour-minute format to float and joining back into one number
    # int value + 0. decimal value
time_num_decimal =  df_time_split[0] + df_time_split[1]/60

# put time decimal into pandas dataframe to join back with original
df_time_decimal = pd.DataFrame({'time_decimal':time_num_decimal})
df_time_int = pd.DataFrame({'time_int':df_time_split[0]})

# joining new column with time decimal and int with main dataframe
df = df.join(df_time_decimal).join(df_time_int)

# %%
# separating dataset with only EV-#, Trip, Departure Time and Decimal Time
cleaned_df = df[['EV Number', 'Trip', 'departure_time', 'time_decimal', 'time_int', 't_dist']]

# splitting coordinate cells by ',' into separate latitude and longitude columns for source and destination
    # remember to convert into float
source_df = df['source'].str.split(pat=",", expand=True).astype('float32')
destination_df = df['destination'].str.split(pat=",", expand=True).astype('float32')

# joining back into separate columns
cleaned_df = pd.concat([cleaned_df, source_df, destination_df], axis=1)

# renaming columns
cleaned_df.columns = ['EV_Number', 'Trip','departure_time', 'time_decimal', 'time_int', 'distance', 'source_lat', 'source_lon', 'destination_lat', 'destination_lon']

cleaned_df.head(10)


# %%
def prep_df(input_df):
    # split departure time into separate columns (2) for hours and minutes
    df_time_split = input_df['departure_time'].str.split(pat=':',expand=True).astype(int)

    # converting hour-minute format to float and joining back into one number
        # int value + 0. decimal value
    time_num_decimal =  df_time_split[0] + df_time_split[1]/60

    # put time decimal into pandas dataframe to join back with original
    df_time_decimal = pd.DataFrame({'time_decimal':time_num_decimal})
    df_time_int = pd.DataFrame({'time_int':df_time_split[0]})

    # joining new column with time decimal and int with main dataframe
    input_df = input_df.join(df_time_decimal).join(df_time_int)

    # separating dataset with only EV-#, Trip, Departure Time and Decimal Time
    clean_df = input_df[['EV Number', 'Trip', 'departure_time', 'time_decimal', 'time_int', 't_dist']]

    # splitting coordinate cells by ',' into separate latitude and longitude columns for source and destination
        # remember to convert into float
    source_df = input_df['source'].str.split(pat=",", expand=True).astype('float32')
    destination_df = input_df['destination'].str.split(pat=",", expand=True).astype('float32')

    # joining back into separate columns
    clean_df = pd.concat([clean_df, source_df, destination_df], axis=1)

    # renaming columns
    clean_df.columns = ['EV_Number', 'Trip','departure_time', 'time_decimal', 'time_int', 'distance', 'source_lat', 'source_lon', 'destination_lat', 'destination_lon']

    return clean_df


# %%
# histogram
sns.histplot(df["time_decimal"])
plt.title("Departure Time Histogram (All Trips)")
plt.xlabel("Time (h)")

# %%
sns.histplot(cleaned_df["distance"])
plt.title("Distance Traveled Histogram (All Trips)")

# %%
def get_coord_df(ev_set):
    # merging source and destination into one series so they are connected in order when graphed
    ev_source = ev_set[['source_lat', 'source_lon']]
    ev_destination = ev_set[['destination_lat', 'destination_lon']]

    # realization: every source is the destination of the previous trip, except for the first one (and last)
    # just need append destination one row to source and join the datasets

    # getting last row of destination
    last_row = ev_destination.iloc[-1]
    # renaming labels to match source df when concatenating so index match on concatenation
    last_row.index = ev_source.columns

    # appending last row to end with concatenate (taking transpose of row)
    return_df = pd.concat([ev_source,last_row.to_frame().T], axis=0, ignore_index=True)
    return_df.columns = ['lat', 'lon']

    return return_df

# %%
# Getting map coordinate boundaries
coord_df = get_coord_df(cleaned_df)

# 0 is 3 o'clock, 1 is 6 o'clock, 2 is 9 o'clock, 3 is 12 o'clock
box_boundaries = [coord_df['lon'].max(), coord_df['lat'].min(), coord_df['lon'].min(), coord_df['lat'].max()]

box_boundaries

# %%
coord_df

# %%
# Separating dataset into morning
morning_cleaned_df = cleaned_df[cleaned_df['time_decimal']<10]

# Separating dataset into midday
midday_cleaned_df = cleaned_df[(cleaned_df['time_decimal']>=10) & (cleaned_df['time_decimal']<17)]

# Separating dataset into evening
evening_cleaned_df = cleaned_df[(cleaned_df['time_decimal']>=17)]

# defining python dict for ease of access
df_blocks = {
    "morning": {
        "name": "morning",
        "dataframe": morning_cleaned_df
        },
    "midday": {
        "name": "midday",
        "dataframe": midday_cleaned_df
        },
    "evening": {
        "name": "evening",
        "dataframe": evening_cleaned_df
        }
}

# %%
# Show Departure Time Histogram for 3 time blocks

block_plot_index = 1

plt.figure(figsize=(15, 10))

for block in df_blocks:
    df = df_blocks[block]["dataframe"]

    # time histogram
    plt.subplot(3, 2, block_plot_index)
    plt.tight_layout()
    sns.histplot(df["time_decimal"])
    plt.title("Departure Time Histogram ({name})".format(name=block))
    plt.xlabel("Time (h)")

    block_plot_index = block_plot_index + 1

    # distance histogram
    plt.subplot(3, 2, block_plot_index)
    plt.tight_layout()
    sns.histplot(df["distance"])
    plt.title("Distance Traveled Histogram ({name})".format(name=block))

    block_plot_index = block_plot_index + 1

plt.show()

# %%
ev1_cleaned_df = cleaned_df[cleaned_df["EV_Number"]=="EV-1"]

# merging source and destination into one series so they are connected in order when graphed
ev1_source = ev1_cleaned_df[['source_lat', 'source_lon']]
ev1_destination = ev1_cleaned_df[['destination_lat', 'destination_lon']]

# realization: every source is the destination of the previous trip, except for the first one
# just need shift by one row and join the datasets

# getting last row of destination
last_row = ev1_destination.iloc[-1]
# renaming labels to match source df when concatenating
last_row.index = pd.Index(['lat', 'lon'])
ev1_source.columns = pd.Index(['lat', 'lon'])

ev1_coord_df = pd.concat([ev1_source,last_row.to_frame().T], axis=0, ignore_index=True)

# reversing because OpenRouteService API expects in (lon, lat) format
ev1_coord_list = ev1_coord_df.values.tolist()
# list comprehension reversing items in coordinate pair 
ev1_coord_list = [list(reversed(coord_pair)) for coord_pair in ev1_coord_list]
ev1_coord_list

# %%
# Accessing OpenRouteService API to get a route for the coordinates of EV1

test_coordinates = ((8.34234,48.23424),(8.34423,48.26424))

# passing key
# client = ors.Client(key='eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6ImZhOGU1ZWU3MzUzNjQ5ODZiYzBjMDA1NTdkMTRhMDEyIiwiaCI6Im11cm11cjY0In0=')

client = ors.Client(base_url='http://localhost:8080/ors')


# %% [markdown]
# ### EV 1 Route Template

# %%
# measuring time for one
ev1_start_time = time.time()

# ----------------------

# getting route from API, returns list with coordinate pair list
route = client.directions(ev1_coord_list, profile="driving-car", format="geojson")["features"][0]["geometry"]["coordinates"]

# putting list into pandas dataframe
ev1_route_df = pd.DataFrame(route, columns=['lon', 'lat'])

# plotting in matplotlib
plt.figure(figsize=(5,5))
plt.tight_layout()
ax = sns.lineplot(data=ev1_route_df, x='lon', y='lat', sort=False, lw=3, estimator=None)
plt.axis('off')

# # aesthetics
# ax.set_title("EV-1 Trip")

ev1_end_time = time.time()
ev1_elapsed_time = ev1_end_time - ev1_start_time

plt.savefig("ev-1-route-test.png", dpi=100)
plt.close()

#-----------------------

print("Elapsed Time(s): {elap}".format(elap=round(ev1_elapsed_time,2)))

# %%
def get_coord_list(df):
    # putting into list because OpenRouteService expects list/tuple
        # ex: [[34.52, -89.34], [34.23, -89.43], [34.56, -89.03]]
    coord_list = df.values.tolist()

    # list comprehension reversing items in coordinate pair 
    # reversing items because OpenRouteService API expects in (lon, lat) format and not (lat, lon)
        # ex: [[-89.34, 34.52, ], [-89.43, 34.23], [-89.03, 34.56]]
    coord_list = [list(reversed(coord_pair)) for coord_pair in coord_list]

    return coord_list

# %%
# get coordinates by time range
    # filter df by time range
    # get route for every motif / row in df
    # append (coords, dept time) to coord list

# can pretty much ignore EV number

"""
returns points in df for time range (inclusive, exclusive]
    like 14:00 to 15:00 (up to 15:00, not included)
"""
def get_points(df, time_range, err=False):
    range_df = df[(df["time_int"] >= time_range[0]) & (df["time_int"] < time_range[1])]

    # 3d list - [lon, lat, time]
        # ex: [[34.52, -89.34, 6], [34.23, -89.43, 7], [34.56, -89.03, 9]]
    list_used = []

    error_coords = []

    for row in range_df.itertuples():
        depart_time = row.time_int
        ev_num = row.EV_Number
        coords_pair = [[row.source_lon, row.source_lat],[row.destination_lon, row.destination_lat]]

        # getting route from API, returns list with coordinate pair list
            # returns something similar to input (ex just above) except with more datapoints
        try:
            # api call to OpenRouteService 
            api_response = client.directions(coords_pair, profile="driving-car", format="geojson")
            # getting just coordinates from simulated path
            route_coord_list = api_response["features"][0]["geometry"]["coordinates"]

            ## Recalculating path so coordinates are evenly distributed along the path ##
            
            # converting to np array and splitting coords into lon and lat
                # note: np.array [:,0] comma - grabs all first elems of all elem in second array
                    # returns [-89.234, -89.2423, -89.5432, etc...]
                # different from expected [:][0] - from all el in first array grab first elem (happens to be arr)
                    # returns [-89.234, 36.543]
            lon = np.array(route_coord_list)[:,0]
            lat = np.array(route_coord_list)[:,1]

            # calculating b-spline along the points of returned path
            tck, u = splprep([lon, lat], s=0)

            # recalculating points along line with new parameter values evenly spaced out
            new_u = np.linspace(0, 1, 100) # 50 points generated, 50 good enough for ~all routes based on random testing
            
            # returns 50 new longitude and latitude coordinates along simulated path
            new_lon, new_lat = splev(new_u, tck)

            ##  --------------------------------------------------------------------- ##                                               

            # append coordinate list to master points list
                # new_lon and new_lat must be same size (should be)
            for i in range(len(new_lon)):
                list_used.append([new_lon[i], new_lat[i], depart_time])

        except ors.exceptions.ApiError as e:
            # catching errors for unreachable routes (mostly ev's that stopped inside airports etc., negligible)
            api_error_code = e.args[0]
            ors_error_code = e.args[1]["error"]["code"]
            ors_error_msg = e.args[1]["error"]["message"]
            if err:
                print("API Error: ", api_error_code)
                print("Error Code: ", ors_error_code)
                print("EV Number: ", ev_num)
                print("Message: ", ors_error_msg)
                print()
        except Exception as e:
            # catching splprep errors for odd datapoints with same source and destination (i.e., near 0 distance travelled)
            if err:
                print("Error: ", e)
                print("EV Number: ", ev_num)
                print()

        
    return list_used

# %%

df_3 = cleaned_df[cleaned_df["EV_Number"]=="EV-3"]

value_arr = cleaned_df.iloc[533].tolist()
coords_pair_3 = [[float(value_arr[7]), float(value_arr[6])], [float(value_arr[9]), float(value_arr[8])]]

api_response_3 = client.directions(coords_pair_3, profile="driving-car", format="geojson")
route_coord_list_3 = api_response_3["features"][0]["geometry"]["coordinates"]

# %%

df_3 = cleaned_df[cleaned_df["EV_Number"]=="EV-3"]

value_arr = cleaned_df.iloc[14].tolist()
coords_pair_3 = [[float(value_arr[7]), float(value_arr[6])], [float(value_arr[9]), float(value_arr[8])]]

api_response_3 = client.directions(coords_pair_3, profile="driving-car", format="geojson")
route_coord_list_3 = api_response_3["features"][0]["geometry"]["coordinates"]

# splitting into x and y 
x_unsorted = np.array(route_coord_list_3)[:,0]
y_unsorted = np.array(route_coord_list_3)[:,1]

# prep
new_t = np.linspace(0, 1, 10)
new_tck, u = splprep([x_unsorted, y_unsorted], s=0)
new_points = splev(new_t, new_tck)

# Plot the original data and the smoothed spline
plt.plot(x_unsorted, y_unsorted, 'o', label='Original Data')
plt.plot(new_points[0], new_points[1], 'ro', label='PrepSpline Original Points')
plt.plot(new_points[0], new_points[1], 'b-', label='PrepSpline Original Line')
plt.legend()
plt.show()

# %%
cleaned_df[cleaned_df["EV_Number"]=="EV-4118"]

# %%
ds_points_6 = get_points(cleaned_df, [0,7], err=True)

# %%
ds_points_6 = get_points(cleaned_df, [0,8])

# %%
# putting returned array into dataframe
augmented_points_df_6 = pd.DataFrame(ds_points_6, columns=['lon', 'lat', 'time_int'])

# setting the time_int column as category type (necessary for Datashade 3D aggregation by time dimension)
augmented_points_df_6['time_int'] = augmented_points_df_6['time_int'].astype('category')

# creating canvas
ds_canvas_6 = ds.Canvas(plot_width=500, plot_height=500)

# making aggregate array
agg_6 = ds_canvas_6.points(augmented_points_df_6, 'lon', 'lat', ds.by('time_int', ds.count()))

# shading original image with black background
img_6 = ds.tf.set_background(ds.tf.shade(ds.tf.spread(agg_6.sel(time_int=6), px=3), cmap=cc.fire), "black")
img_7 = ds.tf.set_background(ds.tf.shade(ds.tf.spread(agg_6.sel(time_int=7), px=2), cmap=cc.fire), "black")

# displaying both images next to each other
ds.tf.Images(img_6, img_7)

# %%
ds_points_6 = get_points(cleaned_df, [0,24])

# %%
# putting returned array into dataframe
augmented_points_df_6 = pd.DataFrame(ds_points_6, columns=['lon', 'lat', 'time_int'])

# setting the time_int column as category type (necessary for Datashade 3D aggregation by time dimension)
augmented_points_df_6['time_int'] = augmented_points_df_6['time_int'].astype('category')

# creating canvas
ds_canvas_6 = ds.Canvas(plot_width=500, plot_height=500)

# making aggregate array
agg_6 = ds_canvas_6.points(augmented_points_df_6, 'lon', 'lat', ds.by('time_int', ds.count()))

cat_to_shade = augmented_points_df_6["time_int"].cat.categories

shade_imgs = []

for cat in cat_to_shade:
    img_name = "hour-{}".format(cat)
    save_path = "./routes-generated/bspline/"
    img = ds.tf.set_background(ds.tf.shade(ds.tf.spread(agg_6.sel(time_int=cat), px=3), cmap=cc.fire), "black")
    ds.utils.export_image(img, filename=img_name, export_path=save_path)


# %%
def generate_maps(start, end, read_path="./final_step_data_collection", save_path="./routes-generated/final"):
    for day in range(start,end):
        # get data from csv file
        csv_path = "{}/day_{}.csv".format(read_path, day)
        
        # get df with relevant data
        day_df = prep_df(pd.read_csv(csv_path))

        # get points for day
        day_points = get_points(day_df, [0,24])

        # putting points back into dataframe for plotting with Datashader
        plotting_df = pd.DataFrame(day_points, columns=['lon', 'lat', 'time_int'])

        # setting the time_int column as category type (necessary for Datashade 3D aggregation by time dimension)
        plotting_df['time_int'] = plotting_df['time_int'].astype('category')

        # creating canvas for Datashader plotting
        ds_canvas = ds.Canvas(plot_width=500, plot_height=500)

        # making aggregate array
        agg = ds_canvas.points(plotting_df, 'lon', 'lat', ds.by('time_int', ds.count()))

        # getting list of categories to iterate through (should be <=24, for 24h)
        cat_to_shade = plotting_df["time_int"].cat.categories

        # plotting image for every category (hour) in dataset
        for cat in cat_to_shade:
            img_name = "day-{}-hour-{}".format(day, cat)
            img = ds.tf.set_background(ds.tf.shade(ds.tf.spread(agg.sel(time_int=cat), px=3), cmap=cc.fire), "black")
            ds.utils.export_image(img, filename=img_name, export_path=save_path)

# %%
generate_maps(1,31, read_path="./final_step_data_collection", save_path="./routes-generated/final")

# %%
ds.tf.set_background(ds.tf.shade(ds.tf.spread(agg_6.sel(time_int=5), px=1), cmap=cc.fire), "black")

# %%
augmented_points_df_6["time_int"].cat.categories



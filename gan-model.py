# %%
# Analysing data set

import pandas as pd
import numpy as np
from scipy.interpolate import splrep, splev, interp1d, splprep, BSpline
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
# settle for the following times
    # 3:00 - 10:00 (Morning)
    # 10:00 - 17:00 (Afternoon)
    # 17:00 - 24:00 (Evenning)

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
# histogram
sns.histplot(df["time_decimal"])
plt.title("Departure Time Histogram (All Trips)")
plt.xlabel("Time (h)")

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

print(morning_cleaned_df.count(), midday_cleaned_df.count(), evening_cleaned_df.count())

# %%
# list for 24h blocks
time_blocks_24 = []

# ranges from 0 to 23 (total 24)
for x in range(0,24):
    next_x = x + 1
    dict = {
        "name" : "hour-{x}".format(x=x),
        "dataframe" : cleaned_df[(cleaned_df['time_decimal']>=x) & (cleaned_df['time_decimal']<next_x)]
    }
    time_blocks_24.append(dict)

# %%
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
# getting EV-1 set as an example to convert into map
ev1_cleaned_df = cleaned_df[cleaned_df['EV_Number']=="EV-1"]

ev1_cleaned_df

# %%
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
time_blocks_24[3]["dataframe"]

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
def get_points(df, time_range):
    range_df = df[(df["time_int"] >= time_range[0]) & (df["time_int"] < time_range[1])]

    # 3d list - [lon, lat, time]
        # ex: [[34.52, -89.34, 6], [34.23, -89.43, 7], [34.56, -89.03, 9]]
    list_used = []

    for row in range_df.itertuples():
        depart_time = row.time_int
        coords_pair = [[row.source_lon, row.source_lat],[row.destination_lon, row.destination_lat]]

        # getting route from API, returns list with coordinate pair list
            # returns something similar to input (ex just above) except with more datapoints
        try:
            api_response = client.directions(coords_pair, profile="driving-car", format="geojson")
            route_coord_list = api_response["features"][0]["geometry"]["coordinates"]

            # append coordinate list to master points list
            for coord_pair in route_coord_list:
                append_trio = coord_pair
                append_trio.append(depart_time)
                list_used.append(append_trio)

        except ors.exceptions.ApiError as e:
            # catching errors for unreachable routes (mostly ev's that stopped inside airports etc., negligible)
            api_error_code = e.args[0]
            ors_error_code = e.args[1]["error"]["code"]
            ors_error_msg = e.args[1]["error"]["message"]
            # print("API Error: ", api_error_code)
            # print("Error Code: ", ors_error_code)
            # print("Message: ", ors_error_msg)
            # print()
        
    return list_used

# %%

df_3 = cleaned_df[cleaned_df["EV_Number"]=="EV-3"]
value_arr = df_3.iloc[0].tolist()
coords_pair_3 = [[float(value_arr[7]), float(value_arr[6])], [float(value_arr[9]), float(value_arr[8])]]

api_response_3 = client.directions(coords_pair_3, profile="driving-car", format="geojson")
route_coord_list_3 = api_response_3["features"][0]["geometry"]["coordinates"]

for coord_paid_arr in route_coord_list_3:
    coord_paid_arr.append(value_arr[4])

# %%
points_3 = route_coord_list_3
points_3_df = pd.DataFrame(points_3, columns=['lon', 'lat', 'time_int'])
points_3_df

# %%
# unsorted values
x_unsorted = points_3_df['lon'].to_numpy()
y_unsorted = points_3_df['lat'].to_numpy()

# prep
new_t = np.linspace(0, 1, 50)
new_tck, u = splprep([x_unsorted, y_unsorted], s=0)
new_points = splev(new_t, new_tck)

# Plot the original data and the smoothed spline
plt.plot(x_unsorted, y_unsorted, 'o', label='Original Data')
plt.plot(new_points[0], new_points[1], 'ro', label='PrepSpline Original Points')
plt.plot(new_points[0], new_points[1], 'b-', label='PrepSpline Original Line')
plt.legend()
plt.show()

# %%
ds_points_6 = get_points(cleaned_df, [0,8])

augmented_points_df_6 = pd.DataFrame(ds_points_6, columns=['lon', 'lat', 'time_int'])
augmented_points_df_6['time_int'] = augmented_points_df_6['time_int'].astype('category')

len(augmented_points_df_6)

# %%
ds_canvas_6 = ds.Canvas(plot_width=500, plot_height=500)

agg_6 = ds_canvas_6.points(augmented_points_df_6, 'lon', 'lat', ds.by('time_int', ds.count()))

# %%
# shading original image with black background
img_6 = ds.tf.set_background(ds.tf.shade(ds.tf.spread(agg_6.sel(time_int='4'), px=2), cmap=cc.fire), "black")
img_7 = ds.tf.set_background(ds.tf.shade(ds.tf.spread(agg_6.sel(time_int='7'), px=2), cmap=cc.fire), "black")

# displaying both images next to each other
ds.tf.Images(img_6, img_7)

# %%
time_list = [6, 9]

time_block_df_6 = time_blocks_24[6]["dataframe"]

time_block_df_9 = time_blocks_24[9]["dataframe"]

ds_points_6 = get_points(time_block_df_6)

ds_points_9 = get_points(time_block_df_9)


augmented_points_df_6 = pd.DataFrame(ds_points_6, columns=['lon', 'lat', 'time_int'])

augmented_points_df_9 = pd.DataFrame(ds_points_9, columns=['lon', 'lat'])


# %%
(agg_6.data.max(), agg_9.data.max())

# %%
# shading original image with black background
img_6 = ds.tf.set_background(ds.tf.shade(agg_9, cmap=cc.fire), "black")
img_9 = ds.tf.set_background(ds.tf.shade(agg_9, cmap=cc.fire), "black")

# displaying both images next to each other
ds.tf.Images(img_6, img_9)

# %%
# morning datashader plotting
morning_points_df = pd.DataFrame(df_blocks["morning"]["point_list"], columns=['lon', 'lat'])
morning_canvas = ds.Canvas(plot_width=500, plot_height=500)
morning_agg = morning_canvas.points(morning_points_df, 'lon', 'lat', ds.count())

# midday datashader plotting
midday_points_df = pd.DataFrame(df_blocks["midday"]["point_list"], columns=['lon', 'lat'])
midday_canvas = ds.Canvas(plot_width=500, plot_height=500)
midday_agg = midday_canvas.points(midday_points_df, 'lon', 'lat', ds.count())

# evening datashader plotting
evening_points_df = pd.DataFrame(df_blocks["evening"]["point_list"], columns=['lon', 'lat'])
evening_canvas = ds.Canvas(plot_width=500, plot_height=500)
evening_agg = evening_canvas.points(evening_points_df, 'lon', 'lat', ds.count())

# %%
# shading original image with black background
original_img = ds.tf.set_background(ds.tf.shade(morning_agg, cmap=cc.fire), "black")

# shading image with spread on pixels to "densify" it
spread_img = ds.tf.set_background(ds.tf.shade(ds.tf.spread(morning_agg, px=3), cmap=cc.fire), "black")

# displaying both images next to each other
ds.tf.Images(original_img, spread_img)



# %%
# shading original image with black background
original_img = ds.tf.set_background(ds.tf.shade(midday_agg, cmap=cc.fire), "black")

# shading image with spread on pixels to "densify" it
spread_img = ds.tf.set_background(ds.tf.shade(ds.tf.spread(midday_agg, px=3), cmap=cc.fire), "black")

# displaying both images next to each other
ds.tf.Images(original_img, spread_img)

# %%
# shading original image with black background
original_img = ds.tf.set_background(ds.tf.shade(evening_agg, cmap=cc.fire), "black")

# shading image with spread on pixels to "densify" it
spread_img = ds.tf.set_background(ds.tf.shade(ds.tf.spread(evening_agg, px=3), cmap=cc.fire), "black")

# displaying both images next to each other
ds.tf.Images(original_img, spread_img)





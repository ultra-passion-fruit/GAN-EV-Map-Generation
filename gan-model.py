# %%
# Analysing data set

import pandas as pd
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
source_as_list = list(df['source'])

split_proper = [coord_pair.split(',') for coord_pair in source_as_list]

split_proper

# %%
# split departure time into separate columns (2) for hours and minutes
df_time_num_temp = df['departure_time'].str.split(pat=':',expand=True).astype(int)

# converting hour-minute format to float and joining back into one number
time_num = df_time_num_temp[0] + df_time_num_temp[1]/60

# %%
# get time block length based on min-max of dataset
# assuming 3 time blocks in a day (morning, afternoon, evening)
time_block_length = (time_num.max()-time_num.min()) / 3

# get range idea to decide block
prev_time_block = time_num.min()
for i in range (1,4):
    second_time_block = prev_time_block + time_block_length
    print("{start} - {end}".format(start=prev_time_block,end=second_time_block))
    prev_time_block = second_time_block


# %%
# settle for the following times
    # 3:00 - 10:00 (Morning)
    # 10:00 - 17:00 (Afternoon)
    # 17:00 - 24:00 (Evenning)

# put time decimal into pandas dataframe to join back with original
df_time_decimal = pd.DataFrame({'time_decimal':time_num})

# joining new column with time decimal with main dataframe
df = df.join(df_time_decimal)

# %%
# histogram
sns.histplot(df["time_decimal"])
plt.title("Departure Time Histogram (All Trips)")
plt.xlabel("Time (h)")

# %%
# separating dataset with only EV-#, Trip, Departure Time and Decimal Time
cleaned_df = df[['EV Number', 'Trip', 'departure_time', 'time_decimal', 't_dist']]

# splitting coordinate cells by ',' into separate latitude and longitude columns for source and destination
    # remember to convert into float
source_df = df['source'].str.split(pat=",", expand=True).astype('float32')
destination_df = df['destination'].str.split(pat=",", expand=True).astype('float32')

# joining back into separate columns
cleaned_df = pd.concat([cleaned_df, source_df, destination_df], axis=1)

# renaming columns
cleaned_df.columns = ['EV_Number', 'Trip','departure_time', 'time_decimal', 'distance', 'source_lat', 'source_lon', 'destination_lat', 'destination_lon']

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

    # return concatenated df
    return pd.concat([ev_source,last_row.to_frame().T], axis=0, ignore_index=True)

# %%
# Getting map coordinate boundaries
coord_df = get_coord_df(cleaned_df)

# 0 is 3 o'clock, 1 is 6 o'clock, 2 is 9 o'clock, 3 is 12 o'clock
box_boundaries = [coord_df['source_lon'].max(), coord_df['source_lat'].min(), coord_df['source_lon'].min(), coord_df['source_lat'].max()]

box_boundaries

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

# %% [markdown]
# ### High-Rez Plotting for Multiple

# %%
# Total 7000 EVs
# EV1 Elapsed: ~0.07s
# Time Estimate: 7000 * 0.07 = 490s = ~8.167 mins

# Lineplotting "High-Rez" Plotting

num_evs = 2000
block_used = df_blocks["morning"]

for i in range(1,num_evs):
    # getting data for one EV
    ev_name = "EV-" + str(i)
    df_block = block_used["dataframe"]
    ev_df = df_block[df_block['EV_Number']==ev_name]

    # check if not empty (i.e., if there is an entry for that EV_#)
    if(not ev_df.empty):
        # get dataframe with just motif coords (lat, lon)
        coord_df = get_coord_df(ev_df)

        # put dataframe into coord list format for ORS API
            # ex: [[34.52, -89.34], [34.23, -89.43], [34.56, -89.03]]
        coord_list = get_coord_list(coord_df)

        # getting route from API, returns list with coordinate pair list
            # returns something similar to input except with more datapoints
        try:
            api_response = client.directions(coord_list, profile="driving-car", format="geojson")
            route_coord_list = api_response["features"][0]["geometry"]["coordinates"]
        except ors.exceptions.ApiError as e:
            api_error_code = e.args[0]
            ors_error_code = e.args[1]["error"]["code"]
            ors_error_msg = e.args[1]["error"]["message"]
            print("API Error: ", api_error_code)
            print("Error Code: ", ors_error_code)
            print("EV Number: ", ev_name)
            print("Message: ", ors_error_msg)
            print()

        # put coordinate pair list back into df for plotting
        route_df = pd.DataFrame(route_coord_list, columns=['lon', 'lat'])

        sns.despine(left=True, bottom=True)
        plt.tight_layout()
        ax = sns.lineplot(data=route_df, x='lon', y='lat', color='blue', alpha=0.3, sort=False, lw=3, estimator=None)
        plt.axis('off')

plt.savefig("routes/{block_name}-routes-{ev_num}.png".format(block_name=block_used["name"], ev_num=num_evs), dpi=100)
plt.close()


# %%
# Collect points for datashader

# Morning

# list to hold datashader points
morning_ds_points = []

# measuring time
block_start_time = time.time()
time_list = []
ev_tl_index = 0
ev_time_list = [100*2**x for x in range(1,7)]

num_evs = 7001
block_used = df_blocks["morning"]
list_used = morning_ds_points

for i in range(1,num_evs):
    # getting data for one EV
    ev_name = "EV-" + str(i)
    df_block = block_used["dataframe"]
    ev_df = df_block[df_block['EV_Number']==ev_name]

    # check if not empty (i.e., if there is an entry for that EV_#)
    if(not ev_df.empty):
        # get dataframe with just motif coords (lat, lon)
        coord_df = get_coord_df(ev_df)

        # put dataframe into coord list format for ORS API
            # ex: [[34.52, -89.34], [34.23, -89.43], [34.56, -89.03]]
        coord_list = get_coord_list(coord_df)

        # getting route from API, returns list with coordinate pair list
            # returns something similar to input except with more datapoints
        try:
            api_response = client.directions(coord_list, profile="driving-car", format="geojson")
            route_coord_list = api_response["features"][0]["geometry"]["coordinates"]

            # append coordinate list to master points list
            for coord_pair in route_coord_list:
                list_used.append(coord_pair)

        except ors.exceptions.ApiError as e:
            api_error_code = e.args[0]
            ors_error_code = e.args[1]["error"]["code"]
            ors_error_msg = e.args[1]["error"]["message"]
            print("API Error: ", api_error_code)
            print("Error Code: ", ors_error_code)
            print("EV Number: ", ev_name)
            print("Message: ", ors_error_msg)
            print()

    if i in ev_time_list:
        end_time = time.time()
        elapsed_time = end_time - block_start_time
        time_list.append([ev_time_list[ev_tl_index], elapsed_time])
        ev_tl_index = ev_tl_index + 1

    if i == num_evs-1:
        end_time = time.time()
        elapsed_time = end_time - block_start_time
        time_list.append([num_evs-1, elapsed_time])
        ev_tl_index = ev_tl_index + 1


# %%
morning_time_run = pd.DataFrame(time_list, columns=["number_of_evs", "time"])
sns.lineplot(morning_time_run, x='number_of_evs', y='time')

# %%
morning_ds_points[0]

# %%
len(morning_ds_points)

# %%
#### Collect points for datashader ###

# measuring time things
block_start_time = time.time()
time_list = []
ev_time_list = [100*2**x for x in range(1,7)]

# number of ev's to go through (max: 7001)
    # 7000 + 1 to account for index
num_evs = 7001

# iterate through all time blocks and generate points
for block in df_blocks:
    block_used = df_blocks[block]
    list_used = []
    ev_tl_index = 0

    # iterating over all ev's in list
    # must go per ev (and not line by line) so as to get the points between an individual ev's routes
    for i in range(1,num_evs):
        # getting data for one EV
        ev_name = "EV-" + str(i)
        df_block = block_used["dataframe"]
        ev_df = df_block[df_block['EV_Number']==ev_name]

        # check if not empty (i.e., if there is an entry for that EV_#)
            # necessary after splitting for time (EV# may not be present in all time blocks)
        if(not ev_df.empty):
            # get dataframe with just motif coords (lat, lon)
            coord_df = get_coord_df(ev_df)

            # put dataframe into coord list format for ORS API
                # ex: [[34.52, -89.34], [34.23, -89.43], [34.56, -89.03]]
            coord_list = get_coord_list(coord_df)

            # getting route from API, returns list with coordinate pair list
                # returns something similar to input (ex just above) except with more datapoints
            try:
                api_response = client.directions(coord_list, profile="driving-car", format="geojson")
                route_coord_list = api_response["features"][0]["geometry"]["coordinates"]

                # append coordinate list to master points list
                for coord_pair in route_coord_list:
                    list_used.append(coord_pair)

            except ors.exceptions.ApiError as e:
                # catching errors for unreachable routes (mostly ev's that stopped inside airports etc.)
                api_error_code = e.args[0]
                ors_error_code = e.args[1]["error"]["code"]
                ors_error_msg = e.args[1]["error"]["message"]
                # print("API Error: ", api_error_code)
                # print("Error Code: ", ors_error_code)
                # print("EV Number: ", ev_name)
                # print("Message: ", ors_error_msg)
                # print()

        # storing time values in a list to plot later
        if i in ev_time_list:
            end_time = time.time()
            elapsed_time = end_time - block_start_time
            time_list.append([ev_time_list[ev_tl_index], elapsed_time])
            ev_tl_index = ev_tl_index + 1
        if i == num_evs-1:
            end_time = time.time()
            elapsed_time = end_time - block_start_time
            time_list.append([num_evs-1, elapsed_time, block])
            ev_tl_index = ev_tl_index + 1

    # saving points to time block list
    df_blocks[block]["point_list"] = list_used

# %%
print(len(df_blocks["morning"]["point_list"]))
print(len(df_blocks["midday"]["point_list"]))
print(len(df_blocks["evening"]["point_list"]))

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



# %%
# # Total 7000 EVs
# # EV1 Elapsed: ~3.65s
# # Time Estimate: 7000 * 3.65 = 25,550s
# # ---> 7h...

# # Low-Polly Plotting lol

# plt.figure(figsize=(17,20))

# num_evs = 30
# evs_per_row = 5

# for i in range(1,num_evs):
#     # getting data for one EV
#     ev_name = "EV-" + str(i)
#     ev_df = cleaned_df[cleaned_df['EV_Number']==ev_name]

#     plotable_set = get_coord_df(ev_df)

#     plt.subplot(int(num_evs/evs_per_row), evs_per_row, i) # 2 rows, 5 columns, ith graph
#     sns.despine(left=True, bottom=True)
#     plt.tight_layout()
#     ax = sns.lineplot(data=plotable_set, x='source_lon', y='source_lat', sort=False, lw=3)
    
#     # aesthetics
#     ax.set_title(ev_name + " Trip")

# plt.savefig("low-polly-routes.png")
# plt.close() 


# %%





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
from tqdm import tqdm
import time


df = pd.read_csv('Daily_Data_EV.csv')

def prep_df(input_df):
    """
    Cleans original dataset from csv and keeps only relevant columns

    Parameters:
        - pandas dataframe

    Retunrs:
        - pandas dataframe (cleaned)
    """
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


# get coordinates by time range
    # filter df by time range
    # get route for every motif / row / source-destin pair in df
    # append (coords, dept time) to coord list

# can pretty much ignore EV number

def get_points(df, time_range, err=False):
    """
    Calculates all routing points in df for time range (inclusive, exclusive]
        like 14:00 to 15:00 (up to 15:00, not included)

    Params:
        - df (pandas dataframe)
        - time_range (list size 2)
        - err=False (whether to show routing errors for routes that fail)

    Returns:
    """
    range_df = df[(df["time_int"] >= time_range[0]) & (df["time_int"] < time_range[1])]

    # 3d list - [lon, lat, time]
        # ex: [[34.52, -89.34, 6], [34.23, -89.43, 7], [34.56, -89.03, 9]]
    list_used = []

    for row in tqdm(range_df.itertuples(), total=range_df.shape[0]):
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


def generate_maps(start, end, size, read_path="./final_step_data_collection", save_path="./routes-generated/final"):
    """
    Generates maps from .csv files for each day and saves to some directory.

    Usage:
        generate_maps(1,31, read_path="./final_step_data_collection", save_path="./routes-generated/final")

        Reads all 30 .csv files in /final_set_data_collection directory and saves to /routes-generated/final

    Parameters:
        - start (start day)
        - end (end day)
        - read_path (directory where .csv files are located)
        - save_path

    Returns: Nothing
    """
    tqdm.write("Shading Maps Process")
    for day in tqdm(range(start,end)):
        # get data from csv file
        csv_path = "{}/day_{}.csv".format(read_path, day)
        
        # get df with relevant data
        day_df = prep_df(pd.read_csv(csv_path))

        tqdm.write(f"\nSimulating Routes in Day {day}")
        # get points for day
        day_points = get_points(day_df, [0,24])

        # putting points back into dataframe for plotting with Datashader
        plotting_df = pd.DataFrame(day_points, columns=['lon', 'lat', 'time_int'])

        # setting the time_int column as category type (necessary for Datashade 3D aggregation by time dimension)
        plotting_df['time_int'] = plotting_df['time_int'].astype('category')

        # creating canvas for Datashader plotting
        ds_canvas = ds.Canvas(plot_width=size, plot_height=size)

        # making aggregate array
        agg = ds_canvas.points(plotting_df, 'lon', 'lat', ds.by('time_int', ds.count()))

        # getting list of categories to iterate through (should be <=24, for 24h)
        cat_to_shade = plotting_df["time_int"].cat.categories
        
        tqdm.write(f"\nShading maps for every hour in Day {day}")
        # plotting image for every category (hour) in dataset
        for cat in tqdm(cat_to_shade):
            img_name = "day-{}-hour-{}".format(day, cat)
            img = ds.tf.set_background(ds.tf.shade(ds.tf.spread(agg.sel(time_int=cat), px=3), cmap=cc.fire), "black")
            ds.utils.export_image(img, filename=img_name, export_path=save_path)

    tqdm.write(f"Finished Shading Process. Saved to {save_path}")
# Creating ors Client object to access API

# passing key, only for web API
# client = ors.Client(key='')

# Passing localhost port 8000 where Docker container is running
client = ors.Client(base_url='http://localhost:8080/ors')

# creating all maps from local directory of .csv files, saving to other local directory
generate_maps(2,4, 256, read_path="./final_step_data_collection", save_path="./routes-generated/test-final")
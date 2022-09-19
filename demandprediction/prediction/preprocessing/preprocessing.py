import logging
import multiprocessing
import os
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from joblib import Parallel
from joblib import delayed
from tqdm import tqdm
from shapely.geometry.polygon import Polygon
import time
import geopandas
from geopandas import GeoDataFrame

from demandprediction.utils.shared_filenames import (
    get_grid_filename,
    get_trips_filename,
    get_events_filename,
    get_grid_with_border_filename,
    get_citywide_filename
)
from demandprediction.utils.utils import get_data_start_date
import demandprediction.prediction.preprocessing.constants as preprocessing_constants


class DataPreprocessor:
    def __init__(self, data_directory: Path, x_limits: Union[tuple, None], y_limits: Union[tuple, None]):
        self.data_directory = data_directory
        self.logger = logging.getLogger(__name__)

        self.x_limits = x_limits
        self.y_limits = y_limits

    def generate_citywide_timeseries(self, stop_date, step_size):
        """
        Checks if citywide timeseries with given step_size exists on hard drive.
        If not the case creates and saves it.
        :param stop_date: Upper boundary of the inspected date range
        :param step_size: Size of the step from t_1 to t_2, from t_2 to t_3, etc.
            e.g.:   30 --> 30 min intervals
                    15 --> 15 min intervals
                    7.5 --> 7.5 min intervals
        :return: Filepath to the saved DataFrame.
        """
        data_filename = get_trips_filename(stop_date=stop_date)

        filepath = os.path.join(
            self.data_directory,
            get_citywide_filename(step_size=step_size, stop_date=stop_date)
        )

        if not os.path.isfile(filepath):
            df = self._get_citywide_df(
                data_filepath=self.data_directory.joinpath(data_filename),
                step_size=step_size,
                stop_date=stop_date
            )
            df.to_hdf(filepath, key='citywide', complevel=6)

        return filepath

    def generate_grid_with_border(self, stop_date, number_of_rows, number_of_columns,
                                  step_size, feature_name, border_width):
        """
        Checks if grid in given resolution with borders is saved on hard drive.
        If not the case creates and saves it.
        :param stop_date: Upper boundary of the inspected date range
        :param number_of_rows: Number of rows of the grid
        :param number_of_columns: Number of columns of the grid
        :param step_size: Size of the step from t_1 to t_2, from t_2 to t_3, etc.
            e.g.:   30 --> 30 min intervals
                    15 --> 15 min intervals
                    7.5 --> 7.5 min intervals
        :param feature_name: Name of the feature from which the data is obtained
            e.g.:   'demand',
                    'event',
                    'weather',
                    etc.
        :param border_width: Width of the border
        :return: Filepath to the saved DataFrame.
        """
        if feature_name == 'demand':
            data_filename = get_trips_filename(stop_date=stop_date)
        elif feature_name == 'event':
            data_filename = get_events_filename(stop_date=stop_date)
        else:
            raise ValueError("results: feature_name must be either demand or event")

        filepath = os.path.join(
            self.data_directory,
            get_grid_with_border_filename(
                feature_name,
                number_of_rows,
                number_of_columns,
                self.x_limits[0],
                self.x_limits[1],
                self.y_limits[0],
                self.y_limits[1],
                step_size,
                border_width,
                stop_date
            )
        )

        # Check if file already exists and create it if it isn´t the case
        if not os.path.isfile(filepath):
            df_border = self._get_border_df(
                data_filepath=os.path.join(
                    self.data_directory,
                    data_filename
                ),
                border_width=border_width,
                step_size=step_size,
                stop_date=stop_date
            )
            df_border.to_hdf(filepath, key='border', complevel=6)

            df_grid = self._get_grid_as_df(
                data_filepath=os.path.join(
                    self.data_directory,
                    data_filename
                ),
                number_of_rows=number_of_rows,
                number_of_columns=number_of_columns,
                step_size=step_size,
                stop_date=stop_date
            )
            df_grid.to_hdf(filepath, key='even_grid', complevel=6)

        return filepath

    def generate_window_grid(self, stop_date, number_of_rows, number_of_columns,
                             step_size, feature_name):
        """
        Checks if grid in given resolution is saved on hard drive. If not the case creates and saves it.
        :param stop_date: Upper boundary of the inspected date range
        :param number_of_rows: Number of rows of the grid
        :param number_of_columns: Number of columns of the grid
        :param step_size: Size of the step from t_1 to t_2, from t_2 to t_3, etc.
            e.g.:   30 --> 30 min intervals
                    15 --> 15 min intervals
                    7.5 --> 15 min intervals
        :param feature_name: Name of the feature from which the data is obtained
            e.g.:   'demand',
                    'event',
                    'weather',
                    etc.
        :return: Filepath to the saved DataFrame.
        """
        # Check if the value for feature_name is appropriate
        if feature_name == 'demand':
            data_filename = get_trips_filename(stop_date=stop_date)
        elif feature_name == 'event':
            data_filename = get_events_filename(stop_date=stop_date)
        else:
            raise ValueError("results: feature_name must be either demand or event")

        filepath = os.path.join(
            self.data_directory,
            get_grid_filename(
                feature_name,
                number_of_rows,
                number_of_columns,
                self.x_limits[0],
                self.x_limits[1],
                self.y_limits[0],
                self.y_limits[1],
                step_size,
                stop_date
            )
        )

        # Check if file already exists and create it if it isn´t the case
        if not os.path.isfile(filepath):
            df = self._get_grid_as_df(
                data_filepath=os.path.join(
                    self.data_directory,
                    data_filename
                ),
                number_of_rows=number_of_rows,
                number_of_columns=number_of_columns,
                step_size=step_size,
                stop_date=stop_date
            )
            df.to_hdf(filepath, key='even_grid', complevel=6)

        return filepath

    def _get_citywide_df(self, data_filepath, step_size, stop_date):
        # noinspection PyTypeChecker
        df = pd.read_hdf(data_filepath)  # type: pd.DataFrame
        df = df[['mercator_start_lat', 'mercator_start_lon', 'timestamp_start']]

        # creating hourly DateTimeIndex from first date to last date
        start_date = pd.to_datetime(get_data_start_date(start_hour=6))
        df = df[df['timestamp_start'] >= start_date]

        date_range = self.get_date_range_between_dates(
            start_date=start_date,
            end_date=stop_date,
            freq=f'{step_size}min',
            end_inclusive=False
        )
        custom_dates = pd.DatetimeIndex(date_range)
        df = self._update_df_index(df, start_date)

        timeseries_df = self._aggregate_by_dt_index(df, custom_dates)
        timeseries_df.columns = ['demand']

        return timeseries_df

    def _get_grid_as_df(self, data_filepath, number_of_rows, number_of_columns,
                        step_size, stop_date) -> pd.DataFrame:
        """
        Calculates the grid timeseries and saves them in one pandas DataFrame.

        :param data_filepath: Filepath to the complete dataset.
        :param number_of_rows: Number of rows of the grid
        :param number_of_columns: Number of columns of the grid
        :param step_size: Size of the step from t_1 to t_2, from t_2 to t_3, etc.
                e.g.:   30 --> 30 min intervals
                        15 --> 15 min intervals
                        7.5 --> 15 min intervals
        :param stop_date: Datetime after which no predictions will be made
        :return: pandas DataFrame where each gridcell timeseries has its own column
        """

        # Extract area
        # noinspection PyTypeChecker
        combined_df = pd.read_hdf(data_filepath)  # type: pd.DataFrame

        combined_df = combined_df[['mercator_start_lat', 'mercator_start_lon', 'timestamp_start']]

        combined_df = combined_df[(combined_df['mercator_start_lon'] >= self.x_limits[0]) &
                                  (combined_df['mercator_start_lon'] <= self.x_limits[1]) &
                                  (combined_df['mercator_start_lat'] >= self.y_limits[0]) &
                                  (combined_df['mercator_start_lat'] <= self.y_limits[1])]

        # creating hourly DateTimeIndex from first date to last date
        start_date = pd.to_datetime(get_data_start_date())

        date_range = self.get_date_range_between_dates(
            start_date=start_date,
            end_date=stop_date,
            freq=f'{step_size}min',
            end_inclusive=False
        )
        custom_dates = pd.DatetimeIndex(date_range)
        combined_df = self._update_df_index(combined_df, start_date)

        # creating the column_names of the gridcells to compute
        gridcell_list = [(row, col) for row in range(number_of_rows) for col in range(number_of_columns)]

        # defining the number of cores to use
        cores = multiprocessing.cpu_count()

        self.logger.info(f'Splitting grid into {number_of_rows * number_of_columns}'
                         f' subcells on {cores} processes.')

        # starting parallel computation
        executor = Parallel(n_jobs=cores, backend='multiprocessing')
        tasks = (delayed(self._timeseries_df_for_gridcell)
                 (combined_df, gridcell, number_of_rows, number_of_columns,
                  custom_dates)
                 for gridcell in tqdm(gridcell_list))

        # retrieving results
        result_list = executor(tasks)
        resulting_dfs, number_of_entries = ([df for df, _ in result_list],
                                            sum(num_of_items for _, num_of_items in result_list))

        # assertion that the number of entries before and after the gridcell computation are the same
        assert number_of_entries == len(combined_df.index)

        # create one pandas DataFrame containing all the timeseries
        endresult = pd.concat(resulting_dfs, axis=1)

        return endresult

    def _get_border_df(self, data_filepath, border_width, step_size, stop_date):
        # noinspection PyTypeChecker
        combined_df = pd.read_hdf(data_filepath)  # type: pd.DataFrame

        combined_df = combined_df[['mercator_start_lat', 'mercator_start_lon', 'timestamp_start']]

        combined_df = combined_df[(combined_df['mercator_start_lon'] >= self.x_limits[0] - border_width) &
                                  (combined_df['mercator_start_lon'] <= self.x_limits[1] + border_width) &
                                  (combined_df['mercator_start_lat'] >= self.y_limits[0] - border_width) &
                                  (combined_df['mercator_start_lat'] <= self.y_limits[1] + border_width)]

        # creating hourly DateTimeIndex from first date to last date
        start_date = pd.to_datetime(get_data_start_date())

        date_range = self.get_date_range_between_dates(
            start_date=start_date,
            end_date=stop_date,
            freq=f'{step_size}min',
            end_inclusive=False
        )
        custom_dates = pd.DatetimeIndex(date_range)
        combined_df = self._update_df_index(combined_df, start_date)
        gdf = GeoDataFrame(combined_df,
                           geometry=geopandas.points_from_xy(
                               combined_df.mercator_start_lon,
                               combined_df.mercator_start_lat
                           ),
                           crs='EPSG:25832')

        # Create polygons for the 4 borders
        north_border = Polygon([
            (self.x_limits[0], self.y_limits[1]),
            (self.x_limits[1], self.y_limits[1]),
            (self.x_limits[1] + border_width, self.y_limits[1] + border_width),
            (self.x_limits[0] - border_width, self.y_limits[1] + border_width)
        ])
        east_border = Polygon([
            (self.x_limits[1], self.y_limits[0]),
            (self.x_limits[1] + border_width, self.y_limits[0] - border_width),
            (self.x_limits[1] + border_width, self.y_limits[1] + border_width),
            (self.x_limits[1], self.y_limits[1])
        ])
        south_border = Polygon([
            (self.x_limits[0] - border_width, self.y_limits[0] - border_width),
            (self.x_limits[1] + border_width, self.y_limits[0] - border_width),
            (self.x_limits[1], self.y_limits[0]),
            (self.x_limits[0], self.y_limits[0])
        ])
        west_border = Polygon([
            (self.x_limits[0] - border_width, self.y_limits[0] - border_width),
            (self.x_limits[0], self.y_limits[0]),
            (self.x_limits[0], self.y_limits[1]),
            (self.x_limits[0] - border_width, self.y_limits[1] + border_width)
        ])
        borders = {
            'north': north_border,
            'east': east_border,
            'south': south_border,
            'west': west_border
        }
        border_dfs = []
        start_time = time.time()
        # Filter the trips based on if they are located in the border areas
        for border_name, border_polygon in borders.items():
            self.logger.info(f'Filtering trips for {border_name} border')
            border_gdf = gdf[gdf.within(border_polygon)].copy()
            self.logger.info(f'{len(border_gdf)} trips found in {border_name} border')
            border_df = pd.DataFrame(border_gdf.drop(columns='geometry'))

            timeseries_border_df = self._aggregate_by_dt_index(border_df, custom_dates)
            timeseries_border_df.columns = [border_name]
            border_dfs.append(timeseries_border_df)

        self.logger.info(f'Took {time.time() - start_time} seconds')

        endresult = pd.concat(border_dfs, axis=1)

        return endresult

    def _timeseries_df_for_gridcell(self, combined_df, gridcell, number_of_rows, number_of_columns,
                                    dt_index):
        """
        Creates timeseries for a given gridcell.

        :param combined_df: pandas DataFrame containing all trips
        :param gridcell: tuple (row, column)
        :param number_of_rows: Number of rows of the grid
        :param number_of_columns: Number of columns of the grid
        :param dt_index: DateTimeIndex to aggregate by
        :return: boolean, specifying whether to focus on center of munich or the whole area
        """
        width = (self.x_limits[1] - self.x_limits[0]) / number_of_columns
        height = (self.y_limits[1] - self.y_limits[0]) / number_of_rows

        # creating limits/borders for current gridcell
        x_limits = (
            self.x_limits[0] + gridcell[1] * width, self.x_limits[0] + (gridcell[1] + 1) * width)
        y_limits = (
            self.y_limits[1] - (gridcell[0] + 1) * height, self.y_limits[1] - gridcell[0] * height)

        # slicing the DataFrame to contain the data for the gridcell
        current_df = combined_df[(x_limits[0] < combined_df['mercator_start_lon']) &
                                 (combined_df['mercator_start_lon'] <= x_limits[1]) &
                                 (y_limits[0] < combined_df['mercator_start_lat']) &
                                 (combined_df['mercator_start_lat'] <= y_limits[1])].copy()

        # create complete timeseries
        timeseries_df = self._aggregate_by_dt_index(current_df, dt_index)

        # rename column of DataFrame
        timeseries_df.columns = [str((gridcell[0], gridcell[1]))]

        # return resulting DataFrame and number of entries
        return timeseries_df, len(current_df.index)

    @staticmethod
    def _aggregate_by_dt_index(df, dt_index) -> pd.DataFrame:
        """
        Creates complete timeseries without missing timesteps.

        :param df: DataFrame of which the timeseries should be created
        :param dt_index: DateTimeIndex to aggregate by
        :return: pandas DataFrame containing the complete timeseries
        """
        # aggregate the DataFrame by the specified datetime index
        current_output = df.groupby([dt_index.searchsorted(df.index)]).count()
        current_output = current_output[['mercator_start_lon']]
        current_output.columns = ['number_of_rides']

        # add missing timesteps with value 0
        dates_to_add = []
        for i in range(1, len(dt_index) + 1):
            if i not in current_output.index:
                dates_to_add.append(i)
        values_list = [0] * len(dates_to_add)
        dates_df = pd.DataFrame(values_list, index=dates_to_add, columns=['number_of_rides'])
        current_output = pd.concat([current_output, dates_df])

        # sort index to ensure the right order of the timeseries
        current_output.sort_index(inplace=True)
        current_output.set_index(dt_index, inplace=True)

        # deal with timechange
        for time_change_date in preprocessing_constants.time_change_dates:
            if time_change_date in current_output.index:
                time_change_pos = current_output.index.get_loc(time_change_date)
                interpolated_rides = (current_output.iloc[time_change_pos - 1]['number_of_rides']
                                      + current_output.iloc[time_change_pos + 1]['number_of_rides']) / 2
                current_output.iloc[time_change_pos]['number_of_rides'] = interpolated_rides

        assert not current_output.isna().values.any()

        return current_output

    @staticmethod
    def check_holiday(row, holiday):
        """
        Function for applying on a pandas Series containing datetimes.
        The function determines if the datetime is inside the given holiday.
        :param row: row of pandas DataFrame
        :param holiday: row of pandas DataFrame
        :return: 1 if datetime is during octoberfest, 0 if datetime isn´t during octoberfest
        """

        for daterange in preprocessing_constants.holidays_dict[holiday]:
            assert daterange[1] > daterange[0]
            if daterange[0] <= row <= daterange[1]:
                return 1
        return 0

    @staticmethod
    def check_octoberfest(row):
        """
        Function for applying on a pandas DataFrame. When the index is a DateTimeIndex,
        the function determines if the datetime is during an octoberfest.
        :param row: row of pandas DataFrame
        :return: 1 if datetime is during octoberfest, 0 if datetime isn´t during octoberfest
        """

        for octoberfest in preprocessing_constants.octoberfest_dates:
            if octoberfest[0] <= row.name <= octoberfest[1]:
                return 1
        return 0

    @staticmethod
    def get_date_range_between_dates(start_date: str, end_date: str, freq: str,
                                     end_inclusive: bool = False):
        if not end_inclusive:
            end_date = pd.to_datetime(end_date)
            end_date = end_date - np.timedelta64(1, 's')
        return pd.date_range(start_date, end_date, freq=freq)

    @staticmethod
    def _update_df_index(df, start_date):
        df.set_index('timestamp_start', inplace=True, drop=True)

        # slightly shift off start_date if it´s exactly the first entry in the DateTimeIndex
        if start_date in df.index:
            df.index = df.index.where(
                df.index != start_date,
                start_date + pd.DateOffset(seconds=1)
            )
        return df

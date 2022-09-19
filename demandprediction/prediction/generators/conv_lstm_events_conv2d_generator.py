import holidays
import pandas as pd
import numpy as np
import tensorflow.keras as keras

from demandprediction.prediction.preprocessing.preprocessing import DataPreprocessor


class ConvLSTMEventsConv2DGenerator(keras.utils.Sequence):
    """
    class inheriting from keras.utils.Sequence. Works as a input generator for the ConvLSTMEventsMLPModel
    It loads only the needed data and is therefore memory saving.
    """

    def __init__(self, trips_filepath, train_val_test, number_of_rows, number_of_columns, lags,
                 batch_size, number_of_prediction_timesteps, events_filepath, bidirectional_events_timesteps,
                 only_input=False):
        """
        Initializes the input generator.

        :param train_val_test: 'train', 'val' or 'test'. Specifying which output is wished
        :param lags: number of lags to use
        :param number_of_rows: number of rows of the grid
        :param number_of_columns: number of columns of the grid
        :param batch_size: desired batch_size
        :param only_input: boolean specifying if only the input or the input and
        the ground truth should be returned
        """

        self.batch_size = batch_size
        self.lags = lags
        self.train_val_test = train_val_test
        self.only_input = only_input
        self.number_of_rows = number_of_rows
        self.number_of_columns = number_of_columns
        self.timesteps_in_test_set = number_of_prediction_timesteps + lags

        self.trips_filepath = trips_filepath

        # Set start and stop index depending on training, validation or test
        if self.train_val_test == 'train':
            self.start = 0
            self.stop = int(len(pd.read_hdf(self.trips_filepath, key='even_grid',
                                            stop=-self.timesteps_in_test_set).index) * 0.67)
        elif self.train_val_test == 'val':
            self.start = int(len(pd.read_hdf(self.trips_filepath, key='even_grid',
                                             stop=-self.timesteps_in_test_set).index) * 0.67)
            self.stop = -self.timesteps_in_test_set
        else:
            self.start = -self.timesteps_in_test_set
            self.stop = None

        self.events_filepath = events_filepath
        self.bidirectional_events_timesteps = bidirectional_events_timesteps

    def __len__(self):
        """
        :return: the number of batches that can be generated per Epoch
        """
        return (len(pd.read_hdf(self.trips_filepath, key='even_grid', start=self.start,
                                stop=self.stop).index) - self.lags) // self.batch_size

    # noinspection PyTypeChecker
    def __getitem__(self, idx):
        """
        prepares the data in the desired batch_size

        :param idx: number of batch to be generated
        :return: batch containing the data in the needed shape
        """
        cur_stop = self.start + (idx + 1) * self.batch_size + self.lags

        if self.train_val_test == 'test':
            if cur_stop == 0:
                cur_stop = None

        # load the needed data
        trips_df = pd.read_hdf(self.trips_filepath, key='even_grid',
                               start=self.start + idx * self.batch_size,
                               stop=cur_stop)  # type: pd.DataFrame
        events_df = pd.read_hdf(self.trips_filepath, key='even_grid',
                                start=self.start + idx * self.batch_size,
                                stop=cur_stop)  # type: pd.DataFrame

        # check if DataFrame has the correct length
        assert len(trips_df.index) == self.lags + self.batch_size
        # initialize the array in the needed shape
        x_convlstm = np.zeros(
            shape=(self.batch_size, self.lags, self.number_of_rows, self.number_of_columns, 1),
            dtype='uint16'
        )

        x_events_conv2d = np.zeros(
            shape=(self.batch_size, self.number_of_rows, self.number_of_columns,
                   2 * self.bidirectional_events_timesteps + 1),
            dtype='uint16'
        )

        # fill the created array
        for timestep in range(self.batch_size):
            cur_timestep_array = trips_df[timestep:timestep + self.lags].values
            cur_timestep_array = cur_timestep_array.reshape(
                self.lags, self.number_of_rows, self.number_of_columns, 1)
            x_convlstm[timestep] = cur_timestep_array

            events_array = events_df[timestep:timestep + self.lags * 2 + 1].values
            events_array = events_array.reshape(events_array.shape[0],
                                                self.number_of_rows, self.number_of_columns)
            events_array = np.moveaxis(events_array, 0, -1)
            cur_timestep_array = np.zeros(shape=(self.number_of_rows, self.number_of_columns,
                                                 2 * self.bidirectional_events_timesteps + 1))
            cur_timestep_array[:, :, :events_array.shape[2]] = events_array
            x_events_conv2d[timestep] = cur_timestep_array

        # generate additional info and save it in the second array
        hour_series = trips_df.index[self.lags:].hour
        hour_series_ohe = np.zeros(shape=(self.batch_size, 23))
        for step, hour in enumerate(hour_series):
            if hour != 0:
                hour_series_ohe[step, hour - 1] = 1
        day_series = trips_df.index[self.lags:].weekday
        day_series_ohe = np.zeros(shape=(self.batch_size, 6))
        for step, day in enumerate(day_series):
            if day != 0:
                day_series_ohe[step, day - 1] = 1

        dt_series = pd.Series(pd.to_datetime(trips_df.index[self.lags:]))
        winter_holiday_series = dt_series.apply(DataPreprocessor.check_holiday, holiday='winter')
        easter_holiday_series = dt_series.apply(DataPreprocessor.check_holiday, holiday='easter')
        whitsun_holiday_series = dt_series.apply(DataPreprocessor.check_holiday, holiday='whitsun')
        summer_holiday_series = dt_series.apply(DataPreprocessor.check_holiday, holiday='summer')
        autumn_holiday_series = dt_series.apply(DataPreprocessor.check_holiday, holiday='autumn')
        christmas_holiday_series = dt_series.apply(DataPreprocessor.check_holiday, holiday='christmas')

        octoberfest_series = trips_df[self.lags:].apply(
            DataPreprocessor.check_octoberfest, axis=1)
        by_holidays = holidays.CountryHoliday('DE', prov='BY')
        holiday_series = pd.Series(trips_df.index.date)[self.lags:].apply(
            lambda x: x in by_holidays
        )

        x_mlp = np.stack((np.array(octoberfest_series), np.array(holiday_series),
                          np.array(winter_holiday_series), np.array(easter_holiday_series),
                          np.array(whitsun_holiday_series), np.array(summer_holiday_series),
                          np.array(autumn_holiday_series), np.array(christmas_holiday_series)), axis=1)
        x_mlp = np.concatenate(
            (hour_series_ohe, day_series_ohe, x_mlp), axis=1)
        x_mlp = x_mlp.reshape(x_mlp.shape[0], 37)

        # deciding whether to only return the inputs or returning the ground truth aswell
        if self.only_input:
            return [x_convlstm, x_mlp, x_events_conv2d]
        else:
            y = trips_df[self.lags:].values
            y = y.reshape(y.shape[0], self.number_of_rows, self.number_of_columns, 1)
            return [x_convlstm, x_mlp, x_events_conv2d], y

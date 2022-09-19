from collections import namedtuple

import holidays
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing as sk_preprocessing

from demandprediction.prediction.preprocessing.preprocessing import DataPreprocessor

BorderData = namedtuple('BorderData', ['train_values', 'val_values', 'test_values', 'scaler'])


class BorderDataHandlerMultiStep:
    # noinspection PyTypeChecker
    def __init__(self, filepath, lags, prediction_horizon, batch_size, number_of_rows, number_of_columns,
                 number_of_prediction_timesteps, shift=1):
        # Calculate number of timesteps in the test set and load according train, val and test original data
        self.offset_from_end = prediction_horizon - 1
        self.timesteps_in_test_set = number_of_prediction_timesteps + lags + self.offset_from_end
        train_and_val_size = len(pd.read_hdf(filepath, key='even_grid',
                                             start=0, stop=-self.timesteps_in_test_set))
        grid_train_df: pd.DataFrame = pd.read_hdf(filepath, key='even_grid', start=0,
                                                  stop=int(0.67 * train_and_val_size))
        grid_val_df: pd.DataFrame = pd.read_hdf(filepath, key='even_grid',
                                                start=int(0.67 * train_and_val_size),
                                                stop=-self.timesteps_in_test_set)
        grid_test_df: pd.DataFrame = pd.read_hdf(filepath, key='even_grid', start=-self.timesteps_in_test_set)

        # Fit scaler based on training data
        self.grid_scaler = sk_preprocessing.StandardScaler()
        self.grid_scaler.fit(grid_train_df.values)

        # Generate metadata
        self.train_metadata_df = self._generate_metadata_df(grid_train_df.index)
        self.val_metadata_df = self._generate_metadata_df(grid_val_df.index)
        self.test_metadata_df = self._generate_metadata_df(grid_test_df.index)

        # Scale train, test and val data
        scaled_train_values = self.grid_scaler.transform(grid_train_df.values)
        self.grid_train_df = pd.DataFrame(scaled_train_values, columns=grid_train_df.columns,
                                          index=grid_train_df.index)
        scaled_val_values = self.grid_scaler.transform(grid_val_df.values)
        self.grid_val_df = pd.DataFrame(scaled_val_values, columns=grid_val_df.columns,
                                        index=grid_val_df.index)
        scaled_test_values = self.grid_scaler.transform(grid_test_df.values)
        self.grid_test_df = pd.DataFrame(scaled_test_values, columns=grid_test_df.columns,
                                         index=grid_test_df.index)

        self.border_data = {
            'north': None,
            'east': None,
            'south': None,
            'west': None
        }

        # Prepare data for the four border areas
        for orientation in self.border_data:
            train_series = pd.read_hdf(filepath, key='border', start=0,
                                       stop=int(0.67 * train_and_val_size))[orientation]
            val_series = pd.read_hdf(filepath, key='border',
                                     start=int(0.67 * train_and_val_size),
                                     stop=-self.timesteps_in_test_set)[orientation]
            test_series = pd.read_hdf(filepath, key='border',
                                      start=-self.timesteps_in_test_set)[orientation]
            scaler = sk_preprocessing.StandardScaler()
            train_values = train_series.values.reshape(-1, 1)
            val_values = val_series.values.reshape(-1, 1)
            test_values = test_series.values.reshape(-1, 1)
            scaler.fit(train_values)

            scaled_train_values = scaler.transform(train_values)
            scaled_val_values = scaler.transform(val_values)
            scaled_test_values = scaler.transform(test_values)

            self.border_data[orientation] = BorderData(
                train_values=scaled_train_values,
                val_values=scaled_val_values,
                test_values=scaled_test_values,
                scaler=scaler
            )

        # Set variables which are needed for correct slicing
        self.lags = lags
        self.shift = shift
        self.prediction_horizon = prediction_horizon

        self.total_window_size = lags + shift

        self.input_slice = slice(0, lags)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.prediction_horizon
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

        self.number_of_rows = number_of_rows
        self.number_of_columns = number_of_columns
        self.batch_size = batch_size

    def split_window(self, features_lstm, features_mlp):
        """
        Set correct shape for LSTM input and labels.
        """
        inputs_lstm = features_lstm[:, self.input_slice, :]
        labels = features_lstm[:, self.labels_slice, :]
        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs_lstm.set_shape([None, self.lags, None])
        labels.set_shape([None, self.prediction_horizon, None])

        return (inputs_lstm, features_mlp), labels

    def split_grid_window(self, features_lstm, features_mlp):
        """
        Set correct shape for ConvLSTM input and labels.
        """
        inputs_lstm = features_lstm[:, self.input_slice, :, :, :]
        labels = features_lstm[:, self.labels_slice, :, :, :]
        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs_lstm.set_shape([None, self.lags, self.number_of_rows, self.number_of_columns, None])
        labels.set_shape([None, self.prediction_horizon, self.number_of_rows, self.number_of_columns, None])

        return (inputs_lstm, features_mlp), labels

    def _generate_metadata_df(self, index: pd.Index):
        """
        Generate the desired metadata and return it as a separate DataFrame.
        """
        metadata_df = pd.DataFrame(index=index)
        metadata_df = self._add_hour_columns(metadata_df)
        metadata_df = self._add_weekday_columns(metadata_df)
        metadata_df = self._add_holidays(metadata_df)

        return metadata_df

    def make_border_dataset(self, lstm_data, metadata, shuffle=True):
        """
        Generate tensorflow dataset for one border area.
        """
        # Put LSTM data to correct shape
        lstm_data = np.array(lstm_data, dtype=np.float32)
        lstm_data = lstm_data.reshape((lstm_data.shape[0], 1))
        # Put metadata to correct shape
        mlp_data = np.array(metadata[self.label_start:], dtype=np.float32)
        # Generate timeseries dataset
        ds_lstm = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=lstm_data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=False,
            batch_size=self.batch_size, )

        # Generate metadata dataset
        ds_mlp = (tf.data.Dataset.from_tensor_slices(mlp_data)
                  .take(mlp_data.shape[0] - self.prediction_horizon + 1)
                  .batch(self.batch_size))

        # Combine timeseries dataset and metadata dataset
        zipped_ds = tf.data.Dataset.zip((ds_lstm, ds_mlp))
        zipped_ds = zipped_ds.map(self.split_window)

        if shuffle:
            zipped_ds = zipped_ds.shuffle(self.batch_size * 8)

        return zipped_ds

    def make_grid_dataset(self, convlstm_data, metadata, shuffle=True):
        """
        Generate tensorflow dataset for the grid.
        """
        # Put ConvLSTM data to correct shape
        convlstm_data = np.array(convlstm_data, dtype=np.float32)
        convlstm_data = convlstm_data.reshape((convlstm_data.shape[0], self.number_of_rows,
                                               self.number_of_columns, self.prediction_horizon))
        # Put metadata to correct shape
        mlp_data = np.array(metadata[self.label_start:], dtype=np.float32)
        # Generate timeseries dataset
        ds_lstm = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=convlstm_data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=False,
            batch_size=self.batch_size, )

        # Generate metadata dataset
        ds_mlp = tf.data.Dataset.from_tensor_slices(mlp_data).batch(self.batch_size)

        # Combine timeseries dataset and metadata dataset
        zipped_ds = tf.data.Dataset.zip((ds_lstm, ds_mlp))
        zipped_ds = zipped_ds.map(self.split_grid_window)

        if shuffle:
            zipped_ds = zipped_ds.shuffle(self.batch_size * 8)

        return zipped_ds

    @property
    def train_grid(self):
        return self.make_grid_dataset(self.grid_train_df, self.train_metadata_df)

    @property
    def val_grid(self):
        return self.make_grid_dataset(self.grid_val_df, self.val_metadata_df)

    @property
    def test_grid(self):
        return self.make_grid_dataset(self.grid_test_df, self.test_metadata_df, shuffle=False)

    def get_border_train(self, orientation):
        return self.make_border_dataset(self.border_data[orientation].train_values, self.train_metadata_df)

    def get_border_val(self, orientation):
        return self.make_border_dataset(self.border_data[orientation].val_values, self.val_metadata_df)

    def get_border_test(self, orientation):
        return self.make_border_dataset(self.border_data[orientation].test_values,
                                        self.test_metadata_df, shuffle=False)

    @staticmethod
    def _add_weekday_columns(df):
        for i in range(7):
            df[f'weekday_{i}'] = np.where(df.index.weekday == i, 1, 0)

        return df

    @staticmethod
    def _add_hour_columns(df):
        for i in range(24):
            df[f'hour_{i}'] = np.where(df.index.hour == i, 1, 0)

        return df

    # noinspection PyTypeChecker
    @staticmethod
    def _add_holidays(df: pd.DataFrame):
        df['winter'] = df.index.to_series().apply(DataPreprocessor.check_holiday, holiday='winter')
        df['easter'] = df.index.to_series().apply(DataPreprocessor.check_holiday, holiday='easter')
        df['whitsun'] = df.index.to_series().apply(DataPreprocessor.check_holiday, holiday='whitsun')
        df['summer'] = df.index.to_series().apply(DataPreprocessor.check_holiday, holiday='summer')
        df['autumn'] = df.index.to_series().apply(DataPreprocessor.check_holiday, holiday='autumn')
        df['christmas'] = df.index.to_series().apply(DataPreprocessor.check_holiday, holiday='christmas')
        df['octoberfest'] = df.apply(DataPreprocessor.check_octoberfest, axis=1)

        by_holidays = holidays.CountryHoliday('DE', prov='BY')
        df['holiday'] = df.apply(lambda row: row.name in by_holidays, axis=1)
        df['holiday'] = df['holiday'].astype(int)

        return df

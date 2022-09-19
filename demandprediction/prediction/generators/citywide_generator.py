import holidays
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing as sk_preprocessing

from demandprediction.prediction.preprocessing.preprocessing import DataPreprocessor


# noinspection PyTypeChecker
class CitywideDataHandler:
    def __init__(self, filepath, number_of_prediction_timesteps, prediction_horizon, input_width,
                 shift, batch_size):
        # Calculate number of timesteps in the test set and load according train, val and test original data
        self.offset_from_end = prediction_horizon - 1
        self.timesteps_in_test_set = number_of_prediction_timesteps + input_width + self.offset_from_end
        train_and_val_size = len(pd.read_hdf(filepath, key='citywide',
                                             start=0, stop=-self.timesteps_in_test_set))

        train_df: pd.DataFrame = pd.read_hdf(filepath, key='citywide', start=0,
                                             stop=int(0.67 * train_and_val_size))
        val_df: pd.DataFrame = pd.read_hdf(filepath, key='citywide', start=int(0.67 * train_and_val_size),
                                           stop=-self.timesteps_in_test_set)
        test_df: pd.DataFrame = pd.read_hdf(filepath, key='citywide', start=-self.timesteps_in_test_set)

        # Fit scaler based on training data
        self.scaler = sk_preprocessing.StandardScaler()
        self.scaler.fit(train_df.values)

        dfs = {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }

        shift_hours = self._get_shift_hours(train_df, prediction_horizon)
        # Scale data and Generate metadata
        for key, df in dfs.items():
            scaled_values = self.scaler.transform(df.values)
            df = pd.DataFrame(scaled_values, columns=df.columns, index=df.index)
            df = self._add_weekday_columns(df)
            df = self._add_month_columns(df)
            df = self._add_holidays(df)
            df = self._add_shift(df, shift_hours)
            dfs[key] = df

        self.train_df = dfs['train']
        self.val_df = dfs['val']
        self.test_df = dfs['test']

        # Work out the window parameters.
        self.input_width = input_width
        self.prediction_horizon = prediction_horizon
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.prediction_horizon
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

        self.batch_size = batch_size

    def split_window(self, features_lstm, features_mlp):
        """
        Set correct shape for LSTM input and labels.
        """
        inputs_lstm = features_lstm[:, self.input_slice, :]
        labels = features_lstm[:, self.labels_slice, :]
        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs_lstm.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.prediction_horizon, None])

        return (inputs_lstm, features_mlp), labels

    def make_dataset(self, data, shuffle=True):
        """
        Generate tensorflow dataset for the citywide LSTM.
        """
        # Put LSTM data to correct shape
        lstm_data = np.array(data.iloc[:, :1], dtype=np.float32)
        # Put metadata to correct shape
        mlp_data = np.array(data.iloc[self.label_start:, 1:], dtype=np.float32)
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

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df, shuffle=False)

    @staticmethod
    def _add_weekday_columns(df):
        for i in range(7):
            df[f'weekday_{i}'] = np.where(df.index.weekday == i, 1, 0)

        return df

    @staticmethod
    def _add_month_columns(df):
        for i in range(12):
            df[f'month_{i}'] = np.where(df.index.month == i, 1, 0)

        return df

    @staticmethod
    def _add_shift(df: pd.DataFrame, shift_hours) -> pd.DataFrame:
        for shift_hour in shift_hours:
            df[f'shift_{shift_hour}'] = np.where(df.index.hour == shift_hour, 1, 0)

        return df

    @staticmethod
    def _get_shift_hours(df: pd.DataFrame, prediction_horizon):
        if prediction_horizon > 1:
            return [
                df.index[i].hour for i in range(prediction_horizon)
            ]
        return []

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

from pathlib import Path


def setup_paths(data_path: Path):
    data_path.mkdir(exist_ok=True)
    data_path.joinpath('visualization').mkdir(exist_ok=True)
    data_path.joinpath('models').mkdir(exist_ok=True)
    data_path.joinpath('predictions').mkdir(exist_ok=True)
    data_path.joinpath('evaluation').mkdir(exist_ok=True)
    data_path.joinpath('prediction_analysis').mkdir(exist_ok=True)
    data_path.joinpath('train_test').mkdir(exist_ok=True)
    data_path.joinpath('logs/fit').mkdir(exist_ok=True, parents=True)


def get_trips_filename(stop_date: str):
    filename = f'demand_{_format_date_for_filenames(stop_date)}.h5'
    return Path('train_test').joinpath(filename)


def get_events_filename(stop_date: str):
    filename = f'events_{_format_date_for_filenames(stop_date)}.h5'
    return Path('train_test').joinpath(filename)


def get_grid_filename(feature_name, number_of_rows, number_of_columns, x_min, x_max, y_min, y_max,
                      step_size, stop_date):
    return Path('train_test').joinpath(f'{feature_name}_grid_{number_of_rows}x{number_of_columns}'
                                       f'_{x_min}_{x_max}_{y_min}_{y_max}'
                                       f'_{step_size}min_steps_'
                                       f'_stop{_format_date_for_filenames(stop_date)}.h5')


def get_grid_with_border_filename(feature_name, number_of_rows, number_of_columns, x_min, x_max, y_min, y_max,
                                  step_size, border_width, stop_date):
    return Path('train_test').joinpath(f'{feature_name}_grid_{number_of_rows}x{number_of_columns}'
                                       f'_{x_min}_{x_max}_{y_min}_{y_max}_{border_width}m_border'
                                       f'_{step_size}min_steps_'
                                       f'stop{_format_date_for_filenames(stop_date)}.h5')


def get_citywide_filename(step_size, stop_date):
    return Path('train_test').joinpath(f'citywide_{step_size}min_steps_'
                                       f'stop{_format_date_for_filenames(stop_date)}.h5')


def get_predictions_filename(model_name, start_time):
    return Path('predictions').joinpath('{}_{date:%Y-%m-%d_%H-%M-%S}.h5'.format(model_name, date=start_time))


def get_score_filename(model_name, start_time):
    return Path('evaluation').joinpath('{}_{date:%Y-%m-%d_%H-%M-%S}.csv'.format(model_name, date=start_time))


def _format_date_for_filenames(date):
    return date.replace(" ", "_").replace(":", "-")

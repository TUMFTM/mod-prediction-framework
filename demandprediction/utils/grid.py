import configparser

import geopandas as gpd
from shapely.geometry.polygon import Polygon


class PredictionArea():
    def __init__(self, x_limits, y_limits, n_rows, n_columns, crs=None, border_width=None):
        self.x_limits = tuple(map(int, x_limits))
        self.y_limits = tuple(map(int, y_limits))
        self.n_rows = int(n_rows)
        self.n_columns = int(n_columns)
        self.border_width = int(border_width)
        self.crs = crs

        # validate inputs
        self._validate_()

        # if min, max values in x,y are swapped, swapp back and give warning
        if self.x_limits[1] < self.x_limits[0]:
            self.x_limits = (self.x_limits[1], self.x_limits[0])
            raise Warning('x_max < x_min, values have been replaced')
        if self.y_limits[1] < self.y_limits[0]:
            self.y_limits = (self.y_limits[1], self.y_limits[0])
            raise Warning('y_max < y_min, values have been replaced')

        # Calc cell width and height
        self._cell_width_ = (self.x_limits[1] - self.x_limits[0]) / self.n_columns
        self._cell_height_ = (self.y_limits[1] - self.y_limits[0]) / self.n_rows

        self.geometry = self._generate_grid_()

        # Add border if necessary
        if border_width is not None:
            self.geometry = self.geometry.append(self._generate_border_())


    @classmethod
    def from_config(cls, config_file):
        parsed_config = configparser.ConfigParser()
        parsed_config.read(config_file)
        config = parsed_config['prediction']
        x_limits = (config['x_min'], config['x_max'])
        y_limits = (config['y_min'], config['y_max'])
        n_rows = config['number_of_grid_rows']
        n_cols = config['number_of_grid_columns']
        border_width = config['border_width']
        crs = config['crs']
        return cls(x_limits=x_limits,
                   y_limits=y_limits,
                   n_rows=n_rows,
                   n_columns=n_cols,
                   crs=crs,
                   border_width=border_width)

    def _validate_(self):
        if not isinstance(self.x_limits, tuple):
            raise TypeError('x_limits must be a tuple of x_min, x_max values')
        if not isinstance(self.y_limits, tuple):
            raise TypeError('x_limits must be a tuple of y_min, y_max values')
        if self.n_rows <= 0:
            raise ValueError('n_rows must be > 0')
        if self.n_columns <= 0:
            raise ValueError('n_columns must be > 0')

    def _generate_grid_(self):
        grid = {}
        for row in range(self.n_rows):
            for col in range(self.n_columns):
                # creating limits/borders for current gridcell
                cell_x_limits = (
                    self.x_limits[0] + col * self._cell_width_, self.x_limits[0] + (col + 1) * self._cell_width_)
                cell_y_limits = (
                    self.y_limits[1] - (row + 1) * self._cell_height_, self.y_limits[1] - row * self._cell_height_)
                coords = [(cell_x_limits[0], cell_y_limits[0]),
                          (cell_x_limits[0], cell_y_limits[1]),
                          (cell_x_limits[1], cell_y_limits[1]),
                          (cell_x_limits[1], cell_y_limits[0])]
                grid[f'({row}, {col})'] = Polygon(coords)
        grid = {'cell': grid.keys(), 'geometry': grid.values()}
        return gpd.GeoDataFrame(grid, crs=self.crs)

    def _generate_border_(self):
        # Border: Create polygons for the 4 borders

        north_border = Polygon([
            (self.x_limits[0], self.y_limits[1]),
            (self.x_limits[1], self.y_limits[1]),
            (self.x_limits[1] + self.border_width, self.y_limits[1] + self.border_width),
            (self.x_limits[0] - self.border_width, self.y_limits[1] + self.border_width)
        ])
        east_border = Polygon([
            (self.x_limits[1], self.y_limits[0]),
            (self.x_limits[1] + self.border_width, self.y_limits[0] - self.border_width),
            (self.x_limits[1] + self.border_width, self.y_limits[1] + self.border_width),
            (self.x_limits[1], self.y_limits[1])
        ])
        south_border = Polygon([
            (self.x_limits[0] - self.border_width, self.y_limits[0] - self.border_width),
            (self.x_limits[1] + self.border_width, self.y_limits[0] - self.border_width),
            (self.x_limits[1], self.y_limits[0]),
            (self.x_limits[0], self.y_limits[0])
        ])
        west_border = Polygon([
            (self.x_limits[0] - self.border_width, self.y_limits[0] - self.border_width),
            (self.x_limits[0], self.y_limits[0]),
            (self.x_limits[0], self.y_limits[1]),
            (self.x_limits[0] - self.border_width, self.y_limits[1] + self.border_width)
        ])

        borders = {
            'north': north_border,
            'east': east_border,
            'south': south_border,
            'west': west_border
        }
        return gpd.GeoDataFrame({'cell':borders.keys(),'geometry':borders.values()}, crs= self.crs)


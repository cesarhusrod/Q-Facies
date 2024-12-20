# -*- coding: utf-8 -*-
'''
                               IGME 
                     Geological Survey of Spain

                             Q-Facies
                            Version 1.3
                            20-12-2024
  Authors:
	M. González-Jiménez   miguel.gonzalez@igme.es
	L. Moreno             l.moreno@igme.es	
	H. Aguilera           h.aguilera@igme.es
	A. De la Losa         a.delalosa@igme.es
	A. Romero             a.romero@igme.es
 
  Revisors:
    C. Husillos-Rodriguez c.husillos@igme.es
    
If you have any problem with the code or want to suggest possible modifications,
please contact H. Aguilera

===============================================================================
===============================================================================
 ------------------------------------------------------------------------------
 --------------------- DATA ENTRY FILE FORMAT ---------------------------------
 ------------------------------------------------------------------------------
Position:           1         2  3   4   5    6     7    8   9
Variable:  ID_group or Date  Ca  Mg  Na  K   HCO3  CO3  SO4  Cl

ASCII file separated by tabs (\t)
The decimal separator is the decimal point (.)
===============================================================================
===============================================================================
'''

__version__ = "1.3"

import configparser
import re
import os
import time
import warnings

import numpy as np
import pandas as pd
from matplotlib import cm

from plot import Overlap, Plot_all, Evolution_time, Evolution_groups
import plot as qplot
from diagram import Diagram
import calculation as cal


def input_converter(value):
    """
    Convert input string values to float, handling special cases.

    This function converts input string values to float. It treats values
    starting with '<' followed by digits as 0.0, and attempts to convert
    all other values to float.

    Parameters:
    value (str): The input string to be converted.

    Returns:
    float: 0.0 if the input matches the pattern '^<[0-9]*$',
           otherwise the float representation of the input.

    """
    patron = re.compile(r'^<[0-9]*$')
    return 0.0 if patron.match(value.strip()) else float(value)


class ReadDataFile(object):
    '''Read the input data file, check for decimal errors, and transform
    units of measurement from ppm or mg/l to miliequivalent percentage (epm).'''

    def __init__(self, fname, way, resample_interval=None,
                 transform=True, kw=None):
        """
        Initialize the ReadDataFile object to read and process input data.

        This method reads the input data file, performs optional resampling for time series data,
        and optionally transforms ionic data from mg/l (or ppm) to milliequivalent percentage (epm).

        Parameters:
        -----------
        fname : str
            Filepath of the input data.
        way : str
            Method of data analysis. Must be either 'by_groups' or 'by_time'.
        resample_interval : str, optional
            Frequency for resampling time series data. Uses pandas frequency aliases.
            Only considered when way = 'by_time'. Default is None.
        transform : bool, optional
            If True, transforms ionic data from mg/l (or ppm) to epm. Default is True.
        kw : dict, optional
            Additional keyword arguments. Should include 'datetime_format' and 'agregation'
            when way = 'by_time'.

        Returns:
        --------
        None

        Notes:
        ------
        - The input file should be a tab-separated CSV file.
        - For time series data (way = 'by_time'), the 'Time' column is converted to datetime.
        - Resampling is performed if resample_interval is provided for time series data.
        - The _converse method is called if transform is True to convert units to epm.
        """
        self.ions = ['Ca', 'Mg', 'Na', 'K', 'HCO3', 'CO3', 'SO4', 'Cl']
        self.cols = {'by_groups': ['Group_ID', *self.ions],
                     'by_time': ['Time', *self.ions]}
        self.way = way
        self.df = pd.read_csv(fname, sep='\t', skiprows=1,
                              names=self.cols[self.way],
                              converters={k: input_converter for k in self.ions})

        if self.way == 'by_time':
            self.df['Time'] = pd.to_datetime(self.df['Time'],
                                             format=kw['datetime_format'])
            if resample_interval:
                trans = {'yearly': 'Y',
                         'monthly': 'M',
                         'weekly': 'W',
                         'daily': 'D'}
                resamp = trans.get(resample_interval, resample_interval)
                self.df = self.df.set_index('Time').resample(trans[resamp],
                                                             origin='start').agg(kw['agregation'])
        if transform:
            self._converse()

    def get(self):
        """
        Retrieve the formatted input file as a pandas DataFrame.

        This method returns the DataFrame containing the input data that has been
        read, processed, and potentially transformed by the ReadDataFile class.

        Returns:
        --------
        pandas.DataFrame
            The DataFrame containing the formatted input data, including any
            transformations or processing that may have been applied during
            the initialization of the ReadDataFile object.
        """
        return self.df

    def _converse(self):
        """
        Convert mg/l or ppm analyses to milliequivalent percentage (epm) values.
        equivalent number (eq) = substance weight () * (valence (V) / molecular mass(Mm))
        Conversion Factor (CF) = V/Mm ----> One per molecule

        This method performs the following steps:
        1. Calculates conversion factors for each ion based on valence and molecular mass.
        2. Converts mass units (mg/l or ppm) to equivalent units (eq).
        3. Calculates the sum of cations and anions.
        4. Converts equivalent units to percentage units (epm).
        5. Updates the DataFrame with new columns and removes intermediate calculations.

        Parameters:
        -----------
        self : object
            The instance of the class containing this method. It should have attributes:
            - df (pandas.DataFrame): The DataFrame containing the ion concentration data.
            - ions (list): A list of ion names used in the calculations.

        Returns:
        --------
        None
            The method modifies the DataFrame in-place, adding new columns for epm values
            and removing original concentration columns.

        Notes:
        ------
        - The method assumes that the input DataFrame (self.df) contains columns for each ion#+
          with concentrations in mg/l or ppm.
        - The resulting DataFrame will contain new columns with '_epm' suffix for each ion,#+
          representing their milliequivalent percentages.
        """
        Mm = {'Ca': 40.078, 'Mg': 24.305, 'Na': 22.989769, 'K': 39.0983,
              'HCO3': 61.01684, 'CO3': 60.0089, 'SO4': 96.0626, 'Cl': 35.453}
        V = {'Ca': 2, 'Mg': 2, 'Na': 1, 'K': 1,
             'HCO3': 1, 'CO3': 2, 'SO4': 2, 'Cl': 1}

        cf = {key: V[key] / Mm[key] for key in V.keys()}

        # --------- Convert mass units into equivalent units (eq) ------------
        # CATIONS
        self.df['Ca_eq'] = (self.df.Ca * cf['Ca']).round(3)
        self.df['Mg_eq'] = (self.df.Mg * cf['Mg']).round(3)
        self.df['Na_eq'] = (self.df.Na * cf['Na']).round(3)
        self.df['K_eq'] = (self.df.K * cf['K']).round(3)

        # ANIONS:
        self.df['HCO3_eq'] = (self.df.HCO3 * cf['HCO3']).round(3)
        self.df['CO3_eq'] = (self.df.CO3 * cf['CO3']).round(3)
        self.df['SO4_eq'] = (self.df.SO4 * cf['SO4']).round(3)
        self.df['Cl_eq'] = (self.df.Cl * cf['Cl']).round(3)

        # Deleting useless fields
        self.df.drop(self.ions, axis=1, inplace=True)
        elements = [f'{ion}_eq' for ion in self.ions]

        # --------- Convert equivalents into percentage units (epm) ----------
        # Summatory:
        self.df['CAT_sum'] = self.df[[
            'Ca_eq', 'Mg_eq', 'Na_eq', 'K_eq']].sum(axis=1)
        self.df['ANI_sum'] = self.df[['HCO3_eq',
                                      'CO3_eq', 'SO4_eq', 'Cl_eq']].sum(axis=1)

        # CATIONS
        self.df['Ca_epm'] = ((self.df.Ca_eq / self.df.CAT_sum) * 100).round(3)
        self.df['Mg_epm'] = ((self.df.Mg_eq / self.df.CAT_sum) * 100).round(3)
        self.df['NaK_epm'] = (
            ((self.df.Na_eq + self.df.K_eq) / self.df.CAT_sum) * 100).round(3)

        # ANIONS:
        self.df['HCO3CO3_epm'] = (
            ((self.df.HCO3_eq + self.df.CO3_eq) / self.df.ANI_sum) * 100).round(3)
        self.df['SO4_epm'] = (
            (self.df.SO4_eq / self.df.ANI_sum) * 100).round(3)
        self.df['Cl_epm'] = ((self.df.Cl_eq / self.df.ANI_sum) * 100).round(3)
        # Deleting useless fields
        self.df.drop(elements, axis=1, inplace=True)


class Dataset(object):
    ''' Main class of the program'''

    def __init__(self, fname, way='by_groups', **kw):
        '''Read the dataset from the 'Data' folder
            way:str {'by_groups', 'by_time'} default 'by_groups'
            Way of analyzing hydrochemical facies evolution.
            'by_groups' -> study the facies of defined groups
            'by_time'   -> study the facies of the temporal series throughout
                           a rolling window. In this case, window parameters need
                           to be set.'''
        new_kw = {k: kw[k] for k in ['datetime_format', 'aggregation']}
        obj_file = ReadDataFile(fname, way, kw.get('resample_interval', None),
                                kw.get('transform', None), new_kw)

        self.fname = os.path.splitext(os.path.basename(fname))[0]
        print('fname =', self.fname)
        self.way = way

        print('Dataframe................................................................')
        self.data = obj_file.get()
        print(self.data.info())
        print(self.data.head(100))
        print('*' * 100)
        self.run(self.way, **kw)

    def run(self, way, **kw):
        """
        Execute the main program workflow for analyzing hydrochemical facies.

        This function performs the following steps:
        1. Groups the data based on the specified method ('by_groups' or 'by_time').
        2. Filters the groups to remove NaN rows.
        3. Generates colors for diagram polygons.
        4. Creates Diagram objects for each group.
        5. Optionally formats data for Excel output.
        6. Generates plots based on the analyzed data.

        Parameters:
        way (str): The method of grouping data. Either 'by_groups' or 'by_time'.
        **kw (dict): Additional keyword arguments including:
            - window_size (int): The size of the time window for 'by_time' grouping.
            - step_length (int): The step length for 'by_time' grouping.
            - folder_color (str): The color scheme for diagram polygons.
            - excel (bool): Whether to generate Excel output.
            Other parameters as required by Diagram and plotting functions.

        Returns:
        None: This function doesn't return a value but updates the object's state.
        """
        print('......... Grouping ...........')
        _g = None
        if way == 'by_groups':
            _g = self.by_groups()
        else:
            _g = self.by_time(
                window_size=kw['window_size'], freq=kw['step_length'])
        # Clean the groups of NaN rows.
        print('.......... Filtering ................')
        self.groups = self._group_filter(_g)

        print('.......... Diagrams ................')
        polygon_colors = self._color_diagram(folder_color=kw['folder_color'])
        # kw.pop('pol_color') # Delete so as not to overwrite kwargs.
        self.diagrams = [Diagram(group, self.way, self.fname, pol_color=color, **kw)
                         for group, color in zip(self.groups, polygon_colors)]
        print(self.diagrams)
        if kw['excel']:
            print('.......... Excel formatting ................')
            self.excel()
        print('-------------- Plotting ------------------')
        self.plotting(**kw)

    def _color_diagram(self, folder_color='Spectral', reverse=False):
        """
        Generate a list of colors for diagram polygons based on a specified colormap.

        This function creates a discretized list of colors from a given colormap. The number
        of colors in the list corresponds to the number of groups to be analyzed. If the
        specified colormap is not found, it returns a list with the same color repeated
        for all groups.

        Parameters:
        folder_color (str, optional): The name of the colormap to use. Defaults to 'Spectral'.
        reverse (bool, optional): If True, reverses the colormap. Defaults to False.

        Returns:
        list: A list of color values, with one color for each group to be analyzed.
              Each color is represented as an RGBA tuple.

        Note:
        The function assumes that `self.groups` is a list or iterable containing the groups
        to be analyzed.
        """
        try:
            folder_color = cm.get_cmap(folder_color)
            folder_color = folder_color.reversed() if not reversed else folder_color
            lsp = np.linspace(0, folder_color.N, len(self.groups)).astype(int)
            return [folder_color(i) for i in lsp]

        except:
            return [folder_color for i in range(len(self.groups))]

    def _group_filter(self, g):
        """
        Filter and clean groups by removing NaN rows, empty groups, and groups with fewer than three points.

        This function processes a list of groups (typically DataFrames) to remove NaN values,
        delete empty groups, and ensure that each remaining group has at least three points.

        Parameters:
        g (list): A list of groups (e.g., pandas DataFrames) to be filtered.

        Returns:
        list: A filtered list of groups, where each group:
              - Has no NaN rows (or at least 8 non-NaN values per row)
              - Is not empty
              - Contains at least three points (rows)

        Raises:
        AssertionError: If any remaining group has fewer than three points.

        Warnings:
        UserWarning: If any empty groups are deleted during the filtering process.
        """
        # Drop NaN rows:
        non_NaN_groups = [i.dropna(axis=0, thresh=8) for i in g]

        # If there are empty groups: delete them and raise a Warning
        # Bool type list of df to remove
        idx = [i.empty for i in non_NaN_groups]
        groups = [i for i, j in zip(non_NaN_groups, idx) if j == False]
        del_groups = len(non_NaN_groups) - len(groups)

        if del_groups > 0:
            warnings.warn(
                '{} empty group(s) has been deleted'.format(del_groups))

        # Raise an exception when groups are formed by less than three points:
        fails = len([i for i in groups if i.shape[0] < 3])
        assert fails == 0, "{a} group(s) have less than three points (minimun \
            requiered). \nTry to define a wider window size (if creating groups\
            by time) or delete those groups conformed by less than 3 analyses \
            (if creating groups by ID_Group).".format(a=fails)

        return groups

    def by_groups(self) -> list:
        """
        Extract all groups contained in the dataset and calculate the facies parameters for each one.

        This method groups the data based on unique Group_ID values. It creates a separate
        DataFrame for each distinct group in the dataset.
        Parameters:
        -----------
        self : Dataset
            The Dataset instance containing the data to be grouped.

        Returns:
        --------
        list
            A list of pandas DataFrame objects. Each DataFrame in the list represents
            a distinct group from the dataset, containing all rows with the same Group_ID.

        Notes:
        ------
        - The grouping is performed based on the 'Group_ID' column in the dataset.
        - Each unique value in 'Group_ID' results in a separate DataFrame in the output list.
        - The method assumes that 'self.data' is a pandas DataFrame with a 'Group_ID' column.
        """
        return [self.data.loc[self.data.Group_ID == i] for i in
                self.data.Group_ID.unique()]

    def by_time(self, window_size=10, freq=5):
        """
        Extract all groups contained in the dataset and calculate the facies parameters for each one.

        This method groups the data based on a rolling time window approach. It creates a separate
        DataFrame for each time window in the dataset.

        Parameters:
        -----------
        window_size : int, optional
            The size of the rolling window (number of data points). Must be greater than 3.
            Default is 10.

        freq : int, optional
            The frequency or step size for the rolling window. Must be less than window_size.
            Default is 5.

        Returns:
        --------
        list
            A list of pandas DataFrame objects. Each DataFrame in the list represents
            a distinct time window from the dataset, containing all rows within that window.

        Raises:
        -------
        AssertionError
            If window_size is 3 or less, or if freq is greater than or equal to window_size.

        Notes:
        ------
        - The method uses a rolling window approach to create time-based groups.
        - Each group must have at least three data points.
        - The window moves forward by 'freq' steps each time.
        - The last group may contain fewer points than the window_size if there are not enough
          remaining data points.
        """
        ''' Extract all groups contained in the dataset and calculate the
        facies parameters for each one. Minimum: every three points.'''
        index = self.data.index.to_numpy()
        assert window_size > 3, "A window group must be conformed by three or more points."
        assert freq < window_size, 'Window size is wider than the step length.\
             There will be analyses not taken into account'

        def rolling_window(array, window_size, freq):
            shape = (array.shape[0] - window_size + 1, window_size)
            strides = (array.strides[0],) + array.strides
            print('$$$$$$$$$$$$ Rolling window $$$$$$$$$$$$$')
            print(shape, strides, array.shape, window_size)
            rolled = np.lib.stride_tricks.as_strided(
                array, shape=shape, strides=strides)
            step_rolled = rolled[np.arange(0, shape[0], freq)]
            start, end = step_rolled[:, 0], step_rolled[:, -1]
            # Mask array for adding 1 to all end-array elements
            mask = np.ones_like(end).astype(bool)
            mask[-1] = False
            return start, np.where(mask, end+1, end)

        start, end = rolling_window(index, window_size, freq)
        return [self.data.iloc[i:j, :] for i, j in zip(start, end)]

    def _info_table(self):
        """
        Incorporate all diagrams' Q-Facies indices into a single DataFrame.

        This method collects the Q-Facies indices from all diagrams associated with
        the current instance and combines them into a single pandas DataFrame.

        Parameters:
        -----------
        self : object
            The instance of the class containing this method.

        Returns:
        --------
        pandas.DataFrame
            A DataFrame containing the Q-Facies indices from all diagrams.
            Each row in the DataFrame represents a set of indices from one diagram,
            and each column represents a specific Q-Facies index.

        Notes:
        ------
        This method assumes that self.diagrams is an iterable containing diagram objects,
        each with a get_params() method that returns a list or dictionary of Q-Facies indices.
        """
        return pd.DataFrame([j for i in self.diagrams for j in i.get_params()])

    def _mapa(self):
        """
        Create a dictionary to map column names to their abbreviated forms.

        This method returns a dictionary that maps full column names to their
        corresponding abbreviated versions. It's used for renaming columns
        in dataframes or other data structures.

        Parameters:
        -----------
        self : object
            The instance of the class containing this method.

        Returns:
        --------
        dict
            A dictionary where keys are the original column names and values
            are their abbreviated forms. The mappings are as follows:
            - 'Area' to 'Ai'
            - 'Shape' to 'Si'
            - 'Angle' to 'Or'
            - 'panel' to 'Panel'
            - 'Blau' to 'Bi'
            - 'points' to 'Points'
            - 'Time' to 'Time'
            - 'Dispersion' to 'Di'

        Note:
        -----
        The 'Angle' key is mapped to 'Or' with a degree symbol, which may
        require special handling when used.
        """
        return {'Area': 'Ai',
                'Shape': 'Si',
                'Angle': 'Or'.format('$^{o}$'),
                'panel': 'Panel',
                'Blau': 'Bi',
                'points': 'Points',
                'Time': 'Time',
                'Dispersion': 'Di'
                }

    def excel(self):
        """
        Create an Excel file with all the groups' Q-Facies indices and save it to the 'Data' folder.

        This method processes the Q-Facies indices data, creates two separate DataFrames with
        different representations of the data, and saves them as two sheets in a single Excel file.

        The method performs the following steps:
        1. Retrieves the Q-Facies indices data.
        2. Renames columns and formats data.
        3. Creates two DataFrames (df1 and df2) with different data representations.
        4. Saves both DataFrames as separate sheets in an Excel file.

        Parameters:
        -----------
        self : object
            The instance of the class containing this method. It should have attributes:
            - fname (str): Used for naming the output Excel file.
            - _info_table() method: To retrieve the Q-Facies indices data.
            - _mapa() method: To get column name mappings.

        Returns:
        --------
        None
            The method doesn't return a value, but creates an Excel file in the 'Data' folder
            named 'Data{self.fname}.xlsx' with two sheets: '{self.fname}_A' and '{self.fname}_B'.

        Notes:
        ------
        - The Excel file is saved in the 'Data' folder.
        - The method uses pandas ExcelWriter to create the Excel file.
        - Floating point numbers in the Excel file are formatted to two decimal places.
        """
        df = self._info_table()
        print('*********** df ********************************')
        print(df.info())
        print(df.head())
        df.rename(mapper=self._mapa(), axis=1, inplace=True)
        df.rename(mapper={'Orientation ($^{o}$)': 'Orientation ({})'.format(u"\u00b0")},
                  axis=1, inplace=True)
        df.Panel = df.Panel.map(lambda x: x.capitalize())  # Format the fields
        # Create two dataframes
        df1 = df.set_index(['Group', 'Panel'])
        df1 = df1[['Points', 'Ai', 'Bi', 'Di', 'Dominant', 'Or', 'Si']]
        print('================================')
        print('Info del Dataframe df1:\n', df1.head())
        a = df.columns.to_list()
        # delete the columns from the first dataframe
        [a.remove(name)
         for name in ('Group', 'Panel', 'A', 'B', 'C', 'D') if name in a]
        # Applying the functions to the second dataframe
        apply_functions = {'Dominant': 'first',
                           'Ai': 'mean',
                           'Bi': 'mean',
                           'Di': 'mean',
                           'Or': 'mean',
                           'Si': 'mean',
                           'Points': 'sum'}
        df2 = df.pivot_table(index=['Group'], columns='Panel', values=a, sort=False,
                             aggfunc=apply_functions).swaplevel(0, 1, axis=1).sort_index(axis=1)
        print('Info del Dataframe df2:\n  ')
        print(df2.info())
        print(df2.head())
        df2 = df2.reindex([('Anion', 'Points'),
                           ('Anion', 'Ai'),
                           ('Anion', 'Bi'),
                           ('Anion', 'Di'),
                           ('Anion', 'Dominant'),
                           ('Anion', 'Or'),
                           ('Anion', 'Si'),
                           ('Cation', 'Points'),
                           ('Cation', 'Ai'),
                           ('Cation', 'Bi'),
                           ('Cation', 'Di'),
                           ('Cation', 'Dominant'),
                           ('Cation', 'Or'),
                           ('Cation', 'Si'),
                           ('Diamond', 'Points'),
                           ('Diamond', 'Ai'),
                           ('Diamond', 'Bi'),
                           ('Diamond', 'Di'),
                           ('Diamond', 'Dominant'),
                           ('Diamond', 'Or'),
                           ('Diamond', 'Si')], axis=1)
        df2.drop(df2.columns[18], axis=1, inplace=True)
        df2.set_index(df.Group.unique(), inplace=True)
        print('Info del Dataframe df2 tras reindexado y filtrado:\n  ')
        print(df2.info())
        print(df2.head())
        # Saving the dataframes as two different sheets of the same Excel file
        writer = pd.ExcelWriter(os.path.join('Data', f'Data{self.fname}.xlsx'))
        df1.to_excel(writer, sheet_name=f'{self.fname}_A', float_format="%.2f")
        df2.to_excel(writer, sheet_name=f'{self.fname}_B', float_format="%.2f")
        writer.close()

    def plotting(self, **kw):
        """
        Generate various plots and figures based on the analyzed data.

        This function creates different types of plots depending on the provided keyword arguments.#+
        It can generate individual figures for each group, evolution figures, overlap figures,#+
        and a comprehensive plot with all data points.#+

        Parameters:
        -----------
        self : object
            The instance of the class containing this method.
        **kw : dict
            Keyword arguments that control the plotting behavior. Expected keys include:
            - 'figs' (bool): If True, creates individual figures for each group.
            - 'evolution_fig' (bool): If True, creates an evolution figure.
            - 'overlap_fig' (bool): If True, creates an overlap figure.
            - 'plot_all' (bool): If True, creates a single figure with all data points (only for 'by_time' mode).
            Additional keyword arguments may be passed to specific plotting functions.

        Returns:
        --------
        None
            This function does not return a value but generates and displays plots based on the input parameters.

        Notes:
        ------
        - The function uses various plotting modules and functions (e.g., Evolution_groups, Evolution_time, Overlap, Plot_all).
        - Different plots are generated based on the analysis mode ('by_groups' or 'by_time').
        - Error handling is implemented for the 'plot_all' option to catch potential NameErrors.
        """
        # Create a Figure for each group if 'Figs' is set to True
        if kw['figs']:
            try:
                import tqdm
                for diag in tqdm.tqdm(self.diagrams):
                    diag.plot()
            except:
                for diag in self.diagrams:
                    diag.plot()
        # Create an Evolution figure:
        if kw['evolution_fig']:
            if self.way == 'by_groups':
                Evolution_groups(self.diagrams, self._info_table(),
                                 self.fname, self.way, self._mapa())
            elif self.way == 'by_time':
                Evolution_time(self.diagrams, self._info_table(),
                               self.fname, self.way, self._mapa())
        # Create an Overlap figure:
        if kw['overlap_fig']:
            Overlap(self.diagrams, self.way,
                    self.fname, **kw)  # ['overlap_var']
        # Create a unique figure with ALL the points:
        if kw['plot_all'] & (self.way == 'by_time'):
            try:
                df_all = Diagram(self.data, 'by_time',
                                 self.fname, **kw).get_all_points()
                Plot_all(df_all, self.fname, **kw)
            except NameError:
                raise ('Cannot graduate by time. Check the time series')


def format_warning(msg, *args, **kwargs):
    """
    Format a warning message by wrapping it with newlines.

    This function is designed to be used as a custom warning formatter.
    It ignores all arguments except the message itself.

    Parameters:
    ----------
    msg (str): The warning message to be formatted.
    *args: Variable length argument list (ignored).
    **kwargs: Arbitrary keyword arguments (ignored).

    Returns:
    str: The formatted warning message, surrounded by newlines.
    """
    return f'\nWarning: {msg}\n'  # ignore everything except the message

# ============================== MAIN =========================================


def main():
    """
    Main function to process hydrochemical data and generate analyses.

    This function reads configuration from a file, sets up parameters,
    processes the dataset, and generates various analyses and plots.

    The function performs the following steps:
    1. Reads configuration from 'Options2.txt'.
    2. Parses configuration values into appropriate data types.
    3. Sets up core parameters and warning configurations.
    4. Processes the dataset using the Dataset class.
    5. Measures and reports execution time.

    Returns:
    int: Returns 0 upon successful completion.

    Note:
    - Requires 'Options.txt' configuration file in the same directory.
    - Uses external modules like configparser, re, warnings, and time.
    - Depends on custom modules/classes like Dataset, cal, and qplot.
    """
    ################### ---------- Variables defining ---------#####################
    kw = dict()
    parser = configparser.ConfigParser()
    parser.read('Options.txt')
    print('Secciones del fichero de configuración =', parser.sections())
    # Patrón para identificar números enteros
    patron_entero = re.compile(r'^[0-9]+$')
    # Patrón para identificar números decimales
    patron_decimal = re.compile(r'^[0-9]+.[0-9]+$')
    for sect in parser.sections():
        for name, value in parser.items(sect):
            # inicializamos los valores del diccionario kw con los valores leídos del fichero
            if value.strip() in ['True', 'False']:
                kw[name] = bool(value.strip())
            elif patron_entero.match(value.strip()):
                kw[name] = int(value.strip())
            elif patron_decimal.match(value.strip()):
                kw[name] = float(value.strip())
            else:
                kw[name] = value.strip()
    # show the input parameters
    print('Input parameters:')
    for name, value in kw.items():
        print(f'{name} = {value}')
    # set primary parameters to specific variables
    fname = kw['fname']
    way = kw['way']
    del kw['fname']
    del kw['way']

    # ================================ CORE =======================================
    if kw['lof']:
        cal.import_skl()
    qplot._extension_graph = kw['extension_graph']

    # ========================= CUSTOM WARNING =====================================
    warnings.formatwarning = format_warning
    if kw['ignore_warnings']:
        warnings.simplefilter('ignore')

    # ========================= PROCESSING CODE ===================================
    start = time.time()
    param_path = os.path.join('Data', fname)
    # Load the dataset
    Dataset(param_path, way, **kw)
    end = time.time()
    d = end - start
    print("Execution time: {} minutes and {} seconds.".format(
        round(d // 60), round(d % 60, 2)))

    return 0


if __name__ == '__main__':
    main()

# =============================================================================
# ======================== END OF THE PROGRAM =================================
# =============================================================================

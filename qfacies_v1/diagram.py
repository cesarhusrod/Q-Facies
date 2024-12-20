# -*- coding: utf-8 -*-
"""
===============================================================================
======================== Module of Q-Facies package ============================
===============================================================================
Contains all the classes related to the elements of the diagrams.
A diagram is always conformed by three panels: Cation, Anion and Diamond (each 
one subclassed from Panel class and associated at Diagram class)

All euclidean transformations requiered for each panel are imported from the
'calculation.py' module.

All plotting methods for visual representation are imported from the 'plot.py' module.

===============================================================================
===============================================================================
"""

import pandas as pd
from plot import Plot_Diagram
from calculation import Indixes, Transform


class Diagram:
    ''' Contains all the information of a classical Piper Diagram'''

    def __init__(self, df, way, df_name, pol_color='orange', **kw):
        ''' Create an object for the Cation, Anion and Diamond panels, and store
        them in a tuple.'''
        self.df, self.way, self.kw = df, way, kw
        self.df_name, self.pol_color = df_name, pol_color
        self.df.dropna(inplace=True)
        # self.df.fillna(0, inplace=True) # same results as dropna
        # Classes agregation:
        self.cation = Cation(df, **kw)
        self.anion = Anion(df, **kw)
        self.diamond = Diamond(df, **kw)
        self.panels = (self.cation, self.anion, self.diamond)

    def __str__(self):
        ''' Set Group name'''
        if self.way == 'by_groups':
            return self.df['Group_ID'].iloc[0]
        elif self.way == 'by_time':
            return (self.df['Time'].min(), self.df['Time'].max())

    def plot(self):
        ''' Plot the Piper diagram with all the information.
            Executed via the 'plot.py' module'''
        Plot_Diagram(self.panels, name=self.__str__(),
                     way=self.way, df_name=self.df_name,
                     pol_color=self.pol_color, **self.kw)

    def get_params(self):
        '''Return a list of dicts with all panels' indices.'''
        d_params = [i.Params.get() for i in self.panels]
        name = '{} to {}'.format(*[i.strftime(format='%d-%m-%Y')
                                   for i in self.__str__()]) if              \
            self.way == 'by_time' else self.__str__()

        [i.update(dict(Group=name)) for i in d_params]
        return d_params

    def get_all_points(self):
        ''' Get all points of the Piper diagram. Columns: xy coordinates and 'ID'
        group or Date column.
        Return a pandas.DataFrame'''
        column = dict(by_time='Time', by_groups='Group_ID')
        return pd.concat([i.get_column(column[self.way]) for i in
                          self.panels], axis=0, ignore_index=True)


class Panel:
    def __init__(self, df):
        """
        Initialize a Panel object.

        This constructor sets up a Panel instance with the provided DataFrame.

        Parameters:
        df (pandas.DataFrame): A DataFrame containing the data for the panel.
                               This should include the necessary columns for
                               the specific panel type (e.g., cation, anion, or diamond).

        Returns:
        None
        """
        self.df = df


class Cation(Panel):
    '''Contains all cation panel data and methods'''

    def __init__(self, df, params=True, **kw):
        """
        Initialize a Cation panel object.

        This constructor sets up a Cation panel instance with the provided DataFrame,
        calculates the panel points, and optionally computes indices.

        Parameters:
        df (pandas.DataFrame): A DataFrame containing the data for the cation panel.
                               This should include 'NaK_epm' and 'Mg_epm' columns.
        params (bool, optional): If True, calculate all indices using the Indixes class.
                                 Defaults to True.
        **kw: Additional keyword arguments to be passed to the Indixes class.

        Returns:
        None

        Attributes:
        panel (str): Set to 'cation' to identify the panel type.
        components (list): List of component names used in the cation panel.
        points (numpy.ndarray): Transformed coordinates of the panel points.
        Params (Indixes): An Indixes object containing calculated indices for the panel.
        """
        super().__init__(df)
        self.panel = 'cation'
        self.components = ['NaK_epm', 'Mg_epm']
        self.points = self.transform(
            df.filter(items=self.components).to_numpy())
        self.Params = Indixes(self.points, df, self.panel, **kw)

    def transform(self, points):
        ''' Make all the required euclidean transformation to the panel's
        points via matrixes-product (@)'''
        T = Transform()
        return points @ T.scale() @ T.t_shear()

    def info(self):
        ''' Contains the info to represent on the diagram'''
        x, y = 0.72, 0.81
        text = '\n'.join([
            "{:^25}".format(r'${}$ ({})'.format(
                'Cation', self.Params.num_points)),
            "{:<}: {:>7.2f}".format('Ai (%)',      self.Params.area),
            "{:<}: {:>8.2f}".format('Si (%)', self.Params.shape_idx),
            "{:<}: {:>8.2f}".format('Or ({})'.format('$^{o}$'),
                                    self.Params.angle),
            "{:<}: {:>7.2f}".format('Bi (%)',      self.Params.blau_idx[0]),
            "{:<}: {:>7.2f}".format('Di (%)',      self.Params.sd)
        ])
        return x, y, text


class Anion(Panel):
    '''Contains all anion panel data and methods'''

    def __init__(self, df, params=True, **kw):
        """
        Initialize an Anion panel object.

        This constructor sets up an Anion panel instance with the provided DataFrame,
        calculates the panel points, and optionally computes indices.

        Parameters:
        df (pandas.DataFrame): A DataFrame containing the data for the anion panel.
                               This should include 'Cl_epm' and 'SO4_epm' columns.
        params (bool, optional): If True, calculate all indices using the Indixes class.
                                 Defaults to True.
        **kw: Additional keyword arguments to be passed to the Indixes class.

        Returns:
        None

        Attributes:
        panel (str): Set to 'anion' to identify the panel type.
        components (list): List of component names used in the anion panel.
        points (numpy.ndarray): Transformed coordinates of the panel points.
        Params (Indixes): An Indixes object containing calculated indices for the panel.
        """
        super().__init__(df)
        self.panel = 'anion'
        self.components = ['Cl_epm', 'SO4_epm']
        self.points = self.transform(
            df.filter(items=self.components).to_numpy())
        self.Params = Indixes(self.points, df, self.panel, **kw)

    def transform(self, points):
        ''' Make all the required euclidean transformation to the panel's
        points'''
        T = Transform()
        return (points @ T.scale() @ T.t_shear()) + T.a_translation()

    def info(self):
        """
        Generate information to be displayed on the anion panel of the Piper diagram.

        This method creates a formatted string containing various parameters and indices
        related to the anion panel, including the number of points, area, shape index,
        orientation angle, Blau index, and standard deviation.

        Returns:
        tuple: A tuple containing three elements:
            - x (float): The x-coordinate for positioning the information on the diagram (0.72).
            - y (float): The y-coordinate for positioning the information on the diagram (0.58).
            - text (str): A formatted string containing the panel information, including:
                - Panel name and number of points
                - Ai (%): Area index
                - Si (%): Shape index
                - Or (°): Orientation angle in degrees
                - Bi (%): Blau index
                - Di (%): Standard deviation
        """
        x, y = 0.72, 0.58
        text = '\n'.join([
            "{:^25}".format(r'${}$ ({})'.format(
                'Anion', self.Params.num_points)),
            "{:<}: {:>7.2f}".format('Ai (%)', self.Params.area),
            "{:<}: {:>8.2f}".format('Si (%)', self.Params.shape_idx),
            "{:<}: {:>8.2f}".format('Or ({})'.format('$^{o}$'),
                                    self.Params.angle),
            "{:<}: {:>7.2f}".format('Bi (%)', self.Params.blau_idx[0]),
            "{:<}: {:>7.2f}".format('Di (%)', self.Params.sd)
        ])
        return x, y, text


class Diamond(Panel):
    '''Contains all diamond panel data and methods'''

    def __init__(self, df, params=True, **kw):
        """
        Initialize a Diamond panel object.

        This constructor sets up a Diamond panel instance with the provided DataFrame,
        calculates the panel points, and optionally computes indices.

        Parameters:
        df (pandas.DataFrame): A DataFrame containing the data for the diamond panel.
                               This should include 'NaK_epm' and 'HCO3CO3_epm' columns.
        params (bool, optional): If True, calculate all indices using the Indixes class.
                                 Defaults to True.
        **kw: Additional keyword arguments to be passed to the Indixes class.

        Returns:
        None

        Attributes:
        panel (str): Set to 'diamond' to identify the panel type.
        components (list): List of component names used in the diamond panel.
        points (numpy.ndarray): Transformed coordinates of the panel points.
        Params (Indixes or None): An Indixes object containing calculated indices for the panel,
                                  or None if params is False.
        """
        super().__init__(df)
        self.panel = 'diamond'
        self.components = ['NaK_epm', 'HCO3CO3_epm']
        points = df.filter(items=self.components)
        points['HCO3CO3_epm'] = points['HCO3CO3_epm'].map(lambda x: 100 - x)
        self.points = self.transform(points.to_numpy())
        self.Params = Indixes(self.points, df, self.panel,
                              **kw) if params else None

    def transform(self, points):
        ''' Make all the required euclidean transformation to the panel's
        points'''
        T = Transform()
        p_trans = (
            points @ T.scale() @ T.d_shear() @ T.rotation()) + T.d_translation()
        return p_trans

    def info(self):
        """
        Generate information to be displayed on the diamond panel of the Piper diagram.

        This method creates a formatted string containing various parameters and indices
        related to the diamond panel, including the number of points, area, shape index,
        orientation angle, Blau index, and standard deviation.

        Returns:
        tuple: A tuple containing three elements:
            - x (float): The x-coordinate for positioning the information on the diagram (0.07).
            - y (float): The y-coordinate for positioning the information on the diagram (0.68).
            - text (str): A formatted string containing the panel information, including:
                - Panel name and number of points
                - Ai (%): Area index
                - Si (%): Shape index
                - Or (°): Orientation angle in degrees
                - Bi (%): Blau index
                - Di (%): Standard deviation
        """
        x, y = 0.07, 0.68
        text = '\n'.join([
            "{:^22}".format(r'${}$ ({})'.format(
                'Diamond', self.Params.num_points)),
            "{:<}: {:>7.2f}".format('Ai (%)', self.Params.area),
            "{:<}: {:>8.2f}".format('Si (%)', self.Params.shape_idx),
            "{:<}: {:>8.2f}".format('Or ({})'.format('$^{o}$'),
                                    self.Params.angle),
            "{:<}: {:>7.2f}".format('Bi (%)', self.Params.blau_idx[0]),
            "{:<}: {:>7.2f}".format('Di (%)', self.Params.sd)
        ])
        return x, y, text

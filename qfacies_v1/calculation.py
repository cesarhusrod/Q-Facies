# -*- coding: utf-8 -*-
"""
===============================================================================
======================== Module of Q-Facies package ============================
===============================================================================
Contains two classes that allow to apply all the euclidean transformations 
requiered, calculate all of the indeces for each panel, and to identify outliers.
===============================================================================
===============================================================================
"""
import numpy as np

_lof = False  # Whether to consider outlier analyses or not.


def import_skl():
    from sklearn.neighbors import LocalOutlierFactor
    from matplotlib.path import Path
    global LocalOutlierFactor, Path
    global _lof
    _lof = True


class Indixes:
    ''' Class that includes all calculation methods of Q-Facies indices.'''

    def __init__(self, points, df, panel, **kw):
        """
        Initialize the Indixes class.

        This method sets up the initial state of the Indixes object, including data preprocessing,
        outlier detection, convex hull calculation, and index creation.

        Parameters:
        -----------
        points : numpy.ndarray
            Array of points representing the data in 2D space.
        df : pandas.DataFrame
            DataFrame containing the original data.
        panel : str
            The type of panel being analyzed ('anion', 'cation', or 'diamond').
        **kw : dict
            Additional keyword arguments. Expected to contain 'lof_neighbours' for LOF calculation.

        Returns:
        --------
        None

        Note:
        -----
        This method modifies the object's state by setting various attributes and
        calling other methods for further calculations.
        """
        global _lof
        self.panel = panel
        self.df = df
        # self.df.dropna(inplace=True)
        self.points = points

        # Calculate convex hull taking into account outliers or not.
        self.LOF(neighbours=kw['lof_neighbours']) if _lof else None
        self.ch_points = self.ConvexHull(self.points)
        # Create indixes
        self.main()

    def main(self):
        """
        Calculate and set various indices and metrics for the dataset.

        This method computes several characteristics of the dataset including:
        - Number of points
        - Area (using Gauss's formula)
        - Perimeter
        - Shape index
        - Blau index
        - Orientation angle
        - Standard distance (dispersion)

        The calculated values are stored as attributes of the class instance.

        Parameters:
        -----------
        None

        Returns:
        --------
        None

        Note:
        -----
        This method modifies the object's state by setting various attributes.
        """
        self.num_points = self.df.shape[0]
        self.area = self.gauss()
        self.perimeter = self.perim()
        self.shape_idx = self.shape(self.area, self.perimeter)
        self.blau_idx = self.blau()  # Tuple (%, values)
        self.angle = self.orientation()
        self.sd = self.dispersion()

    def get(self):
        """
        Retrieve all calculated group parameters for the dataset.

        This method collects various metrics and indices calculated for the dataset
        and returns them as a dictionary.

        Returns:
        --------
        dict
            A dictionary containing the following keys and values:
            - Area: The calculated area of the dataset.
            - Shape: The shape index of the dataset.
            - Angle: The orientation angle of the dataset.
            - Panel: The type of panel being analyzed ('anion', 'cation', or 'diamond').
            - Blau: The normalized Blau index value.
            - Dispersion: The standard distance (dispersion) of the dataset.
            - points: The number of points in the dataset.
            - Dominant: The dominant facies (for cation and anion panels only).
            - Additional keys from self.blau_idx[1]: Various facies proportions.

        Note:
        -----
        The returned dictionary includes all the main characteristics and indices
        calculated for the dataset, providing a comprehensive summary of its properties.
        """
        return dict(Area=self.area, Shape=self.shape_idx,
                    Angle=self.angle, Panel=self.panel,
                    Blau=self.blau_idx[0], Dispersion=self.sd,
                    points=self.df.shape[0], Dominant=self.blau_idx[2],
                    **self.blau_idx[1])

    def ConvexHull(self, points):
        '''
        Create the convex hull of a given array and return the resulting coordinates
        array. The convex hull is calculated by the 'Graham scan' method [1]
        and implemented from RodolfoFerro's code [2].

        [1] Graham, R. L. (1972). An efficient algorithm for determining the
            convex hull of a finite planar set. Info. Pro. Lett., 1, 132-133.

        [2] RodolfoFerro, ConvexHull.GrahamScan (2015), Github.
            https://github.com/RodolfoFerro/ConvexHull
        '''

        def RightTurn(p1, p2, p3):
            ''' Function to determine if we have a counterclock-wise turn.'''
            if (p3[1]-p1[1])*(p2[0]-p1[0]) >= (p2[1]-p1[1])*(p3[0]-p1[0]):
                return False
            return True

        def GrahamScan(P):
            P.sort()  # Sort the x-coordintate of the set of points
            L_upper = [P[0], P[1]]  # Initialize with the left-most point
            # Compute the upper part of the hull
            for i in range(2, len(P)):
                L_upper.append(P[i])
                while len(L_upper) > 2 and not RightTurn(L_upper[-1], L_upper[-2], L_upper[-3]):
                    del L_upper[-2]
            L_lower = [P[-1], P[-2]]  # Initialize the lower part
            # Compute the lower part of the hull
            for i in range(len(P)-3, -1, -1):
                L_lower.append(P[i])
                while len(L_lower) > 2 and not RightTurn(L_lower[-1], L_lower[-2], L_lower[-3]):
                    del L_lower[-2]
            del L_lower[0], L_lower[-1]
            L = L_upper + L_lower		# Build the full hull

            return np.array(L)

        def main(points):
            ''' Execute the Convex-Hull. First, from array to list of tuples.'''
            points = [(x, y) for x, y in zip(points[:, 0], points[:, 1])]
            L = GrahamScan(points)

            return L

        return main(points)

    def centroid(self):
        ''' Centroid coordinates calculation: axis means'''
        return self.points[:, 0].mean(), self.points[:, 1].mean()

    def LOF(self, neighbours=50):
        '''
        Two-steps outlier detection method:
            1_ Unsupervised Outlier Detection using Local Outlier Factor (LOF),
               from sklearn.neighbors.LocalOutlierFactor.
            2_ Exclude from analysis only those outliers outside the
                convex hull polygon.

        Return None. It just removes outlier points by modifying self.points
        attribute and create a new variable to contain them.'''

        mask_1 = LocalOutlierFactor(neighbours).fit_predict(self.points)
        mask_1 = np.where(mask_1 == 1, True, False).astype(bool)

        # Check if preliminary outliers lies within the ch polygon
        initial_inliers = np.compress(mask_1, self.points, axis=0)
        ch_path = Path(self.ConvexHull(initial_inliers))
        isin_pol = np.apply_along_axis(lambda x: ch_path.contains_point(x),
                                       1, self.points).astype(bool)
        # Do not erase outliers if they fall within ch polygon
        mask = np.logical_or(mask_1, isin_pol)
        lof_points = np.compress(mask, self.points, axis=0)
        self.outliers = np.compress(~mask, self.points, axis=0)
        self.points = lof_points  # Redefine points to erase Outliers.

        # Delete selected outliers from df:
        self.df = self.df[mask]

    def gauss(self):
        """
        Calculation of polygon area using Gauss theorem.
        This is computed with the points sorted in a non-clockwise sense.

        a) first diagonal of the determinant b) second one c) Absolute value

        return: polygon area normalized to the polygon extent
        """
        # Last coordinate must be repeted in order to close the polygon.
        points = np.concatenate(([self.ch_points[-1]], self.ch_points))

        a = np.array(
            [i * j for i, j in zip(points[:-1, 0], points[1:, 1])]).sum()
        b = np.array(
            [i * j for i, j in zip(points[1:, 0], points[:-1, 1])]).sum()
        c = abs(a - b)/2

        if self.panel == 'anion' or self.panel == 'cation':  # Area as panel percentage
            # Normalized area to the total triangle area
            return c/(50*np.sin(np.pi/3))
        elif self.panel == 'diamond':
            # Normalized area to the total diamond area
            return c/(100*np.sin(np.pi/3))

    def perim(self):
        """
        Calculate the perimeter of a polygon defined by the convex hull points.

        This method computes the perimeter of the polygon formed by the convex hull points
        of the dataset. It assumes that the points are sorted in a non-clockwise order.

        Parameters:
        -----------
        self : object
            The instance of the class containing the convex hull points (self.ch_points).

        Returns:
        --------
        float
            The calculated perimeter of the polygon.

        Notes:
        ------
        The method closes the polygon by repeating the last point at the beginning,
        calculates the distances between consecutive points using the Pythagorean theorem,
        and sums these distances to obtain the total perimeter.
        """
        # Last coordinate must be repeated in order to close the polygon.
        points = np.concatenate(([self.ch_points[-1]], self.ch_points))
        # Elements differences
        aristas = abs(np.diff(points, axis=0))
        # Pythagorean theorem for the calculation of the hypotenuse
        h = sum([(x**2 + y**2)**0.5 for x, y in zip(aristas[:, 0], aristas[:, 1])])
        return h

    def shape(self, area, perimeter):
        """
        Calculate the Shape Index, introduced by Richardson in 1961.

        This method computes the Shape Index, which provides a measure of how circular
        or linear a shape is. Values close to 100% indicate a circular-like form,
        while values close to zero suggest a more linear-like form.

        Parameters:
        -----------
        area : float
            The area of the shape, typically calculated using the Gauss theorem.
        perimeter : float
            The perimeter of the shape.

        Returns:
        --------
        float
            The Shape Index value, ranging from 0 to 100.
            - Values close to 100 indicate a more circular shape.
            - Values close to 0 indicate a more linear shape.

        Notes:
        ------
        The formula used is: ((4 * π * area) / (perimeter^2)) * 100

        Reference:
        ----------
        Haggett, P., Cliff, A. D., & Frey, A. (1977). Locational analysis in
        human geography (2nd ed.). London: Edward Arnold Ltd.
        """
        area = area*50 if self.area == 'anion' or 'cation' else area*100
        return ((4 * np.pi * area)/(perimeter**2)) * 100

    def orientation(self):
        """
        Calculate the orientation angle of the data points using linear regression.

        This method fits a linear regression model to the data points and calculates
        the angle of the resulting line. The angle is measured in degrees from the
        positive x-axis, with positive angles indicating a counter-clockwise rotation.

        Parameters:
        -----------
        self : object
            The instance of the class containing the data points (self.points).

        Returns:
        --------
        float
            The orientation angle in degrees, ranging from 0 to 360.
            - 0 degrees indicates a horizontal line pointing right.
            - 90 degrees indicates a vertical line pointing up.
            - 180 degrees indicates a horizontal line pointing left.
            - 270 degrees indicates a vertical line pointing down.

        Notes:
        ------
        The method uses numpy's polyfit function to perform the linear regression.
        The resulting angle is adjusted to always be between 0 and 360 degrees.
        """
        x, y = self.points[:, 0], self.points[:, 1]
        coeffs = np.polyfit(x, y, 1)
        angle = (np.arctan(coeffs[0])) * 180/np.pi
        return angle + 180 if angle < 0 else angle

    def blau(self):
        '''
        Calculation of the Blau index [4] that ranges between 0.25 and 1. Values are 
        normalized to the interval [0,100].

        This index gives information about the data dispersion among the four
        possible facies (fs) that conform every panel, being calculated for
        each one of them.

        [4] Blau, P.M. (1977) Inequality and Heterogeneity: A Primitive Theory
                  of Social, Nev York: The Free Press.
        '''
        df = self.df
        total = df.shape[0]     # Total number of samples

        # Facies maps:
        cation_map = dict(A='Magnesium', B='Calcium',
                          C='Sodium-Potassium', D='Mixed')
        anion_map = dict(A='Sulphate', B='Bicarbonate',
                         C='Chloride', D='Mixed')
        panel_map = dict(cation=cation_map, anion=anion_map)

        def get_dominant(facies, panel_map):
            '''Extract the dominant facies on cation and anion panels. Could be
            more than one '''
            max_value = max(facies.values())
            dominants = [key for key, value in facies.items()
                         if value == max_value]
            dominants = [panel_map[x] for x in dominants]
            return '-'.join(dominants) if len(dominants) > 1 else dominants[0]

        # CATION FACIES (facies' points relative frequency).
        if self.panel == 'cation':
            facies = dict(
                A=df.Mg_epm.loc[df.Mg_epm >= 50].count()/total,
                B=df.Ca_epm.loc[df.Ca_epm >= 50].count()/total,
                C=df.NaK_epm.loc[df.NaK_epm >= 50].count()/total)
            facies['D'] = round(1 - sum(facies.values()), 2)

        # ANION FACIES (facies' points relative frequency).
        elif self.panel == 'anion':
            facies = dict(
                A=df.SO4_epm.loc[df.SO4_epm >= 50].count()/total,
                B=df.HCO3CO3_epm.loc[df.HCO3CO3_epm >= 50].count()/total,
                C=df.Cl_epm.loc[df.Cl_epm >= 50].count()/total)
            facies['D'] = round(1 - sum(facies.values()), 2)

        # DIAMOND FACIES (facies' points relative frequency).
        elif self.panel == 'diamond':
            up_carb, down_carb = (df.HCO3CO3_epm >= 50), (df.HCO3CO3_epm < 50)
            up_NaK, down_NaK = (df.NaK_epm >= 50), (df.NaK_epm < 50)

            facies = dict(
                A=df.loc[down_carb & down_NaK].shape[0]/total,
                B=df.loc[down_carb & up_NaK].shape[0]/total,
                C=df.loc[up_carb & up_NaK].shape[0]/total,
                D=df.loc[down_NaK & up_carb].shape[0]/total)

        ''' Blau formula and normalize values to 0 - 100 range'''
        index = 1 - sum([i**2 for i in facies.values()])
        normalized_index = round((index/0.75) * 100, 2)
        dominant = get_dominant(facies, panel_map[self.panel]) if self.panel != \
            'diamond' else None

        return [normalized_index, facies, dominant]

    def dispersion(self):
        """
        Calculate the standard distance index based on Roberto Bachi's 'typical distance' concept.

        This method computes the dispersion of points in the dataset using the standard distance
        formula. The result is normalized to the maximum dispersion value for each panel type.

        Parameters:
        -----------
        self : object
            The instance of the class containing the data points (self.points) and panel type (self.panel).

        Returns:
        --------
        float
            The normalized standard distance index.
            - For cation and anion panels: normalized to 57.73 distance units.
            - For diamond panel: normalized to 70.71 distance units.

        Notes:
        ------
        The calculation is based on:
        Bachi, R. (1963). Standard distance measures and related methods for spatial analysis.
        Papers of the Regional Science Association 10, 83–132. https://doi.org/10.1007/BF01934680

        For a detailed explanation, see:
        https://volaya.github.io/libro-sig/chapters/Estadistica_espacial.html
        """
        x, y = self.points[:, 0], self.points[:, 1]
        sd_distance = np.sqrt(((np.sum(x ** 2) / len(x)) - np.mean(x) ** 2) +
                              ((np.sum(y ** 2) / len(y)) - np.mean(y) ** 2))
        if self.panel == 'anion' or self.panel == 'cation':
            return sd_distance * 100 / 57.735026918962575
        elif self.panel == 'diamond':
            return sd_distance * 100 / 70.71067811865476


class Transform:
    '''
    Affine 2D transformation for plotting points into a Piper diagram.
    Transformation consists in a set of scale, shear, rotation and
    translocation operations that are applied to an xy-array of points.
    '''

    def __init__(self):
        self.offset = 22  # This must match with Plot.Skeleton.offset value.

    def rotation(self):
        '''Rotation transformation of 300 degrees. Apply for diamond panel'''
        return np.array([(np.cos(np.radians(300)), np.sin(np.radians(300))),
                         (-np.sin(np.radians(300)), np.cos(np.radians(300)))])

    def scale(self):
        '''Scale transformation. For cation and anion panels. '''
        return np.array([(1,      0),
                         (0, np.sin(np.pi/3))])

    def t_shear(self):
        '''Shear transformation. For cation and anion panels. '''
        return np.array([(1, 0),
                         (np.tan(np.pi/6), 1)])

    def d_shear(self):
        '''Shear transformation for diamond panel'''
        return np.array([(1, 0),
                         (-np.tan(np.pi/6), 1)])

    def d_translation(self):
        '''Translation of diamond points'''
        Ax = 50 + self.offset/2
        Ay = np.sin(np.pi/3) * (100 + self.offset)
        return np.array([(Ax),
                         (Ay)])

    def a_translation(self):
        '''Translation of anion points'''
        return np.array([(self.offset + 100),
                         (0)])

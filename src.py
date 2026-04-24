import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class MomentumTransformer1D(BaseEstimator, TransformerMixin):
    """
    Transformer that computes polynomial moments for 1D data.

    This class calculates moments (mean, variance, etc.) of the input signal
    over a specified interval, providing a compact representation.
    """

    def __init__(self, order=2, x_bounds=(0, 1)):
        """
        Initialize the transformer.

        Parameters
        ----------
        order : int, default=2
            Maximum order of moments to compute (0 to order).
        x_bounds : tuple of float, default=(0, 1)
            Bounds of the x-axis interval (x_min, x_max).
        """
        self.order = order
        self.n_components = order + 1
        self.x_min = x_bounds[0]
        self.x_max = x_bounds[1]

    def fit(self, X, y=None):
        """
        Fit the transformer. No fitting required for moment computation.

        Parameters
        ----------
        X : array-like
            Input data (ignored).
        y : array-like, optional
            Target values (ignored).

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        return self

    def _get_momentum(self, X, m_order=1, M_1=None):
        """
        Compute a single moment of the given order.

        Parameters
        ----------
        X : ndarray
            Input data of shape (batch, N).
        m_order : int
            Order of the moment (0, 1, or higher).
        M_1 : ndarray, optional
            First moment, used for central moments.

        Returns
        -------
        ndarray
            Computed moment of shape (batch,).
        """
        N = X.shape[-1]  # Number of points in one example
        _argument = np.linspace(self.x_min, self.x_max, N)
        _argument = np.expand_dims(_argument, axis=0)  # Shape: (1, N)

        if m_order == 0:
            return np.sum(X, axis=-1) * (self.x_max - self.x_min) / N  # Total integral
        elif m_order == 1:
            return np.sum(X * _argument, axis=-1) * (self.x_max - self.x_min) / N  # Mean position
        else:
            return (np.sum(X * (_argument - M_1) ** m_order, axis=-1) *
                    (self.x_max - self.x_min) / N) ** (1 / m_order)  # Central moment

    def transform(self, X, y=None):
        """
        Transform input data to moment features.

        Parameters
        ----------
        X : ndarray
            Input data of shape (batch, N).
        y : array-like, optional
            Target values (ignored).

        Returns
        -------
        ndarray
            Transformed data of shape (batch, n_components).
        """
        M_0 = self._get_momentum(X, m_order=0)  # Zero-order moment (integral)
        M_0 = np.expand_dims(M_0, axis=-1)
        if self.order == 0:
            return M_0

        M_1 = self._get_momentum(X, m_order=1)  # First-order moment (mean)
        M_1 = np.expand_dims(M_1, axis=-1)

        l_momentum_X = [M_0, M_1]

        for m_order in range(2, self.order + 1):
            M = self._get_momentum(X, m_order, M_1)  # Higher central moments
            M = np.expand_dims(M, axis=-1)
            l_momentum_X.append(M)

        momentum_X = np.concatenate(l_momentum_X, axis=-1)

        return momentum_X

    def get_pd_table(self, X=None):
        """
        Get moments as a pandas DataFrame.

        Parameters
        ----------
        X : ndarray, optional
            Input data. If None, uses previously transformed data.

        Returns
        -------
        pd.DataFrame
            DataFrame with moment columns.
        """
        X = self.transform(X)
        columns = [f"M_{i}" for i in range(self.n_components)]

        pd_table = pd.DataFrame(X, columns=columns)
        return pd_table


class MomentumTransformer2D(BaseEstimator, TransformerMixin):
    def __init__(self, order=2, x_order=None, y_order=None, x_bounds=(0, 1), y_bounds=(0, 1)):
        if order != None:
            if x_order == None: x_order = order
            if y_order == None: y_order = order
        else:
            order = x_order + y_order

        self.order = order
        self.x_order = x_order
        self.y_order = y_order
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds

        self.x_min = x_bounds[0]
        self.x_max = x_bounds[1]
        self.y_min = y_bounds[0]
        self.y_max = y_bounds[1]

        self.order_pairs = []
        for sum_i in range(0, np.max((order, x_order, y_order))+1):
            for x_i in range(0, sum_i+1):
                y_i = sum_i - x_i
                if (x_i + y_i <= order) and (x_i <= x_order) and (y_i <= y_order):
                    self.order_pairs.append((x_i, y_i))

        self.n_components = len(self.order_pairs)


    def fit(self, X, y=None):
        return self

    def _get_momentum(self, X, m_x_order=1, m_y_order=1, M_x1_y0=None, M_x0_y1=None):
        # X.shape=[batch, N_y, N_x]
        if len(X.shape)==2:
            X = np.array(X)

        N_x = X.shape[-1] # N_x is a number of points along x-axis
        N_y = X.shape[-2] # N_y is a number of points along y-axis

        _x_argument = np.linspace(self.x_min, self.x_max, N_x)
        _x_argument = np.expand_dims(_x_argument, axis=(0,1)) # _x_argument.shape=[1, 1, N_x]
        _y_argument = np.linspace(self.y_min, self.y_max, N_y)
        _y_argument = np.expand_dims(_y_argument, axis=(0,-1)) # _y_argument.shape=[1, N_y, 1]

        d_x_d_y = (self.x_max - self.x_min) / N_x * (self.y_max - self.y_min) / N_y

        if m_x_order+m_y_order==0:
            return np.sum(X, axis=(-1,-2)) * d_x_d_y # shape=[batch]
        elif m_x_order==1 and m_y_order==0:
            return np.sum(X * _x_argument, axis=(-1,-2)) * d_x_d_y # shape=[batch]
        elif m_x_order==0 and m_y_order==1:
            return np.sum(X * _y_argument, axis=(-1,-2)) * d_x_d_y # shape=[batch]
        else:
            M_x1_y0 = np.expand_dims(M_x1_y0, axis=-1) # shape=[batch, 1, 1]
            M_x0_y1 = np.expand_dims(M_x0_y1, axis=-1) # shape=[batch, 1, 1]
            #M = (np.sum(X * (_x_argument - M_x1_y0)**m_x_order * (_y_argument - M_x0_y1)**m_y_order, axis=(-1,-2)) * d_x_d_y)**(1 / (m_x_order + m_y_order)) # shape=[batch]
            #return (np.sum(X * (_x_argument - M_x1_y0)**m_x_order * (_y_argument - M_x0_y1)**m_y_order, axis=(-1,-2)) * d_x_d_y) # shape=[batch]
            M = (np.sum(X * (_x_argument - M_x1_y0)**m_x_order * (_y_argument - M_x0_y1)**m_y_order, axis=(-1,-2)) * d_x_d_y) # shape=[batch]
            sign_M = np.sign(M)
            M = sign_M * np.abs(M)**(1 / (m_x_order + m_y_order))
            return M

    def transform(self, X, y=None):
        M_x0_y0 = self._get_momentum(X, m_x_order=0, m_y_order=0)
        M_x0_y0 = np.expand_dims(M_x0_y0, axis=-1)
        if self.order==0: return M_x0_y0

        M_x1_y0 = self._get_momentum(X, m_x_order=1, m_y_order=0)
        M_x1_y0 = np.expand_dims(M_x1_y0, axis=-1)

        M_x0_y1 = self._get_momentum(X, m_x_order=0, m_y_order=1)
        M_x0_y1 = np.expand_dims(M_x0_y1, axis=-1)

        l_momentum_X = [M_x0_y0, M_x1_y0, M_x0_y1]

        for m_x_order, m_y_order in self.order_pairs[3:]:
            M = self._get_momentum(X, m_x_order, m_y_order, M_x1_y0, M_x0_y1)
            M = np.expand_dims(M, axis=-1)
            l_momentum_X.append(M)

        momentum_X = np.concatenate(l_momentum_X, axis=-1)

        return momentum_X

    def get_pd_table(self, X=None):
        X = self.transform(X)
        columns = [f"M_x{x}_y{y}" for x, y in self.order_pairs]

        pd_table = pd.DataFrame(X, columns=columns)
        return pd_table

    def get_momenta_names(self):
        return [f"M_x{x}_y{y}" for x, y in self.order_pairs]


class LegendreTransformer1D(BaseEstimator, TransformerMixin):
    """
    Transformer that computes Legendre polynomial coefficients for 1D data.

    This class projects 1D signals onto Legendre polynomials on [-1, 1],
    providing orthogonal basis expansion for signal representation.
    """

    def __init__(self, order=2):
        """
        Initialize the transformer.

        Parameters
        ----------
        order : int, default=2
            Maximum degree of Legendre polynomials (0 to order).
        """
        self.order = order
        self.n_components = order + 1

    def fit(self, X, y=None):
        """
        Fit the transformer. No fitting required for Legendre transform.

        Parameters
        ----------
        X : array-like
            Input data (ignored).
        y : array-like, optional
            Target values (ignored).

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        return self

    def transform(self, X, y=None):
        """
        Transform input data to Legendre coefficients.

        Parameters
        ----------
        X : ndarray
            Input data of shape (batch, N).
        y : array-like, optional
            Target values (ignored).

        Returns
        -------
        ndarray
            Legendre coefficients of shape (batch, n_components).
        """
        N = X.shape[-1]  # Number of points in one example

        # Normalize the argument to [-1, 1]
        x_norm = np.linspace(-1, 1, N)
        x_norm = np.expand_dims(x_norm, axis=0)  # Shape: (1, N)

        dx = 2 / N  # Integration step for [-1, 1]

        l_coeffs = []
        for n in range(self.order + 1):
            P_n = np.polynomial.legendre.Legendre.basis(n)  # Legendre polynomial basis
            p_vals = P_n(x_norm)  # Shape: (1, N)

            integral = np.sum(X * p_vals, axis=-1) * dx  # Inner product with basis
            a_n = (2 * n + 1) / 2 * integral  # Legendre coefficient normalization

            a_n = np.expand_dims(a_n, axis=-1)  # Shape: (batch, 1)
            l_coeffs.append(a_n)

        coeffs = np.concatenate(l_coeffs, axis=-1)  # Shape: (batch, order+1)

        return coeffs

    def get_pd_table(self, X=None):
        """
        Get coefficients as a pandas DataFrame.

        Parameters
        ----------
        X : ndarray, optional
            Input data. If None, uses previously transformed data.

        Returns
        -------
        pd.DataFrame
            DataFrame with coefficient columns.
        """
        X = self.transform(X)
        columns = [f"L_{n}" for n in range(self.n_components)]

        pd_table = pd.DataFrame(X, columns=columns)
        return pd_table


class LegendreTransformer2D(BaseEstimator, TransformerMixin):
    def __init__(self, order=2, x_order=None, y_order=None):
        if order is not None:
            if x_order is None: x_order = order
            if y_order is None: y_order = order
        else: 
            order = x_order + y_order
        
        self.order = order
        self.x_order = x_order
        self.y_order = y_order
        
        self.order_pairs = []
        for sum_i in range(0, np.max((order, x_order, y_order)) + 1):
            for x_i in range(0, sum_i + 1):
                y_i = sum_i - x_i
                if (x_i + y_i <= order) and (x_i <= x_order) and (y_i <= y_order):
                    self.order_pairs.append((x_i, y_i))
        
        self.n_components = len(self.order_pairs)
        

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        # X.shape=[batch, N_y, N_x]
        N_y, N_x = X.shape[-2], X.shape[-1]
        
        # Normalize arguments to [-1, 1]
        x_norm = np.linspace(-1, 1, N_x)
        x_norm = np.expand_dims(x_norm, axis=(0, 1))  # [1, 1, N_x]
        y_norm = np.linspace(-1, 1, N_y)
        y_norm = np.expand_dims(y_norm, axis=(0, -1))  # [1, N_y, 1]
        
        dx_dy = 4 / (N_x * N_y)  # Integration area element
        
        l_coeffs = []
        for m, n in self.order_pairs:
            P_m = np.polynomial.legendre.Legendre.basis(m)
            P_n = np.polynomial.legendre.Legendre.basis(n)
            
            p_m_vals = P_m(x_norm)  # [1, 1, N_x]
            p_n_vals = P_n(y_norm)  # [1, N_y, 1]
            
            integral = np.sum(X * p_m_vals * p_n_vals, axis=(-1, -2)) * dx_dy  # [batch]
            a_mn = (2 * m + 1) * (2 * n + 1) / 4 * integral  # Legendre coefficient
            
            a_mn = np.expand_dims(a_mn, axis=-1)  # [batch, 1]
            l_coeffs.append(a_mn)
        
        coeffs = np.concatenate(l_coeffs, axis=-1)  # [batch, n_components]
        
        return coeffs
    
    def get_pd_table(self, X=None):
        X = self.transform(X)
        columns = [f"L_x{x}_y{y}" for x, y in self.order_pairs]
        
        pd_table = pd.DataFrame(X, columns=columns)
        return pd_table


class FourierTransformer1D(BaseEstimator, TransformerMixin):
    """
    Transformer that computes Fourier series coefficients for 1D data.

    This class expands 1D signals into Fourier series on [-1, 1],
    using cosine and sine basis functions for harmonic analysis.
    """

    def __init__(self, order=2):
        """
        Initialize the transformer.

        Parameters
        ----------
        order : int, default=2
            Maximum harmonic order (0 to order for cosines, 1 to order for sines).
        """
        self.order = order
        self.n_components = 2 * order + 1  # a0, a1..a_order, b1..b_order

    def fit(self, X, y=None):
        """
        Fit the transformer. No fitting required for Fourier transform.

        Parameters
        ----------
        X : array-like
            Input data (ignored).
        y : array-like, optional
            Target values (ignored).

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        return self

    def transform(self, X, y=None):
        """
        Transform input data to Fourier coefficients.

        Parameters
        ----------
        X : ndarray
            Input data of shape (batch, N).
        y : array-like, optional
            Target values (ignored).

        Returns
        -------
        ndarray
            Fourier coefficients of shape (batch, n_components).
        """
        N = X.shape[-1]  # Number of points in one example

        # Normalize the argument to [-1, 1]
        x_norm = np.linspace(-1, 1, N)
        x_norm = np.expand_dims(x_norm, axis=0)  # Shape: (1, N)

        dx = 2 / N  # Integration step for [-1, 1]

        l_coeffs = []
        # a0 - constant term
        a0 = np.sum(X, axis=-1) * dx
        a0 = np.expand_dims(a0, axis=-1)
        l_coeffs.append(a0)

        for k in range(1, self.order + 1):
            # a_k - cosine coefficients
            a_k = np.sum(X * np.cos(k * np.pi * x_norm), axis=-1) * dx
            a_k = np.expand_dims(a_k, axis=-1)
            l_coeffs.append(a_k)

            # b_k - sine coefficients
            b_k = np.sum(X * np.sin(k * np.pi * x_norm), axis=-1) * dx
            b_k = np.expand_dims(b_k, axis=-1)
            l_coeffs.append(b_k)

        coeffs = np.concatenate(l_coeffs, axis=-1)  # Shape: (batch, 2*order+1)

        return coeffs

    def get_pd_table(self, X=None):
        """
        Get coefficients as a pandas DataFrame.

        Parameters
        ----------
        X : ndarray, optional
            Input data. If None, uses previously transformed data.

        Returns
        -------
        pd.DataFrame
            DataFrame with coefficient columns.
        """
        X = self.transform(X)
        columns = ['F_a0']
        for k in range(1, self.order + 1):
            columns.append(f'F_a{k}')
            columns.append(f'F_b{k}')

        pd_table = pd.DataFrame(X, columns=columns)
        return pd_table


class FourierTransformer2D(BaseEstimator, TransformerMixin):
    """
    Transformer that computes 2D Fourier series coefficients for 2D data.

    This class expands 2D images into 2D Fourier series on [-1, 1] x [-1, 1],
    using combinations of cosine and sine basis functions for harmonic analysis.
    Coefficients are conditionally included based on m and n values to avoid redundancy.
    """

    def __init__(self, order=2, x_order=None, y_order=None):
        """
        Initialize the transformer.

        Parameters
        ----------
        order : int, default=2
            Maximum total order of 2D Fourier harmonics.
        x_order : int, optional
            Maximum order in x-direction. If None, uses order.
        y_order : int, optional
            Maximum order in y-direction. If None, uses order.
        """
        if order is not None:
            if x_order is None: x_order = order
            if y_order is None: y_order = order
        else:
            order = x_order + y_order

        self.order = order
        self.x_order = x_order
        self.y_order = y_order

        self.order_pairs = []
        self.n_components = 0
        for sum_i in range(0, np.max((order, x_order, y_order)) + 1):
            for x_i in range(0, sum_i + 1):
                y_i = sum_i - x_i
                if (x_i + y_i <= order) and (x_i <= x_order) and (y_i <= y_order):
                    self.order_pairs.append((x_i, y_i))
                    self.n_components += 1
                    if y_i != 0:
                        self.n_components += 1
                    if x_i != 0:
                        self.n_components += 1
                    if x_i != 0 and y_i != 0:
                        self.n_components += 1

    def fit(self, X, y=None):
        """
        Fit the transformer. No fitting required for Fourier transform.

        Parameters
        ----------
        X : array-like
            Input data (ignored).
        y : array-like, optional
            Target values (ignored).

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        return self

    def transform(self, X, y=None):
        """
        Transform input data to 2D Fourier coefficients.

        Parameters
        ----------
        X : ndarray
            Input data of shape (batch, N_y, N_x).
        y : array-like, optional
            Target values (ignored).

        Returns
        -------
        ndarray
            Fourier coefficients of shape (batch, n_components).
        """
        # X.shape=[batch, N_y, N_x]
        N_y, N_x = X.shape[-2], X.shape[-1]

        # Normalize arguments to [-1, 1]
        x_norm = np.linspace(-1, 1, N_x)
        x_norm = np.expand_dims(x_norm, axis=(0, 1))  # [1, 1, N_x]
        y_norm = np.linspace(-1, 1, N_y)
        y_norm = np.expand_dims(y_norm, axis=(0, -1))  # [1, N_y, 1]

        dx_dy = 4 / (N_x * N_y)  # Integration area element

        l_coeffs = []
        for m, n in self.order_pairs:
            a_mn = np.sum(X * np.cos(m * np.pi * x_norm) * np.cos(n * np.pi * y_norm), axis=(-1, -2)) * dx_dy
            a_mn = np.expand_dims(a_mn, axis=-1)
            l_coeffs.append(a_mn)

            if n != 0:
                b_mn = np.sum(X * np.cos(m * np.pi * x_norm) * np.sin(n * np.pi * y_norm), axis=(-1, -2)) * dx_dy
                b_mn = np.expand_dims(b_mn, axis=-1)
                l_coeffs.append(b_mn)

            if m != 0:
                c_mn = np.sum(X * np.sin(m * np.pi * x_norm) * np.cos(n * np.pi * y_norm), axis=(-1, -2)) * dx_dy
                c_mn = np.expand_dims(c_mn, axis=-1)
                l_coeffs.append(c_mn)

            if m != 0 and n != 0:
                d_mn = np.sum(X * np.sin(m * np.pi * x_norm) * np.sin(n * np.pi * y_norm), axis=(-1, -2)) * dx_dy
                d_mn = np.expand_dims(d_mn, axis=-1)
                l_coeffs.append(d_mn)

        coeffs = np.concatenate(l_coeffs, axis=-1)  # [batch, n_components]
        return coeffs

    def get_pd_table(self, X=None):
        """
        Get coefficients as a pandas DataFrame.

        Parameters
        ----------
        X : ndarray, optional
            Input data. If None, uses previously transformed data.

        Returns
        -------
        pd.DataFrame
            DataFrame with coefficient columns.
        """
        X = self.transform(X)
        columns = []
        for m, n in self.order_pairs:
            columns.append(f"F_a_{m}_{n}")
            if n != 0:
                columns.append(f"F_b_{m}_{n}")
            if m != 0:
                columns.append(f"F_c_{m}_{n}")
            if m != 0 and n != 0:
                columns.append(f"F_d_{m}_{n}")

        pd_table = pd.DataFrame(X, columns=columns)
        return pd_table


class ZernikeTransformer2D(BaseEstimator, TransformerMixin):
    """
    Transformer that computes Zernike polynomial coefficients for 2D data.

    This class projects 2D images onto Zernike polynomials on a circle,
    providing orthogonal basis expansion for optical wavefront analysis.
    """

    def __init__(self, order=2, x_bounds=(0, 1), y_bounds=(0, 1), center=None, radius_mode='min'):
        """
        Initialize the transformer.

        Parameters
        ----------
        order : int, default=2
            Maximum radial order of Zernike polynomials.
        x_bounds : tuple of float, default=(0, 1)
            Bounds of the x-axis.
        y_bounds : tuple of float, default=(0, 1)
            Bounds of the y-axis.
        center : tuple of float, optional
            Center coordinates (x0, y0). If None, computed as centroid.
        radius_mode : str, default='min'
            'min' for minimum distance to boundary, 'max' for maximum to corner.
        """
        self.order = order
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.center = center
        self.radius_mode = radius_mode

        # Zernike modes: list of (n, m) pairs
        self.modes = []
        for n in range(order + 1):
            for m in range(-n, n + 1, 2):  # m from -n to n, step 2
                self.modes.append((n, m))
        self.n_components = len(self.modes)

    def _radial_poly(self, n, m, r):
        """
        Compute radial Zernike polynomial R_n^m(r).
        """
        if n == 0:
            return np.ones_like(r)
        R = np.zeros_like(r)
        for k in range((n - abs(m)) // 2 + 1):
            binom1 = np.math.factorial(n - k)
            binom2 = np.math.factorial(k)
            binom3 = np.math.factorial((n + abs(m)) // 2 - k)
            binom4 = np.math.factorial((n - abs(m)) // 2 - k)
            R += ((-1) ** k * binom1 / (binom2 * binom3 * binom4)) * r ** (n - 2 * k)
        return R

    def _zernike_poly(self, n, m, x, y, x0, y0, R):
        """
        Compute Zernike polynomial Z_n^m at points (x, y).
        """
        r = np.sqrt((x - x0)**2 + (y - y0)**2) / R
        theta = np.arctan2(y - y0, x - x0)
        mask = r <= 1  # Only inside circle
        Z = np.zeros_like(r)
        if abs(m) <= n:
            R_nm = self._radial_poly(n, abs(m), r)
            if m >= 0:
                Z[mask] = R_nm[mask] * np.cos(m * theta[mask])
            else:
                Z[mask] = R_nm[mask] * np.sin(abs(m) * theta[mask])
        return Z

    def fit(self, X, y=None):
        """
        Fit the transformer.

        No internal parameters are learned in advance for Zernike expansion.

        Parameters
        ----------
        X : array-like
            Input data (ignored).
        y : array-like, optional
            Target values (ignored).

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        return self

    def transform(self, X, y=None):
        """
        Transform input data to Zernike coefficients.

        The center of mass is computed for each sample when `center` is None.
        The radius is then derived from the chosen radius mode.

        Parameters
        ----------
        X : ndarray
            Input data of shape (batch, N_y, N_x).
        y : array-like, optional
            Target values (ignored).

        Returns
        -------
        ndarray
            Zernike coefficients of shape (batch, n_components).
        """
        batch, N_y, N_x = X.shape

        # Physical grid in x and y
        x = np.linspace(self.x_bounds[0], self.x_bounds[1], N_x)
        y = np.linspace(self.y_bounds[0], self.y_bounds[1], N_y)
        XX, YY = np.meshgrid(x, y)

        dx = x[1] - x[0]
        dy = y[1] - y[0]
        dx_dy = dx * dy

        if self.center is None:
            x_coords = x[None, None, :]
            y_coords = y[None, :, None]
            total = np.sum(X, axis=(-1, -2))
            x0 = np.sum(X * x_coords, axis=(-1, -2)) / np.where(total == 0, 1, total)
            y0 = np.sum(X * y_coords, axis=(-1, -2)) / np.where(total == 0, 1, total)
            x0 = np.where(total == 0,
                          (self.x_bounds[0] + self.x_bounds[1]) / 2,
                          x0)
            y0 = np.where(total == 0,
                          (self.y_bounds[0] + self.y_bounds[1]) / 2,
                          y0)
            centers = np.stack([x0, y0], axis=-1)
        else:
            x0, y0 = self.center
            centers = np.tile(np.array([x0, y0], dtype=float)[None, :], (batch, 1))

        # Determine radius for each sample.
        # For the inscribed unit circle, the limiting radius is the nearest boundary side.
        x_min, x_max = self.x_bounds
        y_min, y_max = self.y_bounds
        side_distances = np.stack([
            centers[:, 0] - x_min,
            x_max - centers[:, 0],
            centers[:, 1] - y_min,
            y_max - centers[:, 1]
        ], axis=-1)

        if self.radius_mode == 'min':
            radii = np.min(side_distances, axis=-1)
        else:
            corners = np.array([[x_min, y_min], [x_min, y_max], [x_max, y_min], [x_max, y_max]])
            distances = np.sqrt(((centers[:, None, :] - corners[None, :, :]) ** 2).sum(axis=-1))
            radii = np.max(distances, axis=1)

        self.last_centers = centers
        self.last_radii = radii

        coeffs = np.zeros((batch, self.n_components), dtype=float)
        for i in range(batch):
            sample = X[i]
            x0_i, y0_i = centers[i]
            R_i = radii[i]
            for j, (n, m) in enumerate(self.modes):
                Z_nm = self._zernike_poly(n, m, XX, YY, x0_i, y0_i, R_i)
                coeffs[i, j] = ((n + 1) / np.pi) * np.sum(sample * Z_nm) * dx_dy / R_i**2

        return coeffs

    def get_pd_table(self, X=None):
        """
        Get coefficients as a pandas DataFrame.

        Parameters
        ----------
        X : ndarray, optional
            Input data. If None, uses previously transformed data.

        Returns
        -------
        pd.DataFrame
            DataFrame with coefficient columns.
        """
        X = self.transform(X)
        columns = [f"Z_{n}_{m}" for n, m in self.modes]

        pd_table = pd.DataFrame(X, columns=columns)
        return pd_table
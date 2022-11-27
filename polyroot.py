# -*- coding: utf-8 -*-
"""

Functions for "Scientific Computing with Python" Assignment 2

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from sympy import Symbol, Poly, printing
plt.rcParams['text.usetex'] = True


def secant(x0, p, *args, **kwargs):
    """
    
    Secant method for finding a root of the polynomial with coefficients p from the starting points x0+-sigma.
    
    Parameters
    ----------
    x0 : float
        Starting guess root location.
    p : array
        Polynomial coefficients.
    *args : 
        None.
    **kwargs :
        epsilon : float, optional
            Accepted accuracy for root (default is 1e-10).
        limit : int, optional
            Highest count n can reach before the convergence attempt is exited (default is 100).
        sigma : float, optional
            Deviation from starting point (default is 0.2).

    Returns
    -------
    float
        Root location. If unexpected zero, limit reached.
    int
        Number of Iterations.
        
    """
    
    # Set epsilon (root accuracy), limit (when to stop the iteration) and sigma (deviation from x0) from keyword arguments (or default).
    epsilon = kwargs.get("epsilon") if kwargs.get("epsilon") != None else 1e-10
    limit   = kwargs.get("limit") if kwargs.get("limit") != None else 100
    sigma   = kwargs.get("sigma") if kwargs.get("sigma") != None else 0.2
    
    # Set the two starting guesses, x=x_{n-2} and y=x_{n-1} as in formula. Begin iteration count at zero.
    x = x0-sigma #x_{n-2}
    y = x0+sigma #x_{n-1}
    n = 0
    
    # Iterate the roots when | polynomial at y | > epsilon (root accuracy) using the secant method.
    while abs(np.polyval(p, y)) > epsilon:
        y, x = y - np.polyval(p, y)*( (y-x)/(np.polyval(p, y)-np.polyval(p, x)) ), y
        
        # Iterate counter and check limit hasn't been reached. If it has, return null values.
        n += 1
        if n > limit:
            return 0, 0
    
    # Return root and number of iterations if the required root accuracy is met.
    return y, n


def newraph(x0, p, *args, **kwargs):
    """
    
    Newton-Raphson method for finding a root of the polynomial with coefficients p from the starting point x0.
    
    Parameters
    ----------
    x0 : float
        Starting guess root location.
    p : array
        Polynomial coefficients.
    *args : 
        None.
    **kwargs :
        epsilon : float, optional
            Accepted accuracy for root (default is 1e-10).
        limit : int, optional
            Highest count n can reach before the convergence attempt is exited (default is 100).

    Returns
    -------
    float
        Root location. If unexpected zero, limit reached.
    int
        Number of Iterations.
        
    """
    
    # Set epsilon (root accuracy) and limit (when to stop the iteration) from keyword arguments (or default).
    epsilon = kwargs.get("epsilon") if kwargs.get("epsilon") != None else 1e-10
    limit   = kwargs.get("limit") if kwargs.get("limit") != None else 100
    
    # Set the starting guess, x = x0 as in formula. Begin iteration count at zero.
    x = x0
    n = 0

    # Iterate the roots when | polynomial at x | > epsilon (root accuracy) using the Newton-Raphson method.
    while abs(np.polyval(p, x)) > epsilon:
        x = x - np.polyval(p, x)/np.polyval(np.polyder(p), x)
        
        # Iterate counter and check limit hasn't been reached. If it has, return null values.
        n += 1
        if n > limit:
            return 0, 0
    
    # Return root and number of iterations if the required root accuracy is met.
    return x, n


def bisection(x0, p, **kwargs):
    """
    
    Bisection method for finding a root of the polynomial with coefficients p from the starting points x0+-sigma.
    
    Parameters
    ----------
    x0 : float
        Starting guess root location.
    p : array
        Polynomial coefficients.
    *args : 
        None.
    **kwargs :
        epsilon : float, optional
            Accepted accuracy for root (default is 1e-10).
        limit : int, optional
            Highest count n can reach before the convergence attempt is exited (default is 100).
        sigma : float, optional
            Deviation from starting point (default is 0.2).

    Returns
    -------
    float
        Root location. If unexpected zero, limit reached.
    int
        Number of Iterations.
        
    """
    
    # Set epsilon (root accuracy), limit (when to stop the iteration) and sigma (deviation from x0) from keyword arguments (or default).
    epsilon = kwargs.get("epsilon") if kwargs.get("epsilon") != None else 1e-10
    limit   = kwargs.get("limit") if kwargs.get("limit") != None else 100
    sigma   = kwargs.get("sigma") if kwargs.get("sigma") != None else 0.2
    
    # Set the upper (xu) and lower (xd) starting guesses. Begin iteration count at zero.
    xd=x0-sigma
    xu=x0+sigma
    n = 0

    # Check if upper and lower are both about a root (sign change). If not, return null values.
    if np.polyval(p, xd)*np.polyval(p, xu) > 0:
        print("Not around a root.")
        return 0, 0

    # Iterate the roots |upper guess - lower guess| > epsilon (root accuracy) using the bisection method (midpoint).
    while abs(xu - xd) > epsilon:
        xmid = (xd+xu)/2
        if np.polyval(p, xu)*np.polyval(p, xmid) > 0:
            xu = xmid
        else:
            xd = xmid
        
        # Iterate counter and check limit hasn't been reached. If it has, return null values.
        n += 1
        if n > limit:
            print("Limit reached in bisection.")
            return 0, 0
    
    # Return root and number of iterations if the required root accuracy is met.
    return xmid, n


def bisecant(x0, p, **kwargs):
    """
    
    Hybrid bisection-secant method for finding a root of the polynomial with coefficients p from the starting points x0+-sigma.
    
    Parameters
    ----------
    x0 : float
        Starting guess root location.
    p : array
        Polynomial coefficients.
    *args : 
        None.
    **kwargs :
        epsilon : float, optional
            Accepted accuracy for root (default is 1e-10).
        limit : int, optional
            Highest count n can reach before the convergence attempt is exited (default is 100).
        sigma : float, optional
            Deviation from starting point (default is 0.2).

    Returns
    -------
    float
        Root location. If unexpected zero, limit reached.
    int
        Number of Iterations.
        
    """
    
    # Set epsilon (root accuracy), limit (when to stop the iteration) and sigma (deviation from x0) from keyword arguments (or default).
    epsilon = kwargs.get("epsilon") if kwargs.get("epsilon") != None else 1e-10
    limit   = kwargs.get("limit") if kwargs.get("limit") != None else 100
    sigma   = kwargs.get("sigma") if kwargs.get("sigma") != None else 0.2
    
    # Set the upper (xu) and lower (xd) starting guesses. Begin iteration count at zero.
    xd=x0-sigma
    xu=x0+sigma
    n = 0

    # Check if upper and lower are both around a root (sign change). If not, return null values.
    if np.polyval(p, xd)*np.polyval(p, xu) > 0:
        print("Not around a root.")
        return 0, 0

    # Iterate the roots |upper guess - lower guess| > epsilon (root accuracy) using the hybrid bisection-secant method, with the midpoint being replaced by the secant method.
    while abs(xu - xd) > epsilon:
        xs = xu - np.polyval(p, xu)*( (xu-xd)/(np.polyval(p, xu) - np.polyval(p, xd)) )
        
        if np.polyval(p, xu)*np.polyval(p, xs) > 0:
            xu = xs
        else:
            xd = xs
        
        # Iterate counter and check limit hasn't been reached. If it has, return null values.
        n += 1
        if n > limit:
            print("Limit reached in bisecant.")
            return 0, 0
    
    # Return root and number of iterations if the required root accuracy is met.
    return xs, n


def convplot(p, func, **kwargs):
    """

    Create a convergence graph for the polynomial with coefficients p based upon the root finding algorithm alg.
    
    Parameters
    ----------
    p : array
        Polynomial coefficients to plot.
    func : function
        Root finding algorithm that returns root, number of iterations.
    **kwargs :
        name : str, optional
            name of root finding method (default is alg.__name__).
        cmap : linearly segmented colormap, optional
            Colour map for the data to be presented in (default is plasma).
        epsilon : float, optional
            Accepted accuracy for root (default is 1e-10).
        width : float, optional
            Width of bars on iteration plot that also corresponds to number of bars (default is 0.05).
        sigma : float, optional
            Deviation from starting point for secant method (default is 0.2).
        limit : int, optional
            Height of bar chart that also corresponds to the highest count n can reach before the convergence attempt is exited (default is 100).
        specroot : int, optional
            Integer value of specific root that must be in colour. Starts at 0, but order is based on algorithm.
        specrootcolor: string, optional 
            Hex value of specific root colour (default is crimson, "#dc143c").
        
    Returns
    ------
    None. Saves figure with given name.
    
    """
    
    # Set name (plot and save name), cmap (colour map for plot), epsilon (root accuracy),
    #   width (width of bars/number of starting guesses), limit (when to stop the iteration)
    #   and sigma (deviation from x0, for secant) from keyword arguments (or default).
    name    = kwargs.get("name") if kwargs.get("name") != None else func.__name__
    cmap    = kwargs.get("cmap") if kwargs.get("cmap") != None else colormaps["plasma"].resampled(256)
    epsilon = kwargs.get("epsilon") if kwargs.get("epsilon") != None else 1e-10
    width   = kwargs.get("width") if kwargs.get("width") != None else 0.05
    limit   = kwargs.get("limit")
    sigma   = kwargs.get("sigma") if kwargs.get("sigma") != None else 0.2
    
    # Change colour map into gray scale if a specific root is selected.
    specroot= kwargs.get("specroot")
    cmap = colormaps["gray_r"].resampled(512) if specroot != None else cmap
    
    # Define dictonary param(eters) to pass into chosen root finding function as kwargs.
    param   = {"epsilon":epsilon, "limit":limit, "sigma":sigma}
    
    # Find the roots of p using an eigenvalue method and generate the starting guess array across these roots.
    roots = np.roots(p)
    x     = np.arange(round(int(1.1*(min(roots)/width-1))*width,4), round(int(1.1*(max(roots)/width+1))*width,4), width)
    
    # Generate a colour map with a unique colour for each root. If a specific root is selected set this root to the given colour.
    cmap = [cmap(round(255/(roots.size+1))*(i+1)) for i in np.arange(0,roots.size,1)]
    if specroot != None:
        cmap[specroot] = kwargs.get("specrootcolor") if kwargs.get("specrootcolor") != None else "#dc143c"

    # Create a subplot figure, including titles, axis labels and height limits.
    fig, axs = plt.subplots(2)
    
    fig.suptitle(r"{} Convergence for $f(x)={}$".format(name, printing.latex(Poly(p, Symbol("x")).as_expr())))
    axs[0].set_xlabel(r"$x$")
    axs[0].set_ylabel(r"$f(x)$")
    axs[1].set_xlabel(r"Starting Guess, $x_0$")
    axs[1].set_ylabel(r"Number of Iterations, $n$")
    if limit:
        axs[1].set_ylim(top = limit+1)
    
    # Plot the top subplot, including the polynomial roots and a line at y=0.
    axs[0].plot(x, np.polyval(p,x), color="black", alpha=0.6, zorder=0)
    axs[0].axhline(y=0, color="black", alpha=0.4, zorder=0, linestyle="--", linewidth=1)
    axs[0].scatter(roots, np.zeros(roots.size), color=cmap)
    
    # Sequence across the starting guess array, placing each guess into the chosen root finding method.
    for i in range(0, x.size):
        r, n = func(x[i], p, **param)
        # Plot the bottom subplot, with a bar at the starting guess of height n in the correct root colour. 
        axs[1].bar(x[i], n, width=width, color=cmap[np.argmin(abs(roots-r))])
    
    # Save the figure etc. with corrected name if specific root
    name = name+" SpecRoot" if specroot != None else name
    fig.tight_layout()
    fig.savefig(name+".pdf")
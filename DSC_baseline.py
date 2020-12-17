"""
Authors: Nicolai Johansen, Simone Orioli
Version: 0.1
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.integrate import simps
import sys
import os

def optimize( data, t_beg, t_end, g, m_beg, m_end, q_beg, q_end ):

    """
    Function that optimizes the estimation of the DSC baseline

    Input
    -----
    data (numpy.ndarray): DSC data
    t_beg (int): index where the initial linear fit stops
    t_end (int): index where the final linear fit begins
    g (scipy.interpolate._cubic.CubicSpline): cubic spline interpolation of the baseline
    m_beg (float): angular coefficient of the initial linear fit
    m_end (float): angular coefficient of the final linear fit
    q_beg (float): y-intersection of the initial linear fit
    q_end (float): y-intersection of the final linear fit

    Returns
    -------
    g_up (np.ndarray): updated baseline
    """

    x = data[:,0][t_beg:t_end] #interpolation range
    diff = data[:,1][t_beg:t_end] - g(x) #difference in the integral
    int_denom = simps( diff, x ) #denominator of g function
    g_up = [] #the updated g

    g_tbeg = g( data[t_beg,0] )
    g_tend = g( data[t_end,0] )
    X = x[-1] - x[0]

    for i in range(1,len(x)-1):
        xi      = x[:i]
        diffi   = data[:,1][t_beg:t_beg+i] - g(xi)
        int_num = simps( diffi, xi )

        gi_0 = g_tbeg + (xi[-1] - xi[0]) * m_beg
        gi_1 = g_tend - (X - xi[-1] + xi[0]) * m_end

        g_i = gi_0 + ( gi_1 - gi_0 ) * int_num / int_denom
        g_up.append( g_i )

    g_up.append( g(data[t_end,0]) )
    g_up.insert(0, g( data[t_beg,0] ) )

    g_up = np.array(g_up)

    return g_up
#-------------------------------------------------------------------------------

def mkdir( dirname ):

    if not os.path.isdir( dirname ):
        os.mkdir( dirname )
        print("# Directory '" + dirname + "' created" )
    else:
        print("# Directory '" + dirname + "' exists" )
#-------------------------------------------------------------------------------

def baseline( xf, yf, tt_b, tt_e, n_iter ):

    args = sys.argv
    res = "results/"

    print("\n# DSC baseline optimization routine")
    print("# ---------------------------------")
    if( len(args) < 6 ):
        print("# Usage: python DSC_baseline.py xfile.format yfile.format t_beg t_end iterations\n")
        sys.exit(-1)

    # #Load data
    x      = np.loadtxt( sys.argv[1] )
    y      = np.loadtxt( sys.argv[2] )
    tt_b   = float( sys.argv[3] )
    tt_e   = float( sys.argv[4] )
    n_iter = int( sys.argv[5] )

    data = np.zeros((len(x),2))
    data[:,0] = x[:]
    data[:,1] = y

    mkdir( res )

    #Select range for linear fit
    t_beg = np.abs(data[:,0] - tt_b).argmin()
    t_end = np.abs(data[:,0] - tt_e).argmin()
    linear1 = data[:t_beg]
    linear2 = data[t_end:]

    #Linear fit
    m_beg, q_beg = np.polyfit(linear1[:,0], linear1[:,1], 1)
    m_end, q_end = np.polyfit(linear2[:,0], linear2[:,1], 1)

    print("# Linear fits: m_beg = {:.2f}, q_beg = {:.2f}, m_end = {:.2f}, q_end = {:.2f}".format(m_beg, q_beg, m_end, q_end) )

    #Define range for spline interpolation
    arr1        = [data[t_beg,0], data[t_end,0]]
    arr2        = [data[t_beg,1], data[t_end,1]]
    data_spline = data[:,0][t_beg:t_end]
    g0          = interpolate.CubicSpline( arr1, arr2, bc_type = ((1,m_beg), (1,m_end)) )

    #Plot the initial interpolated range
    plt.figure(0)
    plt.plot( data[:,0], data[:,1], label = 'Experiment' )
    plt.plot( data_spline, g0(data_spline), label = 'Cubic Interpolation' )

    #plt.xlabel('Something')
    #plt.ylabel('Something')
    plt.legend()
    plt.tight_layout()
    plt.savefig( res + 'cubic_interpolation.pdf', format = 'pdf')
    print("# Saved '" + res + "cubic_interpolation.pdf'" )

    # Optimize
    x = data[:,0][t_beg:t_end]
    g_ups = [g0]

    for i in range(n_iter):
        print("# Iteration {}".format(i))
        g_tmp = optimize( data, t_beg, t_end, g_ups[i], m_beg, m_end, q_beg, q_end )
        g     = interpolate.CubicSpline( x, g_tmp )
        g_ups.append( g )

    plt.figure(1)
    plt.plot( data[:,0], data[:,1], label = 'Experimental' )
    for i in range(n_iter):
        plt.plot( x, g_ups[i](x), label = f'Iteration {i}' )

    #plt.xlabel('Something')
    #plt.ylabel('Something')
    plt.legend()
    plt.tight_layout()
    plt.savefig( res + 'iterations.pdf', format = 'pdf')
    print("# Saved '" + res + "iterations.pdf'" )

    #Subtract baseline
    plt.figure(2)
    plt.plot( data[:,0], data[:,1], 'k--', label = 'Not subtracted' )
    plt.plot( x, data[:,1][t_beg:t_end] - g_ups[-1](x), label = 'Subtracted' )
    plt.xlim( (data[0,0], data[-1,0]) )
    #plt.xlabel('Something')
    #plt.ylabel('Something')
    plt.legend()
    plt.tight_layout()
    plt.savefig( res + 'not_and_subtracted.pdf', format = 'pdf')
    print("# Saved '" + res + "not_and_subtracted.pdf'" )

    #subtracted signal over the whole range
    subtracted = []
    k = 0
    for i in range(len(data)):

        if( i >= t_beg and i < t_end ):
            subtracted.append( data[:,1][t_beg:t_end][k] - g_ups[-1](x)[k] )
            k += 1
        else:
            subtracted.append(0)

    plt.figure(3)
    plt.plot( data[:,0], subtracted )
    #plt.xlabel('Something')
    #plt.ylabel('Something')
    plt.tight_layout()
    plt.savefig( res + 'subtracted_whole_range.pdf', format = 'pdf')
    print("# Saved '" + res + "subtracted_whole_range.pdf'" )

    np.savetxt( res + "subtracted_whole_range.dat", subtracted )
    print("# Saved '" + res + "subtracted_whole_range.dat'" )

#-------------------------------------------------------------------------------

if( __name__ == '__main__' ):
    baseline()

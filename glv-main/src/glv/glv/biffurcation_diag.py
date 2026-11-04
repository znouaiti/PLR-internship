import diffrax as dfx
import jax.numpy as jnp
import jax
import functools
import glv
import matplotlib.pyplot as plt

def extract_trajectory(ys):
    """
    Extracts the mean abundance value at each of the `N` recorded time points from a ys solution of abundances of the glv system
    """
    N=ys.shape[0]
    averages_over_time=jnp.zeros(N)
    for k in range(N):
        average = jnp.mean(ys[k], axis=0)
        averages_over_time = averages_over_time.at[k].set(average)
    return averages_over_time

def extrema_ammount(y):
    """
    Extracts the ammount of extrema and their value from a time series of abundances of the glv system
    takes a vector of mean abundances, returns the ammount of extrema and a list of their values
    """
    N=y.shape[0]
    list_extrema=[]
    for k in range (1,N-1):
        if (y[k]-y[k-1] )*(y[k+1]-y[k]) < 0:
            list_extrema.append(y[k])
    return list_extrema

def biffurcation_diag_m(r, A, m_min, m_max, m_step, y0, N, dT, rtol=1e-5, atol=1e-5, threshold=1e-10, max_steps=10000):
    """
    plots the ammount of extrema and their values in the mean abundance value over time of a group solutions of the gLV system for a range of immigration m. 
    """
    m_values = jnp.arange(m_min, m_max, m_step)
    extrema_values_over_time=[]
    for m in m_values:

        sol = glv.solve_log_RK45(N=N, dT=dT, r=r, A=A, m=m, y0=y0,
                             rtol=rtol, atol=atol,
                             threshold=threshold,
                             max_steps=max_steps)
    
        ys = sol.ys
        mean_abundances = extract_trajectory(ys)
        extrema_values = extrema_ammount(mean_abundances)
        extrema_values_over_time.append(extrema_values)
    plt.plot(m_values, extrema_values_over_time)
    plt.xlabel('Immigration rate (m)')
    plt.ylabel('Extrema values of mean abundance over time')
    plt.show()
    
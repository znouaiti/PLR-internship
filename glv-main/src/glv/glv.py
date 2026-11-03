import diffrax as dfx
import jax.numpy as jnp
import jax
import functools
import matplotlib.pyplot as plt
#----------------------------------------------------------------------------------------
# Model definition
#----------------------------------------------------------------------------------------


def vf(t,y,args):
    """
    rhs of GLV expressed in log-abundances
    """
    r, A, m = args
    return y*(r - A @ y) + m


def vf_log(t,z,args):
    """
    rhs of GLV expressed in log-abundances
    """
    r, A, m = args
    x = jnp.exp(z)
    return r - A @ x + m / x

def jac(t, x, args):
    """
    Jacobian of GLV in linear abundances
    """
    r, A, m = args    
    J = - x[:, None] * A
    J = J.at[jnp.arange(x.size), jnp.arange(x.size)].add(r - A@x)
    return J


#----------------------------------------------------------------------------------------
# Integration schemes
#----------------------------------------------------------------------------------------

_SOLVE_DOC_TEMPLATE = """
Generates a trajectory of the GLV with parameters (r, A, m), sampled `N` times at
intervals of `dT`.

{description}

## Params
- `N` : number of recorded steps (integrator)
- `dT` : continuous time interval between recorded steps
- `r,A,m` : growth rates, interaction matrix, immigration rate
- `y0`: initial condition
- `rtol, atol` relative and absolute tolerances for the adaptive step size
- `threshold` : abundance that fall on or below this value will be frozen
- `max_steps` : maximum number of internal steps for the solver
"""

def make_adaptive_solve(vf, solver, trans=None, description=""):
    """
    Maker for different integration schemes
    
    ## Params
    - `vf` : vecor field, rhs of ODE
    - `solver` : a diffrax solver
    - `trans` : tuple of transform and anti-transform applied to state before and after integration
    """
    
    
    def _solve(N, dT, r, A, m, y0,rtol=1e-9,atol=1e-11,threshold=None,max_steps=None):
            
            if max_steps is None:
                max_steps = int(100*N*dT)

            if threshold is not None:
                threshold_ = trans[0](threshold) if trans is not None else threshold

                def vf_sticky(t, y, args):
                    raw = vf(t, y, args)
                    return jnp.where(y <= threshold_, 0., raw)

                term = dfx.ODETerm(vf_sticky)               
            else:
                term = dfx.ODETerm(vf)
    
            ts = jnp.arange(N+1)*dT
            y0_ = trans[0](y0) if trans is not None else y0
                
            saveat = dfx.SaveAt(ts=ts)
            pid = dfx.PIDController(rtol, atol)
    
            sol = dfx.diffeqsolve(term,solver,0,ts[-1],0.01,y0_,(r,A,m),saveat=saveat,stepsize_controller=pid,max_steps=max_steps)
    
            ts = sol.ts
    
            ys = trans[1](sol.ys) if trans is not None else sol.ys
    
            return ts, ys

    _solve.__doc__ = _SOLVE_DOC_TEMPLATE.format(description=description)

    return _solve


solve_log_RK45 = make_adaptive_solve(
    vf_log, dfx.Tsit5(), (jnp.log, jnp.exp),
    description=(
        "The integration uses an adaptive Runge-Kutta 4/5 in the space of "
        "log-abundances. Note: in the absence of migration, the `threshold`"
        "parameter should be set to absorb abundances at a small but finite value"
    ),
)
    
solve = solve_log_RK45
jit_solve = functools.partial(jax.jit, static_argnums=(0,9))(solve)     # for convenience

solve_RK45 = make_adaptive_solve(
    vf, dfx.Tsit5(), None, description=(
        "The integration uses an adaptive Runke-Kutta 4/5 in linear abundances."
        ),
    )

solve_implicit34 = make_adaptive_solve(
    vf, dfx.Kvaerno4(), None, description=(
        "The integration uses an adaptive and implicit Kvaerno 4/3 method in linear abundances. Abundances"
        "below `cutoff` (if given) are set to zero."
        ),
    )

#----------------------------------------------------------------------------------------
# Batching convenience  
#----------------------------------------------------------------------------------------

def jit_batch(solve_x, N, dT, rtol=1e-9, atol=1e-11, threshold=None, max_steps=None):
    """
    Wrap a glv solve function into a jitted, batched function of (r, A, m, y0).

    - r:  (B, N) or (B)     batched growth rates
    - A:  (B, N, N)         batched interaction matrices
    - m:  (B, N) or (B)     batched mortality/dilution rates
    - y0: (B, N)            batched initial conditions
    """
    def wrapped(r, A, m, y0):
        return solve_x(N, dT, r, A, m, y0, rtol=rtol, atol=atol, threshold=threshold, max_steps=max_steps)

    batched = jax.jit(jax.vmap(wrapped, in_axes=(0, 0, 0, 0)))
    return jax.jit(batched)


def jit_batch_solve(N, dT, rs, As, ms, y0s, rtol=1e-9, atol=1e-11, threshold=None, max_steps=None):
    return jit_batch(solve, N, dT, rtol, atol, threshold, max_steps)(rs, As, ms, y0s) 


def extrema_ammount(y):
    ''' 
    renvoie les extrema de y
    '''
    d=jnp.diff(y)
    mask=d[1:]*d[:-1]<0
    return y[1:-1][mask]

def kl_divergence(y1,y2):
    """
    Computes the Kullback-Leibler divergence between two distributions y1 and y2, et les normalise
    """
    y1=y1/y1.sum()
    y2=y2/y2.sum()
    return jnp.sum(y1 * jnp.log(y1 / y2))

def kl_over_t(list_kl):
    plt.plot(list_kl)
    plt.show()


def sweep_m_values(Nzero,dT,r,A,y0,m_min,m_max,m_step,seuil,*,reverse,return_yf=False):#plot_abundance, return_dist):
    """ 
    Renvoie une liste de doublets (m,val) où m est une valeur du taux d'immigration et val, un extremum de l'abondance totale
    Si return_yf, renvoie aussi le vecteur d'abondances final du balayage: (points, yf)
    """
    m_values=jnp.logspace(m_min,m_max,int((m_max-m_min)/m_step))
    if reverse:
        m_values = m_values[::-1] # sweeping from high to low immigration rate to spot any hysteresis
    N=Nzero
    ldist=[]
    points=[]
    for m in m_values:
        ts,ys = solve_log_RK45(N, dT=dT, r=r, A=A, m=m, y0=y0)#on résout le système pour une certaine valeur de m 
        dist=kl_divergence(ys[-1],y0) #on utilise la divergence de KL pour mesurer la distance entre 2 distributions successives                       
        ldist.append(dist)

        y0=ys[-1] # Use the last state as the initial condition for the next run, principle of an adiabatic sweep
        if dist<seuil:#si les deux dernières distributions consécutives ne sont pas trop différentes
            N=Nzero
            mean_abundances =  ys.mean(axis=1)#on extrait l'abondance moyenne à chaque pas de temps
            extrema_values = extrema_ammount(mean_abundances)#on capture les extrema de l'abondance moyenne
            for val in extrema_values:
                print(f"m={m:.4e}, extrema value={val:.4e}")
                    
                points.append((float(m),float(val)))
        else:
            N=10*Nzero #on augmente le nombre de pas de temps du prochain pour essayer d'atteindre le point fixe       
            print(f"m={m:.4e}, not close enough to fixed point, skipping...")
        print(f"distance to previous point: {dist:.4e}")
    if return_yf:
        return points,y0
    return points


def y0_exploration_sweep(N,dT,r,A,y0,m_min,m_max,m_step,seuil,nb_esp_test,reverse):
    """
    Explore the effect of different initial conditions on the dynamics of the GLV system.
    """
    initial_conditions_vectors=jnp.ones((nb_esp_test, len(y0)))/(10*len(y0))#on crée une matrice où les vecteurs lignes sont les conditions initiales
    for esp in range(len(initial_conditions_vectors)):
        initial_conditions_vectors = initial_conditions_vectors.at[esp, esp].add(0.9)#dans le n eme vecteur de la matice, la neme espèce a une abondance >0.9
    results=[]
    for esp in initial_conditions_vectors:
        pts=sweep_m_values(N, dT, r, A, esp, m_min, m_max, m_step, seuil,reverse=reverse)#on appelle la fonction sweep pour chacune des conditions initiales 
        lm,lval=zip(*pts)
        results.append((lm,lval))
    return results


def BC_dissimilarity(yi,yj):
    Si=sum(yi)
    Sj=sum(yj)
    Smin=jnp.minimum(yi,yj)
    return 1-(2*Smin.sum()/(Si+Sj))

def equilibrium_distinction(N,dT,r,A,y0,m_min,m_max,m_step,seuil,nb_esp_test,reverse ):
    initial_conditions_vectors=jnp.ones((nb_esp_test, len(y0)))/(10*len(y0))#on crée une matrice où les vecteurs lignes sont les conditions initiales
    for esp in range(len(initial_conditions_vectors)):
        initial_conditions_vectors = initial_conditions_vectors.at[esp, esp].add(0.9)#dans le n eme vecteur de la matice, la neme espèce a une abondance >0.9
    lyf=[]
    for esp in initial_conditions_vectors:
        _,yf=sweep_m_values ( N , dT, r, A,esp, m_min, m_max, m_step, seuil,reverse=reverse,return_yf=True)#on appelle la fonction sweep pour chacune des conditions initiales
        lyf.append(yf)
    dissimilarity_matrix=jnp.ones((nb_esp_test,nb_esp_test))
    for l in range(nb_esp_test):
        for c in range(nb_esp_test):
            dissimilarity_matrix=dissimilarity_matrix.at[l,c].set(BC_dissimilarity(lyf[l],lyf[c]))
    return dissimilarity_matrix

def point_inflexion(x,y):
    '''
    renvoie les abscisses des points d'inflexions de y(x)
    '''
    d=jnp.diff(jnp.asarray(y),2)
    mask=d[1:]*d[:-1]<0
    return jnp.asarray(x)[2:-1][mask]


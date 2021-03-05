#!/usr/bin/env python

import numpy as np
import theano.tensor as tt
import pymc3 as pm
import exoplanet as xo

def _set_default(value, default=0.1, sd=1):

    if value is None:
        return np.array([default, sd])

    elif isinstance(value, float):
        return np.array([value, sd])

    elif isinstance(value, list):
        return value


def autofit(x, y, period=None, t0=None,
            Q=None,
            dur=None,
            ror=None,
            ):

    # can provide a number, or list with mean, sd


#            m_star=None,
#            r_star=None):

#    if m_star is None and r_star is None:
#        fit_star = False

#        fit_pars = ["dur", "

    if isinstance(x, np.ndarray):
        x = [x]
        y = [y]

    names = [f"lc_{i}" for i in range(len(x))]

    if period is None and t0 is None:
        raise NotImplementedError("need to implement BLS/LS search")


    dur = _set_default(dur)
    ror = _set_default(ror)



    with pm.Model() as model:

        _period = pm.Lognormal("period", mu=period[0], sd=period[1])
        _t0     = pm.Lognormal("t0",     mu=t0[0],     sd=t0[1])

        _dur    = pm.Lognormal("dur", mu=dur[0], sd=dur[1])
        _ror    = pm.Lognormal("ror", mu=ror[0], sd=ror[1])

        _u = xo.distributions.QuadLimbDark("u")
        _b = xo.distributions.ImpactParameter("b", ror=_ror)

        orbit = xo.orbits.KeplerianOrbit(
           duration=_dur,
           period=_period,
           t0=_t0,
           b=_b
        )

        log_Sw4_all = []
        log_w0_all = []
        log_s2_all = []
        mean_all = []

        for i, name in enumerate(names):

            with pm.Model(name=name, model=model):

                _mean    = pm.Lognormal("mean", mu=np.median(y[i]), sd=5)
                _log_Sw4 = pm.Normal("log_Sw4", mu=np.log(np.var(y[i])), sd=10)
                _log_w0  = pm.Normal("log_w0", mu=np.log(2 * np.pi / 5), sd=10)
                _log_s2    = pm.Normal("log_s2", mu=np.log(np.var(y[i])), sd=10)

                mean_all.append(_mean)
                log_Sw4_all.append(_log_Sw4)
                log_w0_all.append(_log_w0)
                log_s2_all.append(logs2)

        light_curves = (
                xo.LimbDarkLightCurve(u).get_light_curve(
                    orbit=orbit, r=ror, t=x[i]
                )
            )

        light_curve = (pm.math.sum(light_curves, axis=-1) + 1) * _mean

        kernel = xo.gp.terms.SHOTerm(
                                    log_Sw4 = _log_Sw4,
                                    log_w0 = _log_w0,
                                    Q = 1 / np.sqrt(2)
                                    )
        gp = xo.gp.GP(kernel, 
                x[i], 
                light_curve**2 * tt.exp(log_s2) + yerr[i]**2, 
                mean=light_curve
                )
            
        gp.marginal(f"{name}_lc_obs", observed=y[i])
                
        if start is None:
            start = model.test_point

        map_soln = xo.optimize(start=start, vars=mean_all)
        map_soln = xo.optimize(start=map_soln, vars=[period, t0])
        map_soln = xo.optimize(start=map_soln, vars=[ror, dur])
        map_soln = xo.optimize(start=map_soln, vars=[b, u])
        map_soln = xo.optimize(start=map_soln, vars=[period, t0])
        map_soln = xo.optimize(start=map_soln, vars=[ror, dur, b, u])

        for p in log_Sw4_all:
            map_soln = xo.optimize(start=map_soln, vars=[p])
        for p in log_w0_all:
            map_soln = xo.optimize(start=map_soln, vars=[p])
        for p in log_s2_all:
            map_soln = xo.optimize(start=map_soln, vars=[p])

        map_soln = xo.optimize(start=map_soln)







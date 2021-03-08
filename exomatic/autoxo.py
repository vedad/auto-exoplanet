#!/usr/bin/env python

import numpy as np
import theano.tensor as tt
import pymc3 as pm
import exoplanet as xo

import pymc3_ext as pmx
from celerite2.theano import terms, GaussianProcess
from functools import partial

__all__ = ["automodel"]

def _set_default(value, default=0.1, sd=1):

    if value is None:
        return np.array([default, sd])

    elif isinstance(value, float):
        return np.array([value, sd])

    elif isinstance(value, list):
        return value


def automodel(
        dataset_lc       = None,
        dataset_rv       = None,
        period           = None,
        t0               = None,
        tp               = None,
        eccentric        = False,
        optimize         = True,
            ):


    if isinstance(period, float):
        period = [period]
        t0     = [t0]

        # period always a list
#    period = [(mu, sd), (mu, sd)]

    n_pl = len(period)

    dur    = _set_default(None, sd=5)
    ror    = _set_default(None, sd=5)
#    dur    = [_set_default(None, sd=5) for _ in range(n_pl)]
#    ror    = [_set_default(None, sd=5) for _ in range(n_pl)]
#    period = [np.array(_set_default(value)) for value in period])
    period = np.array([period[0]])
    t0     = _set_default(t0[0])

    res = dict()


    if dataset_lc is not None:

        res['lc'] = dict()

        fit_lc = True
#        lcres         = dict()
        res['lc']['gp']   = dict()
        res['lc']['lc']   = dict()
        res['lc']['mean'] = dict()
    else:
        fit_lc = False

    if dataset_rv is not None:

        res['rv'] = dict()
        fit_rv = True
#        rvres          = dict()
        res['rv']['rv']    = dict()
#        rvres['trend'] = dict()
        res['rv']['mean']  = dict()

    else:
        fit_rv = False


    with pm.Model() as model:

#        _period = pm.Lognormal("period", mu=np.log(period[0]), 
#                sd=period[1], shape=n_pl)
        _period = pm.Lognormal("period", mu=np.log(period), 
                sd=np.array([1]), shape=n_pl)

        _t0     = pm.Lognormal("t0", mu=np.log(t0[0]), sd=t0[1])

        if fit_lc:
            _dur = pm.Lognormal("dur", mu=np.log(0.1), sd=1)
            _ror = pm.Lognormal("ror", mu=np.log(0.1), sd=1)
            _b   = xo.distributions.ImpactParameter("b", ror=_ror)

        if fit_rv:
            K = pm.Lognormal("K", mu=np.log(0.05), sigma=5)
#            trend = pm.Normal("trend", mu=0, sd=10.0 ** -np.arange(1,3)[::-1],
#                    shape=2)

        if eccentric:
            _ecs = pmx.UnitDisk("ecs", testval=np.array([0.01, -0.01]))
            _ecc = pm.Deterministic("ecc", tt.sum(_ecs ** 2))
            _omega = pm.Deterministic("omega", tt.arctan2(_ecs[1], _ecs[0]))
        else:
            _ecc = None
            _omega = None

        orbit = xo.orbits.KeplerianOrbit(
           duration = _dur,
           period   = _period,
           ecc      = _ecc,
           omega    = _omega,
           t0       = _t0,
           b        = _b
        )


        u = dict()
        for name, _ in dataset_lc.items():
            u[name] = xo.distributions.QuadLimbDark(f"{name}_u")
            

        parameters = dict()

        if fit_rv:
            for i, (name, (x, y, yerr)) in enumerate(dataset_rv.items()):

                subname = f"{name}"

                with pm.Model(name=subname, model=model):
                
                    _mean_rv   = pm.Normal("mean", mu=np.median(y),
                            sd=0.5)
                    _sigma_rv = pm.HalfNormal("sigma", sd=0.1)

                    parameters[f'{subname}_noise'] = [_mean_rv, _sigma_rv]

                def radial_velocity(mean, K, t):
                    vrad = orbit.get_radial_velocity(t, K * 1e3)
                    if n_pl > 1:
                        return tt.sum(vrad, axis=-1) / 1e3 + mean
                    return vrad / 1e3 + mean

                radial_velocity = partial(radial_velocity, _mean_rv, K)
                res['rv']['rv'][subname] = radial_velocity

                rv_model = radial_velocity(x)
                pm.Normal(f"{subname}_obs",
                        mu       = radial_velocity(x),
                        sd       = tt.sqrt(yerr**2 + _sigma_rv**2),
                        observed = y
                        )
    #                x_rv_ref = 0.5 * (x.min() + x.max())
    #                def radial_velocity_trend(trend, t):
    #                    A = np.vander(t - x_rv_ref, 3)[:,:2]
    #                    bkg = tt.dot(A, trend)


        if fit_lc:
            for name, data in dataset_lc.items():

                for i, (x, y, yerr, texp) in enumerate(data):

                    subname = f"{name}_{i}"

                    with pm.Model(name=subname, model=model):

                        _mean     = pm.Lognormal("mean", mu=np.log(np.median(y)), sd=5)
                        _sigma_gp = pm.Lognormal("sigma_gp", mu=np.log(np.std(y)), sd=10)
                        _rho_gp   = pm.Lognormal("rho_gp", mu=0, sd=10)
                        _sigma    = pm.Lognormal("sigma", mu=np.log(np.var(y)), sd=10)

                        parameters[f'{subname}_noise'] = [_mean, _sigma_gp, 
                                                         _rho_gp, _sigma]

                    star = xo.LimbDarkLightCurve(u[name])

                    def light_curve(mean, star, ror, texp, t):
                        return (tt.sum(
                                    star.get_light_curve(
                                        orbit = orbit,
                                        r     = ror,
                                        t     = t,
                                        texp  = texp
                                        ), axis=-1
                                    ) + 1) * mean


                    light_curve = partial(light_curve, _mean, star, _ror, texp)
                    res['lc']['lc'][subname] = light_curve


                    kernel = terms.SHOTerm(
                                            sigma = _sigma_gp,
                                            rho   = _rho_gp,
                                            Q     = 1 / np.sqrt(2)
                                            )

                    gp = GaussianProcess(
                            kernel, 
                            t     = x,
                            diag  = light_curve(x)**2 * _sigma**2 + yerr**2,
                            mean  = light_curve,
                            quiet = True
                            )
                        
                    gp.marginal(f"{subname}_obs", observed=y)

                    res['lc']['gp'][subname] = gp


        print(model.check_test_point())

        map_soln = model.test_point

        if fit_lc:
            map_soln = xo.optimize(start=map_soln, vars=[_period, _t0])
            map_soln = xo.optimize(start=map_soln, vars=[_ror, _dur])

        if eccentric:
            if fit_rv:
                map_soln = xo.optimize(start=map_soln, vars=[ecs, K])
            else:
                map_soln = xo.optimize(start=map_soln, vars=[ecs])

        map_soln = xo.optimize(start=map_soln, vars=[_b] + list(u.values()))

        map_soln = xo.optimize(start=map_soln, vars=[_period, _t0])
        map_soln = xo.optimize(start=map_soln, 
                vars=[_ror, _dur, _b] + list(u.values()))

        for _,pars in parameters.items():
            map_soln = xo.optimize(start=map_soln, vars=pars)

        map_soln = xo.optimize(start=map_soln, vars=[_period, _t0])
        map_soln = xo.optimize(start=map_soln, vars=[_ror, _dur, _b])
        
        if eccentric:
            if fit_rv:
                map_soln = xo.optimize(start=map_soln, vars=[ecs, K])
            else:
                map_soln = xo.optimize(start=map_soln, vars=[ecs])

        map_soln = xo.optimize(start=map_soln)

    res['model']    = model
    res['map_soln'] = map_soln

#    if fit_lc and not fit_rv:
#        return lcres
#    if fit_rv and not fit_lc:
#        return rvres

    return res



#with model0:
#    trace = pm.sample(
#       tune=10,
#       draws=10,
#       start=map_soln0,
#       chains=2,
#        cores=2,
##        initial_accept=0.8,
##        target_accept=0.95,
#       step=xo.get_dense_nuts_step(target_accept=0.95),
#   )



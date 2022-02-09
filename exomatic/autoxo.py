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

def concatenate_array(x, index):
    return np.concatenate(
            [
                item[i] for item in lst
                for _,lst in x.items()
                ]
            )

def _set_multiplanet(value, n=1):
    if n == 1:
        return value
    return np.array([value] * n)

def plot_fit(res):
    return



def automodel(
        """
        docstr
        """
        dataset_lc       = None,
        dataset_rv       = None,
        period           = None,
        t0               = None,
        tp               = None,
#        transiting       = None,
        eccentric        = False,
        trend_order      = 0,
        optimize         = True,
            ):

    if isinstance(period, float):
        period = [period]
        t0     = [t0]

        # period always a list
#    period = [(mu, sd), (mu, sd)]

    n_pl = len(period)

#    if transiting is None:
#        transiting = [True] * n_pl

    dur    = _set_default(None, sd=5)
    ror    = _set_default(None, sd=5)
#    dur    = [_set_default(None, sd=5) for _ in range(n_pl)]
#    ror    = [_set_default(None, sd=5) for _ in range(n_pl)]
#    period = [np.array(_set_default(value)) for value in period])
    period = np.array([period[0]])
    t0     = _set_default(t0[0])
    
#    print('period', period)
    res = dict()


    if dataset_lc is not None:
        fit_lc = True

        res['lc'] = dict()

#        lcres         = dict()
        # computed GP model on observations per instrument
#        res['lc']['gp']   = dict()
#        res['lc']['lc']   = dict()

        # full model on observations per instrument
        res['lc'] = dict() # res['lc']['tess_0'] etc

        # gp predictive model per ``instrument``
#        res['lc']['gp_pred'] = dict() # res['lc']['gp_pred']['tess_0']
        res['lc']['gp'] = dict() # res['lc']['gp_pred']['tess_0']

        # predictive model per ``instrument`` (really band, but because of texp
        # it needs to be per instrument
        res['lc']['lc_pred'] = dict() # res['lc']['lc_pred']['tess']
        res['lc']['lc_phase_pred'] = dict() # res['lc']['lc_phase_pred']['tess']

    else:
        fit_lc = False

    if dataset_rv is not None:

        # pred should have bkg and rv
        # obs should have total
        fit_rv = True

        # full model on observations per instrument
        res['rv'] = dict() # res['rv']['harps']

        # predictive models for system, single function, not dict
#        res['rv']['rv_pred']
#        res['rv']['rv_phase_pred']
#        res['rv']['bkg_pred']


    else:
        fit_rv = False


    with pm.Model() as model:

        _period = pm.Lognormal("period", mu=np.log(period), 
                sd=np.array([1]), shape=n_pl)

        _t0     = pm.Lognormal("t0", mu=np.log(t0[0]), sd=t0[1])

        if fit_lc:
            _dur = pm.Lognormal("dur", mu=np.log(0.1), sd=1)
            _ror = pm.Lognormal("ror", mu=np.log(0.1), sd=1)
            _b   = xo.distributions.ImpactParameter("b", ror=_ror)

        if fit_rv:
            K = pm.Lognormal("K", mu=np.log(0.05), sigma=5)

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

        orbit_phase = xo.orbits.KeplerianOrbit(
           duration = _dur,
           period   = _period,
           ecc      = _ecc,
           omega    = _omega,
           t0       = 0,
           b        = _b
        )


        u = dict()
        for name, _ in dataset_lc.items():
            u[name] = xo.distributions.QuadLimbDark(f"{name}_u")
            

        parameters = dict()


        if fit_rv:

            if trend_order > 0:

                t_tot = np.concatenate(
                        [
                            item[0] for item in list(dataset_lc.values())[0]
                            ]
                        )
                t_ref = 0.5 * (t_tot.min() + t_tot.max())

                if trend_order == 1:
                    trend = pm.Normal('trend', mu=0, sd=0.1)

                elif trend_order == 2:
                    trend = pm.Normal('trend', mu=np.array([0, 0]), 
                            sd=np.array([0.01, 0.1]), shape=2)

                else:
                    raise notimplementederror(
                    """
                    polynomial orders higher than 2 is not implemented
                    """
                    )


            def get_rv_phase_pred(t):
                return orbit_phase.get_radial_velocity(t, K * 1e3) 

            def get_rv_pred(t):
                return orbit.get_radial_velocity(t, K * 1e3)

            def get_bkg_model_pred(t):
                if trend_order > 0:

                    if trend_order == 1:
                        A = np.vander((t - t_ref), 2)[:,0]

                    elif trend_order == 2:
                        A = np.vander((t - t_ref), 3)[:,:2]

                    return tt.dot(A, trend)

                else:
                    return tt.zeros_like(t)


            # save predictive functions
            res['rv']['rv_pred']       = get_rv_pred
            res['rv']['rv_phase_pred'] = get_rv_phase_pred
            res['rv']['bkg_pred']      = get_bkg_model_pred


            for i, (name, (x, y, yerr)) in enumerate(dataset_rv.items()):

                subname = f"{name}"

                with pm.Model(model=model) as submodel:
                
                    _mean_rv   = pm.Normal(f"mean_{subname}", mu=np.median(y),
                            sd=0.5)
                    _sigma_rv = pm.HalfNormal(f"sigma_{subname}", sd=0.1)

                    parameters[f'noise_{subname}'] = [_mean_rv, _sigma_rv]

                    res[f'model_{subname}'] = submodel


                rvmod = (
                        orbit.get_radial_velocity(x, K * 1e3) / 1e3 + 
                        _mean_rv +
                        get_bkg_model_pred(x)
                        )

                res['rv'][subname] = rvmod

                pm.Normal(f"obs_{subname}",
                        mu       = rvmod,
                        sd       = tt.sqrt(yerr**2 + _sigma_rv**2),
                        observed = y
                        )


        if fit_lc:
            for name, data in dataset_lc.items():

                star = xo.LimbDarkLightCurve(u[name])

                def get_lc_phase_pred(t, texp=None):
                    return (tt.sum(
                                    star.get_light_curve(
                                        orbit = orbit_phase,
                                        r     = _ror,
                                        t     = t,
                                        texp  = texp
                                        ), axis=-1
                                    ) + 1)

                def get_lc_pred(t, texp=None):
                    return (tt.sum(
                                    star.get_light_curve(
                                        orbit = orbit,
                                        r     = _ror,
                                        t     = t,
                                        texp  = texp
                                        ), axis=-1
                                    ) + 1)

                res['lc']['lc_pred'][name] = get_lc_pred
                res['lc']['lc_phase_pred'][name] = get_lc_phase_pred

                for i, (x, y, yerr, texp) in enumerate(data):

                    subname = f"{name}_{i}"

                    with pm.Model(model=model) as submodel:

                        _mean     = pm.Lognormal(f"mean_{subname}", mu=np.log(np.median(y)), sd=5)
                        _sigma_gp = pm.Lognormal(f"sigma_gp_{subname}", mu=np.log(np.std(y)), sd=10)
                        _rho_gp   = pm.Lognormal(f"rho_gp_{subname}", mu=0, sd=10)
                        _sigma    = pm.Lognormal(f"sigma_{subname}", mu=np.log(np.var(y)), sd=10)

                        parameters[f'noise_{subname}'] = [_mean, _sigma_gp, 
                                                         _rho_gp, _sigma]

                        res[f'model_{subname}'] = submodel


                    lcmod = get_lc_pred(x, texp=texp) * _mean

                    res['lc'][subname] = lcmod


                    kernel = terms.SHOTerm(
                                            sigma = _sigma_gp,
                                            rho   = _rho_gp,
                                            Q     = 1 / np.sqrt(2)
                                            )

                    gp = GaussianProcess(
                            kernel, 
                            t     = x,
#                            diag  = light_curve(x)**2 * _sigma**2 + yerr**2,
                            diag  = lcmod**2 * _sigma**2 + yerr**2,
#                            mean  = lcmod,
                            quiet = True
                            )
                        
                    gp.marginal(f"obs_{subname}", observed=y-lcmod)

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

        if trend_order > 0:
            map_soln = xo.optimize(start=map_soln, vars=[trend])

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



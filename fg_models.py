from __future__ import division
import sys

from astropy import units
from astropy.cosmology import Planck15, z_at_value
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf, erfc, erfinv
from scipy.stats import beta, betaprime, gaussian_kde
from scipy.integrate import fixed_quad
import scipy.interpolate

sys.path.append("./gw_event_gen")
sys.path.append("../gw_event_gen")

from netsim import Network
import eventgen

#np.seterr(all='raise')

# ## Universality when slicing in amplitude-dependent parameters
# 
# For a distribution which is uniform in the Euclidean volume, but sliced between $r_1$ and $r_2$, with a reference SNR $\bar{\rho}_0$ at $r_0$ (could be taken to be the horizon, but doesn't really matter), we have the following distribution of detected SNR $\rho = A\bar{\rho}$ ($A$ being an amplitude parameter which scales the SNR down from optimal) in that volume slice:
# 
# $$
# p(\rho) = \frac{3(\bar{\rho}_0r_0)^3}{r_2^3 - r_1^3}\frac{1}{\rho^4}\int_a^b p(A) A^3dA
# $$
# 
# Where the integration limits are $\rho$ dependent:
# 
# $$
# a = \rho / \bar{\rho}_1 \\
# b = \rho / \bar{\rho}_2
# $$

def _generate_sample(n=100000, sys_random_seed=None):
    """
    Generate a large sample of the GW orientation amplitude distribution to reliably and consistently construct a KDE for integral/moments later on. Probably only for internal use.
    """
    # Ensure consistent results
    np.random.seed(0)

    # first we need the distribution of A
    amp = eventgen.EventGenerator()

    amp.append_generator(eventgen._extr, eventgen.random_orientation)
    amp.append_generator(("event_time",), eventgen.uniform_event_time)

    def _amp(e):
        return (eventgen.theta("H1", e.event_time, e.right_ascension, \
                            e.declination, e.polarization, e.inclination) / 4,)

    amp.append_post(("amp",), _amp)

    for e in amp.generate_events(n_event=100000):
        pass

    # Random Seed
    #if sys_random_seed is not None:
    #    np.random.seed(int(sys_random_seed))
    #else:
    #    np.random.seed()


    return amp

def fit_beta_to_theta(sys_random_seed=None):
    """
    Fit the output of the amplitude distribution to a Beta distribution. Note that the distribution obtained in this way is slightly biased, since the distribution is not exactly fit by the Beta distribution. Recommend usage of `fit_kde_to_theta` instead.
    """
    amp = _generate_sample(sys_random_seed)
    return beta.fit(amp["amp"], floc=0)

def fit_kde_to_theta(sys_random_seed=None):
    """
    Dump the output of the amplitude distribution into a KDE. Se also `fit_beta_to_theta` for a less accurate, but faster and more compact answer.
    """
    amp = _generate_sample(sys_random_seed)
    return gaussian_kde(amp["amp"]).pdf

def intr_beta_moment(pdf_args, pdf=None, m=3):
    """
    Get an interpolated vector version of \\langle A^m \\rangle between 0 and 1
    (cumulative integral). m is the desired moment of the distribution (default
    is 3).
    """
    # x values to evaluate
    _x = np.linspace(0, 1 + 2e-3, 1002)
    
    # discretized integrand: p(A) A^3
    if pdf is None:
        beta_moment_interp = beta(*pdf_args).pdf(_x) * _x**m
    else:
        beta_moment_interp = pdf(_x) * _x**m
    
    # Interpolated integral
    return scipy.interpolate.interp1d(_x[:-1], \
                np.cumsum(beta_moment_interp)[:-1] * np.diff(_x))

# Precompute this, because the snr_distr model needs it
# FIXME: Eschew the beta fit for now, go with KDE

_beta_params = fit_beta_to_theta()
kde = fit_kde_to_theta()

# NOTE: the _beta_params are ignored if the kde is supplied, so the moment
# functions are derived from the KDE. Once its interpolated, the computational
# speed is the same, though.
amp_mom_0 = intr_beta_moment(_beta_params, kde, m=0)
amp_mom_3 = intr_beta_moment(_beta_params, kde, m=3)
_amp_mom_cache = {}



_snr_cache = {}

def snr_distr_z(rho, z1, z2, z0, rhob0, mass1, mass2, ifo=['H1'], \
                  spline=None, power=4, renorm=True, universe='lambdacdm', mass_distr='fixed_source'):
    """
    Binned in distance snr distribution, including reduction of SNR from
    geometric effects. Exists only between z1 and z2, with a fiducial SNR rho0
    at r0.

    The default behavior is to assume a LambdaCDM universe described by
    Planck15 (universe='lambdacdm'). Setting universe='euclidean' will turn off
    redshifting, and treat z{0,1,2} as *distances* in Mpc.

    Evaluated at detected SNR rho.
    """
    

    mbid = (mass1, mass2, z1, z2, mass_distr)

    aligo = Network()
    aligo.add_psd('H1')

    if universe == "lambdacdm":
        r0, r1, r2 = Planck15.luminosity_distance((z0, z1, z2)).value

    elif universe == "euclidean":
        r0, r1, r2 = z0, z1, z2
        # Explicitly zero out the redshifts
        z0, z1, z2 = 0., 0., 0.
    else:
        raise ValueError("Universe must be one of (lambdacdm, euclidean), got {0}".format(universe))
    # Cache the computationally difficult stuff so normalization calls are quick
    if mbid in _snr_cache:
        rhob1, rhob2 = _snr_cache[mbid]
    else:
        # Set the masses
        if mass_distr == 'fixed_source':
            m1, m2 = mass1 * (1 + z1), mass2 * (1 + z1)
        else:
            m1, m2 = mass1, mass2    

        if r1 == 0.:
            rhob1 = np.inf
        else:
            tmplt = eventgen.Event.event_template(mass1=m1, mass2=m2, \
                                                    distance=r1)
            rhob1 = aligo.snr_at_ifo(tmplt, optimal=True, ifo="H1")

        tmplt = eventgen.Event.event_template(mass1=m1, mass2=m2, distance=r2)
        rhob2 = aligo.snr_at_ifo(tmplt, optimal=True, ifo="H1")

        _snr_cache[mbid] = rhob1, rhob2

    # Clip to minimum snr threshold
    rhob2 = max(rhob2, rhob0)

    if spline is None:
        exponent = power - 1
        # Set normalization in front of integral
        norm = exponent * (rhob0 * r0)**(exponent) / (r2**(exponent)- r1**(exponent))
        

        # Clip limits limits for allowed amplitude range
        a, b = max(0, rho / rhob1), min(1, rho / rhob2)
        if a >= b:
            return 0.

        try:
            if power in _amp_mom_cache:
                amplitude_moment = _amp_mom_cache[power]
            else:
                amplitude_moment = intr_beta_moment(_beta_params, kde, m=exponent)
                _amp_mom_cache[power] = amplitude_moment

            intg = amplitude_moment(b) - amplitude_moment(a)

        except Exception as e:
            print rho / rhob2, rho / rhob1
            raise e

        return norm * intg * rho**(-power)

    else:
        
        def _int_fg(alpha):
            return spline(rho / alpha) / alpha * kde(alpha)

        # Clip limits limits for allowed amplitude range
        a, b = max(0, rho / rhob1), min(1, rho / rhob2)
        if a >= b:
            return 0.
        return fixed_quad(_int_fg, a, b)[0]




mom_cache = {}
def mass_snr_distr(rho, r1, r2, r0, rhob0, mass_idx, mc_0, power=4, \
                    renorm=True):
    """
    Binned in distance snr distribution, including reduction of SNR from
    geometric effects. Exists only between r1 and r2, with a fiducial SNR rho0
    at r0. Assumes a power law form of the (rescaled) chirp mass distribution,
    with power law index given by mass_idx.
    
    Evaluated at detected SNR rho.
    """


    rhob1 = rhob0 * r0 / r1 if r1 > 0. else np.inf
    rhob2 = rhob0 * r0 / r2
    #print rhob1, rhob2
    norm = 3. * (rhob0 * r0)**3 / (r2**3 - r1**3)

    # Could be calculated but not used since we renormalize anyway
    #norm2 = (1 - mass_idx) / (mc_0**(1 - mass_idx) - mc_max**(1 - mass_idx))

    # Clip limits limits for allowed amplitude range
    a, b = max(0, rho / rhob1), min(1, rho / rhob2)
    if a >= b:
        return 0.

    beta = 6 / 5. * (1 - mass_idx) + 3
    prefac = 6 / 5. * mc_0**(1 - mass_idx) / beta

    try:
        # Because reconstructing the amplitude moment integral is
        # computationally expensive, we have a cache (dictionary stored within
        # the namespace of the module itself) here keyed by the requested
        # moment power. Most of the time, the same mass power and hence beta
        # value is requested, so the first call to this will be slow and
        # subsequent calls much faster.
        if beta + 3 in mom_cache:
            amp_mom_b = mom_cache[beta+3]
        else:
            # FIXME: Could be precomputed for fixed beta
            mom_cache[beta+3] = amp_mom_b = \
                    intr_beta_moment(_beta_params, kde, m=beta+3)
        intg = amp_mom_b(1)
    except Exception as e:
        print rho / rhob2, rho / rhob1
        raise e

    # SNR factors at the limits
    fac1, fac2 = a**beta, b**beta

    #print (rho / rhob1)**(beta + 3), intg1, intg2
    return norm * prefac * (fac2 - fac1) * intg * rho**(-power)

def get_norm_snr_bins(z_low, z_high, z_horiz, snr_min, mass1, mass2, snr_max=np.inf, ifo=['H1'], 
    spline=None, power=4, universe='lambdacdm', mass_distr='fixed_source'):
    """
    As constructed, the SNR-selected foreground distributions are normalized when considered over (0, infty). We renormalize them to meet the SNR selection conditions, which impose a minimum at snr_min. Technically, they are also normalized to snr_max (for the bin itself) but in reality, the distributions themselves fall to negligible mass at high SNR and thus are ~zero above their maximum SNR anyway.
    """

    fcns = []
    # FIXME: add in some dependence on maximum snr?
    #snrbins = np.logspace(np.log10(snr_min), 2, 1000)
    snrbins = np.logspace(np.log10(snr_min), min(4, np.log10(snr_max)), 1000)
    #norm_factor = []
    #for i in range(len(zbins) - 1):
    pdist = [snr_distr_z(s, z_low, z_high, z_horiz, \
                            mass1=mass1, mass2=mass2, \
                            ifo=ifo, rhob0=snr_min, \
                            spline=spline, power=power, \
                            universe=universe, mass_distr=mass_distr) for s in snrbins]


    cdf = np.cumsum(pdist[:-1] * np.diff(snrbins))
    #norm_factor.append(cdf[-1])

    print 'Normalization factor:', cdf[-1]

    return cdf[-1]

def get_norm_mass_snr_bins(r_low, r_high, r_horiz, mc_ref, mass_power, snr_min, power=4):
    """
    As constructed, the SNR-selected foreground distributions are normalized when considered over (0, infty). We renormalize them to meet the SNR selection conditions, which impose a minimum at snr_min. Technically, they are also normalized to snr_max (for the bin itself) but in reality, the distributions themselves fall to negligible mass at high SNR and thus are ~zero above their maximum SNR anyway.

    Parameters:
    -----------
    r_low: float
        Inner distance of bin (Mpc)

    r_high: float
        Outer distance of bin (Mpc)

    r_horiz: float
        horizon distance (Mpc)

    mc_ref: float
        Reference chirp mass in solar masses

    mass_power: float
        power law index of chirp mass distribution

    snr_min: float
        Reference SNR, often the search threshold SNR

    power: float
        Power law index of leading order SNR distribution

    Returns:
    --------
    norm_factor: float
        Normalization factor for one bin

    """

    fcns = []
    # FIXME: not generic
    snrbins = np.logspace(np.log10(snr_min), 2, 50)
    #norm_factor = []
    pdist = [mass_snr_distr(s, r_low, r_high, r_horiz, \
                                power=power, rhob0=snr_min, \
                                mc_0=mc_ref, mass_idx=mass_power) for s in snrbins]

    cdf = np.cumsum(pdist[:-1] * np.diff(snrbins))
    #norm_factor.append(cdf[-1])

    return cdf[-1] 

if __name__ == "__main__":
    amp = _generate_sample()


    exit()


    # confirm statistics
    print 4**2 * np.mean(amp["amp"]**2), (64 / 25.0)

    pdf_args = fit_beta_to_theta()
    _, b, _ = plt.hist(amp["amp"], bins=100, range=(0, 1), normed=True)

    # So, we'll try to use semi-analytic fits to integrals of the Beta function.
    # For later: note that moments of the beta function end up being the beta
    # function with different arguments.
    plt.plot(b, beta(*pdf_args).pdf(b), color='k', label='beta fit');

    # But the skewness isn't quite right and it looks like scipy tries to
    # compensate by extending the range of the distribution past 1, which isn't
    # the correct answer. So we'll try a KDE
    pdf = fit_kde_to_theta()
    plt.plot(b, pdf(b), color='k', linestyle='-.', label='kde');
    plt.xlim(0, 1)
    plt.legend()
    plt.show()

    # Check normalization
    amp_mom_0 = intr_beta_moment(pdf_args, m=0)
    print amp_mom_0(1.0)
    amp_mom_0 = intr_beta_moment(pdf_args, pdf, m=0)
    print amp_mom_0(1.0)

    # Check second moment
    amp_mom_2 = intr_beta_moment(pdf_args, m=2)
    print amp_mom_2(1.0) * 4**2, (64 / 25.)
    amp_mom_2 = intr_beta_moment(pdf_args, pdf, m=2)
    print amp_mom_2(1.0) * 4**2, (64 / 25.)

    amp_mom_3 = intr_beta_moment(pdf_args, m=3)
    print amp_mom_3(1.0)
    amp_mom_3 = intr_beta_moment(pdf_args, pdf, m=3)
    print amp_mom_3(1.0)

    exit()
    #print snr_distr(3.5, dbins[0], dbins[1], dbins[-1], rhob0=snr_min)

    # Check normalization
    norm_factor = []
    for i in range(len(snrs)):
        pdist = [snr_distr(s, dbins[i], dbins[i+1], dbins[-1], rhob0=snr_min) \
                        for s in snrbins]
        cdf = np.cumsum(pdist[:-1] * np.diff(snrbins))
        norm_factor.append(1.0 / cdf[-1])
        print norm_factor[-1]
        plt.plot(snrbins[:-1], cdf)

    plt.semilogx()
    plt.ylim(0, 1)

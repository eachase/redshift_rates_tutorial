from __future__ import division

from astropy.cosmology import Planck15
from compute_lookup_table import decimal_to_string
import fg_models
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import fixed_quad
import scipy.interpolate
from scipy.optimize import curve_fit, fmin, fminbound
from scipy.stats import beta
from scipy.special import erf, erfc, erfinv
from sklearn.neighbors.kde import KernelDensity

import sys
sys.path.append("./gw_event_gen")
from netsim import Network
import netsim
import eventgen

# Default network
aligo = Network()
aligo.add_psd("H1")
ifo = 'H1'

# Variable mass distribution
min_bh_mass = 3
max_bh_mass = 50
min_bh_mchirp_source = (2.0)**(-1./5.) * min_bh_mass
max_bh_mchirp_source = (2.0)**(-1./5.) * max_bh_mass


# Set up caches to store variables
_fg_cache = {}  # Stores normalizations for foreground analytic functions 
_fg_samples_cache = {}  # Stores foreground samples generated from analytic
_theta_cache = {}  # Stores normalizations for foreground analytic functions 

class RedshiftBin(object):
    """
    A range of redshift values with a defined maximum SNR
    """
    def __init__(self, z_low, z_high, snr_thresh, optimal,
        mass_distr, mass1, mass2, universe, z_horiz, 
        equal_mass, rs=None, lookuptable=None):
        """
        Initialize bin object
        
        Parameters:
        -----------
        z_low: float
            Lowest redshift of range
            
        z_high: float
            Highest redshift of range
            
        snr_thresh: float
            minimum threshold SNR   
 
        optimal: Boolean
            If True, all binaries are assumed to be optimally oriented
            in the likelihood computation

        mass_distr: string
            Sets the assumed mass distribution of binaries.
            Options:
                - 'fixed_source': source frame masses are set to mass1 
                and mass2
                - 'fixed_detector': detector frame masses are set to 
                mass1 and mass2
                - 'variable': source frame masses drawn from astrophysical
                distributions; mass1 and mass2 are ignored 

        mass1, mass2: floats
            Masses of larger and smaller binary components 
            in solar masses. If mass_distr='fixed_source', these are
            source frame masses, while mass_distr='fixed_detector' leads
            to fixed detector frame masses.
        
        universe: string
            Performs calculations in either a cosmological ('lambdacdm')
            or Euclidean ('euclidean') universe

        z_horiz: float
            Horizon redshift for a comological universe or 
            horizon distance for a euclidean universe
 
        equal_mass: Boolean
            only generate events with equal component masses

        """

        # Assign attributes
        self.z_low = z_low
        self.z_high = z_high
        self.snr_thresh = snr_thresh
        self.optimal = optimal
        self.universe = universe
        self.z_horiz = z_horiz
        self.fg_samples_for_plotting = None
        self.mass_distr = mass_distr
        self.equal_mass = equal_mass
        self.rs = rs
        self.lookuptable = lookuptable
        # FIXME!!! - also assign network. Currently, we just assume H1.    

        # Compute maximum possible SNR in the bin, corresponding
        # to an optimally aligned binary at the lowest redshift
        self.mass1 = mass1
        self.mass2 = mass2
        self.snr_max = max(self.snr_thresh, compute_max_snr(self.z_low, self.snr_thresh,
            mass_distr=self.mass_distr, mass1=self.mass1, mass2=self.mass2, 
            universe=self.universe, equal_mass=self.equal_mass, rs=self.rs)) 

        # Set minimum possible SNR in the bin
        if optimal and self.mass_distr != 'variable':
            self.snr_min = max(self.snr_thresh, compute_max_snr(self.z_high, self.snr_thresh,
                mass_distr=self.mass_distr, mass1=self.mass1, mass2=self.mass2, 
                universe=self.universe, equal_mass=self.equal_mass, rs=self.rs))

        elif optimal and self.mass_distr == 'variable':
            # Set minimum SNR for optimal, varied mass case (consider minimum SNR instead of max)
            pass

        else:
            self.snr_min = self.snr_thresh


                
        # Set best fit power law for foreground
        if self.universe == 'euclidean':
            self.power_law_fit = 4
        elif self.universe == 'lambdacdm' and self.mass_distr != 'variable':
            self.power_law_fit = self.fit_foreground_model()
            #self.spline = self.fit_foreground_model(spline=True)
        else:
            # FIXME! store a fit for the power_law_fit as a function of det frame chirp mass
            # in a given bin
            self.power_law_fit = 4


    def kde_calc_likelihood(self, all_samples):
        """
        Fit f(rho) for a given redshift bin with a KDE.
        Evaluate this KDE for each sample in range
         
        Parameters:
        ----------
        all_samples: array
            compilation of several SNRs

        Returns:
        --------
        likelihood: array
            Normalized density function (f(rho)) evaluated for 
            each SNR point in an array. Zero at any SNR value
            that is not in range for this bin.

        FIXME:
            - Update for various mass examples
            - Generate events in a way that efficiently samples
              SNR space (not necessarily uniform in Vc)
        """
        from pyqt_fit import kde, kde_methods

        print 'Fitting likelihood with a KDE!'


        # Generate a bunch of events in the bin
        if self.universe == 'lambdacdm':
            events = generate_comoving(z_low=self.z_low,
                z_high=self.z_high, optimal=self.optimal,
                mass_distr=self.mass_distr, mass1=self.mass1, mass2=self.mass2,
                snr_thresh=self.snr_thresh, num_events=1000, 
                equal_mass=self.equal_mass, rs=self.rs)
        elif self.universe == 'euclidean':
            events = generate_uniform_euclidean(dist_low=self.z_low, 
                dist_high=self.z_high, optimal=self.optimal, 
                mass_distr=self.mass_distr, mass1=self.mass1, mass2=self.mass2, 
                snr_thresh=self.snr_thresh, num_events=1000,
                equal_mass=self.equal_mass, rs=self.rs)


        # Fit a KDE to the events 
        snrs = np.sort(events['snr'])
        fg_kde = kde.KDE1D(snrs, lower=self.snr_thresh, upper=self.snr_max,
            method=kde_methods.renormalization)
        # FIXME check KDE with a plot. Check out-of-range points

        # Return likelihood array
        self.likelihood = fg_kde(all_samples)
        return snrs, self.likelihood

    def calc_likelihood(self, all_samples):
        """
        Compute likelihood for a set of samples in a given redshift range
        
        Parameters:
        ----------
        all_samples: array
            compilation of several SNRs

        Returns:
        --------
        likelihood: array
            Normalized density function (f(rho)) evaluated for 
            each SNR point in an array. Zero at any SNR value
            that is not in range for this bin.
        """
        #likelihood = np.zeros_like(all_samples)
        likelihood = np.zeros(all_samples.shape[0])
        # If samples directly provided, assign as fg samples

        # Fixed mass
        if self.mass_distr == 'fixed_source' or self.mass_distr == 'fixed_detector':
            # Only consider SNRs less than the max_snr and greater than the minimum SNR
            samples_in_range_idx = np.where((all_samples <= self.snr_max) & \
                (all_samples >= self.snr_min))[0]


            # Optimal events
            if self.optimal:
                exponent = 1 - self.power_law_fit

                likelihood[samples_in_range_idx] = exponent * \
                    all_samples[samples_in_range_idx]**(-self.power_law_fit) / (
                    self.snr_max**(exponent) - self.snr_min**(exponent))

            # Non-optimal events 
            else:
                # Compute f(rho) for each point in range (not normalized)
                for i in samples_in_range_idx:
                    likelihood[i] = fg_models.snr_distr_z(all_samples[i], self.z_low, self.z_high, 
                        self.z_horiz, self.snr_thresh, ifo=ifo, mass1=self.mass1, 
                        mass2=self.mass2, universe=self.universe, power=self.power_law_fit, 
                        mass_distr=self.mass_distr)

                # Normalize
                bin_key = (self.z_low, self.z_high, self.z_horiz, self.snr_min, self.power_law_fit)
                if bin_key in _fg_cache:
                    norm = _fg_cache[bin_key]
                else:
                    norm = fg_models.get_norm_snr_bins(self.z_low, self.z_high, self.z_horiz,
                        snr_min=self.snr_thresh, ifo=ifo, mass1=self.mass1, mass2=self.mass2, 
                        universe=self.universe, power=self.power_law_fit, 
                        mass_distr=self.mass_distr)
                    _fg_cache[bin_key] = norm
                likelihood /= norm

        # Varied masses
        elif self.mass_distr == 'variable':
            
            # Check for lookuptable
            precomputed = True
            if self.lookuptable is None:
                precomputed = False
               

            # Compute KDE of the bin's detector frame chirp mass distribution
            kde = self.fit_mchirp_distr()

            # Evaluate the KDE at each point in the bin
            mchirp_arr = all_samples['mchirp'].values
            prob_mchirp_det = np.exp(kde.score_samples(mchirp_arr.reshape(-1,1)))


            # For each sample
            for i, sample in all_samples.iterrows():
                mchirp = sample['mchirp']
                snr = sample['snr']

                # Set corresponding component masses (assuming q=1)
                comp_mass = 2**(1/5) * mchirp

   
                # Check if this mchirp and snr combination is in range -- FIXME better way to organize this
                if self.z_low > 0:
                    max_snr_at_mchirp = compute_max_snr(self.z_low, self.snr_thresh,
                        mass_distr=self.mass_distr, mass1=comp_mass, mass2=comp_mass, 
                        universe=self.universe, equal_mass=self.equal_mass, rs=self.rs)
                else:
                    max_snr_at_mchirp = np.inf

                # Find the power law slope for the detector frame chirp mass and evaluate
                if snr > max_snr_at_mchirp or prob_mchirp_det[i] == 0:
                    prob_snr = 0
                    likelihood[i] = 0

                else:
                    power = self.fit_foreground_model(mass1=comp_mass, mass2=comp_mass)

                    # Compute p(rho|M) for only this point (not normalized)
                    prob_snr = fg_models.snr_distr_z(snr, self.z_low, self.z_high, self.z_horiz,
                        self.snr_thresh, ifo=ifo, mass1=comp_mass, mass2=comp_mass,
                        universe=self.universe, power=power, mass_distr='fixed_detector')

                    # Normalize p(rho|M) for all rho
                    if precomputed:
                        norm = self.lookuptable[find_nearest(self.lookuptable[:,0], mchirp),1]
                    else:
                        norm = fg_models.get_norm_snr_bins(self.z_low, self.z_high, self.z_horiz,
                            snr_min=self.snr_thresh, ifo=ifo, mass1=comp_mass, mass2=comp_mass, 
                            universe=self.universe, power=power, mass_distr='fixed_detector')

                    
                    # Set likelihood value
                    if norm == 0:
                        likelihood[i] = 0
                    else:
                        likelihood[i] = prob_mchirp_det[i] * prob_snr / norm
                    
                print snr, mchirp, prob_snr, likelihood[i]


        self.likelihood = likelihood
        return likelihood
 


    def fit_foreground_model(self, mass1=None, mass2=None, spline=False):
        """
        Fit a simple power law (A * rho**-power) to a distribution of 
        events drawn uniformly in comoving volume, with optimal orientations

        Returns:
        --------
        power: float
            negative of the best-fit power law exponent
        """

        # Set random seed
        set_random_seed(self.rs)

        # Check mass values
        if mass1 is None:
            mass1 = self.mass1
        if mass2 is None:
            mass2 = self.mass2

        # Generate optimally oriented events
        events = self.generate_optimal_comoving(mass1=mass1, mass2=mass2)


        # Store SNRs and corresponding probability   
        snrs = events['snr']
        probs = (snrs * (1 + events['z']))**(-4)

        # FIXME!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! This is important
        # What does it mean if generate_optimal_comoving() can't produce
        # any ("enough in a long time"?) events with SNR > thresh?

        if spline:
            print 'Using spline fit!'
            return scipy.interpolate.interp1d(snrs, probs, kind='cubic', 
                fill_value=0, bounds_error=False)

        else:
            # Fit power law to data
            def _powerlaw(snr, amplitude, power): # FIXME! bad practice
                return amplitude * snr**(-power) 
            
            fit_params, covariance = curve_fit(_powerlaw, snrs, probs)

            # Return exponent of fit
            return fit_params[1]


    def generate_optimal_comoving(self, num_events=300, mass1=None, mass2=None):
        """
        Generate a set of events that are distributed uniformly in comoving 
        volume, with only optimal orientations. Events are within the 
        given redshift bin.

        Parameters:
        -----------
        num_events: int
            Number of events to generate

        Returns:
        --------        
        events: EventGenerator object
            set of events
        """
        # Set random seed
        set_random_seed(self.rs)

        if self.mass_distr != 'variable':
            mass1 = self.mass1
            mass2 = self.mass2


        # Initiate event generator
        events = eventgen.EventGenerator()
        
        # All spin components and eccentricity are zero
        events.append_generator(eventgen._spins_cart, eventgen.zero_spin)
        events.append_generator(("eccentricity",), lambda: (0.0,))
     
        # Inclindation, polarization, and phase are zero
        events.append_generator(eventgen._orien, 
            eventgen.optimally_oriented)

        # Fix event time
        events.append_generator(("event_time",), 
            lambda: (1e9 + np.random.uniform(86400 * 24 * 365),))

        # Set source frame masses to fixed values
        events.append_generator(("mass1", "mass2"), 
            eventgen.delta_fcn_comp_mass, mass1=mass1, mass2=mass2)

        # Place events uniformly in comoving volume, out to the horizon redshift
        events.append_generator(("z",), 
            eventgen.updated_uniform_comoving_redshift, z_min=self.z_low, z_max=self.z_high)

        # Set the corresponding distance (Mpc)
        events.append_post(("distance",), eventgen.uniform_luminosity_dist)

        if self.mass_distr == 'fixed_source':
            # Redshift the masses, if appropriate
            events.append_post(eventgen._redshifted_masses,
                eventgen.detector_masses)
            
        # Compute detector frame chirp mass
        events.append_post(("mchirp", "eta"),
            eventgen.mchirp_eta)

        # Compute the SNR
        events.append_post(("snr",), lambda e: (aligo.snr_at_ifo(e, ifo="H1", optimal=True),))

        # Cut off -- FIXME!!! should snr_thresh become snr_min?
        #events.append_conditional(lambda e: e.snr > self.snr_thresh)                     
        
        # Generate events
        for _ in events.generate_events(n_event=num_events):
            pass
        
        return events
        

    def fit_mchirp_distr(self, plot=False, num_events=10000):
        """
        Fit the bin-dependent distribution of detector-frame chirp masses
        to a KDE

        Parameters:
        -----------
        plot: Boolean
            If true, a histogram is produced

        num_events: int
            Number of events to simulate

        Returns:
        --------
        kde: KernelDensity() object
            kde fit to the chirp masses
        """

        # Initiate event generator
        events = eventgen.EventGenerator()

        # All spin components and eccentricity are zero
        events.append_generator(eventgen._spins_cart, eventgen.zero_spin)
        events.append_generator(("eccentricity",), lambda: (0.0,))

        # Inclindation, polarization, and phase are zero
        events.append_generator(eventgen._orien, 
            eventgen.optimally_oriented)

        # Fix event time
        events.append_generator(("event_time",), 
            lambda: (1e9 + np.random.uniform(86400 * 24 * 365),))

        # Draw source frame component masses from probability distributions
        events.append_generator(("mass1", "mass2"), 
           eventgen.power_law_flat, min_bh_mass=min_bh_mass, max_bh_mass=max_bh_mass, equal_mass=self.equal_mass)


        # Place events uniformly in comoving volume, out to the horizon redshift
        events.append_generator(("z",), 
            eventgen.updated_uniform_comoving_redshift, z_min=self.z_low, z_max=self.z_high)

        # Redshift the masses
        events.append_post(eventgen._redshifted_masses,
            eventgen.detector_masses)

        # Compute detector frame chirp mass
        events.append_post(("mchirp", "eta"),
            eventgen.mchirp_eta)

        # Generate events
        for _ in events.generate_events(n_event=num_events):
            pass
        
        # Compute KDE for chrip mass distribution
        mchirp = np.sort(events['mchirp'])
        kde = KernelDensity(bandwidth=1).fit(mchirp.reshape(-1, 1))
       
        if plot:
            max_chirp = max(100, np.ceil(np.max(events['mchirp'])))
            det_mass_bins = np.arange(0, max_chirp, 2)
            plt.figure(figsize=(8,6))
            plt.hist(events['mchirp'], bins=det_mass_bins, histtype='step', color='k', density=True)
            log_dens = kde.score_samples(mchirp.reshape(-1, 1))
            plt.plot(mchirp, np.exp(log_dens), '-')
            plt.xlabel(r'$M_{\odot}$')
            plt.xlim(0, max_chirp)
            plt.title('Detector Frame Chirp Mass')
        
        return kde





    def compute_VT(self, time, z_step=1e-2, mc_step=5):
        """
        Compute accessible time and population averaged spacetime volume for detections

        Parameters:
        -----------
        time: float
            detection time in years

        z_step: float
            integration steps for redshifts (smaller than redshift bin width)

        mc_step: float
            integration steps for chirp mass (Set smaller value for more accurate integration)

        Returns:
        --------
        VT: float
            accessible spacetime volume in Gpc^3 * yr
        """

        # Set random seed
        set_random_seed(self.rs)

        volume = 0
        if self.mass_distr == 'variable':

            # Compute extremes of mass range in bin
            min_bh_mchirp_det = min_bh_mchirp_source * (1 + self.z_low)
            max_bh_mchirp_det = max_bh_mchirp_source * (1 + self.z_high)

            # Split bins into "mini-bins" for both mass and redshift 
            minibins_mchirp = np.arange(min_bh_mchirp_det, max_bh_mchirp_det, mc_step)
            minibins_z = np.arange(self.z_low, self.z_high, z_step)

            # For all masses
            for (mc_low, mc_high) in zip(minibins_mchirp, minibins_mchirp[1:]):

                mc_mid = (mc_low + mc_high) / 2
                comp_mass = 2**(1/5) * mc_mid

                # Find horizon corresponding to the chirp mass
                z_horiz = aligo.redshift_horizon(mass1=comp_mass, mass2=comp_mass,
                    snr_threshold=self.snr_thresh, mass_distr='fixed_detector')["H1"]

                # Compute density function for chirp mass distribution
                kde = self.fit_mchirp_distr()

                volume_massindependent = 0
                # Integrate mass-independent VT
                for (z_low, z_high) in zip(minibins_z, minibins_z[1:]):
                    # Compute detection efficiency: f(z)
                    det_efficiency = angular_detection_efficiency(
                        z_low, z_high, self.z_horiz)
                    # Approximate 1 + z as 1 + (average z for bin)
                    z_mid = (z_low + z_high) / 2
                    
                    # Compute change in accessible comoving volume
                    dV = 4 * np.pi * Planck15.differential_comoving_volume(z_mid).value  # Mpc^3
                    volume_massindependent += det_efficiency * dV * z_step / (1 + z_mid) 
                
            volume += volume_massindependent * np.exp(kde.score_samples(mc_mid.reshape(-1, 1)))[0] * mc_step

        # Non-optimal, lambdacdm universe
        elif not self.optimal and self.universe == 'lambdacdm':
            # Set precomputed detection efficiencies, corresponding to the fraction of detectable events
            # for events of all orientations computed at a given SNR
            bin_key = (self.z_horiz, self.snr_thresh, self.mass1, self.mass2, self.mass_distr)
            if bin_key in _theta_cache:
                fraction_above_thresh = _theta_cache[bin_key]
            else:
                fraction_above_thresh = approx_theta_distr(z_horiz=self.z_horiz, mass_distr=self.mass_distr,
                    mass1=self.mass1, mass2=self.mass2, snr_thresh=self.snr_thresh, rs=self.rs)

                _theta_cache[bin_key] = fraction_above_thresh

            def _int_dV(z):
                # Make an array that includes redshift 0 and the horizon
                return fraction_above_thresh(z) / \
                     (1 + z) * Planck15.differential_comoving_volume(z).value
    
            volume = 4 * np.pi * fixed_quad(_int_dV, self.z_low, self.z_high)[0]

        # Euclidean universe
        elif self.universe == 'euclidean':


            def _int_dV_euclidean(dist):           

                if self.optimal:
                    det_efficiency = dist < self.z_horiz
                else:
                    # Insert 0 and horizon into detection efficiency computation
                    dist_array_for_int = np.insert(np.append(dist, self.z_horiz), 0, 0)
                    det_efficiency = netsim._fit_theta(np.array(dist_array_for_int))[1:-1]
                return dist**2 * det_efficiency
        
            volume = 4 * np.pi * fixed_quad(_int_dV_euclidean, self.z_low, self.z_high)[0]

        else:

            # Split bins into "mini-bins"
            minibins = np.arange(self.z_low, self.z_high, z_step)


            for (z_low, z_high) in zip(minibins, minibins[1:]):
                # Compute detection efficiency: f(z)
                if self.optimal:
                    det_efficiency = flat_detection_efficiency(
                        z_low, z_high, self.z_horiz)
                else:
                    det_efficiency = angular_detection_efficiency(
                        z_low, z_high, self.z_horiz)
                # Approximate 1 + z as 1 + (average z for bin)
                z_mid = (z_low + z_high) / 2
                
                # Compute change in accessible comoving volume
                if self.universe == 'euclidean':
                    dV = 4 * np.pi * z_mid**2
                elif self.universe == 'lambdacdm':
                    dV = 4 * np.pi * Planck15.differential_comoving_volume(z_mid).value  # Mpc^3
                volume += det_efficiency * dV * z_step / (1 + z_mid) 



        self.VT = volume * time / 1e9
        return self.VT
        
   


class SampleCollector(object):
    """
    A collection of SNRs, corresponding to various foreground or background distributions.
    """
    def __init__(self, snr_thresh, bin_edges, optimal=False, 
        mass_distr='fixed_source', mass1=10, mass2=10, universe='lambdacdm', 
        equal_mass=False, rs=None):
        """
        Initialize sample collector
        
        Parameters:
        -----------
        snr_thresh: float
            minimum threshold SNR
            
        bin_edges: array
            Edges of consecutive redshift bins. 
            Length should be the number of bins + 1

        bin_samples: array
            Array of arrays, where each array contains all 
            SNRs for a given redshift bin
        
        optimal: Boolean
            If True, all binaries are assumed to be optimally oriented
            in the likelihood computation

        mass_distr: string
            Sets the assumed mass distribution of binaries.
            Options:
                - 'fixed_source': source frame masses are set to mass1 
                and mass2
                - 'fixed_detector': detector frame masses are set to 
                mass1 and mass2
                - 'variable': source frame masses drawn from astrophysical
                distributions; mass1 and mass2 are ignored 

        mass1, mass2: floats
            Masses of larger and smaller binary components 
            in solar masses. If mass_distr='fixed_source', these are
            source frame masses, while mass_distr='fixed_detector' leads
            to fixed detector frame masses.

        universe: string
            Performs calculations in either a cosmological ('lambdacdm')
            or Euclidean ('euclidean') universe

        equal_mass: Boolean
            only generate events with equal component masses

        rs: int or None
            random seed

        Notes:
        ------
            - mass_distr cannot be set to 'variable' in euclidean universes
        """

        # Assign attributes
        self.snr_thresh = snr_thresh
        self.optimal = optimal
        self.universe = universe 
        self.mass_distr = mass_distr
        self.equal_mass = equal_mass
        self.num_bin = len(bin_edges) - 1
        self.rs = rs
        # FIXME!!! - also assign network. Currently, we just assume H1.    
        
        # Set horizon redshift (if euclidean universe, this is actually distance)  
        if self.universe == 'euclidean' and self.mass_distr != 'variable':
            self.mass1 = mass1
            self.mass2 = mass2
            self.z_horiz = aligo.horizon(mass1=mass1, mass2=mass2,
                snr_threshold=snr_thresh)["H1"]

        # Set masses for either fixed source-frame or fixed detector-frame distributions
        elif self.universe == 'lambdacdm' and (
            self.mass_distr == 'fixed_source' or self.mass_distr == 'fixed_detector'):
             
            self.mass1 = mass1
            self.mass2 = mass2
            self.z_horiz = aligo.redshift_horizon(mass1=mass1, mass2=mass2,
                snr_threshold=snr_thresh, mass_distr=self.mass_distr)["H1"]

        elif self.universe == 'lambdacdm' and self.mass_distr == 'variable':
            # Find largest possible horizon, given many different mass combinations
            self.z_horiz = various_mass_horizon(self.snr_thresh)

            # Set masses to None
            self.mass1 = None
            self.mass2 = None

        else:
            raise ValueError('Bad universe!')

        # Check for lookuptable
        try:
            # Find lookuptable filename
            filename = 'data/%ibin_minz%sp%s_maxz%sp%s_snrthresh%sp%s.npy' % (
                (self.num_bin,) + decimal_to_string(bin_edges[0]) +\
                decimal_to_string(bin_edges[-1]) + decimal_to_string(snr_thresh))

            # Try to open it
            lookuptable = np.load(filename)
            print 'Lookuptable found. This will run a lot faster!'
        except:
            # FIXME in the future, maybe force this to compute a new lookup table
            lookuptable = None

        # Assemble redshift bins
        self.redshift_bins = []
        bintable = None
        for i, (z_low, z_high) in enumerate(zip(bin_edges, bin_edges[1:])):
            if lookuptable is not None:
                bintable = lookuptable[i,:,:]

            self.redshift_bins.append(RedshiftBin(z_low, z_high, snr_thresh,
                optimal=optimal, mass_distr=self.mass_distr, mass1=mass1, mass2=mass2, 
                universe=universe, z_horiz=self.z_horiz, equal_mass=self.equal_mass, 
                rs=self.rs, lookuptable=bintable))
        

        # Prepare sample collection
        if self.mass_distr == 'variable':
            self.samples = pd.DataFrame(columns=['snr', 'mchirp'])
        else:
            self.samples = np.array([])


    def assign_background_samples(self, bg_counts=None, bg_samples=np.array([])):

        """
        Add background samples to the collector.

        Parameters:
        -----------
        bg_counts: int, optional  
            Number of background samples to generate

        bg_samples: array, optional
            Array of samples to assign to background

        Returns:
        --------
        bg_samples: array
            Array of background samples

        Note:
        -----
         * Either bg_counts or bg_samples must be assigned.
         * bg_counts can be set to zero


        FIXME add a way to supply masses when bg_samples are passes
        """

        # Set random seed
        set_random_seed(self.rs)

        # If samples are not directly provided, generate background samples
        if not bg_samples.any() and bg_counts:
            snr_param = self.snr_thresh / np.sqrt(2)
            bg_samples = np.sqrt(2) * erfinv(
                erf(snr_param) + np.random.uniform(size=bg_counts) * \
                erfc(snr_param))
        elif not bg_samples.any() and bg_counts is None:
            raise ValueError('Need to set either bg_counts or bg_samples.')

        # Combine with current samples
        if self.mass_distr == 'variable':
            # FIXME: check that this mass range is appropriate
            max_bh_mchirp_det = max_bh_mchirp_source * (1 + self.z_horiz)
            uniform_masses = np.random.uniform(min_bh_mchirp_source, max_bh_mchirp_det, len(bg_samples))
            self.samples = self.samples.append(pd.DataFrame(data={
                'snr': bg_samples, 'mchirp': uniform_masses}), ignore_index=True)
            return bg_samples, uniform_masses
        else:
            self.samples = np.append(self.samples, bg_samples)

            # Store background samples for plotting
            self.bg_samples_for_plotting = bg_samples

            return bg_samples


    def assign_foreground_samples(self, fg_counts=None, fg_samples=np.array([])):
        """
        Add foreground samples to the collector.

        Parameters:
        -----------
        fg_counts: array, optional
            Number of foreground samples to generate. 
            Should be ordered in the same way that redshift_bins is ordered.
            i.e. [count for bin 1, count for bin 2, etc.]. Length of array
            must equal the number of redshift bins.

        fg_samples: array or EventGenerator() object, optional
            Either an array of only SNR values or an EventGenerator()
            object containing several binary parameters

        Returns:
        --------
        fg_samples: array
            Array of foreground samples

        Note:
        -----
         * Either fg_counts or fg_samples must be assigned
         * If samples are generated, different generation prescriptions 
           will be applied depending on the 'optimal' attribute.
         * fg_counts may be zero.

        FIXME this is hot garbage
        """

        # Set random seed
        set_random_seed(self.rs)

        # Explicitly account for the zero case
        if fg_counts == 0:
            fg_samples = np.array([])                
            return fg_samples
        
        elif type(fg_samples) == np.ndarray:
            if fg_samples != np.array([]):
                print 'Foreground samples provided'

            elif fg_counts is None:
                raise ValueError('Need to set either fg_counts or fg_samples.')


            # For an optimal distribution, draw fg_count samples from a direct distribution
            elif fg_counts and self.optimal and self.mass_distr != 'variable':
                for i, zbin in enumerate(self.redshift_bins):
                    num_events = fg_counts[i]        

                    exponent = 1 - zbin.power_law_fit

                    # Inverse CDF sample
                    zbin_samples = (zbin.snr_min**(exponent) + np.random.uniform(size=num_events) * \
                        (zbin.snr_max**(exponent) - zbin.snr_min**(exponent)))**(1 / exponent)

                    fg_samples = np.append(fg_samples, zbin_samples)

                    # Store bin samples for plotting
                    zbin.fg_samples_for_plotting = zbin_samples

            elif fg_counts and self.optimal and self.mass_distr == 'variable':
                pass
              
            # Generate events uniformly in volume within each bin, if no samples provided
            # for variable masses
            elif fg_counts:
                for i, zbin in enumerate(self.redshift_bins):
                    if fg_counts[i] > 0:
                        # Generate events in bin
                        if self.universe == 'lambdacdm':
                            events = generate_comoving(z_low=zbin.z_low,
                                z_high=zbin.z_high, optimal=self.optimal,
                                mass_distr=self.mass_distr, mass1=self.mass1, mass2=self.mass2,
                                snr_thresh=self.snr_thresh, num_events=fg_counts[i], 
                                equal_mass=self.equal_mass, rs=self.rs)
                        elif self.universe == 'euclidean':
                            events = generate_uniform_euclidean(dist_low=zbin.z_low, dist_high=zbin.z_high,
                                optimal=self.optimal, mass_distr=self.mass_distr, mass1=self.mass1,
                                mass2=self.mass2, snr_thresh=self.snr_thresh, num_events=fg_counts[i],
                                equal_mass=self.equal_mass, rs=self.rs)
                        if self.mass_distr == 'variable':
                            # Add to array of fg events
                            if type(fg_samples) == np.ndarray:
                                fg_samples = events
                            else:
                                fg_samples.append(events)

                        else:
                            fg_samples = np.append(fg_samples, events['snr'])



        # Store samples
        if self.mass_distr == 'variable':
            self.samples = self.samples.append(pd.DataFrame(data={
                'snr': fg_samples['snr'], 
                'mchirp': fg_samples['mchirp']}), ignore_index=True)
        else:
            self.samples = np.append(self.samples, fg_samples)


        return fg_samples



    def generate_from_analytic(self, fg_counts):
        """
        Generate samples straight from the analytic 
        function for each bin.

        Assumes multiple orientations of binaries 

        Parameters:
        -----------
        fg_counts: array 
            Number of foreground samples to generate. 
            Should be ordered in the same way that redshift_bins is ordered.
            i.e. [count for bin 1, count for bin 2, etc.]

        Returns:
        --------
        fg_samples: array
            Array of all foreground samples, regardless of bin or origin

        FIXME!: add condition for fg_counts=0
        FIXME!: add dependence on mass_distr

        """
        

        fg_samples = np.array([])

        # For each bin
        for i, zbin in enumerate(self.redshift_bins):
            num_events = fg_counts[i]
            zbin_samples = []
            snr_max = min(zbin.snr_max, 40)
            bin_key = (zbin.z_low, zbin.z_high, self.z_horiz, zbin.snr_min, zbin.power_law_fit)
            if bin_key in _fg_cache:
                norm = _fg_cache[bin_key]
            else:
                norm = fg_models.get_norm_snr_bins(zbin.z_low, zbin.z_high, self.z_horiz,
                    snr_min=zbin.snr_thresh, ifo=ifo, mass1=self.mass1, mass2=self.mass2, 
                    universe=self.universe, power=zbin.power_law_fit, mass_distr=self.mass_distr)
                _fg_cache[bin_key] = norm
            samples_key = (num_events, zbin.z_low,
                zbin.z_high, self.z_horiz, zbin.snr_min, zbin.power_law_fit)

            if samples_key in _fg_samples_cache:
                zbin_samples = _fg_samples_cache[samples_key]
                print 'Found in cache'

            else:
                # Generate the desired number of samples
                while len(zbin_samples) < num_events:
                    # Generate SNR between threshold and either SNR max or 40
                    rand_snr = np.random.uniform(zbin.snr_min, snr_max)

                    # Compute f(rho)
                    prob_snr = fg_models.snr_distr_z(rand_snr, zbin.z_low, zbin.z_high, 
                        self.z_horiz, zbin.snr_thresh, ifo=ifo, mass1=self.mass1, 
                        mass2=self.mass2, universe=self.universe, power=zbin.power_law_fit, 
                        mass_distr=self.mass_distr) / norm

                    # Generate random number between 0 and 1 = u
                    prob_rand = np.random.uniform()

                    # If u < f(rho), keep it
                    if prob_rand < prob_snr:
                        zbin_samples.append(rand_snr)

                zbin_samples = np.asarray(zbin_samples)
                _fg_samples_cache[samples_key] = zbin_samples

            # Add the bin's samples to the overall collection of fg_samples
            fg_samples = np.append(fg_samples, zbin_samples)

            # Store bin samples for plotting
            zbin.fg_samples_for_plotting = zbin_samples


        return fg_samples





    def calc_bg_likelihood(self, snr):
        """
        Calculate b(rho) for a given SNR values
        
        Parameters:
        -----------
        snr: float or array
            SNR value greater than self.snr_thresh

        Returns:
        --------
        bg_likelihood: float or array
            b(rho) probability for each provided SNR array
        """

        return (np.sqrt(np.pi/2) * erfc(
            self.snr_thresh / np.sqrt(2)))**(-1) * np.exp(-snr**2 / 2)



            
    def compute_likelihood_statistics(self, kde=False):
        """
        Compute likelihood weights for each class.
        
        Background likelihood is calculated separately.
        
        Each redshift bin object has its own likelihood computed,
        given the entire set of unlabeled samples.
        """
        # Background -- FIXME eventually save all samples as pandas DF regardless of mass distr
        if self.mass_distr == 'variable':
            self.bg_likelihood = self.calc_bg_likelihood(self.samples['snr'].values)
        else:
            self.bg_likelihood = self.calc_bg_likelihood(self.samples)

        # Normalize the background values
        snrbins = np.logspace(np.log10(self.snr_thresh), 2, 2000)
        pdist = [self.calc_bg_likelihood(s) for s in snrbins]
        norm_factor = np.trapz(pdist, snrbins)

        self.bg_likelihood /= norm_factor


        # For variable mass distributions, normalize by masses
        if self.mass_distr == 'variable':
            max_bh_mchirp_det = max_bh_mchirp_source * (1 + self.z_horiz)
            # Divide by difference in detector frame chirp mass range 
            # (this should correspond to the range at which background masses were generated) 
            self.bg_likelihood /= max_bh_mchirp_det - min_bh_mchirp_source

        print 'Background likelihood sum: ', np.sum(self.bg_likelihood)

        # Foreground for all redshift bins
        for zbin in self.redshift_bins:
            if kde:
                zbin.kde_calc_likelihood(self.samples)
            else:
                zbin.calc_likelihood(self.samples)
            print 'Bin likelihood sum: ', np.sum(zbin.likelihood)




    def compute_log_likelihood(self, counts):
        """
        Compute log likelihood for a given set of class counts.
        
        Parameters:
        -----------
        counts: array
            each entry is a count for each source type in the following order:
            [background_counts, foreground_counts] where the foreground counts 
            are in ascending redshift order

        Returns:
        --------
        log_like: float
            Logarithm of the likelihood

        Notes:
        ------   
        - counts should be the length of bin counts + 1
        - compute_likelihood_statistics() should be run once before
        this method is called
        """
        if np.all(counts >= 0):
            # Background likelihood
            bg_likelihood = self.bg_likelihood * counts[0]

            # Foreground for all redshift bins
            fg_likelihood = 0
            for idx, zbin in enumerate(self.redshift_bins):
                fg_likelihood += counts[idx+1] * zbin.likelihood
            assert np.all(fg_likelihood + bg_likelihood >= 0)
            lnlike = np.sum(np.log(fg_likelihood + bg_likelihood)) - np.sum(counts)

            return lnlike
        else:
            return -np.inf

    @staticmethod
    def lnprior(counts, minlambda=0, maxlambda=np.inf):
        """
        Log Prior

        Parameters:
        -----------
        counts: array
            each entry is a count for each source type in the following order:
            [background_counts, foreground_counts] where the foreground counts 
            are in ascending redshift order
        
        maxlambda: float
            set upper limit on possible Lambda values
        
        minlambda: float
            set lower limit on possible Lambda values


        Returns:
        --------
        log_prior: float
            Logarithm of the prior

        Notes:
        ------   
        - counts should be the length of bin counts + 1
        - compute_likelihood_statistics() should be run once before
        this method is called
        """
        # Only evaluate prior in range
        if np.all(counts >= minlambda) and np.all(counts <=maxlambda):
            
            # Only compute a prior for valid regions of parameter space
            return -0.5 * (np.log(counts[0]) + np.log(np.sum(counts[1:])))

        else:

            return -np.inf


    def lnprob(self, counts, turn_prior_off=False, minlambda=0, maxlambda=np.inf, turn_like_off=False):
        """
        Combine log likelihood and log prior
 
        Parameters:
        -----------
        counts: array
            each entry is a count for each source type in the following order:
            [background_counts, foreground_counts] where the foreground counts 
            are in ascending redshift order
 
        turn_prior_off: Boolean
            Sets log prior to zero

        turn_like_off: Boolean
            Sets log likelihood to zero
        
        maxlambda: float
            set upper limit on possible Lambda values


        Returns:
        --------
        log_posterior: float
            Log of the posterior

        Notes:
        ------           
        - counts should be the length of bin counts + 1
        - compute_likelihood_statistics() should be run once before
        this method is called
        """

        # Compute prior
        if turn_prior_off:
            prior = 0 
        else:
            prior = self.lnprior(counts, minlambda, maxlambda)
            
        # Compute likelihood
        if turn_like_off:
            likelihood = 0
            # Prevent the prior from growing without bound
            prior = self.lnprior(counts, minlambda, maxlambda)
            if prior < -7:
                prior = -np.inf
        else:
            likelihood = self.compute_log_likelihood(counts)
            

        # Combine to get posterior
        #if not np.isfinite(prior):
        #    return -np.inf
        return prior + likelihood

    def compute_log_likelihood_at_counts(self, counts):
        all_likes = np.zeros((len(np.atleast_2d(counts)), self.bg_likelihood.size, 6))
        # Background likelihood
        all_likes[:,:, 0] = self.bg_likelihood * np.atleast_2d(counts)[:,0].reshape(-1,1)

        # Foreground for all redshift bins
        for idx, zbin in enumerate(self.redshift_bins):
            all_likes[:,:, idx+1] = np.atleast_2d(counts)[:, idx+1].reshape(-1,1) * zbin.likelihood

        lnlike = np.sum(np.log(np.sum(all_likes,2)),1) - np.sum(np.atleast_2d(counts), 1)
        return lnlike
    
    def plot(self, plot_analytic=True, filename='temp.png'):
        """
        Make cumulative histogram of samples.

        Parameters:
        -----------
        plot_analytic: Boolean
            if True, overlay analytic f(rho) curves

        filename: string
            name for output png plot

        """
        snr_max = self.redshift_bins[0].snr_max
        if snr_max <= 100:
            bins = np.linspace(0, self.redshift_bins[0].snr_max, 100)
        else:
            bins = np.linspace(0, 30, 100) 

        plt.figure(figsize=(8,6))
        
        colors = plt.cm.viridis(np.linspace(0, 0.75, len(self.redshift_bins)))
       
        # Plot analytic, non-cumulative histogram
        if plot_analytic:
            # Define an SNR Array
            snr_array_plot = np.linspace(self.snr_thresh, 50, 100)
            #bg_likelihood = (np.sqrt(np.pi/2) * erfc(
            #    self.snr_min / np.sqrt(2)))**(-1) * np.exp(-snr_array_plot**2 / 2)
            bg_likelihood = self.calc_bg_likelihood(snr_array_plot)


            # Normalize the background values
            snrbins = np.logspace(np.log10(self.snr_thresh), 2, 50)
            pdist = [self.calc_bg_likelihood(s) for s in snrbins]
            bg_likelihood /= np.trapz(pdist, snrbins)
            
            plt.hist(self.bg_samples_for_plotting, label='Background',
                             color='red', bins=snr_array_plot,
                             histtype='step', log=True, density=True)

     
            plt.plot(snr_array_plot, bg_likelihood, color='r')

            for i, zbin in enumerate(self.redshift_bins):
                plt.hist(zbin.fg_samples_for_plotting, color=colors[i], bins=snr_array_plot, 
                     histtype='step', log=True, density=True,
                     label=('Distance : %.2f - %.2f'%(zbin.z_low, zbin.z_high)))

                # Compute f(rho)
                #fg_distr = np.asarray([snr_distr(snr, zbin.z_low, zbin.z_high, r_horiz,
                #    zbin.snr_min, ifo=ifo, mass1=mass1, mass2=mass2) for snr in snr_array_plot])
 
                fg_distr = np.asarray([fg_models.snr_distr_z(snr, zbin.z_low, zbin.z_high, 
                    self.z_horiz, zbin.snr_thresh, ifo=ifo, mass1=self.mass1, 
                    mass2=self.mass2, universe=self.universe, power=zbin.power_law_fit, 
                    mass_distr=self.mass_distr) for snr in snr_array_plot])

                # Compute Normalization Factor
                norm = fg_models.get_norm_snr_bins(zbin.z_low, zbin.z_high, self.z_horiz,
                    snr_min=zbin.snr_thresh, ifo=ifo, mass1=self.mass1, mass2=self.mass2, 
                    universe=self.universe, power=zbin.power_law_fit, mass_distr=self.mass_distr)

                # Plot the distribution
                plt.plot(snr_array_plot, fg_distr / norm, color=colors[i])

            plt.xlim(3, 40)
            plt.ylim(1e-4, 1e5)
            plt.ylabel('Normalized Number of Events')

        else:
            plt.hist(self.bg_samples_for_plotting, label='Background',
                             color='red', bins=bins, cumulative=-1,
                             histtype='step')

            for i, zbin in enumerate(self.redshift_bins):
                plt.hist(zbin.fg_samples_for_plotting, color=colors[i], bins=bins, 
                         histtype='step', cumulative=-1,
                         label=('Distance : %.2f - %.2f'%(zbin.z_low, zbin.z_high)))
            plt.xlim(3, None)
            plt.ylim(0.1, None)
            plt.ylabel('Number of Events  with SNR > Corresponding SNR')
 

        plt.legend(loc='upper right')
        #plt.xscale('log', nonposx='clip')
        plt.yscale('log', nonposy='clip')
        plt.xlabel('SNR')
        plt.title('Simulated Samples')
        #plt.show()
        plt.savefig(filename)

def find_horizon(snr_min, mass1=10, mass2=10, network=aligo):
    """
    Compute horizon redshift for a given combination
    of component masses and threshold SNR
    
    Parameters:
    -----------
    snr_min: float
        Threshold SNR
        
    mass1: float
        Larger binary component mass in solar masses
    
    mass2: float
        Smaller binary component mass in solar masses
    
    Returns:
    --------
    float:
        maximum horizon redshift
    """
    
    return network.redshift_horizon(mass1=mass1, mass2=mass2, \
                                            snr_threshold=snr_min)["H1"]
    

def compute_max_snr(z, snr0, mass_distr='fixed_source', mass1=10, mass2=10, 
    universe='lambdacdm', equal_mass=False, rs=None):

    """
    Arbitrarily defines a maximum possible SNR for a given redshift.

    This function assumes a detection network containing the advanced
    LIGO design sensitivity Hanford PSD.

    Parameter:
    ----------
    z: float
        Redshift value. Should not be negative.
        If in a Euclidean universe, this is a distance in Mpc.

    mass_distr: string
        Sets the assumed mass distribution of binaries.
        Options:
            - 'fixed_source': source frame masses are set to mass1 
            and mass2
            - 'fixed_detector': detector frame masses are set to 
            mass1 and mass2
            - 'variable': source frame masses drawn from astrophysical
            distributions; mass1 and mass2 are ignored 

    mass1, mass2: floats
        Masses of larger and smaller binary components 
        in solar masses. If mass_distr='fixed_source', these are
        source frame masses, while mass_distr='fixed_detector' leads
        to fixed detector frame masses.
    
    universe: string
        Performs calculations in either a cosmological ('lambdacdm')
        or Euclidean ('euclidean') universe
    
    equal_mass: Boolean
        only generate events with equal component masses


    Returns:
    --------
    snr_max: float
        SNR of optimally-aligned binary at this redshift.
        If the redshift is zero, then an infinite SNR is returned.

    """
    
    # Set random seed
    set_random_seed(rs)

    if z == 0:
        return np.infty

    # Assume z is some reference distance for a Euclidean universe
    elif universe == 'euclidean':
        r0 = aligo.horizon(mass1=mass1, mass2=mass2, snr_threshold=snr0)["H1"]
        return snr0 * r0 / z
    elif universe == 'lambdacdm':
        # Generate an optimally-oriented event at the lowest redshift
        event = eventgen.EventGenerator()
        event.append_generator(eventgen._spins_cart, eventgen.zero_spin)

        # Set inclination, polarization, phase, and eccentricity to zero
        event.append_generator(('inclination', 'polarization',
            'coa_phase', 'eccentricity'), lambda:(0,0,0,0))

        # Set some event times -- FIXME!!! this random event time will bias SNR ... does this even change anything??
        event.append_generator(("event_time",),
            lambda: (1e9 + np.random.uniform(86400 * 24 * 365),))

        # Set the masses
        if mass_distr == 'variable':
            event.append_generator(("mass1", "mass2"), 
                eventgen.power_law_flat, min_bh_mass=min_bh_mass, max_bh_mass=max_bh_mass, equal_mass=equal_mass)
        else:
            event.append_generator(("mass1", "mass2"), \
                eventgen.delta_fcn_comp_mass, mass1=mass1, mass2=mass2)

        # Set the redshift
        event.append_generator(('z',), lambda:(z,))

        # Set the corresponding distance (Mpc)
        event.append_post(("distance",), eventgen.uniform_luminosity_dist)

        # Redshift the masses, if appropriate
        if mass_distr == 'fixed_source' or mass_distr == 'variable':
            event.append_post(eventgen._redshifted_masses, 
                eventgen.detector_masses)

        # Generate one event for fixed mass distributions or several for variables
        num_events = 1000 if mass_distr == 'variable' else 1 #FIXME this causes a major slow down
        for _ in event.generate_events(n_event=num_events):
            pass

        # For each event, compute the SNR of an optimally aligned binary with these properties
        event_snr = 0
        for e in event:
            event_snr = max(event_snr, aligo.snr_at_ifo(e, ifo='H1', optimal=True))

        return event_snr


    else:
        raise ValueError('Bad Universe!')

    # FIXME! - should I do something special if snr_max < snr_thresh?



def flat_detection_efficiency(z_min, z_max, z_horiz):
    """
    Compute detection efficiency f(z) assuming a flat step function
    cut off after some horizon redshift.
    
    -f(z) is 1 if z_max < z_horiz
    -f(z) is 0 if z_min > z_horiz
    -Special case if z_min < z_horiz < z_max
    
    Parameters:
    -----------
    z_min: float
        Lowest redshift in bin
        
    z_max: float
        Highest redshift in bin
        
    z_horiz: float
        Horizon redshift for a given system of binaries
        
    Returns:
    --------
    float:
        detection efficiency in this bin. Between 0 and 1
    """
    
    #FIXME: if in need of boosting computation efficiency, this function
    # would be a good place to start
    
    #FIXME: add something in to check that z_min and z_max are logical values
    # with z_max > z_min and both values positive
    
    if z_max < z_horiz:
        return 1
    elif z_min > z_horiz:
        return 0
    else:
        return (z_horiz - z_min) / (z_max - z_min)
    
        
def angular_detection_efficiency(z_min, z_max, z_horiz):
    """
    FIXME: Find a better name for this function!!!
    
    Compute detection efficiency f(z) assuming some variation 
    based on binary orientation. Assume equal mass binaries
    
    Parameters:
    -----------
    z_min: float
        Lowest redshift in bin
        
    z_max: float
        Highest redshift in bin
        
    z_horiz: float
        Horizon redshift for a given system of binaries
        
    Returns:
    --------
    float:
        detection efficiency in this bin. Between 0 and 1
    """
    # Find average redshift for the bin
    z_mid = (z_min + z_max) / 2
    
    # middle value of det_eff_arr should correspond to the
    # array of detection efficiency values        
    return netsim._fit_theta(np.array([0, z_mid, z_horiz]))[1]
    

def approx_theta_distr(z_horiz, num_events=1000, mass_distr='fixed_source',
    mass1=10, mass2=10, snr_thresh=3.5, num_points=25, rs=None):
    """
    Empirically trace out the Finn & Chernoff detection efficiency

    Parameters:
    -----------
    z_horiz: float
        Horizon redshift, after which no events can be detected

    num_events: int
        Number of events to generate

    mass_distr: string
        Sets the assumed mass distribution of binaries.
        Options:
            - 'fixed_source': source frame masses are set to mass1 
            and mass2
            - 'fixed_detector': detector frame masses are set to 
            mass1 and mass2
            - 'variable': source frame masses drawn from astrophysical
            distributions; mass1 and mass2 are ignored 


    mass1, mass2: floats
        Masses of larger and smaller binary components 
        in solar masses. If mass_distr='fixed_source', these are
        source frame masses, while mass_distr='fixed_detector' leads
        to fixed detector frame masses.

    snr_thresh: float
        minimum threshold SNR   

    num_points: int
        number of points to include in the interpolation

    Returns:
    --------
    fraction_above_thresh: interp1d object
        interpolation of the detection efficiency, as a function of redshift
    """

    # Set random seed
    set_random_seed(rs)

    # Generate events, regardless of threshold
    events_nothresh = generate_comoving(0, z_high=z_horiz, num_events=num_events, 
        optimal=False, mass_distr=mass_distr, mass1=mass1, mass2=mass2, 
        snr_thresh=0, rs=rs)

    # Set interval to evaluate interpolation at
    z_step = z_horiz/num_points    

    # Sort events into bins (not redshift bins)
    z_array = np.arange(0, z_horiz + 2*z_step, z_step)

    # For each bin, record the fraction of detectable events
    fraction_above_thresh = np.zeros(shape=len(z_array) - 1)
    for i, z_low in enumerate(z_array[:-1]):
        # Total events in bin
        num_events = len(np.where((events_nothresh['z'] > z_low)\
            & (events_nothresh['z'] < z_array[i+1]))[0])
        if num_events == 0:
            fraction_above_thresh[i] = 1 if z_low < z_horiz else 0
        else:
            # Detectable events in bin
            num_detectable_events = len(np.where((events_nothresh['z'] > z_low)\
                & (events_nothresh['z'] < z_array[i+1]) &\
                (events_nothresh['snr'] > snr_thresh))[0])
            fraction_above_thresh[i] = num_detectable_events / num_events


    # Interpolate this function    
    return scipy.interpolate.interp1d(z_array[:-1], fraction_above_thresh)
    

def generate_uniform_euclidean(dist_low=0, dist_high=2000, num_events=300, optimal=True, 
    mass_distr='fixed_source', mass1=10, mass2=10, snr_thresh=3.5, equal_mass=False, rs=None,
    powerlaw=0):
    """
    Generate a set of events that are distributed uniformly in Euclidean
    volume

    Parameters:
    -----------
    dist_low: float
        Minimum luminosity distance in Mpc

    dist_high: float
        Highest luminosity distance in Mpc

    num_events: int
        Number of events to generate

    optimal: Boolean
        If true, events are only optimally-oriented

    mass_distr: string
        Sets the assumed mass distribution of binaries.
        Options:
            - 'fixed_source': source frame masses are set to mass1 
            and mass2
            - 'variable': source frame masses drawn from astrophysical
            distributions; mass1 and mass2 are ignored 

    mass1, mass2: floats
        Masses of larger and smaller binary components 
        in solar masses. If mass_distr='fixed_source', these are
        source frame masses, while mass_distr='fixed_detector' leads
        to fixed detector frame masses.

    snr_thresh: float
        minimum threshold SNR   

    equal_mass: Boolean
        only generate events with equal component masses

    Returns:
    --------        
    events: EventGenerator object
        set of events
    """
    # Set random seed
    set_random_seed(rs)



    assert num_events > 0

    # Initiate event generator
    events = eventgen.EventGenerator()
    
    # All spin components and eccentricity are zero
    events.append_generator(eventgen._spins_cart, eventgen.zero_spin)
    events.append_generator(("eccentricity",), lambda: (0.0,))
 

    # Set orientation parameters and event time
    if optimal: 
        events.append_generator(eventgen._orien, 
            eventgen.optimally_oriented)
        events.append_generator(("event_time",), 
            lambda: (1e9 + np.random.uniform(86400 * 24 * 365),))
    else:
        events.append_generator(eventgen._extr, 
            eventgen.random_orientation)
        events.append_generator(("event_time",), 
            lambda: (1e9 + np.random.uniform(86400 * 24 * 365),))

    # Set mass values
    if mass_distr == 'variable':
        events.append_generator(("mass1", "mass2"), 
            eventgen.power_law_flat, min_bh_mass=min_bh_mass, max_bh_mass=max_bh_mass, equal_mass=equal_mass)
    else:
        events.append_generator(("mass1", "mass2"), 
            eventgen.delta_fcn_comp_mass, mass1=mass1, mass2=mass2)
 

    if powerlaw is 0:
        # Place events uniformly in Euclidean volume, out to the maximum distance
        events.append_generator(("distance",), eventgen.uniform_volume, d_max=dist_high)
    else:       
        # Generate events following a power law in Euclidean distance
        events.append_generator(("distance",), eventgen.uniform_volume, d_max=dist_high, power=powerlaw)

    # Compute the SNR
    events.append_post(("snr",), lambda e: (aligo.snr_at_ifo(e, ifo="H1", optimal=optimal, fmin=10, fmax=2048),))

    # Cut off SNR
    events.append_conditional(lambda e: e.snr > snr_thresh)                     

    # Place lower limit on distance
    events.append_conditional(lambda e: e.distance > dist_low)                     
    
    # Generate events
    for _ in events.generate_events(n_event=num_events):
        pass
   
    return events








def generate_comoving(z_low=0, z_high=2, num_events=300, optimal=True, 
    mass_distr='fixed_source', mass1=10, mass2=10, snr_thresh=3.5, equal_mass=False, rs=None):
    """
    Generate a set of events that are distributed uniformly in comoving 
    volume

    Parameters:
    -----------
    z_low: float
        Lowest redshift events can have

    z_high: float
        Highest redshift events can have

    num_events: int
        Number of events to generate

    optimal: Boolean
        If true, events are only optimally-oriented

    mass_distr: string
        Sets the assumed mass distribution of binaries.
        Options:
            - 'fixed_source': source frame masses are set to mass1 
            and mass2
            - 'fixed_detector': detector frame masses are set to 
            mass1 and mass2
            - 'variable': source frame masses drawn from astrophysical
            distributions; mass1 and mass2 are ignored 

    mass1, mass2: floats
        Masses of larger and smaller binary components 
        in solar masses. If mass_distr='fixed_source', these are
        source frame masses, while mass_distr='fixed_detector' leads
        to fixed detector frame masses.

    snr_thresh: float
        minimum threshold SNR   

    equal_mass: Boolean
        only generate events with equal component masses

    Returns:
    --------        
    events: EventGenerator object
        set of events
    """

    # Set random seed
    set_random_seed(rs)

    assert num_events > 0

    # Initiate event generator
    events = eventgen.EventGenerator()
    
    # All spin components and eccentricity are zero
    events.append_generator(eventgen._spins_cart, eventgen.zero_spin)
    events.append_generator(("eccentricity",), lambda: (0.0,))
 

    # Set orientation parameters and event time
    if optimal: 
        events.append_generator(eventgen._orien, 
            eventgen.optimally_oriented)
        events.append_generator(("event_time",), 
            lambda: (1e9 + np.random.uniform(86400 * 24 * 365),))
        
    else:
        events.append_generator(eventgen._extr, 
            eventgen.random_orientation)
        events.append_generator(("event_time",), 
            lambda: (1e9 + np.random.uniform(86400 * 24 * 365),))

    # Set mass values
    if mass_distr == 'variable':
        events.append_generator(("mass1", "mass2"), 
            eventgen.power_law_flat, min_bh_mass=min_bh_mass, max_bh_mass=max_bh_mass, equal_mass=equal_mass)
    else:
        events.append_generator(("mass1", "mass2"), 
            eventgen.delta_fcn_comp_mass, mass1=mass1, mass2=mass2)



    # Place events uniformly in comoving volume, out to the horizon redshift
    events.append_generator(("z",), 
        eventgen.updated_uniform_comoving_redshift, z_min=z_low, z_max=z_high)

    # Set the corresponding distance (Mpc)
    events.append_post(("distance",), eventgen.uniform_luminosity_dist)

    if mass_distr == 'fixed_source' or mass_distr == 'variable':
        # Redshift the masses, if appropriate
        events.append_post(eventgen._redshifted_masses,
            eventgen.detector_masses)
    
    # Compute detector frame chirp mass
    events.append_post(("mchirp", "eta"),
        eventgen.mchirp_eta)
    
    # Compute the SNR
    events.append_post(("snr",), lambda e: (aligo.snr_at_ifo(e, ifo="H1", optimal=optimal, fmin=10, fmax=2048),))

    # Cut off SNR
    events.append_conditional(lambda e: e.snr > snr_thresh)                     
    
    # Generate events
    for _ in events.generate_events(n_event=num_events):
        pass
    
    return events



def generate_astro(distribution='SFR', z_low=0, z_high=2, num_events=300, optimal=True, 
    mass_distr='fixed_source', mass1=10, mass2=10, snr_thresh=3.5, equal_mass=False, rs=None):
    """
    Generate a set of events that are distributed uniformly in comoving 
    volume

    Parameters:
    -----------
    distribution: string
        indicator of astrophysical model for generating events. Options: SFR

    z_low: float
        Lowest redshift events can have

    z_high: float
        Highest redshift events can have

    num_events: int
        Number of events to generate

    optimal: Boolean
        If true, events are only optimally-oriented

    mass_distr: string
        Sets the assumed mass distribution of binaries.
        Options:
            - 'fixed_source': source frame masses are set to mass1 
            and mass2
            - 'fixed_detector': detector frame masses are set to 
            mass1 and mass2
            - 'variable': source frame masses drawn from astrophysical
            distributions; mass1 and mass2 are ignored 

    mass1, mass2: floats
        Masses of larger and smaller binary components 
        in solar masses. If mass_distr='fixed_source', these are
        source frame masses, while mass_distr='fixed_detector' leads
        to fixed detector frame masses.

    snr_thresh: float
        minimum threshold SNR   

    equal_mass: Boolean
        only generate events with equal component masses

    Returns:
    --------        
    events: EventGenerator object
        set of events
    """

    # Set random seed
    set_random_seed(rs)

    assert num_events > 0

    # Initiate event generator
    events = eventgen.EventGenerator()
    
    # All spin components and eccentricity are zero
    events.append_generator(eventgen._spins_cart, eventgen.zero_spin)
    events.append_generator(("eccentricity",), lambda: (0.0,))
 

    # Set orientation parameters and event time
    if optimal: 
        events.append_generator(eventgen._orien, 
            eventgen.optimally_oriented)
        events.append_generator(("event_time",), 
            lambda: (1e9 + np.random.uniform(86400 * 24 * 365),))
        
    else:
        events.append_generator(eventgen._extr, 
            eventgen.random_orientation)
        events.append_generator(("event_time",), 
            lambda: (1e9 + np.random.uniform(86400 * 24 * 365),))

    # Set mass values
    if mass_distr == 'variable':
        events.append_generator(("mass1", "mass2"), 
            eventgen.power_law_flat, min_bh_mass=min_bh_mass, max_bh_mass=max_bh_mass, equal_mass=equal_mass)
    else:
        events.append_generator(("mass1", "mass2"), 
            eventgen.delta_fcn_comp_mass, mass1=mass1, mass2=mass2)



    # Place events uniformly in comoving volume, out to the horizon redshift
    if distribution == 'SFR':
        events.append_generator(("z",), 
            eventgen.madau_dickinson_redshift, z_min=z_low, z_max=z_high)

    else: #FIXME add a tag here
        events.append_generator(("z",), 
            eventgen.updated_uniform_comoving_redshift, z_min=z_low, z_max=z_high)

    # Set the corresponding distance (Mpc)
    events.append_post(("distance",), eventgen.uniform_luminosity_dist)

    if mass_distr == 'fixed_source' or mass_distr == 'variable':
        # Redshift the masses, if appropriate
        events.append_post(eventgen._redshifted_masses,
            eventgen.detector_masses)
    
    # Compute detector frame chirp mass
    events.append_post(("mchirp", "eta"),
        eventgen.mchirp_eta)
    
    # Compute the SNR
    events.append_post(("snr",), lambda e: (aligo.snr_at_ifo(e, ifo="H1", optimal=optimal, fmin=10, fmax=2048),))

    # Cut off SNR
    events.append_conditional(lambda e: e.snr > snr_thresh)                     
    
    # Generate events
    for _ in events.generate_events(n_event=num_events):
        pass
    
    return events



def various_mass_horizon(snr_thresh):
    """
    Find largest possible redshift horizon for various
    source-frame component masses.

    Assumes lambdacdm universe.

    Parameters:
    -----------
    snr_thresh: float
        minimum threshold SNR   
    
    Returns:
    --------
    z_horiz: float
        horizon redshift

    FIXME To Do:
    - Add dependency on network
    - Consider ranges of possible mass distributions, assert
    that the minimization only considers masses within this range
    """

    def _redshift_horizon_H1(masses, snr_thresh):
        """
        Return negative of horizon redshift for H1 only.
        Used for optimization
        
        Parameters:
        -----------
        masses: iterable of form (mass1, mass2)
            component masses in solar masses
            
        snr_thresh: float
            minimum SNR threshold of the search
            
            
        Returns:
        --------
        neg_z_horiz: float
            negative redshift horizon
        """
        

        mass1, mass2 = masses
        return -aligo.redshift_horizon(mass1=mass1, mass2=mass2,
            snr_threshold=snr_thresh)['H1']


    # Make initial mass guess
    mass_guess = np.array([10,10])

    # Find masses with furthest horizon redshift
    mass_opt = fmin(_redshift_horizon_H1, mass_guess, args=(snr_thresh,), disp=False)

    # Return corresponding horizon redshift
    return aligo.redshift_horizon(mass1=mass_opt[0], mass2=mass_opt[1],
        snr_threshold=snr_thresh)['H1']



def set_equal_VT_bins(num_bins, optimal, mass_distr, mass1, mass2, 
    universe, snr_thresh, equal_mass=False, rs=None, min_z=None, max_z=None):
    """
    Compute bin edges for bins of equal VT.

    Parameters:
    -----------

    num_bins: int
        number of bins

    optimal: Boolean
        If True, all binaries are assumed to be optimally oriented
        in the likelihood computation

    mass_distr: string
        Sets the assumed mass distribution of binaries.
        Options:
            - 'fixed_source': source frame masses are set to mass1 
            and mass2
            - 'fixed_detector': detector frame masses are set to 
            mass1 and mass2
            - 'variable': source frame masses drawn from astrophysical
            distributions; mass1 and mass2 are ignored 

    mass1, mass2: floats
        Masses of larger and smaller binary components 
        in solar masses. If mass_distr='fixed_source', these are
        source frame masses, while mass_distr='fixed_detector' leads
        to fixed detector frame masses.
    
    universe: string
        Performs calculations in either a cosmological ('lambdacdm')
        or Euclidean ('euclidean') universe

    snr_thresh: float
        minimum threshold SNR   

    Returns:
    --------
    bin_edges: array
        Edges of consecutive redshift bins. 
        Length should be the number of bins + 1
    """
    
    # FIXME this is not applicable to a euclidean universe, right now

    
    # Compute max_z/max_dist based on the masses
    if universe == 'lambdacdm':
        z_horiz = aligo.redshift_horizon(mass1=mass1, mass2=mass2,
            snr_threshold=snr_thresh, mass_distr=mass_distr)['H1']
    elif universe == 'euclidean':
        z_horiz = aligo.horizon(mass1=mass1, mass2=mass2,
            snr_threshold=snr_thresh)["H1"]
    else:
        raise ValueError('Bad universe!')

    if max_z is None:
        max_z = z_horiz
    else:
        max_z = float(max_z)

    if min_z is None:
        min_z = 0
    else:
        min_z = float(min_z)



    # Compute total VT
    total_bin = RedshiftBin(z_low=min_z, z_high=max_z,
        snr_thresh=snr_thresh, optimal=optimal,
        mass_distr=mass_distr, mass1=mass1, mass2=mass2,
        universe=universe, z_horiz=z_horiz, equal_mass=equal_mass,
        rs=rs)
    total_VT = total_bin.compute_VT(time=1)


    # Split VT equally among bins
    VT_per_bin = total_VT / num_bins



    def _VT_diff(z_high, z_low, VT):

        # Compute VT between z_low and z_high
        return np.fabs(RedshiftBin(z_low=z_low, z_high=z_high,
            snr_thresh=snr_thresh, optimal=optimal,
            mass_distr=mass_distr, mass1=mass1, mass2=mass2,
            universe=universe, z_horiz=z_horiz, equal_mass=equal_mass).compute_VT(time=1) - VT)

    # Find bin edge which gives the desired VT
    bin_edges = np.zeros(num_bins+1)
    bin_edges[0] = min_z
    bin_edges[-1] = max_z
    for i in np.arange(num_bins-1):
        bin_edges[i+1] = fminbound(_VT_diff, bin_edges[i], max_z, args=(bin_edges[i], VT_per_bin))


    return bin_edges


def compute_cdf(post_samples):
    """
    Compute CDF for a posterior distribution

    FIXME! really shouldn't be in this file anyway
    """

    # Set number of samples
    num_samples = len(post_samples)
    
    # Bin into histogram
    counts, bin_edges = np.histogram(post_samples, 
        bins=num_samples, density=True)
    
    # Interpolate CDF
    cdf = np.cumsum(counts)
    return scipy.interpolate.interp1d(bin_edges[:-1], cdf/cdf[-1], 
        fill_value=0, bounds_error=False)

def set_random_seed(rs):
    """
    Set random seed to the provided value. If no
    value is given, reset the random seed

    Parameters:
    -----------
    rs: int or None
        random seed
    """
    # Set random seed
    if rs is not None:
        rs = int(rs)
        rstate = np.random.RandomState(rs)
        np.random.seed(rs)
    else:
        rstate = np.random.RandomState()
        np.random.seed()

def find_nearest(array, value):
    """
    Helper function to find the index of the closest 
    value to a given value in an array. 
    Assumes the array is sorted
    """

    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or \
        np.fabs(value - array[idx-1]) < np.fabs(value - array[idx])):
        return idx-1
    else:
        return idx

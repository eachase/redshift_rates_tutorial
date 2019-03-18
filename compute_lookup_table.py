from __future__ import division

# Standard python modules
import argparse
import numpy as np


def decimal_to_string(decimal):
    decimal_string = str(decimal)
    split_string = decimal_string.split('.')
    return split_string[0], split_string[1]
            

def make_foreground_fit(mchirp, zbin):
    import fg_models
   
    # Compute exponent
    comp_mass = 2**(1/5) * mchirp
    power = zbin.fit_foreground_model(mass1=comp_mass, mass2=comp_mass)
   
    # Return normalization of PDF
    norm = fg_models.get_norm_snr_bins(zbin.z_low, zbin.z_high, zbin.z_horiz, \
        mass1=comp_mass, mass2=comp_mass, snr_min=snr_thresh, power=power, \
        mass_distr='fixed_detector')
    return norm

if __name__ == '__main__':
    import redshift_rates_tool as rt



    # Inport many arguments
    argp = argparse.ArgumentParser()

    # Manually set bin edges
    argp.add_argument('--bin-edges', default=None, help='Set in the format\
         [z1,z2,z3,z4,etc.] without spaces. Assumed to be redshift values.\
         Flag not valid for euclidean universes.')

    # Define threshold SNR of search
    argp.add_argument('-t', '--snr-thresh', default=3.5, help='Threshold SNR of search.')

    # Number of masses
    argp.add_argument('-M', '--num-masses', default=3, type=int, help='Number of masses')

    args = argp.parse_args()
 
    bin_edges = np.array([float(i) for i in args.bin_edges[1:-1].split(',')])
    num_bins = len(bin_edges) - 1
    min_z = bin_edges[0]
    max_z = bin_edges[-1]

    num_masses = int(args.num_masses)
    snr_thresh = float(args.snr_thresh)
    z_horiz = rt.various_mass_horizon(snr_thresh)


    # Descriptive filename
    filename = 'data/%ibin_minz%sp%s_maxz%sp%s_snrthresh%sp%s.npy' % (
        (num_bins,) + decimal_to_string(min_z) + decimal_to_string(max_z) +\
        decimal_to_string(snr_thresh))

    # Set up array
    lookupdata = np.zeros((num_bins, num_masses, 2))

    # For each bin
    for bin_idx, (z_low, z_high) in enumerate(zip(bin_edges, bin_edges[1:])):
        # Create RedshiftBin object
        zbin = rt.RedshiftBin(z_low, z_high, snr_thresh, optimal=False,
            mass_distr='variable', mass1=10, mass2=10, universe='lambdacdm',
            z_horiz=z_horiz, equal_mass=False, rs=170608)

        # Store array of possible masses on a grid
        min_bh_mchirp_det = rt.min_bh_mchirp_source * (1 + z_low)
        max_bh_mchirp_det = rt.max_bh_mchirp_source * (1 + z_high)
        mchirp_det_range = np.linspace(min_bh_mchirp_det, max_bh_mchirp_det, num_masses)

        # For each mass, compute the normalization factor
        for mc_idx, mchirp in enumerate(mchirp_det_range):
            lookupdata[bin_idx, mc_idx, :] = mchirp, make_foreground_fit(mchirp, zbin)


    # Save
    np.save(filename, lookupdata)










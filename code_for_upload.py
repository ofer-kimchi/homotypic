#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import nupack 
import matplotlib.pyplot as plt
import time
import pickle
import copy
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import linregress
from matplotlib.lines import Line2D
from numba import njit
from scipy.stats import ttest_ind


#%% Metaparameters
min_pal_len = 4
include_GU = True
num_subopt = 10**5

kB = 0.0019872 #units of kcal/(mol*Kelvin)
T = 310.15  # Kelvin   

colors = (['#1E88E5', '#FFC107', '#D81B60', '#004D40', '#501ACA', '#65C598', 
           '#4C1F49', '#A64ADE', '#339010'] + ['#0B3B65', '#8C6902', '#7D0E36', '#03B99A', '#8F6AE2']) * 10

folder = ''   

#%%
def flatten(l): #From StackExchange
    #Flatten a list -- given a list of sublists, concatenate the sublists into one long list
    return([item for sublist in l for item in sublist])

def save(obj, filename):
# =============================================================================
#     Save an object to a file
# =============================================================================
    if not filename[-7:] == '.pickle':
        filename = filename + '.pickle'
    with open(filename, 'wb') as handle:
        pickle.dump(obj, handle, protocol=4) #protocol 4 came out with Python version 3.4
    
def load(filename):
# =============================================================================
#     Load an object from a file
# =============================================================================
    if not filename[-7:] == '.pickle':
        filename = filename + '.pickle'
    with open(filename, 'rb') as handle:
        data = pickle.load(handle)
    return data


#%%
def is_complementary(seq1, seq2):
    if len(seq1) != len(seq2):
        return(False)
    for e1, nt1 in enumerate(seq1):
        nt2 = seq2[-e1 - 1]
        if nt1 == 'A' and not (nt2 == 'U'):
            return(False)
        if nt1 == 'C' and not (nt2 == 'G'):
            return(False)
        if nt1 == 'G' and not (nt2 == 'C' or nt2 == 'U'):
            return(False)
        if nt1 == 'U' and not (nt2 == 'A' or nt2 == 'G'):
            return(False)
    return(True)


def is_complementary_noGU(seq1, seq2):
    if len(seq1) != len(seq2):
        return(False)
    for e1, nt1 in enumerate(seq1):
        nt2 = seq2[-e1 - 1]
        if nt1 == 'A' and not (nt2 == 'U'):
            return(False)
        if nt1 == 'C' and not (nt2 == 'G'):
            return(False)
        if nt1 == 'G' and not (nt2 == 'C'):
            return(False)
        if nt1 == 'U' and not (nt2 == 'A'):
            return(False)
    return(True)


def find_palindromes(
        seq, min_len_palindrome_with_C, min_len_palindrome, 
        max_len_palindrome=500, check_all_comp=True):
    
    if check_all_comp:
        comp_fxn = is_complementary
    else:
        comp_fxn = is_complementary_noGU
        
    if min_len_palindrome_with_C > min_len_palindrome:
        print('Error: min_len_palindrome_with_C cant be larger than min_len_palindrome')
        return(False)
    
    loc_complementary_regions_identical = [[] for _ in range(max_len_palindrome + 1)]
    for len_comp in range(min_len_palindrome_with_C, max_len_palindrome + 1, 2)[::-1]:  # find longest palindromes first
        for start_nt in range(len(seq) - len_comp + 1):
            subseq = seq[start_nt : start_nt + len_comp]
            if len_comp >= min_len_palindrome or 'C' in subseq:
                if comp_fxn(subseq, subseq):
                    shorter_version_of_longer_palindrome = False
                    # Check if this palindrome is already entirely encompassed within a longer palindrome we already found
                    for longer_len_comp in range(len_comp + 2, max_len_palindrome + 1, 2):
                        for start_nt_check in range(start_nt + len_comp - longer_len_comp, start_nt + 1):
                            if start_nt_check in loc_complementary_regions_identical[longer_len_comp]:
                            #if start_nt - (longer_len_comp - len_comp)/2 in loc_complementary_regions_identical[longer_len_comp]:
                            # then this is just the middle of the same (longer) palindrome we found earlier
                                shorter_version_of_longer_palindrome = True        
                    if not shorter_version_of_longer_palindrome:
                        loc_complementary_regions_identical[len_comp] += [start_nt]

    complementary_regions_identical_summary = []
    for len_comp, comp_region_list in enumerate(loc_complementary_regions_identical):
        for comp_start in comp_region_list:
            complementary_regions_identical_summary += [(comp_start, len_comp)]
        
    complementary_regions_identical_summary = sorted(complementary_regions_identical_summary)

    num_complementary_regions_identical = np.array(
        [len(comp_regions_loc) for comp_regions_loc in loc_complementary_regions_identical],
        dtype=float)
    
    return(loc_complementary_regions_identical, complementary_regions_identical_summary, 
           num_complementary_regions_identical)
    

def loc_comp_regions(seq, min_len_comp_region, check_all_comp=True, substems=0):
    
    if check_all_comp:
        DNA = False
    else:
        seq = seq.replace('U', 'T')
        DNA = True
    
    landscapeFold_result = LandscapeFold(seq, minBPInStem=min_len_comp_region, DNA=DNA, 
                  minNtsInHairpin=-500, maxSizeSTable=10**6,noPrintWhatsoever=True,
                  substems=substems,
                  ).makeListOfStemsAndCompatibilities(makeC=False)
    
    return(landscapeFold_result.STableBPs)


#%% Functions for the design of random sequences

def designRandomSeq(len_seq):
    seq_in_numbers = np.random.choice(4, size=len_seq)
    nts = ['A', 'C', 'G', 'U']
    seq = ''
    for e in range(len_seq):
        seq += nts[seq_in_numbers[e]]
    return(seq)


#%% Smith-Waterman score calculation for sequence dissimilarity
MATCH = 1
MISMATCH = -2
GAP = -2

@njit
def smith_waterman_numba_fast(seq1, seq2, match=MATCH, mismatch=MISMATCH, gap=GAP):
    m, n = len(seq1), len(seq2)

    # DP matrix, only scores
    H = np.zeros((m+1, n+1), dtype=np.int16)

    max_score = 0

    for i in range(1, m+1):
        for j in range(1, n+1):
            s = match if seq1[i-1] == seq2[j-1] else mismatch
            diag = H[i-1, j-1] + s
            up   = H[i-1, j] + gap
            left = H[i, j-1] + gap
            val  = max(0, diag, up, left)
            H[i, j] = val
            if val > max_score:
                max_score = val

    return max_score

def smith_waterman_score_fast(seq1, seq2):
    # Convert to NumPy uint8 arrays for speed
    s1 = np.frombuffer(seq1.encode('ascii'), dtype=np.uint8)
    s2 = np.frombuffer(seq2.encode('ascii'), dtype=np.uint8)
    return smith_waterman_numba_fast(s1, s2)


#%%
def get_monomer_landscape_FE(seq,  # the sequence you want to consider; str
                             calculate_pair_probs=False,
                             calculate_subopt=False,
                             num_subopt=10000,
                             nupack_material='rna'
                             ):

    nupack_model = nupack.Model(material=nupack_material,  # 'rna', 'rna95', 'rna99-nupack3', 'rna95-nupack3' are all options. try rna99-nupack3
                                kelvin=310.15
                                )
    nupack_seq = nupack.Strand(seq, name='seq1')
    cset1 = nupack.ComplexSet(strands=[nupack_seq], complexes=nupack.SetSpec(
        max_size=1))
    
    options = {'num_sample': num_subopt}
    
    complex_analysis_result = nupack.complex_analysis(cset1, model=nupack_model, 
                                            compute=['pfunc'] + ['pairs'] * calculate_pair_probs + 
                                            ['sample'] * (calculate_subopt and num_subopt > 0) + 
                                            ['mfe'] * (num_subopt==0),
                                            #['subopt', 'mfe'] * calculate_subopt, 
                                            options=options
                                            )  
    monomer_FE = complex_analysis_result[cset1.complexes[0]].free_energy
    if calculate_subopt:
        # use sample for subopt since subopt and mfe give a small fraction of landscape in terms of boltzmann factors
        if num_subopt > 0:
            sample = complex_analysis_result[cset1.complexes[0]].sample
            if calculate_pair_probs:
                pair_probs = complex_analysis_result[cset1.complexes[0]].pairs.to_array()
                return(monomer_FE, pair_probs, sample #subopt, mfe
                       )
            return(monomer_FE, sample #subopt, mfe
                   )
        else:
            sample = complex_analysis_result[cset1.complexes[0]].mfe
            if calculate_pair_probs:
                pair_probs = complex_analysis_result[cset1.complexes[0]].pairs.to_array()
                return(monomer_FE, pair_probs, sample #subopt, mfe
                       )
            return(monomer_FE, sample #subopt, mfe
                   )
    if calculate_pair_probs:
        pair_probs = complex_analysis_result[cset1.complexes[0]].pairs.to_array()
        return(monomer_FE, pair_probs)
    return(monomer_FE)


def get_equilibrium_prob_of_free_nmers_faster(seq, n_list, num_subopt, return_all_structures=False, nupack_material='rna'):
    
    monomer_FE, sample = get_monomer_landscape_FE(seq, False, True, num_subopt=num_subopt, nupack_material=nupack_material)
    if num_subopt > 0:
        sample_array = np.array([i.pairlist() for i in sample])
    else:
        sample_array = np.array([i.structure.pairlist() for i in sample])
        num_subopt = sample_array.shape[0]
    # Make an array of shape num_subopt x len(seq) which for each structure, for each nt, 
    # gives whether that nt is unpaired (True) or paired (False)
    sample_unpaired = (sample_array == np.arange(len(seq)))
    
    free_nmers_prob = []
    for n in n_list:
        # for each stretch of n nts, determine if they are all unpaired in this structure
        free_nmers_per_structure = np.prod(
            np.array([sample_unpaired[:, i:-n+1+i] for i in range(n-1)] + 
                     [sample_unpaired[:, n-1:]]
                     ), 0)
        if not return_all_structures:
            free_nmers_prob += [np.sum(free_nmers_per_structure, 0) / num_subopt]
        else:
            free_nmers_prob += [free_nmers_per_structure]
    
    return(free_nmers_prob)

#%%
def bondFreeEnergiesRNARNA():
# =============================================================================
#     #define the RNA/RNA bond enthalpy and entropy arrays
#     
#     #Sources: Table 4 of Xia et al Biochemistry '98
#         #Table 4 of Mathews et al. JMB '99
#         #Table 3 of Xia, Mathews, Turner "Thermodynamics of RNA secondary structure formation" in book by Soll, Nishmura, Moore
#     #First index tells you if the first bp of the set is AU (0) CG (1) GC (2) UA (3) GU (4) or UG (5)
#     #Second index tells you if the 3' ntd of the second bp is A (0) C (1) G(2) or U(3) (which row of table 1 in Serra & Turner).
#     #Third index tells you if the 5' ntd of the second bp is A (0) C (1) G(2) or U(3) (which column of table 1 in Serra & Turner).
#     
# #    Had to make this 2D array to be able to use scipy sparse functions, so second/third index is replaced
# #    by (4*second index) + third index
# =============================================================================
    
    bondEnergyMatrixRNARNA = np.array([[-3.9, 2, -3.5, -6.82, -2.3, 6, -11.4, -0.3, -3.1, -10.48, -3.5, -3.21, -9.38, 4.6, -8.81, -1.7],
                                       [-9.1, -5.6, -5.6, -10.44, -5.7, -3.4, -13.39, -2.7, -8.2, -10.64, -9.2, -5.61, -10.48, -5.3, -12.11, -8.6],
                                       [-5.2, -4, -5.6, -12.44, -7.2, 0.5, -14.88, -4.2, -7.1, -13.39, -6.2, -8.33, -11.4, -0.3, -12.59, -5],
                                       [-4, -6.3, -8.9, -7.69, -4.3, -5.1, -12.44, -1.8, -3.8, -10.44, -8.9, -6.99, -6.82, -1.4, -12.83, 1.4],
                                       [3.4, 2, -3.5, -12.83, -2.3, 6, -12.59, -0.3, 0.6, -12.11, -3.5, -13.47, -8.81, 4.6, -14.59, -1.7],
                                       [-4.8, -6.3, -8.9, -6.99, -4.3, -5.1, -8.33, -1.8, -3.1, -5.61, 1.5, -9.26, -3.21, -1.4, -13.47, 1.4]])
    #matrix of enthalpies (deltaH). Units are kcal/mol.
    
    bondEntropyMatrixRNARNA = np.array([[-10.2, 9.6, -8.7, -19, -5.3, 21.6, -29.5, 1.5, -7.3, -27.1, -8.7, -8.6, -26.7, 17.4, -24, -2.7],
                                        [-24.5, -13.5, -13.4, -26.9, -15.2, -7.6, -32.7, -6.3, -21.8, -26.7, -24.6, -13.5, -27.1, -12.6, -32.2, -23.9],
                                        [-13.2, -8.2, -13.9, -32.5, -19.6, 3.9, -36.9, -12.2, -17.8, -32.7, -15.1, -21.9, -29.5, -2.1, -32.5, -14],
                                        [-9.7, -17.1, -25.2, -20.5, -11.6, -14.6, -32.5, -4.2, -8.5, -26.9, -25, -19.3, -19, -2.5, -37.3, 6],
                                        [10, 9.6, -8.7, -37.3, -5.3, 21.6, -32.5, 1.5, 0, -32.2, -8.7, -44.9, -24, 17.4, -51.2, -2.7],
                                        [-12.1, -17.7, -25.2, -19.3, -11.6, -14.6, -21.9, -4.2, -11.2, -13.5, 2.1, -30.8, -8.6, -2.5, -44.9, 6]])
    #matrix of entropies (deltaS). Units are initially eu, but then converted to kcal/(mol*K).
     
    bondEntropyMatrixRNARNA /= 1000 #to convert from eu (entropy units) to kcal/(mol*K)

#    bondFreeEnergyMatrixRNARNA = np.zeros((6,4,4)) #matrix of stacking energies (deltaG). Units are kcal/mol.    
#    bondFreeEnergyMatrixRNARNA[0,:,:] = [[-0.8, -1.0, -0.8, -0.93], [-0.6, -0.7, -2.24, -0.7], [-0.8, -2.08, -0.8, -0.55], [-1.10, -0.8, -1.36, -0.8]]
#    bondFreeEnergyMatrixRNARNA[1,:,:] = [[-1.5, -1.5, -1.4, -2.11], [-1.0, -1.1, -3.26, -0.8], [-1.4, -2.36, -1.6, -1.41], [-2.08, -1.4, -2.11, -1.2]]
#    bondFreeEnergyMatrixRNARNA[2,:,:] = [[-1.1, -1.5, -1.3, -2.35], [-1.1, -0.7, -3.42, -0.5], [-1.6, -3.26, -1.4, -1.53], [-2.24, -1.0, -2.51, -0.7]]
#    bondFreeEnergyMatrixRNARNA[3,:,:] = [[-1.0, -0.8, -1.1, -1.33], [-0.7, -0.6, -2.35, -0.5], [-1.1, -2.11, -1.2, -1.00], [-0.93, -0.6, -1.27, -0.5]]
#    bondFreeEnergyMatrixRNARNA[4,:,:] = [[0.3, -1.0, -0.8, -1.27], [-0.6, -0.7, -2.51, -0.7], [0.6, -2.11, -0.8, -0.500], [-1.36, -0.8, 1.29, -0.8]] 
#    bondFreeEnergyMatrixRNARNA[5,:,:] = [[-1.0, -0.8, -1.1, -1.00], [-0.7, -0.6, -1.53, -0.5], [0.5, -1.41, 0.8, 0.30], [-0.55, -0.6, -0.500, -0.5]]
#    #the -0.500 was actually measured at +0.47 but the authors claim -0.5 is a better estimate.
    
    return(bondEnergyMatrixRNARNA, bondEntropyMatrixRNARNA)


bondEnergyMatrixRNARNA, bondEntropyMatrixRNARNA = bondFreeEnergiesRNARNA()
bondFEMatrixRNARNA = bondEnergyMatrixRNARNA - T * bondEntropyMatrixRNARNA
bond_dict_1 = {('A', 'U'): 0, ('C', 'G'): 1, ('G', 'C'): 2, ('U', 'A'): 3, ('G', 'U'): 4, ('U', 'G'): 5}
bond_dict_2 = {'A': 0, 'C': 1, 'G': 2, 'U': 3}


def build_bond_FE_sum_matrix(a):  # From ChatGPT; verified.
    a = np.asarray(a)
    n = len(a)
    cumsum = np.concatenate(([0], np.cumsum(a)))
    
    # Create all ranges of (i, j) to compute cumsum[j] - cumsum[i]
    starts = np.arange(n).reshape(-1, 1)
    ends = np.arange(n) + 1
    row_idx, col_idx = np.triu_indices(n)
    starts = row_idx
    ends = n - col_idx + row_idx

    M = np.zeros((n, n))
    M[row_idx, col_idx] = cumsum[ends] - cumsum[starts]
    return M


def get_bond_FE_matrix_of_substems(stem_L, stem_R):
    # seq1 and seq2 should be the same length and complementary. For convenience, we input stem_R reversed.
    len_seq = len(stem_L)
    if len_seq != len(stem_R):
        print('Sequences need to be the same length')
        return('ERROR')

    # ignore terminal mismatched nts, dangling ends, etc. for this analysis
    bond_list = [(bond_dict_1[(stem_L[i], stem_R[i])], 
                  bond_dict_2[stem_L[i + 1]], 
                  bond_dict_2[stem_R[i + 1]]) for i in range(len_seq - 1)]
    nn_FE_vec = np.array([bondFEMatrixRNARNA[i, 4*j + k] for i, j, k in bond_list]) # vector of free energies from nn model
    
    nn_FE_mat = build_bond_FE_sum_matrix(nn_FE_vec)
    return(nn_FE_mat)


def get_stem_Z(stem_L, stem_R, stem_L_start, stem_R_start, free_nmers_L, free_nmers_R=None, minBPInStem=2):
    # stem_R is inputted reversed
    if free_nmers_R is None:
        free_nmers_R = free_nmers_L
    nn_FE_mat = get_bond_FE_matrix_of_substems(stem_L, stem_R)
    if stem_L == stem_R[::-1] and stem_L_start == stem_R_start:  # then it's a palindrome
    # this matters since we don't want to double-count the subsequences of a palindrome, since they're equivalent by symmetry
        pal = True
    else:
        pal = False
    
    len_stem = len(stem_L)
    if minBPInStem > 2:
        nn_FE_mat = nn_FE_mat[:-minBPInStem + 2, :-minBPInStem + 2]
    stem_FE = 0
    stem_Zs = []
    stem_free_probs = []
    for i in range(nn_FE_mat.shape[0]):  # i is analogous to 'substems': i.e. i=0 is longest substem; i=1 is substem 1 nt shorter, etc.
        FEs_of_substems = nn_FE_mat[:i+1, i]
        Zs_of_substems = np.exp(-FEs_of_substems / (kB * T))
        prob_substems_free_L = free_nmers_L[len_stem - i - minBPInStem][stem_L_start : stem_L_start + i + 1] 
        prob_substems_free_R = (free_nmers_R[len_stem - i - minBPInStem][stem_R_start : stem_R_start + i + 1])[::-1]
        
        if pal:
            Zs_of_substems = Zs_of_substems[:(len(Zs_of_substems) + 1)//2]
            prob_substems_free_L = prob_substems_free_L[:(len(prob_substems_free_L) + 1)//2]
            prob_substems_free_R = prob_substems_free_R[:(len(prob_substems_free_R) + 1)//2]
        stem_Zs += [Zs_of_substems]
        stem_free_probs += [prob_substems_free_L * prob_substems_free_R]
        stem_FE += np.sum(Zs_of_substems * prob_substems_free_L * prob_substems_free_R)
    return(stem_FE, stem_Zs, stem_free_probs)

#%%
def get_dimer_landscape_FE(
        seqs,  # A list of the two sequences you want to consider
        return_monomer_FEs=False,  # if True, also returns the free energies of the monomers
        return_all_dimer_FEs=False,  # if True, also returns the free energies of the homodimers
        compute=['pfunc'],  # what properties to compute
        ):
    
    if len(seqs) != 2:
        print('The code is currently only set up to handle 2 sequences')
        return(0.)

    if seqs[0] == seqs[1]:  # Then we are actually considering homodimers
        return_all_dimer_FEs = False  # because all dimers are the same
        
    nupack_model = nupack.Model(material='rna', 
                                kelvin=310.15,
                                # ensemble='nostacking'
                                )
    nupack_seqs = []
    for e, seq in enumerate(seqs):
        nupack_seqs += [nupack.Strand(seq, name='seq' + str(e+1))]

    if not return_monomer_FEs:
        to_exclude = [
            nupack.Complex([nupack_seqs[0]], name='(seq1)'),
            nupack.Complex([nupack_seqs[1]], name='(seq2)')
            ]
    elif seqs[0] == seqs[1]:
        to_exclude = [nupack.Complex([nupack_seqs[1]], name='(seq2)')]
    else:
        to_exclude = []

    if not return_all_dimer_FEs:  # only consider the dimer we're interested in; exclude the rest
        to_exclude += [
            nupack.Complex([nupack_seqs[1], nupack_seqs[1]], name='(seq2+seq2)')]
        if seqs[0] != seqs[1]:  # Then we care about the heterodimer
            to_exclude += [nupack.Complex([nupack_seqs[0], nupack_seqs[0]], name='(seq1+seq1)')]
        else:  # Then we care about the homodimer
            to_exclude += [nupack.Complex([nupack_seqs[0], nupack_seqs[1]], name='(seq1+seq2)')]
        
    cset1 = nupack.ComplexSet(strands=nupack_seqs, complexes=nupack.SetSpec(
        max_size=2, exclude=to_exclude))
    complex_analysis_result = nupack.complex_analysis(cset1, model=nupack_model, 
                                            compute=compute)  # , 'mfe'

    if not return_all_dimer_FEs:
        homodimer_1_name = '(seq1+seq1)'
        monomer_1_name = '(seq1)'
        if seqs[0] == seqs[1]:  # Then we are actually considering homodimers
            if compute==['pfunc']:
                homodimer_1_FE = complex_analysis_result[homodimer_1_name][2]  # [1] is the partition function; [2] is the free energy
                if return_monomer_FEs:
                    monomer_1_FE = complex_analysis_result[monomer_1_name][2]
                    return(monomer_1_FE, monomer_1_FE, homodimer_1_FE)
                else:
                    return(homodimer_1_FE)
            else:
                if return_monomer_FEs:
                    return(complex_analysis_result[monomer_1_name], 
                           complex_analysis_result[monomer_1_name], 
                           complex_analysis_result[homodimer_1_name])                
                else:
                    return(complex_analysis_result[homodimer_1_name])
        else:
            heterodimer_name = [i.name for i in cset1 if 
                                'seq1' in i.name and 'seq2' in i.name][0]  # can be either '(seq1+seq2)' or '(seq2+seq1)'
            monomer_2_name = '(seq2)'
            if compute==['pfunc']:
                heterodimer_FE = complex_analysis_result[heterodimer_name][2]
                if return_monomer_FEs:
                    monomer_1_FE = complex_analysis_result[monomer_1_name][2]
                    monomer_2_FE = complex_analysis_result[monomer_2_name][2]
                    return(monomer_1_FE, monomer_2_FE, heterodimer_FE)
                else:
                    return(heterodimer_FE)
            else:
                if return_monomer_FEs:
                    return(complex_analysis_result[monomer_1_name],
                           complex_analysis_result[monomer_2_name],
                           complex_analysis_result[heterodimer_name])                
                else:
                    return(complex_analysis_result[heterodimer_name])

    homodimer_1_name = '(seq1+seq1)'
    homodimer_2_name = '(seq2+seq2)'
    heterodimer_name = [i.name for i in cset1 if 
                        'seq1' in i.name and 'seq2' in i.name][0]  # can be either '(seq1+seq2)' or '(seq2+seq1)'    
    monomer_1_name = '(seq1)'
    monomer_2_name = '(seq2)'
    if compute==['pfunc']:
        homodimer_1_FE = complex_analysis_result[homodimer_1_name][2]  # [1] is the partition function; [2] is the free energy
        homodimer_2_FE = complex_analysis_result[homodimer_2_name][2]
        heterodimer_FE = complex_analysis_result[heterodimer_name][2]
        if return_monomer_FEs:
            monomer_1_FE = complex_analysis_result[monomer_1_name][2]
            monomer_2_FE = complex_analysis_result[monomer_2_name][2]
            return(monomer_1_FE, monomer_2_FE, heterodimer_FE, 
                   homodimer_1_FE, homodimer_2_FE)
        else:
            return(heterodimer_FE, homodimer_1_FE, homodimer_2_FE)
    else:
        if return_monomer_FEs:
            return(complex_analysis_result[monomer_1_name],
                   complex_analysis_result[monomer_2_name],
                   complex_analysis_result[heterodimer_name], 
                   complex_analysis_result[homodimer_1_name],
                   complex_analysis_result[homodimer_2_name])
        else:
            return(complex_analysis_result[heterodimer_name], 
                   complex_analysis_result[homodimer_1_name],
                   complex_analysis_result[homodimer_2_name])


def get_multimer_landscape(seq, max_complex_size=5, conc=1e-6):

    nupack_model = nupack.Model(material='rna', kelvin=310.15#, ensemble='nostacking'
                            )
    nupack_seq = nupack.Strand(seq, name='seq')
    
    tube = nupack.Tube({nupack_seq: conc}, complexes=nupack.SetSpec(max_size=max_complex_size), name='tube')
    
    tube_analysis_result = nupack.tube_analysis([tube], model=nupack_model, 
                                                # compute=['pfunc']  # , 'mfe'
                                                )

    concs = tube_analysis_result.tubes[tube].complex_concentrations
    
    concs_list = [concs[nupack.Complex([nupack_seq]*i)] for i in range(1, max_complex_size + 1)]
    
    return(concs_list)  # return the list of concs


#%% 
def binding_to_free_regions(seq, num_subopt=10000, min_bp_in_stem=6, max_bp_in_stem=12, check_all_comp=True,
                            substems=0, free_nmers=None, STableBPs=None, pal_bps=None, print_times=True):
    # Given a sequence, get the self-binding partition function by summing all the possible regions of binding
    start = time.time()
    if free_nmers is None:
        free_nmers = get_equilibrium_prob_of_free_nmers_faster(
            seq, range(min_bp_in_stem, max_bp_in_stem + 1), num_subopt=num_subopt)
        if print_times:
            print(time.time() - start)

    if STableBPs is None:
        STableBPs = loc_comp_regions(seq, min_len_comp_region=min_bp_in_stem, 
                                     check_all_comp=check_all_comp, substems=substems)
        if print_times:
            print(time.time() - start)

    if pal_bps is None:
        pal_locs, pal_summary, _ = find_palindromes(
            seq, 
            min_len_palindrome_with_C=min_bp_in_stem, 
            min_len_palindrome=min_bp_in_stem, 
            max_len_palindrome=500, 
            check_all_comp=check_all_comp
            )
        pal_bps = [[] for _ in range(len(pal_summary))]
        for e, (pal_start, pal_len) in enumerate(pal_summary):
            pal_bps[e] = np.array([[pal_start + i, pal_start + pal_len - 1 - i] for i in range(pal_len)])

    stem_Zs = np.zeros(len(STableBPs))
    stem_Zs_indiv = [[] for _ in range(len(STableBPs))]
    stem_free_probs_indiv = [[] for _ in range(len(STableBPs))]
    palindromes = np.zeros(len(STableBPs))
    stem_free_probs = np.zeros(len(STableBPs))
    for e, stem in enumerate(STableBPs):
        stem_array = np.array(stem, dtype=int)
        len_stem = stem_array.shape[0]
        
        if len_stem <= max_bp_in_stem:
            stem_L_indices = stem_array[:, 0]
            stem_R_indices = stem_array[:, 1]
            
            stem_free_prob_L = free_nmers[len_stem - min_bp_in_stem][np.min(stem_L_indices)]
            stem_free_prob_R = free_nmers[len_stem - min_bp_in_stem][np.min(stem_R_indices)]
            stem_free_prob = stem_free_prob_L * stem_free_prob_R
            stem_free_probs[e] = stem_free_prob
            
            # stem_FE = get_dimer_landscape_FE(  # if using nupack, should use substems=0 since nupack automatically accounts for substems in FE
            #     [''.join([seq[i] for i in stem_L_indices]), 
            #      ''.join([seq[i] for i in stem_R_indices])
            #      ])/(kB * T)
            
            # stem_Zs[e] = np.exp(-stem_FE) * stem_free_prob

            stem_Zs[e], stem_Zs_indiv[e], stem_free_probs_indiv[e] = get_stem_Z(
                ''.join([seq[i] for i in stem_L_indices]), 
                ''.join([seq[i] for i in stem_R_indices]), 
                np.min(stem_L_indices), 
                np.min(stem_R_indices), 
                free_nmers,
                minBPInStem=min_bp_in_stem)
                        
            # if len(set(stem_L).intersection(set(stem_R))) > 0:
            # Include in palindromes stems that have their left side from one palindrome and their right side from another
            # (i.e. mutually complementary palindromes binding to one another)
            if (any([all([i in j[:, 0] for i in stem_L_indices]) for j in pal_bps]) and
                any([all([i in j[:, 0] for i in stem_R_indices]) for j in pal_bps])):
                palindromes[e] = 1
        
    if print_times:
        print(time.time() - start)
    return(stem_Zs, palindromes, np.sum(stem_Zs), np.sum(stem_Zs * palindromes), stem_free_probs, free_nmers, STableBPs, 
           stem_Zs_indiv, stem_free_probs_indiv)


def calculate_stem_H(stem_Zs):
    stem_Zs = stem_Zs[stem_Zs != 0]
    stem_Z = np.sum(stem_Zs)
    return(-np.sum(stem_Zs / stem_Z * np.log(stem_Zs / stem_Z)))
    # return(-np.nansum(stem_Zs / stem_Z * np.log(stem_Zs / stem_Z)))


#%% 
def make_seq_in_numbers(seq):  # From chatGPT; verified
    # Convert string to np array of bytes
    b = np.fromiter((ord(c) for c in seq), dtype=int)
    
    # Create lookup table
    lookup = np.full(256, -1, dtype=int)
    lookup[ord('A')] = 0
    lookup[ord('C')] = 1
    lookup[ord('G')] = 2
    lookup[ord('U')] = 3
    return lookup[b]


bondFEMatrixRNARNA_full = np.zeros((4, 4, 4, 4))
bondEMatrixRNARNA_full = np.zeros((4, 4, 4, 4))
if include_GU: 
    comp_fxn = is_complementary
else:
    comp_fxn = is_complementary_noGU
for i1 in range(4):
    for i2 in range(4):
        for j1 in range(4): 
            for j2 in range(4):
                if comp_fxn('ACGU'[i1], 'ACGU'[j1]) and comp_fxn('ACGU'[i2], 'ACGU'[j2]):
                    if i1 == 0 and j1 == 3:
                        k1 = 0
                    elif i1 == 1 and j1 == 2:
                        k1 = 1
                    elif i1 == 2 and j1 == 1:
                        k1 = 2
                    elif i1 == 3 and j1 == 0:
                        k1 = 3
                    elif i1 == 2 and j1 == 3:
                        k1 = 4
                    elif i1 == 3 and j1 == 2:
                        k1 = 5
                    else:      
                        print('ERROR in bondFEMatrixRNARNA_full calculation')
                    bondFEMatrixRNARNA_full[i1, i2, j1, j2] = bondFEMatrixRNARNA[k1, 4*i2 + j2]
                    bondEMatrixRNARNA_full[i1, i2, j1, j2] = bondEnergyMatrixRNARNA[k1, 4*i2 + j2]
                else:
                    bondFEMatrixRNARNA_full[i1, i2, j1, j2] = np.inf
                    bondEMatrixRNARNA_full[i1, i2, j1, j2] = np.inf


def binding_FEs_of_subseqs_faster(subseq1, subseq2, use_energy=False):
    # gives an array of nn values for the binding between subseq1 (a sequence in numbers) and subseq2 (same)
    # subseq2 is reversed, so its first element binds to the first element of subseq1, etc.
    a1 = subseq1[:-1]
    a2 = subseq1[1:]
    b1 = subseq2[:-1]
    b2 = subseq2[1:]
    if use_energy:
        return bondEMatrixRNARNA_full[a1, a2, b1, b2]
    return bondFEMatrixRNARNA_full[a1, a2, b1, b2]


def is_palindrome(seq_in_numbers):
    pal_FE = binding_FEs_of_subseqs_faster(seq_in_numbers, seq_in_numbers[::-1])
    if np.isinf(pal_FE).any():
        return(False)
    return(True)


def binding_FEs_of_subseqs(subseq1, subseq2, use_energy=False):
    # subseq1 and subseq2 are sequences in numbers of the same length, 
    # with subseq2 reversed so that its first element binds to the first element of subseq1, etc.
    return(binding_FEs_of_subseqs_faster(subseq1, subseq2, use_energy=use_energy))


#%%
def binding_to_all_regions(seq1, seq2, free_nmers1, free_nmers2, min_n=4, max_n=20, correct_self_binding=True,
                           inputted_seq_in_numbers=False, use_prob_diss=False, prob_diss_tau_A=1e10,
                           only_consider_palindromes=False, pal_summary_1=None, pal_summary_2=None  # pal_summaries only used if only_consider_palindromes==True
                           ):
    # Given two sequences, slides them against one another and calculates all the n-mer partition functions that arise from this sliding.    
    
    len_a, len_b = len(seq1), len(seq2)
    
    ns = np.arange(min_n, max_n + 1)
    Z_n = np.zeros(len(ns))
    Zs_n = []

    if inputted_seq_in_numbers:
        seq1_in_numbers = seq1
        seq2_in_numbers = seq2[::-1]
    else:
        seq1_in_numbers = make_seq_in_numbers(seq1)
        seq2_in_numbers = make_seq_in_numbers(seq2)[::-1]  # take the reverse for convenience

    if only_consider_palindromes:
        if pal_summary_1 is None:
            _, pal_summary_1, _ = find_palindromes(
                seq1, 
                min_len_palindrome_with_C=min_pal_len, 
                min_len_palindrome=min_pal_len, 
                max_len_palindrome=500, 
                check_all_comp=include_GU
                )
        if pal_summary_2 is None:
            _, pal_summary_2, _ = find_palindromes(
                seq2, 
                min_len_palindrome_with_C=min_pal_len, 
                min_len_palindrome=min_pal_len, 
                max_len_palindrome=500, 
                check_all_comp=include_GU
                )
        pal_locs_1 = [np.zeros(max(0, len(seq1) - n + 1)) for n in ns]
        pal_locs_2 = [np.zeros(max(0, len(seq2) - n + 1)) for n in ns]
        ns_list = [n for n in ns]
        for pal_start_1, pal_len_1 in pal_summary_1:
            if pal_len_1 in ns_list:
                pal_locs_1[ns_list.index(pal_len_1)][pal_start_1] = 1
        for pal_start_2, pal_len_2 in pal_summary_2:
            if pal_len_2 in ns_list:
                pal_locs_2[ns_list.index(pal_len_2)][pal_start_2] = 1
            
    
    binding_FEs_lists = []
    a_start_lists = []
    b_start_lists = []
    
    for shift in range(-len_b + 2, len_a-1):  # need each subseq to be at least length 2
        # Compute overlap indices
        a_start = max(0, shift)
        b_start = max(0, -shift)
        overlap_len = min(len_a - a_start, len_b - b_start)
        
        if overlap_len > 0:
            overlap_a = seq1_in_numbers[a_start:a_start + overlap_len]
            overlap_b = seq2_in_numbers[b_start:b_start + overlap_len]
            binding_FEs_lists.append(binding_FEs_of_subseqs(
                overlap_a, overlap_b, use_energy=False))
            a_start_lists.append(a_start)
            b_start_lists.append(b_start)
    # print(binding_FEs_lists)
   
    for e_n, n in enumerate(ns):
        if n <= len_a and n <= len_b:
            lo = n - 2
            hi = -(n - 2) if (n - 2) != 0 else None

            binding_FEs_lists_n = binding_FEs_lists[lo : hi]
            a_start_lists_n = a_start_lists[lo : hi]
            b_start_lists_n = b_start_lists[lo : hi]
            
            if seq1 == seq2 and correct_self_binding:  # we're double-counting non-palindromic stems, so we need to only consider half of each sliding window
                first_half = lambda x: x[:(len(x)+1)//2]
            else:
                first_half = lambda x: x
                
            free_nmers1_n = np.concatenate([first_half(
                free_nmers1[n - min_n][a_start : a_start + len(binding_FEs_lists[n - 2 + e]) - n + 2])
                for e, a_start in enumerate(a_start_lists_n)])
    
            free_nmers2_n = np.concatenate([first_half((free_nmers2[n - min_n]
                [len_b - b_start - len(binding_FEs_lists[n - 2 + e]) - 1: len_b - b_start - n + 1])[::-1])
                for e, b_start in enumerate(b_start_lists_n)])

            if only_consider_palindromes:
                pal_locs_1_n = np.concatenate([first_half(
                    pal_locs_1[n - min_n][a_start : a_start + len(binding_FEs_lists[n - 2 + e]) - n + 2])
                    for e, a_start in enumerate(a_start_lists_n)])
                pal_locs_2_n = np.concatenate([first_half(
                    pal_locs_2[n - min_n][a_start : a_start + len(binding_FEs_lists[n - 2 + e]) - n + 2])
                    for e, a_start in enumerate(a_start_lists_n)])
                
            # faster version
            out = []
            for arr in binding_FEs_lists_n:
                # Use sliding window view for efficient summing
                if len(arr) >= n - 1:
                    windows = sliding_window_view(arr, n - 1)
                    out2 = windows.sum(axis=1)
                else:
                    out2 = np.array([])
        
                if seq1 == seq2 and correct_self_binding:
                    out2 = out2[: (len(out2) + 1) // 2]
        
                out.append(out2)
        
            if out:
                FEs = np.concatenate(out)
            else:
                FEs = np.array([])


            if use_prob_diss:
                FEs_Z = np.exp(-prob_diss_tau_A * np.exp(FEs/(kB * T))) #/ (len_a * len_b)  
                # FEs come here with + sign because \Delta G for dissociation is measured in opposite direction
            else:
                FEs_Z = np.exp(-FEs/(kB * T))
            
            if only_consider_palindromes:
                Zs_n += [free_nmers1_n * free_nmers2_n * FEs_Z * pal_locs_1_n * pal_locs_2_n]
            else:
                Zs_n += [free_nmers1_n * free_nmers2_n * FEs_Z]
            # if use_prob_diss:
            #     Z_n[e_n] = 1 - np.prod(1 - Zs_n[-1])
            # else:
            Z_n[e_n] = np.sum(Zs_n[-1])

    return(Z_n, Zs_n)
        
    
#%%
def binding_to_all_regions_and_return_indiv(
        seq1, seq2, free_nmers1, free_nmers2, min_n=4, max_n=20, 
        correct_self_binding=True, inputted_seq_in_numbers=False, 
        only_consider_palindromes=False, pal_summary_1=None, pal_summary_2=None  # pal_summaries only used if only_consider_palindromes==True
        ):
    # Given two sequences, slides them against one another and calculates all the n-mer partition functions that arise from this sliding.    
    # Also return the STable equivalent
    
    len_a, len_b = len(seq1), len(seq2)
    
    ns = np.arange(min_n, max_n + 1)
    FEs_list_n = []
    free_nmers1_list_n = []
    free_nmers2_list_n = []
    seq1_positions_list_n = []
    seq2_positions_list_n = []

    if inputted_seq_in_numbers:
        seq1_in_numbers = seq1
        seq2_in_numbers = seq2[::-1]
    else:
        seq1_in_numbers = make_seq_in_numbers(seq1)
        seq2_in_numbers = make_seq_in_numbers(seq2)[::-1]  # take the reverse for convenience

    if only_consider_palindromes:
        if pal_summary_1 is None:
            _, pal_summary_1, _ = find_palindromes(
                seq1, 
                min_len_palindrome_with_C=min_pal_len, 
                min_len_palindrome=min_pal_len, 
                max_len_palindrome=500, 
                check_all_comp=include_GU
                )
        if pal_summary_2 is None:
            _, pal_summary_2, _ = find_palindromes(
                seq2, 
                min_len_palindrome_with_C=min_pal_len, 
                min_len_palindrome=min_pal_len, 
                max_len_palindrome=500, 
                check_all_comp=include_GU
                )
        pal_locs_1 = [np.zeros(max(0, len(seq1) - n + 1)) for n in ns]
        pal_locs_2 = [np.zeros(max(0, len(seq2) - n + 1)) for n in ns]
        ns_list = [n for n in ns]
        for pal_start_1, pal_len_1 in pal_summary_1:
            if pal_len_1 in ns_list:
                pal_locs_1[ns_list.index(pal_len_1)][pal_start_1] = 1
        for pal_start_2, pal_len_2 in pal_summary_2:
            if pal_len_2 in ns_list:
                pal_locs_2[ns_list.index(pal_len_2)][pal_start_2] = 1
            
    
    binding_FEs_lists = []
    a_start_lists = []
    b_start_lists = []
    
    for shift in range(-len_b + 2, len_a-1):  # need each subseq to be at least length 2
        # Compute overlap indices
        a_start = max(0, shift)
        b_start = max(0, -shift)
        overlap_len = min(len_a - a_start, len_b - b_start)
        
        if overlap_len > 0:
            overlap_a = seq1_in_numbers[a_start:a_start + overlap_len]
            overlap_b = seq2_in_numbers[b_start:b_start + overlap_len]
            binding_FEs_lists.append(binding_FEs_of_subseqs(
                overlap_a, overlap_b, checking=False, use_energy=False))
            a_start_lists.append(a_start)
            b_start_lists.append(b_start)
   
    for e_n, n in enumerate(ns):
        if n <= len_a and n <= len_b:
            lo = n - 2
            hi = -(n - 2) if (n - 2) != 0 else None

            binding_FEs_lists_n = binding_FEs_lists[lo : hi]
            a_start_lists_n = a_start_lists[lo : hi]
            b_start_lists_n = b_start_lists[lo : hi]
            
            if seq1 == seq2 and correct_self_binding:  # we're double-counting non-palindromic stems, so we need to only consider half of each sliding window
                first_half = lambda x: x[:(len(x)+1)//2]
            else:
                first_half = lambda x: x
                
            free_nmers1_n = np.concatenate([first_half(
                free_nmers1[n - min_n][a_start : a_start + len(binding_FEs_lists[n - 2 + e]) - n + 2])
                for e, a_start in enumerate(a_start_lists_n)])
    
            free_nmers2_n = np.concatenate([first_half((free_nmers2[n - min_n]
                [len_b - b_start - len(binding_FEs_lists[n - 2 + e]) - 1: len_b - b_start - n + 1])[::-1])
                for e, b_start in enumerate(b_start_lists_n)])

            seq1_positions_n = np.concatenate([first_half(
                np.arange(a_start, a_start + len(binding_FEs_lists[n - 2 + e]) - n + 2))
                for e, a_start in enumerate(a_start_lists_n)])
            
            seq2_positions_n = np.concatenate([first_half(
                np.arange(len_b - b_start - len(binding_FEs_lists[n - 2 + e]) - 1, len_b - b_start - n + 1)[::-1])
                for e, b_start in enumerate(b_start_lists_n)])

            if only_consider_palindromes:
                pal_locs_1_n = np.concatenate([first_half(
                    pal_locs_1[n - min_n][a_start : a_start + len(binding_FEs_lists[n - 2 + e]) - n + 2])
                    for e, a_start in enumerate(a_start_lists_n)])
                pal_locs_2_n = np.concatenate([first_half(
                    pal_locs_2[n - min_n][a_start : a_start + len(binding_FEs_lists[n - 2 + e]) - n + 2])
                    for e, a_start in enumerate(a_start_lists_n)])
                
            out = []
            for arr in binding_FEs_lists_n:
                # Use sliding window view for efficient summing
                if len(arr) >= n - 1:
                    windows = sliding_window_view(arr, n - 1)
                    out2 = windows.sum(axis=1)
                else:
                    out2 = np.array([])
        
                if seq1 == seq2 and correct_self_binding:
                    out2 = out2[: (len(out2) + 1) // 2]
        
                out.append(out2)
        
            if out:
                FEs = np.concatenate(out)
            else:
                FEs = np.array([])

            if only_consider_palindromes:
                stem_mask = (FEs < np.inf) & (pal_locs_1_n == 1) & (pal_locs_2_n == 1)
            else:
                stem_mask = (FEs < np.inf)
            FEs_list_n += [FEs[stem_mask]]
            free_nmers1_list_n += [free_nmers1_n[stem_mask]]
            free_nmers2_list_n += [free_nmers2_n[stem_mask]]
            seq1_positions_list_n += [seq1_positions_n[stem_mask]]
            seq2_positions_list_n += [seq2_positions_n[stem_mask]]


    return(FEs_list_n, free_nmers1_list_n, free_nmers2_list_n, seq1_positions_list_n, seq2_positions_list_n)
   
    
def group_stems_by_substems(FEs_list_n, free_nmers1_list_n, free_nmers2_list_n, seq1_positions_list_n, seq2_positions_list_n, 
                            min_n=4, max_n=20):
    seq1_positions_list_groupedby_stems = []
    seq2_positions_list_groupedby_stems = []
    FEs_list_groupedby_stems = []
    free_nmers1_list_groupedby_stems = []
    free_nmers2_list_groupedby_stems = []
    stems_used = set()  # O(1) lookups

    pal_lengths = list(range(min_n, max_n + 1))
    for e_n in reversed(range(len(pal_lengths))):
        seq1_e_n = seq1_positions_list_n[e_n]
        seq2_e_n = seq2_positions_list_n[e_n]
        FE_e_n = FEs_list_n[e_n]
        free1_e_n = free_nmers1_list_n[e_n]
        free2_e_n = free_nmers2_list_n[e_n]

        for e, (seq1_elem, seq2_elem, FE_elem, free1_elem, free2_elem) in enumerate(
                zip(seq1_e_n, seq2_e_n, FE_e_n, free1_e_n, free2_e_n)):
            if (e_n, e) in stems_used:
                continue

            stems_used.add((e_n, e))
            seq1_group = [seq1_elem]
            seq2_group = [seq2_elem]
            FE_group = [FE_elem]
            free1_group = [free1_elem]
            free2_group = [free2_elem]

            for e_n_2 in range(min_n, e_n):
                seq1_e_n2 = seq1_positions_list_n[e_n_2]
                seq2_e_n2 = seq2_positions_list_n[e_n_2]
                FE_e_n2 = FEs_list_n[e_n_2]
                free1_e_n2 = free_nmers1_list_n[e_n_2]
                free2_e_n2 = free_nmers2_list_n[e_n_2]

                diff_len = e_n - e_n_2
                for i in range(diff_len + 1):
                    mask = ((seq1_e_n2 == seq1_elem + i) &
                            (seq2_e_n2 == seq2_elem + diff_len - i))

                    if not np.any(mask):  # not really necessary
                        continue

                    index_of_substem = np.nonzero(mask)[0]
                    stems_used.update((e_n_2, idx) for idx in index_of_substem)

                    seq1_group.extend(seq1_e_n2[index_of_substem])
                    seq2_group.extend(seq2_e_n2[index_of_substem])
                    FE_group.extend(FE_e_n2[index_of_substem])
                    free1_group.extend(free1_e_n2[index_of_substem])
                    free2_group.extend(free2_e_n2[index_of_substem])

            seq1_positions_list_groupedby_stems.append(seq1_group)
            seq2_positions_list_groupedby_stems.append(seq2_group)
            FEs_list_groupedby_stems.append(FE_group)
            free_nmers1_list_groupedby_stems.append(free1_group)
            free_nmers2_list_groupedby_stems.append(free2_group)
            
    return(FEs_list_groupedby_stems, free_nmers1_list_groupedby_stems, free_nmers2_list_groupedby_stems, 
           seq1_positions_list_groupedby_stems, seq2_positions_list_groupedby_stems)


# This code runs faster
def get_Z_from_lists_groupedby_stems(FEs_list_groupedby_stems, free_nmers1_list_groupedby_stems,
                                     free_nmers2_list_groupedby_stems, use_prob_diss=False, prob_diss_tau_A=1e10):
    if len(FEs_list_groupedby_stems) == 0:
        return(0.)
    inv_kBT = 1.0 / (kB * T)

    # Flatten all values into one big array
    lengths = [len(stem) for stem in FEs_list_groupedby_stems]
    FE_all = np.concatenate(FEs_list_groupedby_stems).astype(float)
    p1_all = np.concatenate(free_nmers1_list_groupedby_stems).astype(float)
    p2_all = np.concatenate(free_nmers2_list_groupedby_stems).astype(float)

    if not use_prob_diss:
        return np.sum(np.exp(-FE_all * inv_kBT) * p1_all * p2_all)
    else:
        # Compute values for all substems
        vals = np.exp(-prob_diss_tau_A * np.exp(FE_all * inv_kBT)) * p1_all * p2_all
        # Group max per stem
        idx = np.cumsum([0] + lengths[:-1])
        stem_max = np.maximum.reduceat(vals, idx)
        return np.sum(stem_max)


def get_Z_from_lists_groupedby_stems_for_diff_prob_diss(FEs_list_groupedby_stems, free_nmers1_list_groupedby_stems,
                                                        free_nmers2_list_groupedby_stems, prob_diss_tau_As):
    if len(FEs_list_groupedby_stems) == 0:
        return(np.zeros(len(prob_diss_tau_As) + 1))

    # Pre-flatten once for efficiency
    lengths = [len(stem) for stem in FEs_list_groupedby_stems]
    FE_all = np.concatenate(FEs_list_groupedby_stems).astype(float)
    p1_all = np.concatenate(free_nmers1_list_groupedby_stems).astype(float)
    p2_all = np.concatenate(free_nmers2_list_groupedby_stems).astype(float)
    idx = np.cumsum([0] + lengths[:-1])
    inv_kBT = 1.0 / (kB * T)

    # Base Z
    Zs = [np.sum(np.exp(-FE_all * inv_kBT) * p1_all * p2_all)]

    # Prob_diss versions
    for tau_A in prob_diss_tau_As:
        vals = np.exp(-tau_A * np.exp(FE_all * inv_kBT)) * p1_all * p2_all
        Zs.append(np.sum(np.maximum.reduceat(vals, idx)))

    return np.array(Zs)


#%% Compare binding between two sequences to binding of each sequence to itself
def self_binding_for_seq_list(seq_list, min_pal_len=4, num_subopt=10**5,
                              free_nmers_filenames=None, free_nmers_cutoff=0,
                              use_prob_diss=False, prob_diss_tau_A=1e10,
                              only_consider_palindromes=False, pal_summary_filenames=None):
    binding_mat_ii = np.zeros(len(seq_list))
    for i in range(len(seq_list)):
        seq = seq_list[i]
        
        if free_nmers_filenames is not None:
            try:
                free_nmers = load(free_nmers_filenames[i])
            except:
                try:
                    free_nmers = load(free_nmers_filenames[i] + '_again')
                except:
                    print("Couldn't load free_nmers_filenames", i, 'again', free_nmers_filenames[i])
                    free_nmers = get_equilibrium_prob_of_free_nmers_faster(
                        seq, range(min_pal_len, 20 + 1), num_subopt=num_subopt)
                    if free_nmers_filenames[i]:
                        save(free_nmers, free_nmers_filenames[i])

        else:
            free_nmers = get_equilibrium_prob_of_free_nmers_faster(
                seq, range(min_pal_len, 20 + 1), num_subopt=num_subopt)    

        for e in range(len(free_nmers)):
            free_nmers[e][free_nmers[e] < free_nmers_cutoff] = 0

        if only_consider_palindromes:
            if pal_summary_filenames is not None:
                try: 
                    pal_summary = load(pal_summary_filenames[i])
                except:
                    try:
                        pal_summary = load(pal_summary_filenames[i] + '_again')
                    except:
                        print("Couldn't load pal_summary_filenames", i, 'again', pal_summary_filenames[i])
                        _, pal_summary, _ = find_palindromes(
                            seq, 
                            min_len_palindrome_with_C=min_pal_len, 
                            min_len_palindrome=min_pal_len, 
                            max_len_palindrome=500, 
                            check_all_comp=include_GU
                            )

                        if pal_summary_filenames[i]:
                            save(pal_summary, pal_summary_filenames[i])
            else:
                _, pal_summary, _ = find_palindromes(
                    seq, 
                    min_len_palindrome_with_C=min_pal_len, 
                    min_len_palindrome=min_pal_len, 
                    max_len_palindrome=500, 
                    check_all_comp=include_GU
                    )
        else:
            pal_summary = None

        (Z_ns_ii, Zs_ns_ii) = binding_to_all_regions(
            seq, seq, free_nmers, free_nmers, min_n=min_pal_len,
            use_prob_diss=use_prob_diss, prob_diss_tau_A=prob_diss_tau_A,
            only_consider_palindromes=only_consider_palindromes, 
            pal_summary_1=pal_summary, pal_summary_2=pal_summary)
    
        # if use_prob_diss:
        #     binding_mat_ii[i] = 1 - np.prod(1 - Z_ns_ii)
        # else:
        binding_mat_ii[i] = np.sum(Z_ns_ii)         
        
    return(binding_mat_ii)
    

def binding_between_two_seq_lists(seq_list_1, seq_list_2, min_pal_len=4, num_subopt=10**5,
                                  free_nmers_1_filenames=None, free_nmers_2_filenames=None,
                                  free_nmers_1_cutoff=0, free_nmers_2_cutoff=0, 
                                  set_free_nmers_to_1=False,
                                  use_prob_diss=False, prob_diss_tau_A=1e10, 
                                  only_consider_palindromes=False, 
                                  pal_summary_1_filenames=None, pal_summary_2_filenames=None):
    start = time.time()
    
    if seq_list_1 == seq_list_2:
        self_binding = True
    else:
        self_binding = False
    
    start2 = time.time()
    binding_mat_ii = self_binding_for_seq_list(
        seq_list_1, min_pal_len=min_pal_len, num_subopt=num_subopt,
        free_nmers_filenames=free_nmers_1_filenames, free_nmers_cutoff=free_nmers_1_cutoff,
        use_prob_diss=use_prob_diss, prob_diss_tau_A=prob_diss_tau_A, 
        only_consider_palindromes=only_consider_palindromes, pal_summary_filenames=pal_summary_1_filenames)
    print('i', np.log10(binding_mat_ii), time.time() - start2, time.time() - start)    
    
    if self_binding:
        binding_mat_jj = binding_mat_ii
    else:
        start2 = time.time()
        binding_mat_jj = self_binding_for_seq_list(
            seq_list_2, min_pal_len=min_pal_len, num_subopt=num_subopt,
            free_nmers_filenames=free_nmers_2_filenames, free_nmers_cutoff=free_nmers_2_cutoff,
            use_prob_diss=use_prob_diss, prob_diss_tau_A=prob_diss_tau_A,
            only_consider_palindromes=only_consider_palindromes, pal_summary_filenames=pal_summary_2_filenames)
        print('j', np.log10(binding_mat_jj), time.time() - start2, time.time() - start)
            
    binding_mat_ij = np.zeros((len(seq_list_1), len(seq_list_2)))
    for i in range(len(seq_list_1)):
        start2 = time.time()
        seq1 = seq_list_1[i]
        
        if free_nmers_1_filenames is not None:
            try:
                free_nmers_1 = load(free_nmers_1_filenames[i])
            except:
                print("Couldn't load free_nmers_1_filenames", i)
                free_nmers_1 = get_equilibrium_prob_of_free_nmers_faster(
                    seq1, range(min_pal_len, 20 + 1), num_subopt=num_subopt)
                if free_nmers_1_filenames[i]:
                    save(free_nmers_1, free_nmers_1_filenames[i])
        else:
            free_nmers_1 = get_equilibrium_prob_of_free_nmers_faster(
                seq1, range(min_pal_len, 20 + 1), num_subopt=num_subopt)    


        for e in range(len(free_nmers_1)):
            if set_free_nmers_to_1:
                free_nmers_1[e] = free_nmers_1[e]**0
            free_nmers_1[e][free_nmers_1[e] < free_nmers_1_cutoff] = 0


        if only_consider_palindromes:
            if pal_summary_1_filenames is not None:
                try: 
                    pal_summary_1 = load(pal_summary_1_filenames[i])
                except:
                    try:
                        pal_summary_1 = load(pal_summary_1_filenames[i] + '_again')
                    except:
                        print("Couldn't load pal_summary_1_filenames", i, 'again', pal_summary_1_filenames[i])
                        _, pal_summary_1, _ = find_palindromes(
                            seq1, 
                            min_len_palindrome_with_C=min_pal_len, 
                            min_len_palindrome=min_pal_len, 
                            max_len_palindrome=500, 
                            check_all_comp=include_GU
                            )

                        if pal_summary_1_filenames[i]:
                            save(pal_summary_1, pal_summary_1_filenames[i])
            else:
                _, pal_summary_1, _ = find_palindromes(
                    seq1, 
                    min_len_palindrome_with_C=min_pal_len, 
                    min_len_palindrome=min_pal_len, 
                    max_len_palindrome=500, 
                    check_all_comp=include_GU
                    )
        else:
            pal_summary_1 = None


        for j in range(len(seq_list_2)):
            seq2 = seq_list_2[j]
            
            if i > j and self_binding: 
                binding_mat_ij[i, j] = binding_mat_ij[j, i] 
            else:
                if free_nmers_2_filenames is not None:
                    try:
                        free_nmers_2 = load(free_nmers_2_filenames[j])
                    except:
                        print("Couldn't load free_nmers_2_filenames", j)
                        free_nmers_2 = get_equilibrium_prob_of_free_nmers_faster(
                            seq2, range(min_pal_len, 20 + 1), num_subopt=num_subopt)
                        if free_nmers_2_filenames[j]:
                            save(free_nmers_2, free_nmers_2_filenames[j])
                else:
                    free_nmers_2 = get_equilibrium_prob_of_free_nmers_faster(
                        seq2, range(min_pal_len, 20 + 1), num_subopt=num_subopt)
    
                for e in range(len(free_nmers_2)):
                    if set_free_nmers_to_1:
                        free_nmers_2[e] = free_nmers_2[e]**0
                    free_nmers_2[e][free_nmers_2[e] < free_nmers_2_cutoff] = 0

                if only_consider_palindromes:
                    if pal_summary_2_filenames is not None:
                        try: 
                            pal_summary_2 = load(pal_summary_2_filenames[j])
                        except:
                            try:
                                pal_summary_2 = load(pal_summary_2_filenames[j] + '_again')
                            except:
                                print("Couldn't load pal_summary_2_filenames", j, 'again', pal_summary_2_filenames[j])
                                _, pal_summary_2, _ = find_palindromes(
                                    seq2, 
                                    min_len_palindrome_with_C=min_pal_len, 
                                    min_len_palindrome=min_pal_len, 
                                    max_len_palindrome=500, 
                                    check_all_comp=include_GU
                                    )
        
                                if pal_summary_2_filenames[j]:
                                    save(pal_summary_2, pal_summary_2_filenames[j])
                    else:
                        _, pal_summary_2, _ = find_palindromes(
                            seq2, 
                            min_len_palindrome_with_C=min_pal_len, 
                            min_len_palindrome=min_pal_len, 
                            max_len_palindrome=500, 
                            check_all_comp=include_GU
                            )
                else:
                    pal_summary_2 = None

                (Z_ns_ij, Zs_ns_ij) = binding_to_all_regions(
                    seq1, seq2, free_nmers_1, free_nmers_2, min_n=min_pal_len,
                    use_prob_diss=use_prob_diss, prob_diss_tau_A=prob_diss_tau_A,
                    only_consider_palindromes=only_consider_palindromes, 
                    pal_summary_1=pal_summary_1, pal_summary_2=pal_summary_2)
        
                # if use_prob_diss:
                #     binding_mat_ij[i, j] = 1 - np.prod(1 - Z_ns_ij)
                # else:
                binding_mat_ij[i, j] = np.sum(Z_ns_ij) 
                
            
        print(i, np.log10(binding_mat_ij[i,:]), time.time() - start2, time.time() - start)
    
    return(binding_mat_ij, binding_mat_ii, binding_mat_jj)


#%% Compare binding between two sequences to binding of each sequence to itself, doing a better job of calculating with use_prob_diss=True 
def self_binding_for_seq_list_with_indiv(
        seq_list, min_pal_len=4, num_subopt=10**5,
        free_nmers_filenames=None, free_nmers_cutoff=0,
        prob_diss_tau_As=[], only_consider_palindromes=False, pal_summary_filenames=None):
    
    binding_mat_ii = np.zeros((len(seq_list), len(prob_diss_tau_As) + 1))
    
    for i in range(len(seq_list)):
        seq = seq_list[i]
        
        if free_nmers_filenames is not None:
            try:
                free_nmers = load(free_nmers_filenames[i])
            except:
                try:
                    free_nmers = load(free_nmers_filenames[i] + '_again')
                except:
                    print("Couldn't load free_nmers_filenames", i, 'again', free_nmers_filenames[i])
                    free_nmers = get_equilibrium_prob_of_free_nmers_faster(
                        seq, range(min_pal_len, 20 + 1), num_subopt=num_subopt)
                    if free_nmers_filenames[i]:
                        save(free_nmers, free_nmers_filenames[i])

        else:
            free_nmers = get_equilibrium_prob_of_free_nmers_faster(
                seq, range(min_pal_len, 20 + 1), num_subopt=num_subopt)    

        for e in range(len(free_nmers)):
            free_nmers[e][free_nmers[e] < free_nmers_cutoff] = 0

        if only_consider_palindromes:
            if pal_summary_filenames is not None:
                try: 
                    pal_summary = load(pal_summary_filenames[i])
                except:
                    try:
                        pal_summary = load(pal_summary_filenames[i] + '_again')
                    except:
                        print("Couldn't load pal_summary_filenames", i, 'again', pal_summary_filenames[i])
                        _, pal_summary, _ = find_palindromes(
                            seq, 
                            min_len_palindrome_with_C=min_pal_len, 
                            min_len_palindrome=min_pal_len, 
                            max_len_palindrome=500, 
                            check_all_comp=include_GU
                            )

                        if pal_summary_filenames[i]:
                            save(pal_summary, pal_summary_filenames[i])
            else:
                _, pal_summary, _ = find_palindromes(
                    seq, 
                    min_len_palindrome_with_C=min_pal_len, 
                    min_len_palindrome=min_pal_len, 
                    max_len_palindrome=500, 
                    check_all_comp=include_GU
                    )
        else:
            pal_summary = None

        (FEs_list_n, free_nmers1_list_n, free_nmers2_list_n, seq1_positions_list_n, seq2_positions_list_n
         ) = binding_to_all_regions_and_return_indiv(
            seq, seq, free_nmers, free_nmers, min_n=min_pal_len,
            only_consider_palindromes=only_consider_palindromes, 
            pal_summary_1=pal_summary, pal_summary_2=pal_summary)
    
        if len(prob_diss_tau_As) == 0:  # then we don't need to go through grouping by substems
            binding_mat_ii[i, 0] = get_Z_from_lists_groupedby_stems(FEs_list_n, free_nmers1_list_n, free_nmers2_list_n)         
            # get_Z_from_lists_groupedby_stems works for use_prob_diss=False even if grouped by n rather than by stems
        else:
            FEs_list_groupedby_stems, free_nmers1_list_groupedby_stems, free_nmers2_list_groupedby_stems, _, _ = group_stems_by_substems(
                FEs_list_n, free_nmers1_list_n, free_nmers2_list_n, seq1_positions_list_n, seq2_positions_list_n)
            binding_mat_ii[i, :] = get_Z_from_lists_groupedby_stems_for_diff_prob_diss(
                FEs_list_groupedby_stems, free_nmers1_list_groupedby_stems, free_nmers2_list_groupedby_stems, prob_diss_tau_As)
            
    return(binding_mat_ii)
    

def binding_between_two_seq_lists_with_indiv(
        seq_list_1, seq_list_2, min_pal_len=4, num_subopt=10**5,
        free_nmers_1_filenames=None, free_nmers_2_filenames=None,
        free_nmers_1_cutoff=0, free_nmers_2_cutoff=0, 
        set_free_nmers_to_1=False, prob_diss_tau_As=[], only_consider_palindromes=False, 
        pal_summary_1_filenames=None, pal_summary_2_filenames=None):
    
    start = time.time()
    
    if seq_list_1 == seq_list_2:
        self_binding = True
    else:
        self_binding = False
    
    start2 = time.time()
    binding_mat_ii = self_binding_for_seq_list_with_indiv(
        seq_list_1, min_pal_len=min_pal_len, num_subopt=num_subopt,
        free_nmers_filenames=free_nmers_1_filenames, free_nmers_cutoff=free_nmers_1_cutoff,
        prob_diss_tau_As=prob_diss_tau_As, only_consider_palindromes=only_consider_palindromes, 
        pal_summary_filenames=pal_summary_1_filenames)
    
    print('i', np.transpose(np.log10(binding_mat_ii[:, :2])), time.time() - start2, time.time() - start)    
    
    if self_binding:
        binding_mat_jj = binding_mat_ii
    else:
        start2 = time.time()
        binding_mat_jj = self_binding_for_seq_list_with_indiv(
            seq_list_2, min_pal_len=min_pal_len, num_subopt=num_subopt,
            free_nmers_filenames=free_nmers_2_filenames, free_nmers_cutoff=free_nmers_2_cutoff,
            prob_diss_tau_As=prob_diss_tau_As, only_consider_palindromes=only_consider_palindromes, 
            pal_summary_filenames=pal_summary_2_filenames)
        print('j', np.transpose(np.log10(binding_mat_jj[:, :2])), time.time() - start2, time.time() - start)
            
    binding_mat_ij = np.zeros((len(seq_list_1), len(seq_list_2), len(prob_diss_tau_As) + 1))
    for i in range(len(seq_list_1)):
        start2 = time.time()
        seq1 = seq_list_1[i]
        
        if free_nmers_1_filenames is not None:
            try:
                free_nmers_1 = load(free_nmers_1_filenames[i])
            except:
                print("Couldn't load free_nmers_1_filenames", i)
                free_nmers_1 = get_equilibrium_prob_of_free_nmers_faster(
                    seq1, range(min_pal_len, 20 + 1), num_subopt=num_subopt)
                if free_nmers_1_filenames[i]:
                    save(free_nmers_1, free_nmers_1_filenames[i])
        else:
            free_nmers_1 = get_equilibrium_prob_of_free_nmers_faster(
                seq1, range(min_pal_len, 20 + 1), num_subopt=num_subopt)    


        for e in range(len(free_nmers_1)):
            if set_free_nmers_to_1:
                free_nmers_1[e] = free_nmers_1[e]**0
            free_nmers_1[e][free_nmers_1[e] < free_nmers_1_cutoff] = 0


        if only_consider_palindromes:
            if pal_summary_1_filenames is not None:
                try: 
                    pal_summary_1 = load(pal_summary_1_filenames[i])
                except:
                    try:
                        pal_summary_1 = load(pal_summary_1_filenames[i] + '_again')
                    except:
                        print("Couldn't load pal_summary_1_filenames", i, 'again', pal_summary_1_filenames[i])
                        _, pal_summary_1, _ = find_palindromes(
                            seq1, 
                            min_len_palindrome_with_C=min_pal_len, 
                            min_len_palindrome=min_pal_len, 
                            max_len_palindrome=500, 
                            check_all_comp=include_GU
                            )

                        if pal_summary_1_filenames[i]:
                            save(pal_summary_1, pal_summary_1_filenames[i])
            else:
                _, pal_summary_1, _ = find_palindromes(
                    seq1, 
                    min_len_palindrome_with_C=min_pal_len, 
                    min_len_palindrome=min_pal_len, 
                    max_len_palindrome=500, 
                    check_all_comp=include_GU
                    )
        else:
            pal_summary_1 = None


        for j in range(len(seq_list_2)):
            seq2 = seq_list_2[j]
            
            if i > j and self_binding: 
                binding_mat_ij[i, j] = binding_mat_ij[j, i] 
            else:
                if free_nmers_2_filenames is not None:
                    try:
                        free_nmers_2 = load(free_nmers_2_filenames[j])
                    except:
                        print("Couldn't load free_nmers_2_filenames", j)
                        free_nmers_2 = get_equilibrium_prob_of_free_nmers_faster(
                            seq2, range(min_pal_len, 20 + 1), num_subopt=num_subopt)
                        if free_nmers_2_filenames[j]:
                            save(free_nmers_2, free_nmers_2_filenames[j])
                else:
                    free_nmers_2 = get_equilibrium_prob_of_free_nmers_faster(
                        seq2, range(min_pal_len, 20 + 1), num_subopt=num_subopt)
    
                for e in range(len(free_nmers_2)):
                    if set_free_nmers_to_1:
                        free_nmers_2[e] = free_nmers_2[e]**0
                    free_nmers_2[e][free_nmers_2[e] < free_nmers_2_cutoff] = 0

                if only_consider_palindromes:
                    if pal_summary_2_filenames is not None:
                        try: 
                            pal_summary_2 = load(pal_summary_2_filenames[j])
                        except:
                            try:
                                pal_summary_2 = load(pal_summary_2_filenames[j] + '_again')
                            except:
                                print("Couldn't load pal_summary_2_filenames", j, 'again', pal_summary_2_filenames[j])
                                _, pal_summary_2, _ = find_palindromes(
                                    seq2, 
                                    min_len_palindrome_with_C=min_pal_len, 
                                    min_len_palindrome=min_pal_len, 
                                    max_len_palindrome=500, 
                                    check_all_comp=include_GU
                                    )
        
                                if pal_summary_2_filenames[j]:
                                    save(pal_summary_2, pal_summary_2_filenames[j])
                    else:
                        _, pal_summary_2, _ = find_palindromes(
                            seq2, 
                            min_len_palindrome_with_C=min_pal_len, 
                            min_len_palindrome=min_pal_len, 
                            max_len_palindrome=500, 
                            check_all_comp=include_GU
                            )
                else:
                    pal_summary_2 = None

                (FEs_list_n, free_nmers1_list_n, free_nmers2_list_n, seq1_positions_list_n, seq2_positions_list_n
                 ) = binding_to_all_regions_and_return_indiv(
                    seq1, seq2, free_nmers_1, free_nmers_2, min_n=min_pal_len,
                    only_consider_palindromes=only_consider_palindromes, 
                    pal_summary_1=pal_summary_1, pal_summary_2=pal_summary_2)
            
                if len(prob_diss_tau_As) == 0:  # then we don't need to go through grouping by substems
                    binding_mat_ij[i, j, 0] = get_Z_from_lists_groupedby_stems(FEs_list_n, free_nmers1_list_n, free_nmers2_list_n)         
                    # get_Z_from_lists_groupedby_stems works for use_prob_diss=False even if grouped by n rather than by stems
                else:
                    FEs_list_groupedby_stems, free_nmers1_list_groupedby_stems, free_nmers2_list_groupedby_stems, _, _ = group_stems_by_substems(
                        FEs_list_n, free_nmers1_list_n, free_nmers2_list_n, seq1_positions_list_n, seq2_positions_list_n)
                    binding_mat_ij[i, j, :] = get_Z_from_lists_groupedby_stems_for_diff_prob_diss(
                        FEs_list_groupedby_stems, free_nmers1_list_groupedby_stems, free_nmers2_list_groupedby_stems, prob_diss_tau_As)
            
        print(i, np.transpose(np.log10(binding_mat_ij[i,:, :2])), time.time() - start2, time.time() - start)
    
    return(binding_mat_ij, binding_mat_ii, binding_mat_jj)


#%% get palindrome enrichment
def expected_number_of_pals(len_seq, len_pal, include_GU):
    l = len_pal / 2
    p = (3/8) if include_GU else (1/4)
    return(len_seq / (l + 1 / ((1 - p) * p**l)))

def palindrome_enrichment(seq, min_pal_len=4, include_GU=include_GU):
    _, _, pal_nums = find_palindromes(seq, 
                                          min_len_palindrome_with_C=min_pal_len, 
                                          min_len_palindrome=min_pal_len, 
                                          max_len_palindrome=500, 
                                          check_all_comp=include_GU
                                          )
    return(pal_nums[min_pal_len : 21 : 2] / expected_number_of_pals(
        len(seq), np.arange(min_pal_len, 21, 2), include_GU))


#%%
def get_indiv_stem_info(seq, free_nmers=None, pal_summary=None):
    if free_nmers is None:
        free_nmers = get_equilibrium_prob_of_free_nmers_faster(
            seq, range(min_pal_len, 20 + 1), num_subopt=num_subopt)

    if pal_summary is None:
        _, pal_summary, _ = find_palindromes(
            seq, 
            min_len_palindrome_with_C=min_pal_len, 
            min_len_palindrome=min_pal_len, 
            max_len_palindrome=500, 
            check_all_comp=include_GU
            )
    
    STableBPs_pal = []
    pal_bps = [[] for _ in range(len(pal_summary))]
    for e, (pal_start, pal_len) in enumerate(pal_summary):
        STableBPs_pal += [[[pal_start + i, pal_start + pal_len - i - 1] for i in range(pal_len)]]    
        pal_bps[e] = np.array([[pal_start + i, pal_start + pal_len - 1 - i] for i in range(pal_len)])

    (stem_Zs_mel_n_pal, _, stem_Z_mel_n_pal, _, stem_free_mel_n_pal, _, _, 
     stem_Zs_indiv_mel_n_pal, stem_free_indiv_mel_n_pal) = binding_to_free_regions(
        seq, check_all_comp=include_GU, min_bp_in_stem=min_pal_len, max_bp_in_stem=20, num_subopt=100000,
        STableBPs=STableBPs_pal, free_nmers=free_nmers)

    return(stem_Zs_mel_n_pal, stem_Z_mel_n_pal, stem_free_mel_n_pal, stem_Zs_indiv_mel_n_pal, 
           stem_free_indiv_mel_n_pal, free_nmers, pal_summary)


def get_substem_weight(stem_FEs_indiv_pal_vec, stem_free_indiv_pal_vec, Gstar):
    stem_free_indiv_pal_vec_cutoff = copy.deepcopy(stem_free_indiv_pal_vec)

    for e in range(len(stem_FEs_indiv_pal_vec)):
        for i in range(len(stem_FEs_indiv_pal_vec[e])):
            for j in range(len(stem_FEs_indiv_pal_vec[e][i])):
                fe_array = np.asarray(stem_FEs_indiv_pal_vec[e][i][j], dtype=float)
                stem_free_indiv_pal_vec_cutoff[e][i][j] = (
                    np.asarray(stem_free_indiv_pal_vec_cutoff[e][i][j], dtype=float) * 
                    np.exp(-np.exp((fe_array - Gstar) / (kB * T)))
                    )

    return stem_free_indiv_pal_vec_cutoff


def get_max_substem_weight(stem_FEs_indiv_pal_vec, stem_free_indiv_pal_vec, Gstar):
    stem_free_indiv_pal_vec_cutoff = get_substem_weight(
        stem_FEs_indiv_pal_vec, stem_free_indiv_pal_vec, Gstar)
    substem_max_probs = [[] for _ in range(len(stem_free_indiv_pal_vec_cutoff))]
    for e in range(len(stem_free_indiv_pal_vec_cutoff)):
        substem_max_probs[e] = [np.max(flatten(stem_free_indiv_pal_vec_cutoff[e][i])) 
                                if len(stem_free_indiv_pal_vec_cutoff[e][i]) 
                                else 0 
                                for i in range(len(stem_free_indiv_pal_vec_cutoff[e]))]
    return(substem_max_probs)


# Get info on how many accessible and strongly binding stems nanos and pgc have
def get_substems_with_strong_binding(stem_FEs_indiv_pal_vec, stem_free_indiv_pal_vec, FE_cutoff, weak_binding_indicator=0):
    stem_free_indiv_pal_vec_cutoff = copy.deepcopy(stem_free_indiv_pal_vec)

    for e in range(len(stem_FEs_indiv_pal_vec)):
        for i in range(len(stem_FEs_indiv_pal_vec[e])):
            for j in range(len(stem_FEs_indiv_pal_vec[e][i])):
                fe_array = np.asarray(stem_FEs_indiv_pal_vec[e][i][j], dtype=float)
                mask = fe_array > FE_cutoff
                free_array = np.asarray(stem_free_indiv_pal_vec_cutoff[e][i][j], dtype=float)
                free_array[mask] = weak_binding_indicator
                stem_free_indiv_pal_vec_cutoff[e][i][j] = free_array

    return stem_free_indiv_pal_vec_cutoff


def get_max_prob_of_substems_with_strong_binding(stem_FEs_indiv_pal_vec, stem_free_indiv_pal_vec, FE_cutoff, weak_binding_indicator=0):
    stem_free_indiv_pal_vec_cutoff = get_substems_with_strong_binding(
        stem_FEs_indiv_pal_vec, stem_free_indiv_pal_vec, FE_cutoff, 
        weak_binding_indicator=weak_binding_indicator)
    substem_max_probs = [[] for _ in range(len(stem_free_indiv_pal_vec_cutoff))]
    for e in range(len(stem_free_indiv_pal_vec_cutoff)):
        substem_max_probs[e] = [np.max(flatten(stem_free_indiv_pal_vec_cutoff[e][i])) 
                                if len(stem_free_indiv_pal_vec_cutoff[e][i]) 
                                else weak_binding_indicator 
                                for i in range(len(stem_free_indiv_pal_vec_cutoff[e]))]
    return(substem_max_probs)


#%%
def get_sim_len_seqs(seq, other_seqs, name, len_diff_cutoff=8):
    seq_sim_len = [i for i in other_seqs if (
        len(i) > len(seq) - len_diff_cutoff and len(i) < len(seq) + len_diff_cutoff
        ) and i != seq]
    
    seqs_to_use_prelim = [seq] + seq_sim_len
    
    start = time.time()
    smith_waterman = np.zeros((len(seqs_to_use_prelim), len(seqs_to_use_prelim)))
    for e, i in enumerate(seqs_to_use_prelim):
        start2 = time.time()
        for e2, j in enumerate(seqs_to_use_prelim):
            # if e == e2:
            #     smith_waterman_gcl[e, e2] = 0
            if e2 < e:
                smith_waterman[e, e2] = smith_waterman[e2, e]
            else:
                smith_waterman[e, e2] = smith_waterman_score_fast(i, j)  # smith_waterman_score_numba(i, j)
    print('smith_waterman', name, time.time() - start2, time.time() - start)
    
    duplicates = np.array([(i, j) for (i, j) in zip(*np.where(smith_waterman > 40)) 
                                if i < j])
    
    seqs_to_use = [i for e, i in enumerate(seqs_to_use_prelim) if e not in duplicates[:, 1]]
    print(len(seqs_to_use))
    
    seqs_filenames = ([folder + 
                           'free_nmers_' + name + '_nodup_nsubopt' + str(num_subopt) + '_mBP' + str(min_pal_len)] +
                          [folder + 
                           'free_nmers_' + name + '_nodup_rand' + str(len_diff_cutoff) + '_' + 
                           str(i - 1) + '_nsubopt' + str(num_subopt) + '_mBP' + str(min_pal_len)
                           for i in range(1, len(seqs_to_use))])
    
    pal_summary_filenames = ([folder + 
                           'pal_summary_' + name + '_nodup' + '_mBP' + str(min_pal_len)] +
                          [folder + 
                           'pal_summary_' + name + '_nodup_rand' + str(len_diff_cutoff) + '_' + 
                           str(i - 1) + '_mBP' + str(min_pal_len)
                           for i in range(1, len(seqs_to_use))])
    
    start = time.time()
    for e, seq in enumerate(seqs_to_use):
        try:
            free_nmers = load(seqs_filenames[e])
        except:
            free_nmers = get_equilibrium_prob_of_free_nmers_faster(
                seq, range(min_pal_len, 20 + 1), num_subopt=num_subopt)
            save(free_nmers, seqs_filenames[e])
        del free_nmers

        try:
            pal_summary = load(pal_summary_filenames[e])
        except:
            _, pal_summary, _ = find_palindromes(
                seq, 
                min_len_palindrome_with_C=min_pal_len, 
                min_len_palindrome=min_pal_len, 
                max_len_palindrome=500, 
                check_all_comp=include_GU
                )
            save(pal_summary, pal_summary_filenames[e])
            print(e, time.time() - start)
        del pal_summary

    return(seqs_to_use, seqs_filenames, pal_summary_filenames)


def get_Wpal_comp(seqs_to_use, seqs_filenames, pal_summary_filenames):
    stem_Zs_pal_vec = [[] for _ in range(len(seqs_to_use))]
    stem_Z_pal_vec = [[] for _ in range(len(seqs_to_use))]
    stem_free_pal_vec = [[] for _ in range(len(seqs_to_use))]
    stem_Zs_indiv_pal_vec = [[] for _ in range(len(seqs_to_use))]
    stem_free_indiv_pal_vec = [[] for _ in range(len(seqs_to_use))]

    for e_seq, seq in enumerate(seqs_to_use):
        free_nmers = load(seqs_filenames[e_seq])
        pal_summary = load(pal_summary_filenames[e_seq])
        
        (stem_Zs_pal_vec[e_seq], stem_Z_pal_vec[e_seq], 
         stem_free_pal_vec[e_seq], stem_Zs_indiv_pal_vec[e_seq], 
         stem_free_indiv_pal_vec[e_seq], free_nmers, pal_summary) = get_indiv_stem_info(
                   seq, free_nmers=free_nmers, pal_summary=pal_summary)
    
    # There's a problem, which is that I'm considering a stem of length L as well as substems of length e.g. L-2 contained within it.
    # I could just consider the longest stem, but it's possible that the longest stem is inaccessible, while a shorter substem
    # is both accessible and still a strong enough binder. Therefore, for each longest stem, I consider all its substems,
    # and take the maximum weight given by a substem.
    stem_FEs_indiv_pal_vec = [[[-kB * T * np.log(i) for i in j] for j in k] for k in stem_Zs_indiv_pal_vec]
    
    # This is what we actually use for paper
    FEs_to_plot = np.linspace(-13, -4, 101)
    substem_max_weight_as_fxn_of_FE = [get_max_substem_weight(
        stem_FEs_indiv_pal_vec, stem_free_indiv_pal_vec, Gstar) for Gstar in FEs_to_plot]
    return(FEs_to_plot, substem_max_weight_as_fxn_of_FE)
    
    
def plot_Wpal_comp(FEs_to_plot, substem_max_weight_as_fxn_of_FE, len_seqs_to_use, name):
    plt.figure(figsize=(5.5, 4))
    ax_main = plt.gca()
    for i in range(1, len_seqs_to_use):
        ax_main.plot(FEs_to_plot, 
                 [np.sum(substem_max_weight_as_fxn_of_FE[e][i]) for e in range(len(FEs_to_plot))],
                 color='orange', alpha=0.2, label=rf'$\it{{{name}}}$-like' if i==1 else '')
    ax_main.plot(FEs_to_plot, 
             [np.sum(substem_max_weight_as_fxn_of_FE[e][0]) for e in range(len(FEs_to_plot))],
             color='blue', label=rf'$\it{{{name}}}$')
    
    ax_main.set_xlabel(r'$\Delta G^{\!\!\ast}$ (kcal/mol)', fontsize=14)
    ax_main.set_ylabel(r'$W^\mathrm{pal}$', fontsize=14)
    ax_main.tick_params(axis='both', labelsize=14)
    # ax_main.set_ylim([5e-5, 1e1])
    ax_main.set_yscale('log')
    leg2 = ax_main.legend(loc='lower right', #loc='lower right', 
               fontsize=12)
    for lh in leg2.legendHandles:
        lh.set_alpha(1) 
    plt.show()
    
def get_sim_seqs_Wpal_comp_and_plot(seq, other_seqs, save_name, plot_name, len_diff_cutoff=8):
    seqs_to_use, seqs_filenames, pal_summary_filenames = get_sim_len_seqs(
        seq, other_seqs, save_name, len_diff_cutoff=len_diff_cutoff)
    FEs_to_plot, substem_max_weight_as_fxn_of_FE = get_Wpal_comp(seqs_to_use, seqs_filenames, pal_summary_filenames)
    plot_Wpal_comp(FEs_to_plot, substem_max_weight_as_fxn_of_FE, len(seqs_to_use), plot_name)
    return(seqs_to_use, seqs_filenames, pal_summary_filenames, FEs_to_plot, substem_max_weight_as_fxn_of_FE)


def get_prob_diss_tau_As(seqs_to_use_1, seqs_to_use_2, free_nmers_1_filenames, free_nmers_2_filenames,
                         save_name_1, save_name_2, min_pal_len=4, num_subopt=10**5, 
                         to_plot=True, plot_all=True, plot_name_1='', plot_name_2='', to_load=True):
    
    prob_diss_tau_As = np.array([np.exp(i/(kB * T)) for i in np.arange(4, 16.1, 0.2)])

    binding_mat_filename_11 = (save_name_1 + '_' + save_name_1 + 'binding_mat_A_nsubopt' + 
                               str(num_subopt) + '_mBP' + str(min_pal_len))
    binding_mat_filename_22 = (save_name_2 + '_' + save_name_2 + 'binding_mat_A_nsubopt' + 
                               str(num_subopt) + '_mBP' + str(min_pal_len))
    binding_mat_filename_12 = (save_name_1 + '_' + save_name_2 + 'binding_mat_A_nsubopt' + 
                               str(num_subopt) + '_mBP' + str(min_pal_len))
    
    try:
        if to_load:
            binding_mat_12 = load(folder + binding_mat_filename_12)
            binding_mat_11 = load(folder + binding_mat_filename_11)
            binding_mat_22 = load(folder + binding_mat_filename_22)
        else:
            raise RuntimeError("to_load is False")
    except:
        binding_mat_12, binding_mat_11, binding_mat_22 = (
            binding_between_two_seq_lists_with_indiv(seqs_to_use_1, seqs_to_use_2, 
                                          min_pal_len=min_pal_len, num_subopt=num_subopt,
                                          free_nmers_1_filenames=free_nmers_1_filenames, 
                                          free_nmers_2_filenames=free_nmers_2_filenames,
                                          free_nmers_1_cutoff=0, free_nmers_2_cutoff=0, 
                                          prob_diss_tau_As=prob_diss_tau_As))
        
        save(binding_mat_12, folder + binding_mat_filename_12)
        save(binding_mat_11, folder + binding_mat_filename_11)
        save(binding_mat_22, folder + binding_mat_filename_22)

    
    FE_binding_mat_12 = -kB * T * np.log(binding_mat_12)
    FE_binding_mat_11 = -kB * T * np.log(binding_mat_11)
    FE_binding_mat_22 = -kB * T * np.log(binding_mat_22)
    
    print(FE_binding_mat_12[0, 0, :2])
    print(FE_binding_mat_11[0, :2])
    print(FE_binding_mat_22[0, :2])
    
    FE_binding_mat_comp_to_homo = (
        FE_binding_mat_12 * 2 - (
            np.expand_dims(FE_binding_mat_11, 1) + 
            np.expand_dims(FE_binding_mat_22, 0)))
    
    # binding_mat_comp_to_homo = np.exp(-FE_binding_mat_comp_to_homo / (kB * T))
    
    if to_plot:
        if plot_all:
            is_to_plot = range(FE_binding_mat_comp_to_homo.shape[-1])
        else:
            is_to_plot = [41]
        for i in is_to_plot: 
            plt.figure()
            mn = np.min(FE_binding_mat_comp_to_homo[:, :, i])
            mx = np.max(FE_binding_mat_comp_to_homo[:, :, i])
            plt.hist(FE_binding_mat_comp_to_homo[:, :, i].flatten(),
                     np.linspace(mn, mx, 41), density=True, alpha=0.5, 
                     label=plot_name_1 + '-like / ' + plot_name_2 + '-like')
            plt.hist(FE_binding_mat_comp_to_homo[0, :, i].flatten(), 
                     np.linspace(mn, mx, 41), density=True, alpha=0.5, 
                     label=plot_name_1 + ' / ' + plot_name_2 + '-like')
            plt.hist(FE_binding_mat_comp_to_homo[:, 0, i].flatten(), 
                     np.linspace(mn, mx, 41), density=True, alpha=0.5, 
                     label=plot_name_2 + ' / ' + plot_name_1 + '-like')
            plt.xlabel(
                r'$2\Delta G^{\mathrm{non\!-\!eq}}_{12} - \Delta G^{\mathrm{non\!-\!eq}}_{11} - \Delta G^{\mathrm{non\!-\!eq}}_{22}$'
                + ' (kcal/mol)',
                fontsize=14
            )
            plt.ylabel('Frequency', fontsize=14)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.title('Z' if i==0 else r'$\Delta G^\ast = $' + str(np.round(-kB * T * np.log(prob_diss_tau_As[i - 1]), 2)))
            plt.legend(fontsize=12, loc='upper left')
            
            ymax = plt.gca().get_ylim()[1]
            y_arrow = 0.25 * ymax   # place arrows low, away from legend
            
            plt.annotate("",
                         xy=(mn, y_arrow), xytext=((mn+mx)/2, y_arrow),
                         arrowprops=dict(arrowstyle="->", lw=1.5))
            plt.text((3*mn+mx)/4, y_arrow + 0.08*ymax, "more heterotypic",
                     ha="center", va="top", fontsize=12)
            
            # right arrow: points right (away from 0)
            plt.annotate("",
                         xy=(mx, y_arrow), xytext=((mn+mx)/2, y_arrow),
                         arrowprops=dict(arrowstyle="->", lw=1.5))
            plt.text((3*mx+mn)/4, y_arrow + 0.08*ymax, "more homotypic",
                     ha="center", va="top", fontsize=12)
    
            plt.show()

    return(FE_binding_mat_comp_to_homo)


#%% Design sequence with no palindromes
def designAllComplementsRNARNA(seq):
    # Take an RNA sequence and design all of its reverse complements as RNA sequences
    complements = ['']
    for nt in seq:
        if nt == 'A':
            complements = ['U' + comp for comp in complements]
        elif nt == 'C':
            complements = ['G' + comp for comp in complements]
        elif nt == 'G':
            complements *= 2
            num_comp = len(complements)
            complements = (['C' + comp for comp in complements[:int(num_comp/2)]] + 
                           ['U' + comp for comp in complements[int(num_comp/2):]])
        elif nt == 'U':
            complements *= 2
            num_comp = len(complements)
            complements = (['A' + comp for comp in complements[:int(num_comp/2)]] + 
                           ['G' + comp for comp in complements[int(num_comp/2):]])
    return(complements)


def equiv_up_to_i_nts(seq1, seq2, i):
    # Can you get seq1 to equal seq2 by removing up to i nts from each?
    
    def single_nt_removed(new_seq):
        # Generate all sequences from new_seq that have one nt removed
        removed_nt_seqs = []
        for j in range(len(new_seq)):
            removed_nt_seqs += [new_seq[:j] + new_seq[j + 1:]]
        return(removed_nt_seqs)
        
    def remove_nts(seq, num_nts_to_remove):
        # Generate all sequences from seq that have up to num_nts_to_remove nts removed
        
        if num_nts_to_remove >= len(seq):
            num_nts_to_remove = len(seq) - 1
            # So you always have sequences with at least one character
            
        new_seqs = [seq]
        num_nts_removed = 0
        while num_nts_removed < num_nts_to_remove:
            new_seqs_store = copy.copy(new_seqs)
            for new_seq in new_seqs_store:
                new_seqs += single_nt_removed(new_seq)
            new_seqs = list(np.unique(new_seqs))  # for speed-up
            num_nts_removed += 1
        return(new_seqs)
        # Check: [len(remove_nts('ABCDE',i)) for i in range(5)] == list(np.cumsum([1, 5, 10, 10, 5]))
    
    seq1_i = remove_nts(seq1, i)  # generate all sequences starting from seq1 that have at most i nts removed
    seq2_i = remove_nts(seq2, i)  # generate all sequences starting from seq2 that have at most i nts removed
    return(any([i == j for i in seq1_i for j in seq2_i]))
    
def is_complementary_allowing_mismatches(seq1, seq2, allowed_num_mismatches_per_strand=0):
    complements = designAllComplementsRNARNA(seq1)
    if any([seq2 == comp for comp in complements]):  # then a perfect complement exists
        return(True)
    for i in range(1, allowed_num_mismatches_per_strand + 1):
        for comp in complements:
            # Check if sticker and comp can be the same thing if you remove up to i nts from each
            if equiv_up_to_i_nts(seq2, comp, i):
                return(True)
    return(False)

def dimer_differential_FE(subseq, ignore_monomer=False):
    nupack_model = nupack.Model(material='rna', kelvin=310.15#, ensemble='nostacking'
                            )
    nupack_seqs = [nupack.Strand(subseq, name='seq1')]

    cset1 = nupack.ComplexSet(strands=nupack_seqs, complexes=nupack.SetSpec(max_size=2))
    complex_analysis_result = nupack.complex_analysis(cset1, model=nupack_model, 
                                            compute=['pfunc'])  # , 'mfe'
    
    dimer_FE = complex_analysis_result['(seq1+seq1)'][2]   # [1] is the partition function; [2] is the free energy
    monomer_FE = complex_analysis_result['(seq1)'][2]
    if ignore_monomer:
        monomer_FE = 0
    
    if dimer_FE >= 100:
        dimer_FE = 100
    if monomer_FE >= 100:
        monomer_FE = 100
    dimer_FE_differential = dimer_FE - 2 * monomer_FE
    return(dimer_FE, dimer_FE_differential)


def decide_if_to_scramble_region(region, allow_AUG=True, allow_CUG=True, allow_G_quad=True,
                                 max_FE=-10, max_dFE=-10, 
                                 shortest_disallowed_pal_length=4, shortest_disallowed_near_pal_length=6,
                                 max_num_consecutive_nts=100):
    
    # Returns True if the region needs to be scrambled; else returns False

    # Check if there are any AUG
    if not allow_AUG:
        if 'AUG' in region:
            return(True)

    if not allow_CUG:
        if 'CUG' in region:
            return(True)
    
    # Check if there are any G-quadruplexes
    if not allow_G_quad:
        if 'GGGG' in region:
            return(True)
    
    nts = ['A', 'C', 'G', 'U']
    for nt in nts:
        if nt * (max_num_consecutive_nts + 1) in region:
            return(True)
    
    # Check if there are any exact palindromes of length > max_exact_pal_length
    if shortest_disallowed_pal_length <= len(region):
        for start_nt in range(len(region) - shortest_disallowed_pal_length + 1):
            subregion = region[start_nt : start_nt + shortest_disallowed_pal_length]
            if 'C' in subregion or shortest_disallowed_pal_length > 4:  # only care about short palindromes if they have a GC bond
                if is_complementary(subregion, subregion):
                    return(True)
        
    # Check if there are any near-exact palindromes of length > max_near_pal_length
    if shortest_disallowed_near_pal_length <= len(region):
        for near_pal_length in [shortest_disallowed_near_pal_length, shortest_disallowed_near_pal_length + 1]:
            # Not sure if this is redundant or necessary to use both lengths, but whatever
            for start_nt in range(len(region) - near_pal_length + 1):
                subregion = region[start_nt : start_nt + near_pal_length]
                if is_complementary_allowing_mismatches(subregion, subregion, 
                                                        allowed_num_mismatches_per_strand=1):
                    return(True)

    # Check if the dimer FE < max_FE or if the dimer_FE - 2*monomer_FE < max_dFE
    dimer_FE, dimer_FE_differential = dimer_differential_FE(
        region, ignore_monomer=False)
    if dimer_FE < max_FE or dimer_FE_differential < max_dFE:
        return(True)
    
    return(False)


def scramble_regions(seq, len_region=12, allow_AUG=True, allow_CUG=True, allow_G_quad=True,
                     max_FE=-10, max_dFE=-10, 
                     shortest_disallowed_pal_length=4, shortest_disallowed_near_pal_length=6,
                     to_print=True, max_num_consecutive_nts=100):
    scrambled_seq = copy.deepcopy(seq)
    start = time.time()
    remaining_regions_to_scramble = True
    counter = 0
    while remaining_regions_to_scramble:
        num_scrambled_regions = 0
        counter += 1
        for start_nt in range(len(scrambled_seq) - len_region + 1):
            subseq = scrambled_seq[start_nt : start_nt + len_region]
                
            need_to_scramble_region = decide_if_to_scramble_region(
                subseq, allow_AUG=allow_AUG, allow_CUG=allow_CUG, allow_G_quad=allow_G_quad, 
                max_FE=max_FE, max_dFE=max_dFE, 
                shortest_disallowed_pal_length=shortest_disallowed_pal_length, 
                shortest_disallowed_near_pal_length=shortest_disallowed_near_pal_length,
                max_num_consecutive_nts=max_num_consecutive_nts)
            
            num_times_scrambled = 0
            num_unique_subseq_shuffles = np.math.factorial(len_region) / np.prod(
                    [np.math.factorial(c) for c in [subseq.count(i) for i in ['A', 'C', 'G', 'U']]])
            while need_to_scramble_region and num_times_scrambled < 4 * num_unique_subseq_shuffles:
                if num_times_scrambled == 0:
                    num_scrambled_regions += 1
                num_times_scrambled += 1
                if to_print and num_times_scrambled % 10000 == 0 and num_times_scrambled > 0:
                    print(start_nt, num_times_scrambled, subseq)
                if num_times_scrambled == 4 * num_unique_subseq_shuffles:
                    print('!may be impossible for this subsequence to not make new palindromes!')
                
                subseq = ''.join(np.random.choice(
                    [i for i in subseq],  # scramble self-comp region
                    size=len_region, 
                    replace=False))
                need_to_scramble_region = decide_if_to_scramble_region(
                    subseq, allow_AUG=allow_AUG, allow_CUG=allow_CUG, allow_G_quad=allow_G_quad,
                    max_FE=max_FE, max_dFE=max_dFE, 
                    shortest_disallowed_pal_length=shortest_disallowed_pal_length, 
                    shortest_disallowed_near_pal_length=shortest_disallowed_near_pal_length,
                    max_num_consecutive_nts=max_num_consecutive_nts)

            scrambled_seq = scrambled_seq[: start_nt] + subseq + scrambled_seq[start_nt + len_region :]
        if to_print:
            print(len_region, counter, num_scrambled_regions, np.round(time.time() - start, 2))
        if num_scrambled_regions == 0:
            remaining_regions_to_scramble = False
    return(scrambled_seq)


#%% Examine nanos and pgc in D. melanogaster, as well as in different Drosophila strains

#https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000001215.4/
melanogaster_seqs = []
melanogaster_seq_names = []
with open(folder + 'd_melanogaster.txt') as f:
    for line in f:
        if line[0] == '>':
            melanogaster_seq_names += [line[1:-1]]
            melanogaster_seqs += ['']
        else:
            melanogaster_seqs[-1] += line[:-1].replace('T', 'U')  # remove \n character
f.close()
num_melanogaster_seqs = len(melanogaster_seqs)

nanos_melanogaster_seq = melanogaster_seqs[[(e, i) for e, i in enumerate(melanogaster_seq_names) if 'nanos' in i][0][0]]  # 2 options, but the first is the one we've been using
# nanos_melanogaster_seq_2 = melanogaster_seqs[[(e, i) for e, i in enumerate(melanogaster_seq_names) if 'nanos' in i][1][0]]  # 2 options
pgc_melanogaster_seq = melanogaster_seqs[[(e, i) for e, i in enumerate(melanogaster_seq_names) if 'pgc' in i][0][0]]  # Liz wrote on Nov 7, 2022 that pgc is 680 nts long 
gcl_melanogaster_seq = melanogaster_seqs[[(e, i) for e, i in enumerate(melanogaster_seq_names) if 'germ cell-less (gcl)' in i][0][0]]  # 2 options, not sure which is right


#https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_009870125.1/
pseudoobscura_seqs = []
pseudoobscura_seq_names = []
with open(folder + 'd_pseudoobscura.txt') as f:
    for line in f:
        if line[0] == '>':
            pseudoobscura_seq_names += [line[1:-1]]
            pseudoobscura_seqs += ['']
        else:
            pseudoobscura_seqs[-1] += line[:-1].replace('T', 'U')  # remove \n character
f.close()
num_pseudoobscura_seqs = len(pseudoobscura_seqs)

nanos_pseudoobscura_seq = pseudoobscura_seqs[[(e, i) for e, i in enumerate(pseudoobscura_seq_names) if 'nanos' in i][0][0]]
pgc_pseudoobscura_seq = pseudoobscura_seqs[[(e, i) for e, i in enumerate(pseudoobscura_seq_names) if 'pgc' in i][0][0]]
gcl_pseudoobscura_seq = pseudoobscura_seqs[[(e, i) for e, i in enumerate(pseudoobscura_seq_names) if 'germ cell-less (gcl)' in i][0][0]]


virilis_seqs = []
virilis_seq_names = []
with open(folder + 'd_virilis.txt') as f:
    for line in f:
        if line[0] == '>':
            virilis_seq_names += [line[1:-1]]
            virilis_seqs += ['']
        else:
            virilis_seqs[-1] += line[:-1].replace('T', 'U')  # remove \n character
f.close()
num_virilis_seqs = len(virilis_seqs)

nanos_virilis_seq_1 = virilis_seqs[[(e, i) for e, i in enumerate(virilis_seq_names) if 'nanos' in i][0][0]]  # 2 options
nanos_virilis_seq_2 = virilis_seqs[[(e, i) for e, i in enumerate(virilis_seq_names) if 'nanos' in i][1][0]]  # 2 options
pgc_virilis_seq = virilis_seqs[[(e, i) for e, i in enumerate(virilis_seq_names) if 'pgc' in i][0][0]]
gcl_virilis_seq = virilis_seqs[[(e, i) for e, i in enumerate(virilis_seq_names) if 'germ cell-less (gcl)' in i][0][0]]

# C. elegans homotypic clustering mRNAs
elegans_seqs = []
elegans_seq_names = []
with open(folder + 'c_elegans.txt') as f:
    for line in f:
        if line[0] == '>':
            elegans_seq_names += [line[1:-1]]
            elegans_seqs += ['']
        else:
            elegans_seqs[-1] += line[:-1].replace('T', 'U')  # remove \n character
f.close()
num_elegans_seqs = len(elegans_seqs)

clu1_seq = elegans_seqs[[(e, i) for e, i in enumerate(elegans_seq_names) if 'clu-1' in i][0][0]]  
chs1_seq = elegans_seqs[[(e, i) for e, i in enumerate(elegans_seq_names) if 'chs-1' in i][0][0]]  


nanos_gfp_hybrid = 'TTAGTTGGCGCGTAGCTTTACCACAAAATTCCTGGAATTGCCGTACGCTTCGCAGTTGTTTCAAGTTGTCTAAGGGACATACGATTTTTTTTGCCTCTGCGTCACGATTTTAACCCAAAAGCGAGTTTAGTTACATGTACATTATTATTAGATAAAGAAGTATCGCGAATACTTCAGTTGAATAAACTGTGCTTGGTTTTTGGGTGAGGATTTGTGGAAAGTAGAGTGCGCGATAACCGTAACTTTCGACCCGGATTTTCGCCATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTGTCCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAGCTGACCCTGAAGTTCATCTGCACCACCGGCAAGCTGCCCGTGCCCTGGCCCACCCTCGTGACCACCCTGACCTACGGCGTGCAGTGCTTCAGCCGCTACCCCGACCACATGAAGCAGCACGACTTCTTCAAGTCCGCCATGCCCGAAGGCTACGTCCAGGAGCGCACCATCTTCTTCAAGGACGACGGCAACTACAAGACCCGCGCCGAGGTGAAGTTCGAGGGCGACACCCTGGTGAACCGCATCGAGCTGAAGGGCATCGACTTCAAGGAGGACGGCAACATCCTGGGGCACAAGCTGGAGTACAACTACAACAGCCACAACGTCTATATCATGGCCGACAAGCAGAAGAACGGCATCAAGGTGAACTTCAAGATCCGCCACAACATCGAGGACGGCAGCGTGCAGCTCGCCGACCACTACCAGCAGAACACCCCCATCGGCGACGGCCCCGTGCTGCTGCCCGACAACCACTACCTGAGCACCCAGTCCGCCCTGAGCAAAGACCCCAACGAGAAGCGCGATCACATGGTCCTGCTGGAGTTCGTGACCGCCGCCGGGATCACTCTCGGCATGGACGAGCTGTACAAGTTCCGCAGCAACTTGGAGGGCAGTGGCGCAGCAGCAGTAGGTGTTGCAAATCCCCCCTCGTTGGCTCAGTCTGGAAAGATTTTCCAATTGCAGGATAACTTTTCTGCTTTTCACGCCAGAGGAGGGCTCAACATTCTGGGCCTGCAGGACATGTATTTGGATACCAGTGGGGCCAACTCGTCGGCCACTTTGAGTCCGCCCATTACGCCGGTGACCCCTGACCCGTCGACGTCTGCGCAGTCGACGCACTTCCCTTTTCTGGCCGACAGCGCAGCCACCGCCAATTCGCTCCTTATGCAGCGACAGTACCACTACCACTTGCTGCTCCAGCAGCAGCAGCAACTGGCCATGGCGCAGCACCAATTGGCGCTGGCTGCATCAGCGGCAGCGGCTAGTGCGAGTCACCAGCAAACGGACGAGATTGCGCGATCCTTGAAAATCTTTGCGCAGGTGACGACGGGCGCAGCAGAAAATGCGGCTGGCTCGATGCAGGATGTGATGCAGGAGTTCGCGACCAATGGCTATGCCAGCGATGATCTCGGTCGCATGTCCTACGGGAGTGCTCCGCCACAGGTGCAAATGCCACCGCAGCAGCAGCATCAGCAACAGCAGGGGCTGCACCTGCCACTGGGCCGCAATCCTGCCCAGCTGCAGACCAATGGCGGCAACTTAATGCCCATTCCACTCGCCACCCACTGGCTGAACAACTACCGGGAGCATCTGAACAACGTGTGGCGAAACATGTCGTATATACCAGCCGCTCCCAATACAATGGGTTTGCAGGCCCAAACAGCGGCCACTGTGTCCACCAATCTCGGCGTGGGAATGGGTCTGGGATTGCCCGTGCAGGGCGAACAGCTGCGCGGAGCTTCCAATTCCAGTAACAATAATAACAACAACAACAAGGTGTACAAGCGTTACAACAGCAAGGCCAAAGAGATCAGCCGCCACTGCGTCTTTTGTGAGAATAACAACGAACCAGAGGCGGTTATCAATAGCCACTCAGTGCGAGATAACTTTAACCGAGTGCTGTGCCCCAAACTACGCACCTACGTGTGCCCCATCTGCGGGGCATCTGGGGACTCGGCGCACACGATTAAGTACTGCCCCAAGAAGCCGATCATCACCATGGAGGATGCGATCAAGGCGGAATCGTTCCGCCTAGCCAAGAGCAGTTACTACAAGCAACAGATGAAGGTTTAGAATTCGCGAATCCAGCTCTGGAGCAGAGGCTCTGGCAGCTTTTGCAGCGTTTATATAACATGAAATATATATACGCATTCCGATCAAAGCTGGGTTAACCAGATAGATAGATAGTAACGTTTAAATAGCGCCTGGCGCGTTCGATTTTAAAGAGATTTAGAGCGTTATCCCGTGCCTATAGATCTTATAGTATAGACAACGAACGATCACTCAAATCCAAGTCAATAATTCAAGAATTTATGTCTGTTTCTGTGAAAGGGAAACTAATTTTGTTAAAGAAGACTTACAATATCGTAATACTTGTTCAATCGTCGTGGCCGATAGAAATATCTTACAATCCGAAAGTTGATGAATGGAATTGGTCTGCAACTGGTCGCCTTCATTTCGTAAAATGTTCGCTTGCGGCCGAAAAATTTCGATATATCTACAATTGATCTACAATCTTTACTAAATTTTGAAAAAGGAACACTTTGAATTTCGAACTGTCAATCGTATCATTAGAATTTAATCTAAATTTAAATCTTGCTAAAGGAAATAGCAAGGAACACTTTCGTCGTCGGCTACGCATTCATTGTAAAATTTTAAATTTTGACATTCCGCACTTTTTGATAGATAAGCGAAGAGTATTTTTATTACATGTATCGCAAGTATTCATTTCAACACACATATCTATATATATATATATATATATATATATATATATATATGTTATATATTTATTCAATTTTGTTTACCATTGATCAATTTTTCACACATGAAACAACCGCCAGCATTATATAATTTTTTTATTTTTTTAAAAAATGTGTACACATATTCTGAAAATGAAAAATTCAATGGCTCGAGTGCCAAATAAAGAAATGGTTACAATTTAAG'.replace('T', 'U')


#%% 
pause

#%% Find Drosophila sequences of similar lengths to pgc
pgc_sim_len_mel = [i for i in melanogaster_seqs if (
    len(i) > len(pgc_melanogaster_seq) - 5 and len(i) < len(pgc_melanogaster_seq) + 5
    ) and i != pgc_melanogaster_seq]

pgc_seqs_to_use_prelim = [pgc_melanogaster_seq] + pgc_sim_len_mel

start = time.time()
smith_waterman_pgc = np.zeros((len(pgc_seqs_to_use_prelim), len(pgc_seqs_to_use_prelim)))
for e, i in enumerate(pgc_seqs_to_use_prelim):
    start2 = time.time()
    for e2, j in enumerate(pgc_seqs_to_use_prelim):
        # if e == e2:
        #     smith_waterman_pgc[e, e2] = 0
        if e2 < e:
            smith_waterman_pgc[e, e2] = smith_waterman_pgc[e2, e]
        else:
            smith_waterman_pgc[e, e2] = smith_waterman_score_fast(i, j)  # smith_waterman_score_numba(i, j)
print('smith_waterman_pgc', time.time() - start2, time.time() - start)

duplicate_pgc = np.array([(i, j) for (i, j) in zip(*np.where(smith_waterman_pgc > 40)) 
                            if i < j])

pgc_seqs_to_use = [i for e, i in enumerate(pgc_seqs_to_use_prelim) if e not in duplicate_pgc[:, 1]]

pgc_seqs_filenames = ([folder + 
                       'free_nmers_mel_pgc_nodup_nsubopt' + str(num_subopt) + '_mBP' + str(min_pal_len)] +
                      [folder + 
                       'free_nmers_mel_pgc_nodup_rand5_' + str(i - 1) + '_nsubopt' + str(num_subopt) + '_mBP' + str(min_pal_len)
                       for i in range(1, len(pgc_seqs_to_use))])

pgc_pal_summary_filenames = ([folder + 
                       'pal_summary_mel_pgc_nodup' + '_mBP' + str(min_pal_len)] +
                      [folder + 
                       'pal_summary_mel_pgc_nodup_rand5_' + str(i - 1) + '_mBP' + str(min_pal_len)
                       for i in range(1, len(pgc_seqs_to_use))])

start = time.time()
for e, seq in enumerate(pgc_seqs_to_use):
    try:
        free_nmers = load(pgc_seqs_filenames[e])
    except:
        free_nmers = get_equilibrium_prob_of_free_nmers_faster(
            seq, range(min_pal_len, 20 + 1), num_subopt=num_subopt)
        save(free_nmers, pgc_seqs_filenames[e])

    try:
        pal_summary = load(pgc_pal_summary_filenames[e])
    except:
        _, pal_summary, _ = find_palindromes(
            seq, 
            min_len_palindrome_with_C=min_pal_len, 
            min_len_palindrome=min_pal_len, 
            max_len_palindrome=500, 
            check_all_comp=include_GU
            )
        save(pal_summary, pgc_pal_summary_filenames[e])

        print(e, time.time() - start)


#%% Find Drosophila sequences of similar lengths to nanos
nanos_sim_len_mel = [i for i in melanogaster_seqs if (
    len(i) > len(nanos_melanogaster_seq) - 8 and len(i) < len(nanos_melanogaster_seq) + 8
    ) and i != nanos_melanogaster_seq]

nanos_seqs_to_use_prelim = [nanos_melanogaster_seq] + nanos_sim_len_mel

start = time.time()
smith_waterman_nanos = np.zeros((len(nanos_seqs_to_use_prelim), len(nanos_seqs_to_use_prelim)))
for e, i in enumerate(nanos_seqs_to_use_prelim):
    start2 = time.time()
    for e2, j in enumerate(nanos_seqs_to_use_prelim):
        # if e == e2:
        #     smith_waterman_nanos[e, e2] = 0
        if e2 < e:
            smith_waterman_nanos[e, e2] = smith_waterman_nanos[e2, e]
        else:
            smith_waterman_nanos[e, e2] = smith_waterman_score_fast(i, j)  # smith_waterman_score_numba(i, j)
print('smith_waterman_nanos', time.time() - start2, time.time() - start)

duplicate_nanos = np.array([(i, j) for (i, j) in zip(*np.where(smith_waterman_nanos > 40)) 
                            if i < j])

nanos_seqs_to_use = [i for e, i in enumerate(nanos_seqs_to_use_prelim) if e not in duplicate_nanos[:, 1]]

nanos_seqs_filenames = ([folder + 
                       'free_nmers_mel_nanos_nodup_nsubopt' + str(num_subopt) + '_mBP' + str(min_pal_len)] +
                      [folder + 
                       'free_nmers_mel_nanos_nodup_rand8_' + str(i - 1) + '_nsubopt' + str(num_subopt) + '_mBP' + str(min_pal_len)
                       for i in range(1, len(nanos_seqs_to_use))])

nanos_pal_summary_filenames = ([folder + 
                       'pal_summary_mel_nanos_nodup' + '_mBP' + str(min_pal_len)] +
                      [folder + 
                       'pal_summary_mel_nanos_nodup_rand8_' + str(i - 1) + '_mBP' + str(min_pal_len)
                       for i in range(1, len(nanos_seqs_to_use))])

start = time.time()
for e, seq in enumerate(nanos_seqs_to_use):
    try:
        free_nmers = load(nanos_seqs_filenames[e])
    except:
        free_nmers = get_equilibrium_prob_of_free_nmers_faster(
            seq, range(min_pal_len, 20 + 1), num_subopt=num_subopt)
        save(free_nmers, nanos_seqs_filenames[e])

    try:
        pal_summary = load(nanos_pal_summary_filenames[e])
    except:
        _, pal_summary, _ = find_palindromes(
            seq, 
            min_len_palindrome_with_C=min_pal_len, 
            min_len_palindrome=min_pal_len, 
            max_len_palindrome=500, 
            check_all_comp=include_GU
            )
        save(pal_summary, nanos_pal_summary_filenames[e])

        print(e, time.time() - start)
        

#%% Find Drosophila sequences of similar lengths to gcl
gcl_sim_len_mel = [i for i in melanogaster_seqs if (
    len(i) > len(gcl_melanogaster_seq) - 8 and len(i) < len(gcl_melanogaster_seq) + 8
    ) and i != gcl_melanogaster_seq]

gcl_seqs_to_use_prelim = [gcl_melanogaster_seq] + gcl_sim_len_mel

start = time.time()
smith_waterman_gcl = np.zeros((len(gcl_seqs_to_use_prelim), len(gcl_seqs_to_use_prelim)))
for e, i in enumerate(gcl_seqs_to_use_prelim):
    start2 = time.time()
    for e2, j in enumerate(gcl_seqs_to_use_prelim):
        # if e == e2:
        #     smith_waterman_gcl[e, e2] = 0
        if e2 < e:
            smith_waterman_gcl[e, e2] = smith_waterman_gcl[e2, e]
        else:
            smith_waterman_gcl[e, e2] = smith_waterman_score_fast(i, j)  # smith_waterman_score_numba(i, j)
print('smith_waterman_gcl', time.time() - start2, time.time() - start)

duplicate_gcl = np.array([(i, j) for (i, j) in zip(*np.where(smith_waterman_gcl > 40)) 
                            if i < j])

gcl_seqs_to_use = [i for e, i in enumerate(gcl_seqs_to_use_prelim) if e not in duplicate_gcl[:, 1]]
print(len(gcl_seqs_to_use))

gcl_seqs_filenames = ([folder + 
                       'free_nmers_mel_gcl_nodup_nsubopt' + str(num_subopt) + '_mBP' + str(min_pal_len)] +
                      [folder + 
                       'free_nmers_mel_gcl_nodup_rand8_' + str(i - 1) + '_nsubopt' + str(num_subopt) + '_mBP' + str(min_pal_len)
                       for i in range(1, len(gcl_seqs_to_use))])

gcl_pal_summary_filenames = ([folder + 
                       'pal_summary_mel_gcl_nodup' + '_mBP' + str(min_pal_len)] +
                      [folder + 
                       'pal_summary_mel_gcl_nodup_rand8_' + str(i - 1) + '_mBP' + str(min_pal_len)
                       for i in range(1, len(gcl_seqs_to_use))])

start = time.time()
for e, seq in enumerate(gcl_seqs_to_use):
    try:
        free_nmers = load(gcl_seqs_filenames[e])
    except:
        free_nmers = get_equilibrium_prob_of_free_nmers_faster(
            seq, range(min_pal_len, 20 + 1), num_subopt=num_subopt)
        save(free_nmers, gcl_seqs_filenames[e])

    try:
        pal_summary = load(gcl_pal_summary_filenames[e])
    except:
        _, pal_summary, _ = find_palindromes(
            seq, 
            min_len_palindrome_with_C=min_pal_len, 
            min_len_palindrome=min_pal_len, 
            max_len_palindrome=500, 
            check_all_comp=include_GU
            )
        save(pal_summary, gcl_pal_summary_filenames[e])

        print(e, time.time() - start)


#%%
prob_diss_tau_As = np.array([np.exp(i/(kB * T)) for i in np.arange(4, 16.1, 0.2)])

try: 
    nanos_pgc_mel_rand_binding_mat = load(folder + 
          'FE_nanos_pgc_mel_rand_binding_mat_nodup_A_nsubopt' + str(num_subopt) + '_mBP' + str(min_pal_len))
    pgc_pgc_mel_rand_binding_mat = load(folder + 
          'FE_pgc_pgc_mel_rand_binding_mat_nodup_A_nsubopt' + str(num_subopt) + '_mBP' + str(min_pal_len))
    nanos_nanos_mel_rand_binding_mat = load(folder + 
          'FE_nanos_nanos_mel_rand_binding_mat_nodup_A_nsubopt' + str(num_subopt) + '_mBP' + str(min_pal_len))
except:
    nanos_pgc_mel_rand_binding_mat, pgc_pgc_mel_rand_binding_mat, nanos_nanos_mel_rand_binding_mat = (
        binding_between_two_seq_lists_with_indiv(pgc_seqs_to_use, nanos_seqs_to_use, 
                                      min_pal_len=4, num_subopt=10**5,
                                      free_nmers_1_filenames=pgc_seqs_filenames, 
                                      free_nmers_2_filenames=nanos_seqs_filenames,
                                      free_nmers_1_cutoff=0, free_nmers_2_cutoff=0, 
                                      prob_diss_tau_As=prob_diss_tau_As))
    
    save(nanos_pgc_mel_rand_binding_mat, folder + 
          'FE_nanos_pgc_mel_rand_binding_mat_nodup_A_nsubopt' + str(num_subopt) + '_mBP' + str(min_pal_len))
    save(pgc_pgc_mel_rand_binding_mat, folder + 
          'FE_pgc_pgc_mel_rand_binding_mat_nodup_A_nsubopt' + str(num_subopt) + '_mBP' + str(min_pal_len))
    save(nanos_nanos_mel_rand_binding_mat, folder + 
          'FE_nanos_nanos_mel_rand_binding_mat_nodup_A_nsubopt' + str(num_subopt) + '_mBP' + str(min_pal_len))

FE_nanos_pgc_mel_rand_binding_mat = -kB * T * np.log(nanos_pgc_mel_rand_binding_mat)
FE_pgc_pgc_mel_rand_binding_mat = -kB * T * np.log(pgc_pgc_mel_rand_binding_mat)
FE_nanos_nanos_mel_rand_binding_mat = -kB * T * np.log(nanos_nanos_mel_rand_binding_mat)

FE_nanos_pgc_mel_rand_binding_mat_comp_to_homo = (
    FE_nanos_pgc_mel_rand_binding_mat * 2 - (
        np.expand_dims(FE_pgc_pgc_mel_rand_binding_mat, 1) + 
        np.expand_dims(FE_nanos_nanos_mel_rand_binding_mat, 0)))

nanos_pgc_mel_rand_binding_mat_comp_to_homo = np.exp(-FE_nanos_pgc_mel_rand_binding_mat_comp_to_homo / (kB * T))


for i in [41]: 
    plt.figure()
    plt.hist(FE_nanos_pgc_mel_rand_binding_mat_comp_to_homo[:, :, i].flatten(),
             np.linspace(-7, 4.5, 41), density=True, alpha=0.5, label=r'$\it{pgc}$-like / $\it{nanos}$-like')
    plt.hist(FE_nanos_pgc_mel_rand_binding_mat_comp_to_homo[0, :, i].flatten(), 
             np.linspace(-7, 4.5, 41), density=True, alpha=0.5, label=r'$\it{pgc}$ / $\it{nanos}$-like')
    plt.hist(FE_nanos_pgc_mel_rand_binding_mat_comp_to_homo[:, 0, i].flatten(), 
             np.linspace(-7, 4.5, 41), density=True, alpha=0.5, label=r'$\it{nanos}$ / $\it{pgc}$-like')
    plt.xlabel(
        r'$2\Delta G^{\mathrm{non\!-\!eq}}_{12} - \Delta G^{\mathrm{non\!-\!eq}}_{11} - \Delta G^{\mathrm{non\!-\!eq}}_{22}$'
        + ' (kcal/mol)',
        fontsize=14
    )
    plt.ylabel('Frequency', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.title('Z' if i==0 else r'$\Delta G^\ast = $' + str(np.round(-kB * T * np.log(prob_diss_tau_As[i - 1]), 2)))
    plt.legend(fontsize=12, loc='upper left')
    
    ymax = plt.gca().get_ylim()[1]
    y_arrow = 0.25 * ymax   # place arrows low, away from legend
    
    plt.annotate("",
                 xy=(-7, y_arrow), xytext=(0, y_arrow),
                 arrowprops=dict(arrowstyle="->", lw=1.5))
    plt.text(-4, y_arrow + 0.08*ymax, "more heterotypic",
             ha="center", va="top", fontsize=12)
    
    # right arrow: points right (away from 0)
    plt.annotate("",
                 xy=(4.7, y_arrow), xytext=(0, y_arrow),
                 arrowprops=dict(arrowstyle="->", lw=1.5))
    plt.text(2.5, y_arrow + 0.08*ymax, "more homotypic",
             ha="center", va="top", fontsize=12)

    plt.show()



#%% Look at the palindromic sequences of nanos and pgc more carefully
nupack_material = 'rna'

stem_Zs_mel_n_pal_vec = [[] for _ in range(len(nanos_seqs_to_use))]
stem_Z_mel_n_pal_vec = [[] for _ in range(len(nanos_seqs_to_use))]
stem_free_mel_n_pal_vec = [[] for _ in range(len(nanos_seqs_to_use))]
stem_Zs_indiv_mel_n_pal_vec = [[] for _ in range(len(nanos_seqs_to_use))]
stem_free_indiv_mel_n_pal_vec = [[] for _ in range(len(nanos_seqs_to_use))]

for e_seq, seq in enumerate(nanos_seqs_to_use):
    free_nmers = load(nanos_seqs_filenames[e_seq])
    pal_summary = load(nanos_pal_summary_filenames[e_seq])
    
    (stem_Zs_mel_n_pal_vec[e_seq], stem_Z_mel_n_pal_vec[e_seq], 
     stem_free_mel_n_pal_vec[e_seq], stem_Zs_indiv_mel_n_pal_vec[e_seq], 
     stem_free_indiv_mel_n_pal_vec[e_seq], free_nmers, pal_summary) = get_indiv_stem_info(
               seq, free_nmers=free_nmers, pal_summary=pal_summary)


stem_Zs_mel_p_pal_vec = [[] for _ in range(len(pgc_seqs_to_use))]
stem_Z_mel_p_pal_vec = [[] for _ in range(len(pgc_seqs_to_use))]
stem_free_mel_p_pal_vec = [[] for _ in range(len(pgc_seqs_to_use))]
stem_Zs_indiv_mel_p_pal_vec = [[] for _ in range(len(pgc_seqs_to_use))]
stem_free_indiv_mel_p_pal_vec = [[] for _ in range(len(pgc_seqs_to_use))]

for e_seq, seq in enumerate(pgc_seqs_to_use):
    free_nmers = load(pgc_seqs_filenames[e_seq])
    pal_summary = load(pgc_pal_summary_filenames[e_seq])
    
    (stem_Zs_mel_p_pal_vec[e_seq], stem_Z_mel_p_pal_vec[e_seq], 
     stem_free_mel_p_pal_vec[e_seq], stem_Zs_indiv_mel_p_pal_vec[e_seq], 
     stem_free_indiv_mel_p_pal_vec[e_seq], free_nmers, pal_summary) = get_indiv_stem_info(
               seq, free_nmers=free_nmers, pal_summary=pal_summary)


# There's a problem, which is that I'm considering a stem of length L as well as substems of length e.g. L-2 contained within it.
# I could just consider the longest stem, but it's possible that the longest stem is inaccessible, while a shorter substem
# is both accessible and still a strong enough binder. Therefore, for each longest stem, I consider all its substems,
# and take the maximum weight given by a substem.
stem_FEs_indiv_mel_n_pal_vec = [[[-kB * T * np.log(i) for i in j] for j in k] for k in stem_Zs_indiv_mel_n_pal_vec]
stem_FEs_indiv_mel_p_pal_vec = [[[-kB * T * np.log(i) for i in j] for j in k] for k in stem_Zs_indiv_mel_p_pal_vec]

# This is what we actually use for paper
FEs_to_plot = np.linspace(-13, -4, 101)
substem_max_weight_as_fxn_of_FE_n = [get_max_substem_weight(
    stem_FEs_indiv_mel_n_pal_vec, stem_free_indiv_mel_n_pal_vec, Gstar) for Gstar in FEs_to_plot]
substem_max_weight_as_fxn_of_FE_p = [get_max_substem_weight(
    stem_FEs_indiv_mel_p_pal_vec, stem_free_indiv_mel_p_pal_vec, Gstar) for Gstar in FEs_to_plot]

fig, ax_main = plt.subplots(figsize=(5.5, 4))

for i in range(1, len(nanos_seqs_to_use)):
    ax_main.plot(FEs_to_plot, 
             [np.sum(substem_max_weight_as_fxn_of_FE_n[e][i]) for e in range(len(FEs_to_plot))],
             color='orange', alpha=0.2, label='$\it{nanos}$-like' if i==1 else '')
ax_main.plot(FEs_to_plot, 
         [np.sum(substem_max_weight_as_fxn_of_FE_n[e][0]) for e in range(len(FEs_to_plot))],
         color='blue', label='$\it{nanos}$')

ax_main.set_xlabel(r'$\Delta G^{\!\!\ast}$ (kcal/mol)', fontsize=14)
ax_main.set_ylabel(r'$W^\mathrm{pal}$', fontsize=14)
ax_main.tick_params(axis='both', labelsize=14)
ax_main.set_ylim([5e-6, 5e1])
leg2 = ax_main.legend(loc='lower left', #loc='lower right', 
           fontsize=12)
for lh in leg2.legendHandles:
    lh.set_alpha(1) 
ax_main.set_yscale('log')

ax_inset = ax_main.inset_axes([0.55, 0.15, 0.35, 0.35]) #[0.15, 0.5, 0.4, 0.4])

ax_inset.patch.set_alpha(0.7)

for i in range(1, len(pgc_seqs_to_use)):
    ax_inset.plot(FEs_to_plot, 
             [np.sum(substem_max_weight_as_fxn_of_FE_p[e][i]) for e in range(len(FEs_to_plot))],
             color=colors[5], alpha=0.2, label='$\it{pgc}$-like' if i==1 else '')
ax_inset.plot(FEs_to_plot, 
         [np.sum(substem_max_weight_as_fxn_of_FE_p[e][0]) for e in range(len(FEs_to_plot))],
         color=colors[6], label='$\it{pgc}$')

ax_inset.set_yscale('log')
ax_inset.set_xticks([-12, -8, -4], labelsize=10)
ax_inset.tick_params(axis='both', labelsize=10)
leg_inset = ax_inset.legend(fontsize=10)
for lh in leg_inset.legendHandles:
    lh.set_alpha(1)   # force full opacity in legend

plt.tight_layout()
plt.show()


# Get info on how many accessible and strongly binding stems nanos and pgc have
substem_max_probs_as_fxn_of_FE_p = [get_max_prob_of_substems_with_strong_binding(
    stem_FEs_indiv_mel_p_pal_vec, stem_free_indiv_mel_p_pal_vec, FE_cutoff) for FE_cutoff in FEs_to_plot]
substem_max_probs_as_fxn_of_FE_n = [get_max_prob_of_substems_with_strong_binding(
    stem_FEs_indiv_mel_n_pal_vec, stem_free_indiv_mel_n_pal_vec, FE_cutoff) for FE_cutoff in FEs_to_plot]

print(r'$\Delta G^\ast$', 'values for which pgc (nanos) has more accessible stems than 95% of the other sequences considered')
pstar = 1e-1
print(FEs_to_plot[np.array([np.mean([
    np.sum(np.array(substem_max_probs_as_fxn_of_FE_p[e][i]) > pstar**2) <=
    np.sum(np.array(substem_max_probs_as_fxn_of_FE_p[e][0]) > pstar**2) for i in range(1, len(pgc_seqs_to_use))])
    for e in range(len(FEs_to_plot))]) >=0.95])

print(FEs_to_plot[np.array([np.mean([
    np.sum(np.array(substem_max_probs_as_fxn_of_FE_n[e][i]) > pstar**2) <=
    np.sum(np.array(substem_max_probs_as_fxn_of_FE_n[e][0]) > pstar**2) for i in range(1, len(nanos_seqs_to_use))])
    for e in range(len(FEs_to_plot))]) >=0.95])


#%% Check whether p(homotypic) increases with palindrome content
num_seqs_per_len = 20
len_seqs = [70, 150, 300, 400]
num_len_seqs = len(len_seqs)

np.random.seed(1)
rand_seq_pairs = [[[designRandomSeq(len_seq), designRandomSeq(len_seq)] for _ in range(num_seqs_per_len)] for len_seq in len_seqs]
dros_seqs = [[i for i in melanogaster_seqs if (len(i) > len_seq - 6 and len(i) < len_seq + 6)] for len_seq in len_seqs]
dros_seq_pairs = [[[dros_seqs[e_len_seq][2 * i], dros_seqs[e_len_seq][2 * i + 1]] for i in range(num_seqs_per_len)] for e_len_seq in range(num_len_seqs)]

max_len_pal_to_consider_here = 26
num_pal_lens_to_consider_here = (max_len_pal_to_consider_here-4)//2 + 1
seq_pair_fes = np.zeros((2, num_len_seqs, num_seqs_per_len, 3))  # heterodimer, homodimer1, homodimer2
seq_pair_monomer_fes = np.zeros((2, num_len_seqs, num_seqs_per_len, 2))  # monomer1, monomer2
seq_pair_num_pals = np.zeros((2, num_len_seqs, num_seqs_per_len, 2, num_pal_lens_to_consider_here))  # seq1, seq2; length of pal
seq_pair_pal_FEs = np.ones((2, num_len_seqs, num_seqs_per_len, 2, num_pal_lens_to_consider_here))  # seq1, seq2
for e_r in range(2):
    if e_r == 0:
        seq_pairs = rand_seq_pairs
    elif e_r == 1:
        seq_pairs = dros_seq_pairs
    for e_len_seq, len_seq in enumerate(len_seqs):
        start = time.time()
        for i in range(num_seqs_per_len):
            if (i + 1) % 1 == 0:
                print('Starting pair #' + str(i + 1) + '; time elapsed = ' + str(time.time() - start))
            seq_pair_fes[e_r, e_len_seq, i, 0], seq_pair_fes[e_r, e_len_seq, i, 1,], seq_pair_fes[e_r, e_len_seq, i, 2] = get_dimer_landscape_FE(
                 [seq_pairs[e_len_seq][i][0], seq_pairs[e_len_seq][i][1]], return_all_dimer_FEs=True)
            seq_pair_monomer_fes[e_r, e_len_seq, i, 0] = get_monomer_landscape_FE(seq_pairs[e_len_seq][i][0])
            seq_pair_monomer_fes[e_r, e_len_seq, i, 1] = get_monomer_landscape_FE(seq_pairs[e_len_seq][i][1])
    
            for e_seq in range(2):
                _, pal_summary, _ = find_palindromes(
                    seq_pairs[e_len_seq][i][e_seq], 
                    min_len_palindrome_with_C=4, 
                    min_len_palindrome=4, 
                    max_len_palindrome=500, 
                    check_all_comp=include_GU
                    )
                for e, (pal_start, pal_len) in enumerate(pal_summary):
                    pal_seq = ''.join([seq_pairs[e_len_seq][i][e_seq][pal_start + j] for j in range(pal_len)])
                    seq_pair_num_pals[e_r, e_len_seq, i, e_seq, pal_len//2 - 2] += 1
                    seq_pair_pal_FEs[e_r, e_len_seq, i, e_seq, pal_len//2 - 2] += np.exp(-get_dimer_landscape_FE(
                         [pal_seq, pal_seq], return_all_dimer_FEs=False) / (kB * T))
    

colors_to_use = [colors[5], colors[7], colors[8], colors[14]]
fig, ax = plt.subplots(figsize=(5.5, 5))

all_concatenated_seq_pair_pal_FEs = []
all_concatenated_seq_pair_fes = []

for e_r in range(2):
    for e_len_seq, len_seq in enumerate(len_seqs):
        concatenated_seq_pair_pal_FEs = np.concatenate(
            [-kB * T * np.log(np.sum(seq_pair_pal_FEs[e_r, e_len_seq, :, 0, :], 1)),
             -kB * T * np.log(np.sum(seq_pair_pal_FEs[e_r, e_len_seq, :, 1, :], 1))])
        all_concatenated_seq_pair_pal_FEs = np.concatenate(
            (all_concatenated_seq_pair_pal_FEs, concatenated_seq_pair_pal_FEs))
        
        concatenated_seq_pair_fes = np.concatenate(
            [seq_pair_fes[e_r, e_len_seq, :, 1] - (2 * seq_pair_monomer_fes[e_r, e_len_seq, :, 0]),
             seq_pair_fes[e_r, e_len_seq, :, 2] - (2 * seq_pair_monomer_fes[e_r, e_len_seq, :, 1])])
        all_concatenated_seq_pair_fes = np.concatenate(
            (all_concatenated_seq_pair_fes, concatenated_seq_pair_fes))
        
        ax.plot(concatenated_seq_pair_pal_FEs,
                concatenated_seq_pair_fes,
                color=colors_to_use[e_len_seq],
                marker=['o','^'][e_r],
                linestyle='')

# Linear regression and fit line
slope, intercept, r_value, p_value, std_err = linregress(
    all_concatenated_seq_pair_pal_FEs, all_concatenated_seq_pair_fes)

x_fit = np.linspace(np.min(all_concatenated_seq_pair_pal_FEs),
                    np.max(all_concatenated_seq_pair_pal_FEs), 51)
ax.plot(x_fit, intercept + slope * x_fit, '--k')

ax.annotate(r'$r=$' + f'${r_value:.2f}$\n' + r'$p=$' + f'${p_value:.1e}'.replace("e", r" \times 10^{") + "}$", 
            xy=(-14, -20.5),
            fontsize=12)

ax.set_xlabel(r'$\Delta G^\mathrm{pal}$' + ' (kcal/mol)', fontsize=14)
ax.set_ylabel(r'$\Delta G_{11} - 2\Delta G_1$ (kcal/mol)', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)

# --- Inset plot ---
# Upper left, nudged rightward by increasing borderpad
ax_inset = ax.inset_axes([0.15, 0.7, 0.25, 0.25])

# Semi-transparent background
ax_inset.patch.set_alpha(0.7)

all_x, all_y, all_c, all_m = [], [], [], []
for e_r in range(2):
    for e_len_seq, len_seq in enumerate(len_seqs):
        concatenated_seq_pair_homodimer_fes = np.concatenate(
            [seq_pair_fes[e_r, e_len_seq, :, 1],
             seq_pair_fes[e_r, e_len_seq, :, 2]])
        concatenated_seq_pair_monomer_fes = np.concatenate(
            [(2 * seq_pair_monomer_fes[e_r, e_len_seq, :, 0]),
             (2 * seq_pair_monomer_fes[e_r, e_len_seq, :, 1])])
        all_x.extend(concatenated_seq_pair_monomer_fes)
        all_y.extend(concatenated_seq_pair_homodimer_fes)
        all_c.extend([colors_to_use[e_len_seq]] * len(concatenated_seq_pair_homodimer_fes))
        all_m.extend(['o', '^'][e_r] * len(concatenated_seq_pair_homodimer_fes))
        
# convert to arrays for easy shuffling
all_x = np.array(all_x)
all_y = np.array(all_y)
all_c = np.array(all_c)
all_m = np.array(all_m)

# shuffle order to intermix visually
shuffle_idx = np.random.permutation(len(all_x))
all_x, all_y, all_c, all_m = all_x[shuffle_idx], all_y[shuffle_idx], all_c[shuffle_idx], all_m[shuffle_idx]

for i in range(len(all_x)):
    ax_inset.scatter(all_x[i], all_y[i], c=all_c[i], marker=all_m[i], alpha=0.25, s=10)

ax_inset.set_xlabel(r'$2\Delta G_1$ (kcal/mol)', fontsize=11)
ax_inset.set_ylabel(r'$\Delta G_{11}$ (kcal/mol)', fontsize=11, labelpad=-2)
ax_inset.set_xticks([-300, -200, -100, 0])
ax_inset.set_yticks([-300, -200, -100, 0])
ax_inset.set_ylim([-310, 10])
ax_inset.set_xlim([-310, 10])
ax_inset.tick_params(axis='both', which='major', labelsize=9)

# --- Legends ---
# Legend 1: markers (random vs drosophila)
marker_handles = [
    Line2D([0], [0], marker='o', color='k', linestyle='', label='Random'),
    Line2D([0], [0], marker='^', color='k', linestyle='', label=r'$\it{Drosophila}$')
]
legend1 = ax.legend(handles=marker_handles, loc='upper right',
                    fontsize=10, title="Sequence origin", title_fontsize=11)

# Legend 2: colors (sequence lengths)
color_handles = [
    Line2D([0], [0], marker='s', color=c, linestyle='', label=f'{ls}')
    for c, ls in zip(colors_to_use[:len(len_seqs)], len_seqs)
]
legend2 = ax.legend(handles=color_handles, loc='lower left',
                    fontsize=10, title="Sequence length", title_fontsize=11)

ax.add_artist(legend1)
ax.add_artist(legend2)

plt.tight_layout()
plt.show()


#%% Look at multimerization properties as a function of palindromes
def multimerization_as_fxn_of_length(L, num_mult_seqs=1000):
    seqs = [designRandomSeq(L) for _ in range(num_mult_seqs)]

    seq_conc = np.zeros((num_mult_seqs, max_complex_size))
    total_pal_Zs = np.zeros(num_mult_seqs)
    total_seq_concs = np.zeros(num_mult_seqs)
    
    start = time.time()
    for i, seq in enumerate(seqs):
        total_seq_concs[i] = conc * (20 / L)**2.5

        if (i+1) % 100 == 0 and i > 0:
            print(f'Length {L}: seq #{i+1}; time elapsed = {time.time() - start:.1f} s')

        concs = get_multimer_landscape(seq, max_complex_size=max_complex_size, conc=total_seq_concs[i])

        _, pal_summary, _ = find_palindromes(
            seq,
            min_len_palindrome_with_C=4,
            min_len_palindrome=4,
            max_len_palindrome=500,
            check_all_comp=include_GU
        )

        for pal_start, pal_len in pal_summary:
            pal_seq = ''.join([seq[pal_start + j] for j in range(pal_len)])
            total_pal_Zs[i] += np.exp(-get_dimer_landscape_FE(
                [pal_seq, pal_seq], return_all_dimer_FEs=False) / (kB * T))

        seq_conc[i, :] = concs

    total_pal_FEs = -kB * T * np.log(total_pal_Zs)

    frac_multimers = (
        np.sum((seq_conc * np.expand_dims((np.arange(max_complex_size) + 1), 0))[:, 3:], 1)
        / total_seq_concs
    )

    return total_pal_FEs, frac_multimers

def plot_multimerization(results, colors, bin_width=2.0, min_bin_count=3):
    plt.figure(figsize=(5.5,5))

    # --- gather all scatter points into one list ---
    all_x, all_y, all_c = [], [], []
    for idx, L in enumerate(results.keys()):
        total_pal_FEs, frac_multimers = results[L]
        all_x.extend(total_pal_FEs)
        all_y.extend(frac_multimers)
        all_c.extend([colors[idx]] * len(total_pal_FEs))

    # convert to arrays for easy shuffling
    all_x = np.array(all_x)
    all_y = np.array(all_y)
    all_c = np.array(all_c)

    # shuffle order to intermix visually
    shuffle_idx = np.random.permutation(len(all_x))
    all_x, all_y, all_c = all_x[shuffle_idx], all_y[shuffle_idx], all_c[shuffle_idx]

    # plot all scatters at once
    plt.scatter(all_x, all_y, c=all_c, alpha=0.1, s=10)

    # --- then plot curves for each length ---        
    for idx, L in enumerate(results.keys()):
        total_pal_FEs, frac_multimers = results[L]

        # binning
        bins = np.arange(np.min(total_pal_FEs), bin_width + 1e-9, bin_width)
        bin_indices = np.digitize(total_pal_FEs, bins)
        bin_means_x, bin_means_y = [], []

        for b in range(1, len(bins)):
            mask = bin_indices == b
            if np.sum(mask) >= min_bin_count:
                bin_means_x.append(np.mean(total_pal_FEs[mask]))
                bin_means_y.append(np.mean(frac_multimers[mask]))

        plt.plot(
            bin_means_x,
            bin_means_y,
            '-o',
            color=colors[idx],
            lw=2,
            label=f"{L} nt"
        )

    plt.ylabel('Fraction of strands\nin higher-order multimers', fontsize=14)
    plt.xlabel(r'$\Delta G^\mathrm{pal}$' + ' (kcal/mol)', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.legend(title="RNA length", fontsize=14, title_fontsize=15)
    plt.tight_layout()
    plt.show()


num_mult_seqs = 500
rna_lengths = [20, 60, 100, 150, 200]
max_complex_size = 10
conc = 4e-3

# Generate data
np.random.seed(1)
multimerization_results = {}
for L in rna_lengths:
    multimerization_results[L] = multimerization_as_fxn_of_length(L, num_mult_seqs=num_mult_seqs)

# Plot results
bin_width = 2.0      # user-specified bin width (kcal/mol)
min_bin_count = 3    # minimum number of points per bin
plot_multimerization(multimerization_results, colors, bin_width=bin_width, min_bin_count=min_bin_count)

all_pal_FEs = np.concatenate([multimerization_results[i][0] for i in rna_lengths])
all_multimerization_fracs = np.concatenate([multimerization_results[i][1] for i in rna_lengths])
slope, intercept, r_value, p_value, std_err = linregress(
   all_pal_FEs[all_pal_FEs < np.inf], 
   all_multimerization_fracs[all_pal_FEs < np.inf])
print('p = ', p_value)


#%% Show that palindromes increase probability of homodimers vs heterodimers
num_seq_pairs = 10000
np.random.seed(1)
len_seq = 30
max_num_impossible_times_to_try = 50
shortest_disallowed_near_pal_length = 40  # 6
seq_pair_fes = np.zeros((num_seq_pairs, 3, 2))  # heterodimer, homodimer1, homodimer2; either random or remove_palindromes
for e_r, remove_palindromes in enumerate([False, True]):
    seq_pairs = [[designRandomSeq(len_seq), designRandomSeq(len_seq)] for _ in range(num_seq_pairs)]
    if remove_palindromes:
        start = time.time()
        for e_p, pair in enumerate(seq_pairs):
            if (e_p + 1) % 1000 == 0:
                print('Removing palindromes from pair #' + str(e_p + 1) + '; time elapsed = ' + str(time.time() - start))
                print('max_num_impossible_times_to_try =', max_num_impossible_times_to_try)
            for e_s, seq in enumerate(pair):
                while np.sum(find_palindromes(seq, 4, 4, 500, True)[-1]) > 0:  # remove 4- and 6-nt palindromes
                    seq = designRandomSeq(len_seq)
                seq_pairs[e_p][e_s] = seq
    
    max_len_self_comp_region = np.zeros((num_seq_pairs, 2))
    mfe_self_comp_region = np.zeros((num_seq_pairs, 2))
    total_fe_self_comp_region = np.zeros((num_seq_pairs, 2))
    start = time.time()
    for i in range(num_seq_pairs):
        if (i + 1) % 1000 == 0:
            print('Starting pair #' + str(i + 1) + '; time elapsed = ' + str(time.time() - start))
        seq_pair_fes[i, 0, e_r], seq_pair_fes[i, 1, e_r], seq_pair_fes[i, 2, e_r] = get_dimer_landscape_FE(
             [seq_pairs[i][0], seq_pairs[i][1]], return_all_dimer_FEs=True)
    
    plt.figure()
    plt.bar(['Homodimer', 'Heterodimer'], 
            [np.mean(2 * seq_pair_fes[:, 0, e_r] >= seq_pair_fes[:, 1, e_r] + seq_pair_fes[:, 2, e_r]), 
             np.mean(2 * seq_pair_fes[:, 0, e_r] < seq_pair_fes[:, 1, e_r] + seq_pair_fes[:, 2, e_r])], 
            yerr=np.array(
                [np.std(2 * seq_pair_fes[:, 0, e_r] >= seq_pair_fes[:, 1, e_r] + seq_pair_fes[:, 2, e_r]), 
                 np.std(2 * seq_pair_fes[:, 0, e_r] < seq_pair_fes[:, 1, e_r] + seq_pair_fes[:, 2, e_r])]
                ) / np.sqrt(num_seq_pairs)
            , 
            color=['#004D40', '#FFC107'][remove_palindromes]
            )
    plt.ylabel('Fraction')
    if remove_palindromes:
        if shortest_disallowed_near_pal_length <= len_seq:
            plt.title('After removing palindromes and near-palindromes')
        else:
            plt.title('Sequences with no palindromes')
    else:
        plt.title('Random sequences')#, fontdict={'color': ['#1E88E5', '#FFC107'][remove_palindromes]})
    plt.show()
    
    
    # Check number of palindromes
    num_6mers = np.zeros(num_seq_pairs * 2)
    counter = 0
    for pair in seq_pairs:
        for seq in pair:
            num_6mers[counter] = find_palindromes(seq, 6, 6, 6, True)[2][-1]
            counter += 1
            
    plt.figure()
    plt.hist(num_6mers, 
             np.linspace(0, max(num_6mers) + 1, int(max(num_6mers) + 2)))
    plt.xlabel('Number of 6-nt palindromes')
    plt.ylabel('Count')
    plt.show()
    
plt.figure(figsize=(2.3, 4))  # default is (6, 4)
categories = ['Random', 'No\npalindromes']
homodimer_freqs = [np.mean(2 * seq_pair_fes[:, 0, 0] >= seq_pair_fes[:, 1, 0] + seq_pair_fes[:, 2, 0]), 
                   np.mean(2 * seq_pair_fes[:, 0, 1] >= seq_pair_fes[:, 1, 1] + seq_pair_fes[:, 2, 1])]
heterodimer_freqs = [np.mean(2 * seq_pair_fes[:, 0, 0] < seq_pair_fes[:, 1, 0] + seq_pair_fes[:, 2, 0]), 
                     np.mean(2 * seq_pair_fes[:, 0, 1] < seq_pair_fes[:, 1, 1] + seq_pair_fes[:, 2, 1])]
plt.bar(categories, 
        homodimer_freqs, 
        label='Homodimer', 
        color='#004D40'
        )
plt.bar(categories,
        heterodimer_freqs,
        bottom=homodimer_freqs,
        label='Heterodimer',
        color='#FFC107')
plt.gca().set_xticks([0, 1], ['Random', 'No\npalindromes'], fontsize=12)
plt.gca().set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], [0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=12)
plt.ylabel('Fraction', fontsize=14)
plt.xlabel('Sequences', fontsize=14)
plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
plt.savefig('/Users/Ofer/Dropbox/LSI 2021/homotypic clusters/Figures/homodimer_heterodimer', 
            bbox_inches='tight', dpi=300)
plt.show()


#%% Reviewer responses 1

(pseudo_nanos_seqs, pseudo_nanos_seqs_filenames, pseudo_nanos_pal_summary_filenames, 
  FEs_to_plot, substem_max_weight_as_fxn_of_FE_pseudo_nanos) = get_sim_seqs_Wpal_comp_and_plot(
      nanos_pseudoobscura_seq, pseudoobscura_seqs, 'pseudo_nanos', 'nanos', len_diff_cutoff=11)

(pseudo_pgc_seqs, pseudo_pgc_seqs_filenames, pseudo_pgc_pal_summary_filenames, 
  FEs_to_plot, substem_max_weight_as_fxn_of_FE_pseudo_pgc) = get_sim_seqs_Wpal_comp_and_plot(
      pgc_pseudoobscura_seq, pseudoobscura_seqs, 'pseudo_pgc', 'pgc', len_diff_cutoff=6)


(vir_nanos_seqs, vir_nanos_seqs_filenames, vir_nanos_pal_summary_filenames, 
  FEs_to_plot, substem_max_weight_as_fxn_of_FE_vir_nanos) = get_sim_seqs_Wpal_comp_and_plot(
      nanos_virilis_seq_1, virilis_seqs, 'vir_nanos', 'nanos', len_diff_cutoff=11)  

(vir_pgc_seqs, vir_pgc_seqs_filenames, vir_pgc_pal_summary_filenames, 
  FEs_to_plot, substem_max_weight_as_fxn_of_FE_vir_pgc) = get_sim_seqs_Wpal_comp_and_plot(
      pgc_virilis_seq, virilis_seqs, 'vir_pgc', 'pgc', len_diff_cutoff=7)


binding_mat_comp_pgc_mel_nanos_mel = get_prob_diss_tau_As(
    pgc_seqs_to_use, nanos_seqs_to_use, pgc_seqs_filenames, nanos_seqs_filenames,
    'pgc_mel_' + str(len(pgc_seqs_to_use)), 'nanos_mel_' + str(len(nanos_seqs_to_use)),
    plot_name_1=r'$\it{pgc}$', plot_name_2=r'$\it{nanos}$')

binding_mat_comp_pgc_pseudo_nanos_pseudo = get_prob_diss_tau_As(
    pgc_seqs_to_use, nanos_seqs_to_use, pgc_seqs_filenames, nanos_seqs_filenames,
    'pgc_pseudo_' + str(len(pgc_seqs_to_use)), 'nanos_pseudo_' + str(len(nanos_seqs_to_use)),
    plot_name_1=r'$\it{pgc}$', plot_name_2=r'$\it{nanos}$')

binding_mat_comp_pgc_vir_nanos_vir = get_prob_diss_tau_As(
    pgc_seqs_to_use, nanos_seqs_to_use, pgc_seqs_filenames, nanos_seqs_filenames,
    'pgc_vir_' + str(len(pgc_seqs_to_use)), 'nanos_vir_' + str(len(nanos_seqs_to_use)),
    plot_name_1=r'$\it{pgc}$', plot_name_2=r'$\it{nanos}$')



#%%Reviewer responses 2


#%% Binding between nanos & nanos-gfp hybrid
nanosgfp_free_nmers = get_equilibrium_prob_of_free_nmers_faster(
    nanos_gfp_hybrid, range(min_pal_len, 20 + 1), num_subopt=num_subopt)
save(nanosgfp_free_nmers, 
      folder + 
      'free_nmers_mel_nanosgfp_nsubopt' + str(num_subopt) + '_mBP' + str(min_pal_len))

binding_mat_comp_nanos_mel_nanosgfp_mel = get_prob_diss_tau_As(
    [nanos_melanogaster_seq], [nanos_gfp_hybrid], 
    [folder + 
     'free_nmers_mel_nanosagain_nsubopt' + str(num_subopt) + '_mBP' + str(min_pal_len)], 
    [folder + 
     'free_nmers_mel_nanosgfp_nsubopt' + str(num_subopt) + '_mBP' + str(min_pal_len)],
    'nanos_mel_' + str(1), 'nanosgfp_mel_' + str(1), 
    to_plot=False)

prob_diss_tau_As = np.array([np.exp(i/(kB * T)) for i in np.arange(4, 16.1, 0.2)])
plt.figure()
plt.plot(-kB * T * np.log(prob_diss_tau_As), 
         binding_mat_comp_nanos_mel_nanosgfp_mel[0, 0, 1:], label='Non-equilibrium')
plt.plot(-kB * T * np.log(prob_diss_tau_As), 
         [binding_mat_comp_nanos_mel_nanosgfp_mel[0, 0, 0]] * len(prob_diss_tau_As),
         label='Quasi-equilibrium')
plt.legend(title='Prediction', fontsize=10, title_fontsize=12)
plt.ylabel(
    r'$2\Delta G^{\mathrm{non\!-\!eq}}_{12} - \Delta G^{\mathrm{non\!-\!eq}}_{11} - \Delta G^{\mathrm{non\!-\!eq}}_{22}$'
    + ' (kcal/mol)',
    fontsize=10
)
plt.xlabel(r'$\Delta G^\ast$' + ' (kcal/mol)', fontsize=14)
plt.title(r'$\it{nanos}$' + ' / ' + r'$\it{nanos}$' + '-' + r'$\it{gfp}$', 
          fontsize=14)
plt.show()

binding_mat_comp_nanos_mel_pgc_mel = get_prob_diss_tau_As(
    [nanos_melanogaster_seq], [pgc_melanogaster_seq], 
    [folder + 'free_nmers_' + 'mel_nanos' + '_nodup_nsubopt' + str(num_subopt) + '_mBP' + str(min_pal_len)],
    [folder + 'free_nmers_' + 'mel_pgc' + '_nodup_nsubopt' + str(num_subopt) + '_mBP' + str(min_pal_len)],
    'nanos_mel_' + str(1), 'pgc_mel_' + str(1), 
    to_plot=False)


#%% Binding between random sequences with no palindromes
len_seq = 1000
min_disallowed_pal_length = 6

np.random.seed(1)
for i in range(21):
    start = time.time()
    counter = 0
    rand_no_pal_seq = designRandomSeq(len_seq)
    if len_seq > 200 and min_disallowed_pal_length < 8:
        rand_no_pal_seq = scramble_regions(designRandomSeq(len_seq), to_print=True)
    else:
        while np.sum(find_palindromes(
                rand_no_pal_seq, min_disallowed_pal_length, min_disallowed_pal_length, 500, True)[-1]) > 0:  
            rand_no_pal_seq = designRandomSeq(len_seq)
            counter += 1
            if counter % 1000 == 0:
                print(i, counter, time.time() - start)
    save(rand_no_pal_seq, 
          folder + 
          'no_pal_' + str(min_disallowed_pal_length) + '_seq_' + str(len_seq) + '_' + str(i))
    rand_no_pal_free_nmers = get_equilibrium_prob_of_free_nmers_faster(
        rand_no_pal_seq, range(min_pal_len, 20 + 1), num_subopt=num_subopt)
    rand_no_pal_free_nmers_filename = (
        folder + 
        'free_nmers_' + 'no_pal_' + str(min_disallowed_pal_length) + '_seq_' + str(len_seq) + 
        '_' + str(i) + '_nsubopt' + str(num_subopt) + '_mBP' + str(min_pal_len))
    save(rand_no_pal_free_nmers, rand_no_pal_free_nmers_filename)


binding_mat_comp_rand_nopal = get_prob_diss_tau_As(
    [load(folder + 
    'no_pal_' + str(min_disallowed_pal_length) + '_seq_' + str(len_seq) + '_' + str(i))
     for i in range(10)], 
    [load(folder + 
    'no_pal_' + str(min_disallowed_pal_length) + '_seq_' + str(len_seq) + '_' + str(i))
     for i in range(10, 21)], 
    [folder + 
        'free_nmers_' + 'no_pal_' + str(min_disallowed_pal_length) + '_seq_' + str(len_seq) + 
        '_' + str(i) + '_nsubopt' + str(num_subopt) + '_mBP' + str(min_pal_len)
        for i in range(10)],
    [folder + 
        'free_nmers_' + 'no_pal_' + str(min_disallowed_pal_length) + '_seq_' + str(len_seq) + 
        '_' + str(i) + '_nsubopt' + str(num_subopt) + '_mBP' + str(min_pal_len)
        for i in range(10, 21)],
    'randnopal_' + str(min_disallowed_pal_length) + '_' + str(len_seq) + '_' + str(10), 
    'randnopal_' + str(min_disallowed_pal_length) + '_' + str(len_seq) + '_' + str(11), 
    to_plot=False, to_load=False)

prob_diss_tau_As = np.array([np.exp(i/(kB * T)) for i in np.arange(4, 16.1, 0.2)])
plt.figure()
plt.plot(-kB * T * np.log(prob_diss_tau_As), 
         binding_mat_comp_rand_nopal[0, 0, 1:], label='Eq. 2')
plt.plot(-kB * T * np.log(prob_diss_tau_As), 
         [binding_mat_comp_rand_nopal[0, 0, 0]] * len(prob_diss_tau_As),
         label='Eq. RR1')
plt.legend(title='Prediction', fontsize=10, title_fontsize=12)
plt.ylabel(
    r'$2\Delta G^{\mathrm{non\!-\!eq}}_{12} - \Delta G^{\mathrm{non\!-\!eq}}_{11} - \Delta G^{\mathrm{non\!-\!eq}}_{22}$'
    + ' (kcal/mol)',
    fontsize=10
)
plt.xlabel(r'$\Delta G^\ast$' + ' (kcal/mol)', fontsize=14)
plt.title('Random, no palindromes / Random, no palindromes', 
          fontsize=14)
plt.show()


#%% Binding between lots of pairs of RNAs 
prob_diss_tau_As = np.array([np.exp(i/(kB * T)) for i in np.arange(4, 16.1, 0.2)])

binding_mat_comp_nanos_mel_pgc_mel = get_prob_diss_tau_As(
    [nanos_melanogaster_seq], [pgc_melanogaster_seq], 
    [folder + 'free_nmers_' + 'mel_nanos' + '_nodup_nsubopt' + str(num_subopt) + '_mBP' + str(min_pal_len)],
    [folder + 'free_nmers_' + 'mel_pgc' + '_nodup_nsubopt' + str(num_subopt) + '_mBP' + str(min_pal_len)],
    'nanos_mel_' + str(1), 'pgc_mel_' + str(1), 
    to_plot=False)
binding_mat_comp_nanos_mel_gcl_mel = get_prob_diss_tau_As(
    [nanos_melanogaster_seq], [gcl_melanogaster_seq], 
    [folder + 'free_nmers_' + 'mel_nanos' + '_nodup_nsubopt' + str(num_subopt) + '_mBP' + str(min_pal_len)],
    [folder + 'free_nmers_' + 'mel_gcl' + '_nodup_nsubopt' + str(num_subopt) + '_mBP' + str(min_pal_len)],
    'nanos_mel_' + str(1), 'gcl_mel_' + str(1), 
    to_plot=False)
binding_mat_comp_pgc_mel_gcl_mel = get_prob_diss_tau_As(
    [pgc_melanogaster_seq], [gcl_melanogaster_seq], 
    [folder + 'free_nmers_' + 'mel_pgc' + '_nodup_nsubopt' + str(num_subopt) + '_mBP' + str(min_pal_len)],
    [folder + 'free_nmers_' + 'mel_gcl' + '_nodup_nsubopt' + str(num_subopt) + '_mBP' + str(min_pal_len)],
    'pgc_mel_' + str(1), 'gcl_mel_' + str(1), 
    to_plot=False)
binding_mat_comp_nanos_vir_pgc_vir = get_prob_diss_tau_As(
    [nanos_virilis_seq_1], [pgc_virilis_seq], 
    [folder + 'free_nmers_' + 'vir_nanos' + '_nodup_nsubopt' + str(num_subopt) + '_mBP' + str(min_pal_len)],
    [folder + 'free_nmers_' + 'vir_pgc' + '_nodup_nsubopt' + str(num_subopt) + '_mBP' + str(min_pal_len)],
    'nanos_vir_' + str(1), 'pgc_vir_' + str(1), 
    to_plot=False)
binding_mat_comp_nanos_vir_gcl_vir = get_prob_diss_tau_As(
    [nanos_virilis_seq_1], [gcl_virilis_seq], 
    [folder + 'free_nmers_' + 'vir_nanos' + '_nodup_nsubopt' + str(num_subopt) + '_mBP' + str(min_pal_len)],
    [folder + 'free_nmers_' + 'vir_gcl' + '_nodup_nsubopt' + str(num_subopt) + '_mBP' + str(min_pal_len)],
    'nanos_vir_' + str(1), 'gcl_vir_' + str(1), 
    to_plot=False)
binding_mat_comp_pgc_vir_gcl_vir = get_prob_diss_tau_As(
    [pgc_virilis_seq], [gcl_virilis_seq], 
    [folder + 'free_nmers_' + 'vir_pgc' + '_nodup_nsubopt' + str(num_subopt) + '_mBP' + str(min_pal_len)],
    [folder + 'free_nmers_' + 'vir_gcl' + '_nodup_nsubopt' + str(num_subopt) + '_mBP' + str(min_pal_len)],
    'pgc_vir_' + str(1), 'gcl_vir_' + str(1), 
    to_plot=False)
binding_mat_comp_nanos_pseudo_pgc_pseudo = get_prob_diss_tau_As(
    [nanos_pseudoobscura_seq], [pgc_pseudoobscura_seq], 
    [folder + 'free_nmers_' + 'pseudo_nanos' + '_nodup_nsubopt' + str(num_subopt) + '_mBP' + str(min_pal_len)],
    [folder + 'free_nmers_' + 'pseudo_pgc' + '_nodup_nsubopt' + str(num_subopt) + '_mBP' + str(min_pal_len)],
    'nanos_pseudo_' + str(1), 'pgc_pseudo_' + str(1), 
    to_plot=False)
binding_mat_comp_nanos_pseudo_gcl_pseudo = get_prob_diss_tau_As(
    [nanos_pseudoobscura_seq], [gcl_pseudoobscura_seq], 
    [folder + 'free_nmers_' + 'pseudo_nanos' + '_nodup_nsubopt' + str(num_subopt) + '_mBP' + str(min_pal_len)],
    [folder + 'free_nmers_' + 'pseudo_gcl' + '_nodup_nsubopt' + str(num_subopt) + '_mBP' + str(min_pal_len)],
    'nanos_pseudo_' + str(1), 'gcl_pseudo_' + str(1), 
    to_plot=False)
binding_mat_comp_pgc_pseudo_gcl_pseudo = get_prob_diss_tau_As(
    [pgc_pseudoobscura_seq], [gcl_pseudoobscura_seq], 
    [folder + 'free_nmers_' + 'pseudo_pgc' + '_nodup_nsubopt' + str(num_subopt) + '_mBP' + str(min_pal_len)],
    [folder + 'free_nmers_' + 'pseudo_gcl' + '_nodup_nsubopt' + str(num_subopt) + '_mBP' + str(min_pal_len)],
    'pgc_pseudo_' + str(1), 'gcl_pseudo_' + str(1), 
    to_plot=False)

binding_mat_no_pal_1000 = get_prob_diss_tau_As(
        [], [], [], [],
        'randnopal_' + str(6) + '_' + str(1001) + '_' + str(10), 
        'randnopal_' + str(6) + '_' + str(1001) + '_' +str(11), 
        to_plot=False)

binding_mat_comp_clu_ele_chs_ele = get_prob_diss_tau_As(
    [clu1_seq], [chs1_seq], 
    [folder + 'free_nmers_' + 'ele_clu' + '_nodup_nsubopt' + str(num_subopt) + '_mBP' + str(min_pal_len)],
    [folder + 'free_nmers_' + 'ele_chs' + '_nodup_nsubopt' + str(num_subopt) + '_mBP' + str(min_pal_len)],
    'clu_ele_' + str(1), 'chs_ele_' + str(1), 
    to_plot=False,
    plot_name_1=r'$\it{clu}$' + '-1', plot_name_2=r'$\it{chs}$' + '-1')

plt.figure()
plt.plot(-kB * T * np.log(prob_diss_tau_As), 
         binding_mat_comp_nanos_mel_pgc_mel[0, 0, 1:], 
         color=colors[0], linestyle='-',
         label=r'$\it{nanos / pgc}$')
plt.plot(-kB * T * np.log(prob_diss_tau_As), 
         binding_mat_comp_nanos_mel_gcl_mel[0, 0, 1:], 
         color=colors[1], linestyle='-',
         label=r'$\it{nanos / gcl}$')
plt.plot(-kB * T * np.log(prob_diss_tau_As), 
         binding_mat_comp_pgc_mel_gcl_mel[0, 0, 1:], 
         color=colors[2], linestyle='-',
         label=r'$\it{pgc / gcl}$')
plt.plot(-kB * T * np.log(prob_diss_tau_As), 
         binding_mat_comp_nanos_vir_pgc_vir[0, 0, 1:], 
         color=colors[0], linestyle='--',
         # label=r'$\it{nanos, vir / pgc, vir}$'
         )
plt.plot(-kB * T * np.log(prob_diss_tau_As), 
         binding_mat_comp_nanos_vir_gcl_vir[0, 0, 1:], 
         color=colors[1], linestyle='--',
         # label=r'$\it{nanos, vir / gcl, vir}$'
         )
plt.plot(-kB * T * np.log(prob_diss_tau_As), 
         binding_mat_comp_pgc_vir_gcl_vir[0, 0, 1:], 
         color=colors[2], linestyle='--',
         # label=r'$\it{pgc, vir / gcl, vir}$'
         )
plt.plot(-kB * T * np.log(prob_diss_tau_As), 
         binding_mat_comp_nanos_pseudo_pgc_pseudo[0, 0, 1:], 
         color=colors[0], linestyle=':',
         # label=r'$\it{nanos, pseudo / pgc, pseudo}$'
         )
plt.plot(-kB * T * np.log(prob_diss_tau_As), 
         binding_mat_comp_nanos_pseudo_gcl_pseudo[0, 0, 1:], 
         color=colors[1], linestyle=':',
         # label=r'$\it{nanos, pseudo / gcl, pseudo}$'
         )
plt.plot(-kB * T * np.log(prob_diss_tau_As), 
         binding_mat_comp_pgc_pseudo_gcl_pseudo[0, 0, 1:], 
         color=colors[2], linestyle=':',
         # label=r'$\it{pgc, pseudo / gcl, pseudo}$'
         )
plt.plot(-kB * T * np.log(prob_diss_tau_As), 
         binding_mat_comp_clu_ele_chs_ele[0, 0, 1:], 
         color=colors[3], linestyle='-',
         label=r'$\it{clu}$' + '-1/' + r'$\it{chs}$' + '-1'
         )

y_mean = np.mean(binding_mat_no_pal_1000[:, :, 1:], (0,1))
plt.plot(-kB * T * np.log(prob_diss_tau_As), 
          y_mean, 
          color='gray', linestyle='-.',
          label='no palindromes'
          )
first_legend = plt.legend(title='Sequence pair', fontsize=10, title_fontsize=12, 
           loc='upper right', # bbox_to_anchor=(1.05, 1),
           )
plt.gca().add_artist(first_legend)

plt.ylabel(
    r'$2\Delta G^{\mathrm{non\!-\!eq}}_{12} - \Delta G^{\mathrm{non\!-\!eq}}_{11} - \Delta G^{\mathrm{non\!-\!eq}}_{22}$'
    + ' (kcal/mol)', fontsize=10
)
plt.xlabel(r'$\Delta G^\ast$' + ' (kcal/mol)', fontsize=14, )
plt.ylim([-2.7, 4.5])

custom_lines = [
    Line2D([0], [0], color="black", linestyle="-"),   # mel
    Line2D([0], [0], color="black", linestyle="--"),  # vir
    Line2D([0], [0], color="black", linestyle=":")    # pseu
]
second_legend = plt.gca().legend(
    custom_lines,
    [r"$\it{D. melanogaster}$", r"$\it{D. virilis}$", r"$\it{D. pseudoobscura}$"],
    title="Species", title_fontsize=12,
    loc="lower right"
)
second_legend.get_frame().set_alpha(0.3)

plt.gca().add_artist(second_legend)

plt.savefig(folder + 'RR2.png', dpi=400)
plt.show()


nat_seq_G12 = np.array([binding_mat_comp_nanos_mel_pgc_mel[0, 0, 41], 
               binding_mat_comp_nanos_mel_gcl_mel[0, 0, 41], 
               binding_mat_comp_pgc_mel_gcl_mel[0,0,41], 
               binding_mat_comp_nanos_vir_pgc_vir[0,0,41],
               binding_mat_comp_nanos_vir_gcl_vir[0,0,41],  
               binding_mat_comp_pgc_vir_gcl_vir[0,0,41], 
               binding_mat_comp_nanos_pseudo_pgc_pseudo[0,0,41],
               binding_mat_comp_nanos_pseudo_gcl_pseudo[0,0,41], 
               binding_mat_comp_pgc_pseudo_gcl_pseudo[0,0,41], 
               binding_mat_comp_clu_ele_chs_ele[0,0,41]])
neg_ctrl_G12 = np.array(flatten(binding_mat_no_pal_1000[:, :, 41]))

print(ttest_ind(nat_seq_G12, neg_ctrl_G12, alternative='greater'))



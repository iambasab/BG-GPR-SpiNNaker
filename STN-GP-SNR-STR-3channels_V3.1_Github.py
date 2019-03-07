"""

21st February 2019: Uploading on Github - Basabdatta
*********************************************
UPDATES ON 30TH MAY:
I HAD SENT THE DOCUMENT TO SEB ON FRIDAY LAST I.E. 26TH MAY - AND AWAITING RESPONSE.

MEANWHILE, I HAD FOOLISHLY NOT UPDATED THE STATUS HERE, AND GOT CONFUSED ON THE DIFFERENCE BETWEEN THIS FILE
AND V2.2. TOOK THE PRINTS OF BOTH AND CHECKED LINE BY LINE:
THE ONLY DIFFERENCE IS THAT THIS CODE IS A MULTIPLE TRIAL (LOOP-ED) VERSION, WHILE V2.2 IS A SINGLE TRIAL VERSION.

STAMPED AND DATED AS FINAL VERSION.
*******************************************************
UPDATES ON 8TH MARCH

Modified the code to make it spynnaker8 compatible.
"""


# !/usr/bin/python

import spynnaker8 as p
import matplotlib.pyplot as plt
import numpy as np
from numpy import *
import time
start_time = time.time()

TotalDuration = 10000####10000  ### TOTAL RUN TIME
TimeInt = 0.1    ### SIMULATION TIME STEP
TotalDataPoints = int(TotalDuration*(1/TimeInt))


'''PARAMETERS USED IN DEFINING THE BASIC MODEL UNITS: IZHIKEVICH NEURONS'''
tau_ampa = 6.0### ampa synapse time constant
E_ampa = 0.0 ## ampa synapse reversal potential


tau_gabaa= 4.0### gaba synapse time constant
E_gabaa = -80.0 ## gaba synapse reversal potential

strd1_a=0.02
strd1_b=0.2
strd1_c=-65.0
strd1_d=8.0
strd1_v_init = -80.0
strd1_u_init = strd1_b * strd1_v_init


strd2_a=0.02
strd2_b=0.2
strd2_c=-65.0
strd2_d=8.0
strd2_v_init = -80.0
strd2_u_init = strd2_b * strd2_v_init

current_bias_str = -30.0

fsi_a=0.1
fsi_b=0.2
fsi_c=-65.0
fsi_d=8.0
fsi_v_init = -70.0
fsi_u_init = fsi_b * fsi_v_init

current_bias_fsi = -10.0

gpe_a=0.005
gpe_b=0.585
gpe_c=-65.0
gpe_d=4.0
gpe_v_init = -70.0
gpe_u_init = gpe_b * gpe_v_init

current_bias_gpe = 2.0

snr_a = 0.005
snr_b = 0.32
snr_c = -65.0
snr_d = 2.0
snr_v_init = -70.0
snr_u_init = snr_b * snr_v_init

current_bias_snr = 5.0

stn_a=0.005
stn_b=0.265
stn_c=-65.0
stn_d=2.0
stn_v_init = -60.0
stn_u_init = stn_b * stn_v_init

current_bias_stn = 5.0

'''SETTING NUMBER OF NEURONS PER CHANNEL'''

numCellsPerCol_STR = 1255 ##1255 #### 90% of 50% of 2790000 = 1255500
numCellsPerCol_FSI = 84####139
numCellsPerCol_STN = 14 ###13560
numCellsPerCol_SNR = 27 ####26320
numCellsPerCol_GPe = 46 ###45960
numPoissonInput = 25


'''DEFINING FUNCTIONS FOR DATA STRUCTURING AND SAVING'''
my_resol = 10 ## 1000 DATAPOINTS IS EQUIVALENT TO 100 MS
checkpoint = int(TotalDataPoints/my_resol) ## checking for each 100 ms


def my_condition(x, lower_bound, upper_bound):
    return x>=lower_bound and x<upper_bound

def my_firingrate(my_data, checkpoint, my_resol):
    lower_bound = 0
    my_hist = np.zeros((checkpoint))
    for loop in range(0,checkpoint):
        upper_bound = lower_bound + my_resol
        my_hist[loop] = sum(1 for x in my_data if my_condition(x, lower_bound, upper_bound))
    #     y[lower_bound:upper_bound]=my_hist[loop] ## assigning same values to the indices in the filler zone
        lower_bound = upper_bound
    return my_hist

'''NOW START RUNNING MULTIPLE TRIALS, AND INITIALISE THE ARRAYS'''
numtrials = 10

gpe_hist1 = np.zeros((numtrials, checkpoint))
gpe_hist2 = np.zeros((numtrials, checkpoint))
gpe_hist3 = np.zeros((numtrials, checkpoint))
snr_hist1 = np.zeros((numtrials, checkpoint))
snr_hist2 = np.zeros((numtrials, checkpoint))
snr_hist3 = np.zeros((numtrials, checkpoint))
stn_hist1 = np.zeros((numtrials, checkpoint))
stn_hist2 = np.zeros((numtrials, checkpoint))
stn_hist3 = np.zeros((numtrials, checkpoint))
fsi_hist1 = np.zeros((numtrials, checkpoint))
fsi_hist2 = np.zeros((numtrials, checkpoint))
fsi_hist3 = np.zeros((numtrials, checkpoint))


for thisTrial in range(0, numtrials):

    ''' SET UP SPINNAKER AND BEGIN SIMULATION'''
    p.setup(timestep=0.1, min_delay=1.0, max_delay=14.0)

    '''STRIATUM OF THE BASAL GANGLIA: MEDIUM SPINY NEURONS (MSN - D1 / D2)'''

    strd1_cell_params = {'a': strd1_a,
                         'b': strd1_b,
                         'c': strd1_c,
                         'd': strd1_d,
                         'v_init': strd1_v_init,
                         'u_init': strd1_u_init,
                         'tau_syn_E': tau_ampa,
                         'tau_syn_I': tau_gabaa,
                         'i_offset': current_bias_str,
                         'isyn_exc': E_ampa,
                         'isyn_inh': E_gabaa
                        }

    strd2_cell_params = {'a': strd2_a,
                         'b': strd2_b,
                         'c': strd2_c,
                         'd': strd2_d,
                         'v_init': strd2_v_init,
                         'u_init': strd2_u_init,
                         'tau_syn_E': tau_ampa,
                         'tau_syn_I': tau_gabaa,
                         'i_offset': current_bias_str,
                         'isyn_exc': E_ampa,
                         'isyn_inh': E_gabaa
                        }

    '''FAST SPIKING INTERNEURONS OF THE STRIATUM'''

    fsi_cell_params = { 'a': fsi_a,
                        'b': fsi_b,
                        'c': fsi_c,
                        'd': fsi_d,
                        'v_init': fsi_v_init,
                        'u_init': fsi_u_init,
                        'tau_syn_E': tau_ampa,
                        'tau_syn_I': tau_gabaa,
                        'i_offset': current_bias_fsi,
                        'isyn_exc': E_ampa,
                        'isyn_inh': E_gabaa
                     }


    '''GLOBAL PALLIDUS - EXTERNA OF THE BASAL GANGLIA'''

    gpe_cell_params = {'a': gpe_a,
                       'b': gpe_b,
                       'c': gpe_c,
                       'd': gpe_d,
                       'v_init': gpe_v_init,
                       'u_init': gpe_u_init,
                       'tau_syn_E': tau_ampa,
                       'tau_syn_I': tau_gabaa,
                       'i_offset': current_bias_gpe,
                       'isyn_exc': E_ampa,
                       'isyn_inh': E_gabaa
                       }


    '''SUBSTANTIA NIAGRA OF THE BASAL GANGLIA'''

    snr_cell_params = {'a': snr_a,
                       'b': snr_b,
                       'c': snr_c,
                       'd': snr_d,
                       'v_init': snr_v_init,
                       'u_init': snr_u_init,
                       'tau_syn_E': tau_ampa,
                       'tau_syn_I': tau_gabaa,
                       'i_offset': current_bias_snr,
                       'isyn_exc': E_ampa,
                       'isyn_inh': E_gabaa
                     }

    '''SUB-THALAMIC NUCLEUS OF THE BASAL GANGLIA'''

    stn_cell_params = {'a': stn_a,
                       'b': stn_b,
                       'c': stn_c,
                       'd': stn_d,
                       'v_init': stn_v_init,
                       'u_init': stn_u_init,
                       'tau_syn_E': tau_ampa,
                       'tau_syn_I': tau_gabaa,
                       'i_offset': current_bias_stn,
                       'isyn_exc': E_ampa,
                       'isyn_inh': E_gabaa
                      }


    ''' THE FIRST CHANNEL'''
    strd1_pop1 = p.Population(numCellsPerCol_STR, p.Izhikevich(**strd1_cell_params), label='strd1_pop1')
    strd2_pop1 = p.Population(numCellsPerCol_STR, p.Izhikevich(**strd2_cell_params), label='strd2_pop1')
    gpe_pop1 = p.Population(numCellsPerCol_GPe, p.Izhikevich(**gpe_cell_params), label='gpe_pop1')
    snr_pop1 = p.Population(numCellsPerCol_SNR, p.Izhikevich(**snr_cell_params), label='snr_pop1')
    stn_pop1 = p.Population(numCellsPerCol_STN, p.Izhikevich(**stn_cell_params), label='stn_pop1')
    fsi1_pop1 = p.Population(numCellsPerCol_FSI, p.Izhikevich(**fsi_cell_params), label='fsi1_pop1')


    ''' THE SECOND CHANNEL'''
    strd1_pop2 = p.Population(numCellsPerCol_STR, p.extra_models.Izhikevich_cond, strd1_cell_params, label='strd1_pop2')
    strd2_pop2 = p.Population(numCellsPerCol_STR, p.extra_models.Izhikevich_cond, strd2_cell_params, label='strd2_pop2')
    gpe_pop2 = p.Population(numCellsPerCol_GPe, p.extra_models.Izhikevich_cond, gpe_cell_params, label='gpe_pop2')
    snr_pop2 = p.Population(numCellsPerCol_SNR, p.extra_models.Izhikevich_cond, snr_cell_params, label='snr_pop2')
    stn_pop2 = p.Population(numCellsPerCol_STN, p.extra_models.Izhikevich_cond, stn_cell_params, label='stn_pop2')
    fsi1_pop2 = p.Population(numCellsPerCol_FSI, p.extra_models.Izhikevich_cond, fsi_cell_params, label='fsi1_pop2')


    ''' THE THIRD CHANNEL'''
    strd1_pop3 = p.Population(numCellsPerCol_STR, p.extra_models.Izhikevich_cond, strd1_cell_params, label='strd1_pop3')
    strd2_pop3 = p.Population(numCellsPerCol_STR, p.extra_models.Izhikevich_cond, strd2_cell_params, label='strd2_pop3')
    gpe_pop3 = p.Population(numCellsPerCol_GPe, p.extra_models.Izhikevich_cond, gpe_cell_params, label='gpe_pop3')
    snr_pop3 = p.Population(numCellsPerCol_SNR, p.extra_models.Izhikevich_cond, snr_cell_params, label='snr_pop3')
    stn_pop3 = p.Population(numCellsPerCol_STN, p.extra_models.Izhikevich_cond, stn_cell_params, label='stn_pop3')
    fsi1_pop3 = p.Population(numCellsPerCol_FSI, p.extra_models.Izhikevich_cond, fsi_cell_params, label='fsi1_pop3')


    '''SETTING THE DOPAMINE LEVELS AND CONDUCTANCE PARAMETERS'''
    g_ampa = 0.5

    mod_ampa_d2 = 0.2  ###00.156 in humphries nnet 2009

    phi_max_dop = 5  ##(Scaled within 0 to 5)
    phi_msn_dop = 0.55 * phi_max_dop
    phi_fsi_dop = 0.75 * phi_max_dop
    phi_stn_dop = 0.4 * phi_max_dop  ###(Note that this is scaled between 0 and 16.67)

    '''SETTING NETWORK CONDUCTANCE PARAMETERS'''

    g_cort2strd1 = g_ampa
    g_cort2strd2 = g_ampa * (1 - (mod_ampa_d2 * phi_msn_dop))
    g_cort2fsi = g_ampa * (1 - (mod_ampa_d2 * phi_fsi_dop))
    g_cort2stn = g_ampa * (1 - (mod_ampa_d2 * phi_stn_dop))

    #################DEFINING DISTRIBUTION OF DELAY PARAMETERS
    distr_strd1 = p.RandomDistribution("uniform", [9, 12])

    distr_strd2 = p.RandomDistribution("uniform", [9, 12])

    distr_stn = p.RandomDistribution("uniform", [9, 12])

    distr_fsi  = p.RandomDistribution("uniform", [9, 12])

    pconn_cort2str = 0.15
    pconn_cort2stn = 0.2

    poplist_ch1 = [strd1_pop1, strd2_pop1, fsi1_pop1, stn_pop1]
    poplist_ch2 = [strd1_pop2, strd2_pop2, fsi1_pop2, stn_pop2]
    poplist_ch3 = [strd1_pop3, strd2_pop3, fsi1_pop3, stn_pop3]

    g_pop = [g_cort2strd1, g_cort2strd2, g_cort2fsi, g_cort2stn]
    distr_pop = [distr_strd1, distr_strd2, distr_fsi, distr_stn]



    '''BASE POISSON INPUTS TO ALL CHANNELS FOR THE ENTIRE SIMULATION DURATION OF 5 SECONDS'''
    Rate_Poisson_Inp_base = 3
    start_Poisson_Inp_base = p.RandomDistribution("uniform", [500, 700]) ###50
    Duration_Poisson_Inp_base = 9200
    spike_source_Poisson_base1 = p.Population(numPoissonInput, p.SpikeSourcePoisson, {'rate': Rate_Poisson_Inp_base, 'duration': Duration_Poisson_Inp_base,'start': start_Poisson_Inp_base}, label='spike_source_Poisson_base1')
    spike_source_Poisson_base2 = p.Population(2, p.SpikeSourcePoisson, {'rate': Rate_Poisson_Inp_base, 'duration': Duration_Poisson_Inp_base,'start': start_Poisson_Inp_base}, label='spike_source_Poisson_base2')

    ######projections for CHANEL 1
    for count1 in range(0, 4):
        if count1 < 3:
            p.Projection(spike_source_Poisson_base1, poplist_ch1[count1],
                         p.FixedProbabilityConnector(p_connect=pconn_cort2str),
                         p.StaticSynapse(weight=g_pop[count1], delay=distr_pop[count1]),
                         receptor_type='excitatory')
        else:
            p.Projection(spike_source_Poisson_base2, poplist_ch1[count1],
                         p.FixedProbabilityConnector(p_connect=pconn_cort2stn),
                         p.StaticSynapse(weight=g_pop[count1], delay=distr_pop[count1]),
                         receptor_type ='excitatory')

    ######projections for CHANEL 2
    for count2 in range(0, 4):
        if count2 < 3:
            p.Projection(spike_source_Poisson_base1, poplist_ch2[count2],
                         p.FixedProbabilityConnector(p_connect=pconn_cort2str),
                         p.StaticSynapse(weight=g_pop[count2], delay=distr_pop[count2]),
                         receptor_type='excitatory')
        else:
            p.Projection(spike_source_Poisson_base2, poplist_ch2[count2],
                         p.FixedProbabilityConnector(p_connect=pconn_cort2stn),
                         p.StaticSynapse(weight=g_pop[count2], delay=distr_pop[count2]),
                         receptor_type='excitatory')

    ######projections for CHANEL 3
    for count3 in range(0, 4):
        if count3 < 3:
            p.Projection(spike_source_Poisson_base1, poplist_ch3[count3],
                         p.FixedProbabilityConnector(p_connect=pconn_cort2str),
                         p.StaticSynapse(weight=g_pop[count3], delay=distr_pop[count3]),
                         receptor_type='excitatory')
        else:
            p.Projection(spike_source_Poisson_base2, poplist_ch3[count3],
                         p.FixedProbabilityConnector(p_connect=pconn_cort2stn),
                         p.StaticSynapse(weight=g_pop[count3], delay=distr_pop[count3]),
                         receptor_type ='excitatory')

    '''CHANNEL 1 RECEIVES FIIRST COMPETING INPUT '''

    Rate_Poisson_Inp_first = 15
    start_Poisson_Inp_first = p.RandomDistribution("uniform", [3000, 3200])
    Duration_Poisson_Inp_first = 6700
    spike_source_Poisson_first1 = p.Population(numPoissonInput, p.SpikeSourcePoisson, {'rate': Rate_Poisson_Inp_first, 'duration': Duration_Poisson_Inp_first,'start': start_Poisson_Inp_first}, label='spike_source_Poisson_first1')
    spike_source_Poisson_first2 = p.Population(2, p.SpikeSourcePoisson, {'rate': Rate_Poisson_Inp_first, 'duration': Duration_Poisson_Inp_first,'start': start_Poisson_Inp_first}, label='spike_source_Poisson_first2')

    ######projections for CHANEL 1
    for count00 in range(0, 4):
        if count00 < 3:
            p.Projection(spike_source_Poisson_first1, poplist_ch1[count00],
                         p.FixedProbabilityConnector(p_connect=pconn_cort2str),
                         p.StaticSynapse(weight=g_pop[count00], delay=distr_pop[count00]),
                         receptor_type='excitatory')
        else:
            p.Projection(spike_source_Poisson_first2, poplist_ch2[count00],
                         p.FixedProbabilityConnector(p_connect=pconn_cort2stn),
                         p.StaticSynapse(weight=g_pop[count00], delay=distr_pop[count00]),
                         receptor_type='excitatory')

    '''CHANNEL 2 RECEIVES SECOND COMPETING INPUT'''

    Rate_Poisson_Inp_second = 25
    start_Poisson_Inp_second = p.RandomDistribution("uniform", [6000, 6200])
    Duration_Poisson_Inp_second = 3700
    spike_source_Poisson_second1 = p.Population(numPoissonInput, p.SpikeSourcePoisson, {'rate': Rate_Poisson_Inp_second, 'duration': Duration_Poisson_Inp_second,'start': start_Poisson_Inp_second}, label='spike_source_Poisson_second1')
    spike_source_Poisson_second2 = p.Population(2, p.SpikeSourcePoisson, {'rate': Rate_Poisson_Inp_second, 'duration': Duration_Poisson_Inp_second,'start': start_Poisson_Inp_second}, label='spike_source_Poisson_second2')

    ######projections for CHANEL 2
    for count01 in range(0, 4):
        if count01 < 3:
            p.Projection(spike_source_Poisson_second1, poplist_ch2[count01],
                         p.FixedProbabilityConnector(p_connect=pconn_cort2str),
                         p.StaticSynapse(weight=g_pop[count01], delay = distr_pop[count01]),
                         receptor_type='excitatory')
        else:
            p.Projection(spike_source_Poisson_second2, poplist_ch2[count01],
                         p.FixedProbabilityConnector(p_connect=pconn_cort2stn),
                         p.StaticSynapse(weight=g_pop[count01], delay=distr_pop[count01]),
                         receptor_type='excitatory')

    '''INTRA-CHANNEL PROJECTIONS'''

    ################     EFFERENTS OF STRIATUM        ####################


    g_gaba = 0.5 * g_ampa  ### the gaba conductance

    mod_gaba = 0.073  ##0.625 ### the level of modulation of dopamine of gaba via the D2 and D1 receptors

    g_strd12snr = g_gaba * (1 + mod_gaba * phi_msn_dop)
    g_strd22gpe = g_gaba * (1 - mod_gaba * phi_msn_dop)
    g_str2str = (1.0/2.55) * g_gaba



    distr_strd12snr = p.RandomDistribution("uniform", [5, 7])
    distr_strd22gpe = p.RandomDistribution("uniform", [5, 7])
    distr_str2str = p.RandomDistribution("uniform", [2, 3])

    '''projections of chanel1'''

    p.Projection(strd1_pop1, snr_pop1,
                 p.FixedProbabilityConnector(p_connect=0.15),
                 p.StaticSynapse(weight = g_strd12snr, delay=distr_strd12snr),
                 receptor_type='inhibitory')
    p.Projection(strd2_pop1, gpe_pop1,
                 p.FixedProbabilityConnector(p_connect=0.15),
                 p.StaticSynapse(weight = g_strd22gpe, delay=distr_strd22gpe),
                 receptor_type='inhibitory')
    p.Projection(strd1_pop1, strd1_pop1,
                 p.FixedProbabilityConnector(p_connect=0.1),
                 p.StaticSynapse(weight = g_str2str, delay=distr_str2str),
                 receptor_type='inhibitory')
    p.Projection(strd1_pop1, strd2_pop1,
                 p.FixedProbabilityConnector(p_connect=0.1),
				 p.StaticSynapse(weight = g_str2str, delay=distr_str2str),
                 receptor_type='inhibitory')
    p.Projection(strd2_pop1, strd1_pop1,
                 p.FixedProbabilityConnector(p_connect=0.1),
				 p.StaticSynapse(weight = g_str2str, delay=distr_str2str),
                 receptor_type='inhibitory')
    p.Projection(strd2_pop1, strd2_pop1,
                 p.FixedProbabilityConnector(p_connect=0.1),
				 p.StaticSynapse(weight = g_str2str, delay=distr_str2str),
                 receptor_type='inhibitory')

    '''projections of chanel2'''

    p.Projection(strd1_pop2, snr_pop2,
                 p.FixedProbabilityConnector(p_connect=0.15),
				 p.StaticSynapse(weight = g_strd12snr, delay=distr_strd12snr),
                 receptor_type='inhibitory')
    p.Projection(strd2_pop2, gpe_pop2,
                 p.FixedProbabilityConnector(p_connect=0.15),
				 p.StaticSynapse(weight = g_strd22gpe, delay=distr_strd22gpe),
                 receptor_type='inhibitory')
    p.Projection(strd1_pop2, strd1_pop2,
                 p.FixedProbabilityConnector(p_connect=0.1),
				 p.StaticSynapse(weight = g_str2str, delay=distr_str2str),
                 receptor_type='inhibitory')
    p.Projection(strd1_pop2, strd2_pop2,
                 p.FixedProbabilityConnector(p_connect=0.1),
				 p.StaticSynapse(weight = g_str2str, delay=distr_str2str),
                 receptor_type='inhibitory')
    p.Projection(strd2_pop2, strd1_pop2,
                 p.FixedProbabilityConnector(p_connect=0.1),
				 p.StaticSynapse(weight = g_str2str, delay=distr_str2str),
                 receptor_type='inhibitory')
    p.Projection(strd2_pop2, strd2_pop2,
                 p.FixedProbabilityConnector(p_connect=0.1),
				 p.StaticSynapse(weight = g_str2str, delay=distr_str2str),
                 receptor_type='inhibitory')

    '''projections of chanel-3'''

    p.Projection(strd1_pop3, snr_pop3,
                 p.FixedProbabilityConnector(p_connect=0.15),
				 p.StaticSynapse(weight = g_strd12snr, delay=distr_strd12snr),
                 receptor_type='inhibitory')
    p.Projection(strd2_pop3, gpe_pop3,
                 p.FixedProbabilityConnector(p_connect=0.15),
				 p.StaticSynapse(weight = g_strd22gpe, delay=distr_strd22gpe),
                 receptor_type='inhibitory')
    p.Projection(strd1_pop3, strd1_pop3,
                 p.FixedProbabilityConnector(p_connect=0.1),
				 p.StaticSynapse(weight = g_str2str, delay=distr_str2str),
                 receptor_type='inhibitory')
    p.Projection(strd1_pop3, strd2_pop3,
                 p.FixedProbabilityConnector(p_connect=0.1),
				 p.StaticSynapse(weight = g_str2str, delay=distr_str2str),
                 receptor_type='inhibitory')
    p.Projection(strd2_pop3, strd1_pop3,
                 p.FixedProbabilityConnector(p_connect=0.1),
				 p.StaticSynapse(weight = g_str2str, delay=distr_str2str),
                 receptor_type='inhibitory')
    p.Projection(strd2_pop3, strd2_pop3,
                 p.FixedProbabilityConnector(p_connect=0.1),
				 p.StaticSynapse(weight = g_str2str, delay=distr_str2str),
                 receptor_type='inhibitory')

    ################     EFFERENTS OF FAST SPIKING INTERNEURONS       ####################

    '''projections in chanel 1'''
    p.Projection(fsi1_pop1, strd1_pop1,
                 p.FixedProbabilityConnector(p_connect=0.1),
				 p.StaticSynapse(weight = g_str2str, delay=distr_str2str),
                 receptor_type='inhibitory')

    p.Projection(fsi1_pop1, strd2_pop1,
                 p.FixedProbabilityConnector(p_connect=0.1),
				 p.StaticSynapse(weight = g_str2str, delay=distr_str2str),
                 receptor_type='inhibitory')


    p.Projection(fsi1_pop1, fsi1_pop1,
                 p.FixedProbabilityConnector(p_connect=0.1),
				 p.StaticSynapse(weight = g_str2str, delay=distr_str2str),
                 receptor_type='inhibitory')


    '''projections in chanel 2'''
    p.Projection(fsi1_pop2, strd1_pop2,
                 p.FixedProbabilityConnector(p_connect=0.1),
				 p.StaticSynapse(weight = g_str2str, delay=distr_str2str),
                 receptor_type='inhibitory')

    p.Projection(fsi1_pop2, strd2_pop2,
                 p.FixedProbabilityConnector(p_connect=0.1),
				 p.StaticSynapse(weight = g_str2str, delay=distr_str2str),
                 receptor_type='inhibitory')


    p.Projection(fsi1_pop2, fsi1_pop2,
                 p.FixedProbabilityConnector(p_connect=0.1),
				 p.StaticSynapse(weight = g_str2str, delay=distr_str2str),
                 receptor_type='inhibitory')


    '''projections in chanel 3'''
    p.Projection(fsi1_pop3, strd1_pop3,
                 p.FixedProbabilityConnector(p_connect=0.1),
				 p.StaticSynapse(weight = g_str2str, delay=distr_str2str),
                 receptor_type='inhibitory')

    p.Projection(fsi1_pop3, strd2_pop3,
                 p.FixedProbabilityConnector(p_connect=0.1),
				 p.StaticSynapse(weight = g_str2str, delay=distr_str2str),
                 receptor_type='inhibitory')


    p.Projection(fsi1_pop3, fsi1_pop3,
                 p.FixedProbabilityConnector(p_connect=0.1),
				 p.StaticSynapse(weight = g_str2str, delay=distr_str2str),
                 receptor_type='inhibitory')


    ################     EFFERENTS OF GLOBAL PALLIDUS - EXTERNA       ####################

    g_gaba_gpe = (1.0 / 1.75) * g_gaba
    g_gpe2stn = g_gaba_gpe
    g_gpe2gpe = g_gaba_gpe
    g_gpe2snr = g_gaba_gpe
    g_gpe2fsi = g_gaba_gpe

    distr_gpe2stn = p.RandomDistribution("uniform", [5, 7])
    distr_gpe2gpe = p.RandomDistribution("uniform", [2, 3])
    distr_gpe2snr = p.RandomDistribution("uniform", [5, 7])
    distr_gpe2fsi = p.RandomDistribution("uniform", [5, 7])

    '''projections in chanel 1'''
    p.Projection(gpe_pop1, stn_pop1,
                 p.FixedProbabilityConnector(p_connect=0.25),
				 p.StaticSynapse(weight=g_gpe2stn, delay=distr_gpe2stn),
                 receptor_type='inhibitory')
    p.Projection(gpe_pop1, gpe_pop1,
                 p.FixedProbabilityConnector(p_connect=0.25),
				 p.StaticSynapse(weight=g_gpe2gpe, delay=distr_gpe2gpe),
                 receptor_type='inhibitory')
    p.Projection(gpe_pop1, snr_pop1,
                 p.FixedProbabilityConnector(p_connect=0.25),
				 p.StaticSynapse(weight=g_gpe2snr, delay=distr_gpe2snr),
                 receptor_type='inhibitory')
    p.Projection(gpe_pop1, fsi1_pop1,
                 p.FixedProbabilityConnector(p_connect=0.05),
				 p.StaticSynapse(weight=g_gpe2fsi, delay=distr_gpe2fsi),
                 receptor_type='inhibitory')


    '''projections in chanel 2'''
    p.Projection(gpe_pop2, stn_pop2,
                 p.FixedProbabilityConnector(p_connect=0.25),
				 p.StaticSynapse(weight=g_gpe2stn, delay=distr_gpe2stn),
                 receptor_type='inhibitory')
    p.Projection(gpe_pop2, gpe_pop2,
                 p.FixedProbabilityConnector(p_connect=0.25),
				 p.StaticSynapse(weight=g_gpe2gpe, delay=distr_gpe2gpe),
                 receptor_type='inhibitory')
    p.Projection(gpe_pop2, snr_pop2,
                 p.FixedProbabilityConnector(p_connect=0.25),
				 p.StaticSynapse(weight=g_gpe2snr, delay=distr_gpe2snr),
                 receptor_type='inhibitory')
    p.Projection(gpe_pop1, fsi1_pop2,
                 p.FixedProbabilityConnector(p_connect=0.05),
				 p.StaticSynapse(weight=g_gpe2fsi, delay=distr_gpe2fsi),
                 receptor_type='inhibitory')



    '''projections in chanel 3'''
    p.Projection(gpe_pop3, stn_pop3,
                 p.FixedProbabilityConnector(p_connect=0.25),
				 p.StaticSynapse(weight=g_gpe2stn, delay=distr_gpe2stn),
                 receptor_type='inhibitory')
    p.Projection(gpe_pop3, gpe_pop3,
                 p.FixedProbabilityConnector(p_connect=0.25),
				 p.StaticSynapse(weight=g_gpe2gpe, delay=distr_gpe2gpe),
                 receptor_type='inhibitory')
    p.Projection(gpe_pop3, snr_pop3,
                 p.FixedProbabilityConnector(p_connect=0.25),
				 p.StaticSynapse(weight=g_gpe2snr, delay=distr_gpe2snr),
                 receptor_type='inhibitory')
    p.Projection(gpe_pop1, fsi1_pop3,
                 p.FixedProbabilityConnector(p_connect=0.05),
				 p.StaticSynapse(weight=g_gpe2fsi, delay=distr_gpe2fsi),
                 receptor_type='inhibitory')


    ################     EFFERENTS OF SUBSTANTIA NIGRA PARS RETICULATA       ####################

    g_gaba_snr = (1 / 1.75) * g_gaba
    g_snr2snr = g_gaba_snr

    distr_snr2snr = p.RandomDistribution("uniform", [2, 3])
    '''projections in chanel 1'''
    p.Projection(snr_pop1, snr_pop1,
                 p.FixedProbabilityConnector(p_connect=0.25),
				 p.StaticSynapse(weight=g_snr2snr, delay=distr_snr2snr),
                 receptor_type='inhibitory')

    '''projections in chanel 2'''
    p.Projection(snr_pop2, snr_pop2,
                 p.FixedProbabilityConnector(p_connect=0.25),
				 p.StaticSynapse(weight=g_snr2snr, delay=distr_snr2snr),
                 receptor_type='inhibitory')

    '''projections in chanel 3'''
    p.Projection(snr_pop3, snr_pop3,
                 p.FixedProbabilityConnector(p_connect=0.25),
				 p.StaticSynapse(weight=g_snr2snr, delay=distr_snr2snr),
                 receptor_type='inhibitory')


    ##################################################################
    ################     EFFERENTS OF SUB-THALAMIC NUCLEUS       ####################

    '''INTER-CHANNEL CONNECTIVITY: DIFFUSE EFFERENTS FROM THE STN'''


    distr_stn2gpe_diffuse = p.RandomDistribution("uniform", [9, 12])
    distr_stn2snr_diffuse = p.RandomDistribution("uniform", [9, 12])
    distr_stn2gpe = p.RandomDistribution("uniform", [5, 7])
    distr_stn2snr = p.RandomDistribution("uniform", [5, 7])

    p_conn_diffuse=0.5

    g_stn2snr_diffuse = (g_ampa * (1 - (mod_ampa_d2 * phi_stn_dop))) / 6.0
    g_stn2gpe_diffuse = (g_ampa * (1 - (mod_ampa_d2 * phi_stn_dop))) / 6.0

    '''projections from chanel 1 to chanel 1'''
    p.Projection(stn_pop1, gpe_pop1,
                 p.FixedProbabilityConnector(p_connect=p_conn_diffuse),
				 p.StaticSynapse(weight=g_stn2gpe_diffuse, delay=distr_stn2gpe),
                 receptor_type='excitatory')
    p.Projection(stn_pop1, snr_pop1,
                 p.FixedProbabilityConnector(p_connect=p_conn_diffuse),
				 p.StaticSynapse(weight=g_stn2snr_diffuse, delay=distr_stn2snr),
                 receptor_type='excitatory')

    '''projections from chanel 1 to chanel 2'''
    p.Projection(stn_pop1, gpe_pop2,
                 p.FixedProbabilityConnector(p_connect=p_conn_diffuse),
				 p.StaticSynapse(weight=g_stn2gpe_diffuse, delay=distr_stn2gpe_diffuse),
                 receptor_type='excitatory')
    p.Projection(stn_pop1, snr_pop2,
                 p.FixedProbabilityConnector(p_connect=p_conn_diffuse),
				 p.StaticSynapse(weight=g_stn2snr_diffuse, delay=distr_stn2snr_diffuse),
                 receptor_type='excitatory')

    '''projections from chanel 1 to chanel 3'''
    p.Projection(stn_pop1, gpe_pop3,
                 p.FixedProbabilityConnector(p_connect=p_conn_diffuse),
				 p.StaticSynapse(weight=g_stn2gpe_diffuse, delay=distr_stn2gpe_diffuse),
                 receptor_type='excitatory')
    p.Projection(stn_pop1, snr_pop3,
                 p.FixedProbabilityConnector(p_connect=p_conn_diffuse),
				 p.StaticSynapse(weight=g_stn2snr_diffuse, delay=distr_stn2snr_diffuse),
                 receptor_type='excitatory')


    '''projections from chanel 2 to chanel 1'''
    p.Projection(stn_pop2, gpe_pop1,
                 p.FixedProbabilityConnector(p_connect=p_conn_diffuse),
				 p.StaticSynapse(weight=g_stn2gpe_diffuse, delay=distr_stn2gpe_diffuse),
                 receptor_type='excitatory')
    p.Projection(stn_pop2, snr_pop1,
                 p.FixedProbabilityConnector(p_connect=p_conn_diffuse),
				 p.StaticSynapse(weight=g_stn2snr_diffuse, delay=distr_stn2snr_diffuse),
                 receptor_type='excitatory')

    '''projections from chanel 2 to chanel 2'''
    p.Projection(stn_pop2, gpe_pop2,
                 p.FixedProbabilityConnector(p_connect=p_conn_diffuse),
				 p.StaticSynapse(weight=g_stn2gpe_diffuse, delay=distr_stn2gpe),
                 receptor_type='excitatory')
    p.Projection(stn_pop2, snr_pop2,
                 p.FixedProbabilityConnector(p_connect=p_conn_diffuse),
				 p.StaticSynapse(weight=g_stn2snr_diffuse, delay=distr_stn2snr),
                 receptor_type='excitatory')

    '''projections from chanel 2 to chanel 3'''
    p.Projection(stn_pop2, gpe_pop3,
                 p.FixedProbabilityConnector(p_connect=p_conn_diffuse),
				 p.StaticSynapse(weight=g_stn2gpe_diffuse, delay=distr_stn2gpe_diffuse),
                 receptor_type='excitatory')
    p.Projection(stn_pop2, snr_pop3,
                 p.FixedProbabilityConnector(p_connect=p_conn_diffuse),
				 p.StaticSynapse(weight=g_stn2snr_diffuse, delay=distr_stn2snr_diffuse),
                 receptor_type='excitatory')

    '''projections from chanel 3 to chanel 1'''
    p.Projection(stn_pop3, gpe_pop1,
                 p.FixedProbabilityConnector(p_connect=p_conn_diffuse),
				 p.StaticSynapse(weight=g_stn2gpe_diffuse, delay=distr_stn2gpe_diffuse),
                 receptor_type='excitatory')
    p.Projection(stn_pop3, snr_pop1,
                 p.FixedProbabilityConnector(p_connect=p_conn_diffuse),
				 p.StaticSynapse(weight=g_stn2snr_diffuse, delay=distr_stn2snr_diffuse),
                 receptor_type='excitatory')


    '''projections from chanel 3 to chanel 2'''
    p.Projection(stn_pop3, gpe_pop2,
                 p.FixedProbabilityConnector(p_connect=p_conn_diffuse),
				 p.StaticSynapse(weight=g_stn2gpe_diffuse, delay=distr_stn2gpe_diffuse),
                 receptor_type='excitatory')
    p.Projection(stn_pop3, snr_pop2,
                 p.FixedProbabilityConnector(p_connect=p_conn_diffuse),
				 p.StaticSynapse(weight=g_stn2snr_diffuse, delay=distr_stn2snr_diffuse),
                 receptor_type='excitatory')

    '''projections from chanel 3 to chanel 3'''
    p.Projection(stn_pop3, gpe_pop3,
                 p.FixedProbabilityConnector(p_connect=p_conn_diffuse),
				 p.StaticSynapse(weight=g_stn2gpe_diffuse, delay=distr_stn2gpe),
                 receptor_type='excitatory')
    p.Projection(stn_pop3, snr_pop3,
                 p.FixedProbabilityConnector(p_connect=p_conn_diffuse),
				 p.StaticSynapse(weight=g_stn2snr_diffuse, delay=distr_stn2snr),
                 receptor_type='excitatory')


    '''RECORD THE SPIKE RASTER'''

    # spike_source_Poisson_base1.record(['spikes'])
    # strd1_pop1.record(['spikes'])
    # strd2_pop1.record(['spikes'])
    fsi1_pop1.record(['spikes'])
    # fsi2_pop1.record(['spikes'])
    gpe_pop1.record(['spikes'])
    snr_pop1.record(['spikes'])
    stn_pop1.record(['spikes'])


    # strd1_pop2.record(['spikes'])
    # strd2_pop2.record(['spikes'])
    fsi1_pop2.record(['spikes'])
    # fsi2_pop2.record(['spikes'])
    gpe_pop2.record(['spikes'])
    snr_pop2.record(['spikes'])
    stn_pop2.record(['spikes'])

    # strd1_pop3.record(['spikes'])
    # strd2_pop3.record(['spikes'])
    fsi1_pop3.record(['spikes'])
    # fsi2_pop3.record(['spikes'])
    gpe_pop3.record(['spikes'])
    snr_pop3.record(['spikes'])
    stn_pop3.record(['spikes'])

    # '''RECORD THE MEMBRANE VOLTAGE'''
    # # strd1_pop1.record_v()
    # # strd2_pop1.record_v()
    # # fsi1_pop1.record_v()
    # fsi2_pop1.record_v()
    # gpe_pop1.record_v()
    # snr_pop1.record_v()
    # stn_pop1.record_v()
    #
    # # strd1_pop2.record_v()
    # # strd2_pop2.record_v()
    # # fsi1_pop2.record_v()
    # fsi2_pop2.record_v()
    # gpe_pop2.record_v()
    # snr_pop2.record_v()
    # stn_pop2.record_v()

    p.run(TotalDuration)

    '''DOWNLOAD THE SPIKE RASTER'''
    # spike_source_Poisson_raster1 = np.asarray(spike_source_Poisson_base1.spinnaker_get_data("spikes"))
    #
    # strd1_spike_raster1 = np.asarray(strd1_pop1.spinnaker_get_data("spikes"))
    # strd2_spike_raster1 = np.asarray(strd2_pop1.spinnaker_get_data("spikes"))
    stn_spike_raster1 = np.asarray(stn_pop1.spinnaker_get_data("spikes"))
    gpe_spike_raster1 = np.asarray(gpe_pop1.spinnaker_get_data("spikes"))

    snr_spike_raster1 = np.asarray(snr_pop1.spinnaker_get_data("spikes"))
    fsi1_spike_raster1 = np.asarray(fsi1_pop1.spinnaker_get_data("spikes"))


    # strd1_spike_raster2 = np.asarray(strd1_pop2.spinnaker_get_data("spikes"))
    # strd2_spike_raster2 = np.asarray(strd2_pop2.spinnaker_get_data("spikes"))
    stn_spike_raster2 = np.asarray(stn_pop2.spinnaker_get_data("spikes"))
    gpe_spike_raster2 = np.asarray(gpe_pop2.spinnaker_get_data("spikes"))
    snr_spike_raster2 = np.asarray(snr_pop2.spinnaker_get_data("spikes"))
    fsi1_spike_raster2 = np.asarray(fsi1_pop2.spinnaker_get_data("spikes"))

    # strd1_spike_raster3 = np.asarray(strd1_pop3.spinnaker_get_data("spikes"))
    # strd2_spike_raster3 = np.asarray(strd2_pop3.spinnaker_get_data("spikes"))
    stn_spike_raster3 = np.asarray(stn_pop3.spinnaker_get_data("spikes"))
    gpe_spike_raster3 = np.asarray(gpe_pop3.spinnaker_get_data("spikes"))
    snr_spike_raster3 = np.asarray(snr_pop3.spinnaker_get_data("spikes"))
    fsi1_spike_raster3 = np.asarray(fsi1_pop3.spinnaker_get_data("spikes"))

    # '''RECORD THE MEMBRANE VOLTAGE'''
    # # strd1_membrane_volt1 =  strd1_pop1.get_v()
    # # strd2_membrane_volt1 =  strd2_pop1.get_v()
    # # fsi1_membrane_volt1 =  fsi1_pop1.get_v()
    # fsi2_membrane_volt1 =  fsi2_pop1.get_v()
    # gpe_membrane_volt1 =  gpe_pop1.get_v()
    # stn_membrane_volt1 =  stn_pop1.get_v()
    # snr_membrane_volt1 =  snr_pop1.get_v()
    #
    # # strd1_membrane_volt2 =  strd1_pop2.get_v()
    # # strd2_membrane_volt2 =  strd2_pop2.get_v()
    # # fsi1_membrane_volt2 =  fsi1_pop2.get_v()
    # fsi2_membrane_volt2 =  fsi2_pop2.get_v()
    # gpe_membrane_volt2 =  gpe_pop2.get_v()
    # stn_membrane_volt2 =  stn_pop2.get_v()
    # snr_membrane_volt2 =  snr_pop2.get_v()

    ''' NOW GENERATE AND SAVE THE HISTOGRAM OF FIRING RATES. THIS IS THEN USED TO PLOT THE FIRING RATES AS IN
    FIGURE 6 OF [TS13]'''

    ## SAVING THE GPe
    my_data = gpe_spike_raster1[:, 1]  ## just extract the times of each spike
    my_data = my_data * 10  ## the times are generated at resolution of 0.1,i.e. with a decimal. remove this

    gpe_hist1[thisTrial, :] = my_firingrate(my_data, checkpoint, my_resol)


    my_data = gpe_spike_raster2[:, 1]  ## just extract the times of each spike
    my_data = my_data * 10  ## the times are generated at resolution of 0.1,i.e. with a decimal. remove this

    gpe_hist2[thisTrial, :] = my_firingrate(my_data, checkpoint, my_resol)


    my_data = gpe_spike_raster3[:, 1]  ## just extract the times of each spike
    my_data = my_data * 10  ## the times are generated at resolution of 0.1,i.e. with a decimal. remove this

    gpe_hist3[thisTrial, :] = my_firingrate(my_data, checkpoint, my_resol)



    ## SAVING THE SNR
    my_data = snr_spike_raster1[:, 1]  ## just extract the times of each spike
    my_data = my_data * 10  ## the times are generated at resolution of 0.1,i.e. with a decimal. remove this

    snr_hist1[thisTrial, :] = my_firingrate(my_data, checkpoint, my_resol)


    my_data = snr_spike_raster2[:, 1]  ## just extract the times of each spike
    my_data = my_data * 10  ## the times are generated at resolution of 0.1,i.e. with a decimal. remove this

    snr_hist2[thisTrial, :] = my_firingrate(my_data, checkpoint, my_resol)


    my_data = snr_spike_raster3[:, 1]  ## just extract the times of each spike
    my_data = my_data * 10  ## the times are generated at resolution of 0.1,i.e. with a decimal. remove this

    snr_hist3[thisTrial, :] = my_firingrate(my_data, checkpoint, my_resol)


    ## SAVING THE STN
    my_data = stn_spike_raster1[:, 1]  ## just extract the times of each spike
    my_data = my_data * 10  ## the times are generated at resolution of 0.1,i.e. with a decimal. remove this

    stn_hist1[thisTrial, :] = my_firingrate(my_data, checkpoint, my_resol)


    my_data = stn_spike_raster2[:, 1]  ## just extract the times of each spike
    my_data = my_data * 10  ## the times are generated at resolution of 0.1,i.e. with a decimal. remove this

    stn_hist2[thisTrial, :] = my_firingrate(my_data, checkpoint, my_resol)


    my_data = stn_spike_raster3[:, 1]  ## just extract the times of each spike
    my_data = my_data * 10  ## the times are generated at resolution of 0.1,i.e. with a decimal. remove this

    stn_hist3[thisTrial, :] = my_firingrate(my_data, checkpoint, my_resol)

    ## SAVING THE FSI
    my_data = fsi1_spike_raster1[:, 1]  ## just extract the times of each spike
    my_data = my_data * 10  ## the times are generated at resolution of 0.1,i.e. with a decimal. remove this

    fsi_hist1[thisTrial, :] = my_firingrate(my_data, checkpoint, my_resol)

    my_data = fsi1_spike_raster2[:, 1]  ## just extract the times of each spike
    my_data = my_data * 10  ## the times are generated at resolution of 0.1,i.e. with a decimal. remove this

    fsi_hist2[thisTrial, :] = my_firingrate(my_data, checkpoint, my_resol)

    my_data = fsi1_spike_raster3[:, 1]  ## just extract the times of each spike
    my_data = my_data * 10  ## the times are generated at resolution of 0.1,i.e. with a decimal. remove this

    fsi_hist3[thisTrial, :] = my_firingrate(my_data, checkpoint, my_resol)

    '''RELEASE THE MACHINE'''
    p.end()

folderpath = './DataFolder'

np.savetxt(folderpath + 'GPehist1.csv', gpe_hist1)
np.savetxt(folderpath + 'GPehist2.csv', gpe_hist2)
np.savetxt(folderpath + 'GPehist3.csv', gpe_hist3)
np.savetxt(folderpath + 'SNRhist1.csv', snr_hist1)
np.savetxt(folderpath + 'SNRhist2.csv', snr_hist2)
np.savetxt(folderpath + 'SNRhist3.csv', snr_hist3)
np.savetxt(folderpath + 'STNhist1.csv', stn_hist1)
np.savetxt(folderpath + 'STNhist2.csv', stn_hist2)
np.savetxt(folderpath + 'STNhist3.csv', stn_hist3)
np.savetxt(folderpath + 'FSIhist1.csv', fsi_hist1)
np.savetxt(folderpath + 'FSIhist2.csv', fsi_hist2)
np.savetxt(folderpath + 'FSIhist3.csv', fsi_hist3)



print("--- %s SECONDS ELAPSED ---\n \n \n" % (time.time() - start_time))

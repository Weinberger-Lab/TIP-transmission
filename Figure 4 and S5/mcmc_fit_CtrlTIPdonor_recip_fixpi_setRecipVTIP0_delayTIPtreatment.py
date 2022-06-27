# 2021 - 2022 Michael Pablo
import pandas as pd
import matplotlib.pyplot as plt
from numba import jit
import numpy as np
from scipy.integrate import solve_ivp
from pymcmcstat import MCMC, structures, propagation
import seaborn as sns
import matplotlib
matplotlib.rcParams['pdf.fonttype']=42
matplotlib.rcParams['ps.fonttype']=42
matplotlib.rcParams['axes.labelsize']=24
matplotlib.rcParams['xtick.labelsize']=16
matplotlib.rcParams['ytick.labelsize']=16
matplotlib.rcParams['font.sans-serif']="Arial"
matplotlib.rcParams['font.family']="sans-serif"

@jit
def dual_inf_ODE(t, y, params):
    """
    Model ODES, written with scipy.integrate.solve_ivp() in mind. Infection by wildtype SARS-CoV-2 + TIP treatment
    :param t: Time, days
    :param y: Within-host microbiological states (cells & virions by physiological location)
    :param params: Model parameters, see below for descriptive names.
    :return dydt: Rate change for each microbiological state.
    """
    # Unpack states ------------------------------------------------------------------------------------
    # The states are as follows. All quantities are absolute numbers, not concentrations, so no volume scaling is made
    # during transport between the upper respiratory tract (URT) and the lower respiratory tract (LRT)
    # 0. T - target cells
    # 1. E - eclipse-stage cells (infected but not producing virions)
    # 2. I - infected cells producing virions
    # 3. V - virions SAMPLED by nasal wash
    T, E, I, V, TTIP, ETIP, ITIP, VTIP = y

    # Unpack parameters ------------------------------------------------------------------------------------
    # beta = target cell infection rate constant, divided by sampling fraction f1 (beta_T = beta1/f1)
    # k = eclipse phase rate constant, for transition from nonproductive infected cell to productive.
    # pi = virion production rate by infected cells, multiplied by sampling fraction f1 (pi_T = f1*p1)
    # delta = clearance rate of virion-producing infected cells
    # c = clearance rate constant for virions
    beta, k, pi, delta, c, rho, psi = params

    n_entries = 8
    dydt = np.zeros(n_entries)

    # dT/dt
    dydt[0] = -beta*V*T - beta*VTIP*T

    # dE/dt
    dydt[1] = +beta*V*T - k*E

    # dI/dt
    dydt[2] = +k*E - delta*I

    # dV/dt
    dydt[3] = pi*I - c*V + psi*pi*ITIP

    # dTTIP/dt
    dydt[4] = beta*VTIP*T - beta*TTIP*V

    # dETIP/dt
    dydt[5] = beta*TTIP*V - k*ETIP

    # dITIP/dt
    dydt[6] = k*ETIP - delta*ITIP

    # dVTIP/dt
    dydt[7] = rho*pi*ITIP - c*VTIP

    return dydt

def do_score(theta, data):
    times = np.ravel(data.xdata[0])
    log10pfus = np.ravel(data.ydata[0])
    beta, k, pi, delta, c, rho, psi, log10T0, log10V0, log10VTIP0 = theta
    treat, housing, id, VTIP0_recip = data.user_defined_object[0]

    if treat == 'ctrl' or (treat == 'TIP' and housing == 'recip'):
        VTIP0 = VTIP0_recip
    else:
        VTIP0 = np.power(10, log10VTIP0)

    if treat == 'ctrl' and VTIP0 != 0:
        raise Exception('Model setup error: attempted to set nonzero VTIP0 for control-treated individual')

    V0 = np.power(10, log10V0)
    T0 = np.power(10, log10T0)

    try:
        odeparams = (beta, k, pi, delta, c, rho, psi)

        # I evaluate at high temporal resolution for peak time evaluation to work.
        sim_times = np.arange(0, 7.5, 0.01)

        if VTIP0 == 0: # No need to adjust for timing of TIP administration
            ode_Y0 = [T0, 0, 0, V0, 0, 0, 0, VTIP0]
            sol = solve_ivp(lambda t, y: dual_inf_ODE(t, y, odeparams),
                            [sim_times[0], sim_times[-1]], ode_Y0, t_eval=sim_times)

        else: # Solve in two stages -- [0, 0.25] and [0.25, 7.5] then stitch together. TIP administered at 0.25d (6hpi)
            ode_Y0 = [T0, 0, 0, V0, 0, 0, 0, 0]
            sol1 = solve_ivp(lambda t, y: dual_inf_ODE(t, y, odeparams),
                             [sim_times[0], sim_times[25]], ode_Y0, t_eval=sim_times[0:26])
            ode_Y0 = sol1.y[:, -1]
            ode_Y0[7] = VTIP0
            sol2 = solve_ivp(lambda t, y: dual_inf_ODE(t, y, odeparams),
                             [sim_times[25], sim_times[-1]], ode_Y0, t_eval=sim_times[25:len(sim_times)])
            sol = lambda:0 # define generic object to assign .y and .t fields to for consistent use of 'sol' variable
            sol.y = np.concatenate((sol1.y[:,0:25],sol2.y),axis=1)
            sol.success = sol1.success & sol2.success
            sol.t = sim_times


        if sol.success: # No errors during simulation
            # Figure out where the viral load peak is (simulating from day 0 to day 7)
            tpeak = sol.t[np.argmax(sol.y[3, :])]

            # If we're looking at recipients, we want to compare against days 2.5, 3.5, 4.5
            if housing == 'recip':
                matchtimes = np.argwhere((sim_times == 2.5) | \
                                         (sim_times == 3.5) | \
                                         (sim_times == 4.5))
                matchtimes = np.ravel(matchtimes)
                Vsim = sol.y[3, matchtimes]

                # Censor data.
                cutlog10 = lambda x: np.log10(x) if x > 500 else np.log10(500)
                Vsim_cutlog10 = np.array([cutlog10(x) for x in Vsim])\
                # enforce penalty to have recipient peak should be on day 3.
                scalefac = 10
                if tpeak < (3-1) or tpeak > (3+1):
                    peak_penalty = scalefac*np.power((tpeak - 3), 2)
                else:
                    peak_penalty = 0
            elif housing == 'donor':
                matchtimes = np.argwhere((sim_times == 3) | \
                                         (sim_times == 4) | \
                                         (sim_times == 5))
                matchtimes = np.ravel(matchtimes)
                Vsim = sol.y[3, matchtimes]

                # Censor data.
                cutlog10 = lambda x: np.log10(x) if x > 500 else np.log10(500)
                Vsim_cutlog10 = np.array([cutlog10(x) for x in Vsim])\

                # enforce penalty to have donor peak should be on day 1.5.
                scalefac = 10
                if tpeak < (2-1) or tpeak > (2+1):
                    peak_penalty = scalefac*np.power((tpeak - 2), 2)
                else:
                    peak_penalty = 0

            else:
                raise IndexError('Invalid housing identifier.')

            MSE = peak_penalty + np.mean(np.power(Vsim_cutlog10 - log10pfus, 2))
        else:
            MSE = np.inf
    except:
        MSE = np.inf

    # print(MSE)
    return MSE

def posterior_sim_fn(theta, data):
    beta, k, pi, delta, c, rho, psi, log10T0, log10V0, log10VTIP0 = theta
    treat, housing, id, VTIP0_recip = data.user_defined_object[0]

    if treat == 'ctrl' or (treat == 'TIP' and housing == 'recip'):
        VTIP0 = VTIP0_recip
    else:
        VTIP0 = np.power(10, log10VTIP0)

    V0 = np.power(10, log10V0)
    T0 = np.power(10, log10T0)
    odeparams = (beta, k, pi, delta, c, rho, psi)

    sim_times = np.arange(0, 13.5, 0.01)
    ode_Y0 = [T0, 0, 0, V0, 0, 0, 0, VTIP0]
    sol = solve_ivp(lambda t, y: dual_inf_ODE(t, y, odeparams),
                    [sim_times[0], sim_times[-1]], ode_Y0, t_eval=sim_times)

    Vsim = sol.y[3, :]

    cutlog10 = lambda x: np.log10(x) if x > 500 else np.log10(500)
    Vsim_cutlog10 = [cutlog10(x) for x in Vsim]

    return np.array([Vsim_cutlog10]).T

def calculate_posterior_intervals(pdata, result):
    intervals = propagation.calculate_intervals(result['chain'][1000:], result, pdata, posterior_sim_fn,
                                                waitbar=True, nsample=1000, s2chain=result['s2chain'][1000:])

    return intervals

def get_MCMC_plot_settings():
    data_display = dict(marker='o',
                        color='k',
                        mfc='none',
                        label='Data')
    model_display = dict(color='r')
    interval_display = dict(alpha=0.5)

    return data_display, model_display, interval_display

def render_prediction_intervals(intervals, mcstat, figurepath, theta_best, theta_median):
    # Creates...
    # - '*_intervals.png' (95% credible interval + median prediction - uses 2.5, 50, and 97.5 percentiles of posterior prediction)
    # - '*_intervals_timezoom.png' (zoomed in to nicer time range)
    # - '*_traces.png' (shows individual predictions sampled from MCMC posterior)
    # - '*_theta_best.png' (shows prediction corresponding to a parameter set with lowest-sampled MSE from MCMC posterior)
    # - '*_theta_median.png' (shows prediction corresponding to parameters set made by taking the median of each posterior parameter distribution)

    sim_times = np.arange(0, 13.5, 0.01)

    hi = np.percentile(intervals['credible'], q=97.5, axis=0)
    lo = np.percentile(intervals['credible'], q=2.5, axis=0)
    md = np.percentile(intervals['credible'], q=50, axis=0)
    if mcstat.data.user_defined_object[0][0] == 'TIP':
        maincolor = '#ff3399'
    else:
        maincolor = '#333399'

    plt.figure(figsize=(5, 3))
    plt.fill_between(sim_times, lo, hi, alpha=0.2, label='Sim. 95% CI', color=maincolor)
    plt.plot(sim_times, md, label='Sim. Median', color=maincolor)
    plt.plot(mcstat.data.xdata[0], mcstat.data.ydata[0][:, 0], 'ko', label='Data', markersize=10)
    plt.ylim(np.log10(500), 7.5)
    plt.xlim(0, 10.5)
    plt.ylabel('')
    if mcstat.data.user_defined_object[0][1] == 'recip':
        matchtimes = np.argwhere((sim_times == 0) | \
                                 (sim_times == 0.33))
        plt.fill_between(sim_times, 0, 1,
                         where=np.logical_and(sim_times >= sim_times[0],
                                              sim_times <= sim_times[-1]),
                         color='#CBCBCA', alpha=0.23, transform=plt.gca().get_xaxis_transform(),
                         label='Co-housing', linestyle='None')
        plt.fill_between(sim_times, 0, 1,
                         where=np.logical_and(sim_times >= sim_times[matchtimes[0]],
                                              sim_times <= sim_times[matchtimes[1]]),
                         color='#83C2C9', alpha=1, transform=plt.gca().get_xaxis_transform(),
                         label='Co-housing', linestyle='None', edgecolor='#83C2C9')

    plt.xlabel('time past exposure (days)', fontsize=22)
    plt.ylabel('viral load\n (log10 PFU/mL)', fontsize=22)
    plt.gca().set_xticks([0, 2, 4, 6, 8, 10])
    plt.gca().set_yticks([3, 4, 5, 6, 7])
    plt.gca().set_xticklabels(labels=['0', '2', '4', '6', '8', '10'], fontsize=20)
    plt.gca().set_yticklabels(labels=['3', '4', '5', '6', '7'], fontsize=20)
    plt.tight_layout()
    plt.savefig(figurepath+'_intervals.png', dpi=300)

    plt.figure(figsize=(6, 3))
    plt.fill_between(sim_times, lo, hi, alpha=0.2, label='Sim. 95% CI', color=maincolor)
    plt.plot(sim_times, md, label='Sim. Median', color=maincolor)
    plt.plot(mcstat.data.xdata[0], mcstat.data.ydata[0][:, 0], 'ko', label='Data', markersize=10)
    plt.ylabel('')

    plt.xlabel('time past exposure (days)', fontsize=22)
    plt.ylabel('viral load\n (log10 PFU/mL)', fontsize=22)
    plt.gca().set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    plt.gca().set_yticks([3, 4, 5, 6, 7])
    plt.gca().set_xticklabels(labels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'], fontsize=18)
    plt.gca().set_yticklabels(labels=['3', '4', '5', '6', '7'], fontsize=18)
    plt.ylim(np.log10(500), 7.5)
    plt.xlim(0, 7.5)
    plt.tight_layout()
    plt.savefig(figurepath+'_intervals_timezoom.png', dpi=300)

    plt.figure(figsize=(6, 3))
    for line in intervals['credible']:
        plt.plot(sim_times, line, alpha=0.1, color=maincolor)
    plt.plot(mcstat.data.xdata[0], mcstat.data.ydata[0][:, 0], 'ko', label='Data')
    plt.ylabel('')

    plt.xlabel('time past exposure (days)', fontsize=22)
    plt.ylabel('viral load\n (log10 PFU/mL)', fontsize=22)
    plt.ylim(np.log10(500), 7.5)
    plt.xlim(0, 12.5)
    plt.gca().set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    plt.gca().set_yticks([3, 4, 5, 6, 7])
    plt.gca().set_xticklabels(labels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'], fontsize=18)
    plt.gca().set_yticklabels(labels=['3', '4', '5', '6', '7'], fontsize=18)
    plt.tight_layout()
    plt.savefig(figurepath+'_traces.png', dpi=300)

    best_pred = posterior_sim_fn(theta_best, mcstat.data)
    plt.figure(figsize=(6, 3))
    plt.plot(sim_times, best_pred, label='Model (best)', color=maincolor)
    plt.plot(mcstat.data.xdata[0], mcstat.data.ydata[0][:, 0], 'ko', label='Data')
    plt.ylabel('')

    plt.xlabel('time past exposure (days)', fontsize=22)
    plt.ylabel('viral load\n (log10 PFU/mL)', fontsize=22)
    plt.ylim(np.log10(500), 7.5)
    plt.xlim(0, 12.5)
    plt.gca().set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    plt.gca().set_yticks([3, 4, 5, 6, 7])
    plt.gca().set_xticklabels(labels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'], fontsize=18)
    plt.gca().set_yticklabels(labels=['3', '4', '5', '6', '7'], fontsize=18)
    plt.tight_layout()
    plt.savefig(figurepath+'_best.png', dpi=300)
    np.savetxt(figurepath+'_theta_best.txt', theta_best)

    median_pred = posterior_sim_fn(theta_median, mcstat.data)
    plt.figure(figsize=(6, 3))
    plt.plot(sim_times, median_pred, label='Model (median)')
    plt.plot(mcstat.data.xdata[0], mcstat.data.ydata[0][:, 0], 'ko', label='Data')
    plt.ylabel('')
    plt.ylim(np.log10(500), 7.5)
    plt.xlim(0, 12.5)
    plt.gca().set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    plt.gca().set_yticks([3, 4, 5, 6, 7])
    plt.gca().set_xticklabels(labels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'], fontsize=18)
    plt.gca().set_yticklabels(labels=['3', '4', '5', '6', '7'], fontsize=18)
    plt.xlabel('time past exposure (days)')
    plt.ylabel('viral load\n (log10 PFU/mL)', fontsize=22)
    plt.tight_layout()
    plt.savefig(figurepath+'_median.png', dpi=300)
    np.savetxt(figurepath+'_theta_median.txt', theta_median)
    plt.close('all')

def fit_model(pathname, savename, target_hamster_name, VTIP0_recip):
    pfu_data = pd.read_csv('pooled_donor_recip corrected 2.csv')

    # strings matched to hamster_id and expected peak time. Provide "TIP" and "donor/recip" identifier to switch between models and peak time constraints
    target_hamster = {'TIP-donor-1': ['TIP', 'donor', 1],
                      'TIP-donor-2': ['TIP', 'donor', 2],
                      'TIP-donor-3': ['TIP', 'donor', 3],
                      'TIP-donor-4': ['TIP', 'donor', 4],
                      'TIP-donor-5': ['TIP', 'donor', 5],
                      'TIP-recip-1': ['TIP', 'recip', 9],
                      'TIP-recip-2': ['TIP', 'recip', 8],
                      'TIP-recip-3': ['TIP', 'recip', 7],
                      'TIP-recip-4': ['TIP', 'recip', 6],
                      'TIP-recip-5': ['TIP', 'recip', 10],
                      'ctrl-donor-1': ['ctrl', 'donor', 1],
                      'ctrl-donor-2': ['ctrl', 'donor', 2],
                      'ctrl-donor-3': ['ctrl', 'donor', 3],
                      'ctrl-donor-4': ['ctrl', 'donor', 4],
                      'ctrl-donor-5': ['ctrl', 'donor', 5],
                      'ctrl-recip-1': ['ctrl', 'recip', 9],
                      'ctrl-recip-2': ['ctrl', 'recip', 10],
                      'ctrl-recip-3': ['ctrl', 'recip', 8],
                      'ctrl-recip-4': ['ctrl', 'recip', 6],
                      'ctrl-recip-5': ['ctrl', 'recip', 7]}

    treat = target_hamster[target_hamster_name][0]
    housing = target_hamster[target_hamster_name][1]
    id = target_hamster[target_hamster_name][2]

    if VTIP0_recip != 0 and treat == 'ctrl':
        raise Exception('Model setup error: attempted to set nonzero VTIP0 for control-treated individual')

    data_x = pfu_data[(pfu_data['hamster_id'] == id) & (pfu_data['treat'] == treat)]['day'].values
    data_y = pfu_data[(pfu_data['hamster_id'] == id) & (pfu_data['treat'] == treat)]['log10_pfu'].values

    mcstat = MCMC.MCMC()
    # Note -- we use the user_defined_object VTIP0_recip only for TIP-treated recipients.
    # This is a workaround: we normally perform MCMC sampling on log10VTIP0, but
    # this makes it impossible for us to set VTIP0 = 0.
    mcstat.data.add_data_set(x=data_x,
                             y=data_y,
                             user_defined_object=(treat, housing, id, VTIP0_recip))
    mcstat.parameters.add_model_parameter(name='beta', theta0=0.000031,
                                          minimum=0, maximum=1,
                                          prior_mu=0.000031, prior_sigma=0.00001,
                                          sample=True)
    mcstat.parameters.add_model_parameter(name='k', theta0=4,
                                          minimum=0, maximum=10,
                                          prior_mu=4, #prior_sigma=0.00001,
                                          sample=False)
    mcstat.parameters.add_model_parameter(name='pi', theta0=10,#11.09,
                                          minimum=0, maximum=100,
                                          prior_mu=10,#11.09, #prior_sigma=1,
                                          sample=False)#True)
    mcstat.parameters.add_model_parameter(name='delta', theta0=1.87,
                                          minimum=0, maximum=100,
                                          prior_mu=1.87, #prior_sigma=0.1,
                                          sample=True)
    mcstat.parameters.add_model_parameter(name='c', theta0=62.21,
                                          minimum=0, maximum=100,
                                          prior_mu=62.21, #prior_sigma=10,
                                          sample=True)
    mcstat.parameters.add_model_parameter(name='rho', theta0=1.5,
                                          minimum=0, maximum=10,
                                          prior_mu=1.5, #prior_sigma=0.1,
                                          sample=False)
    mcstat.parameters.add_model_parameter(name='psi', theta0=0.02,
                                          minimum=0, maximum=1,
                                          prior_mu=0.02, #prior_sigma=0.1,
                                          sample=False)
    mcstat.parameters.add_model_parameter(name='log10T0', theta0=np.log10(1e7),
                                          minimum=np.log10(5e5), maximum=np.log10(5e8),
                                          prior_mu=np.log10(1e7), #prior_sigma=1e7,
                                          sample=False)
    # Prior: the viral inoculum is larger for the donor versus the recipient
    # Prior: we'll fix the TIP recieved.
    if housing == 'donor':
        mcstat.parameters.add_model_parameter(name='log10V0', theta0=np.log10(1e6),
                                              minimum=np.log10(1), maximum=np.log10(1e7),
                                              prior_mu=np.log10(1e6), #prior_sigma=1e5,
                                              sample=True)
        mcstat.parameters.add_model_parameter(name='log10VTIP0', theta0=np.log10(1e6),
                                              minimum=np.log10(1), maximum=np.log10(1e7),
                                              prior_mu=np.log10(1e6), #prior_sigma=1e3,
                                              sample=True)
    elif housing == 'recip':
        mcstat.parameters.add_model_parameter(name='log10V0', theta0=np.log10(1e3),
                                              minimum=np.log10(1), maximum=np.log10(1e5),
                                              prior_mu=np.log10(1e3), #prior_sigma=1e5,
                                              sample=True)
        # Note that VTIP0 should not be used as a parameter for recipients, or for any ctrl-treated.
        # We keep it here to make variable plotting consistent. (see user_defined_object for more)
        mcstat.parameters.add_model_parameter(name='log10VTIP0', theta0=0,
                                              minimum=0, maximum=1,
                                              prior_mu=0,  # prior_sigma=1e3,
                                              sample=True)

    mcstat.simulation_options.define_simulation_options(nsimu=10000,
                                                        updatesigma=True,
                                                        verbosity=True,
                                                        save_to_json=True,
                                                        save_lightly=False,
                                                        results_filename=savename+'.json',
                                                        savedir=pathname)
    if housing == 'donor':
        mcstat.model_settings.define_model_settings(sos_function=do_score,
                                                    S20=0.1,
                                                    N0=20)
    elif housing == 'recip':
        mcstat.model_settings.define_model_settings(sos_function=do_score,
                                                    S20=0.1,
                                                    N0=20)
    mcstat.run_simulation()



def visualize_result(pathname, savename, target_hamster_name, VTIP0_recip, interval_file=None):
    # IF the posterior interval file is not supplied, samples posterior intervals
    # (*_VL_intervals.txt) based on MCMC fit file (*.json) and saves it.
    # Then calls render_prediction_intervals() to create plots.
    # Plots the MCMC error chain (*_error_TossedBurnIn.png)
    # Also computes several statistics (time to reach LOD, viral load at peak, viral load during cohousing).
    result = structures.ResultsStructure.ResultsStructure.load_json_object(pathname)
    result['chain'] = np.array(result['chain'])
    result['parind'] = np.array(result['parind'])
    result['theta'] = np.array(result['theta'])
    result['s2chain'] = np.array(result['s2chain'])
    beta, k, pi, delta, c, rho, psi, T0, V0, VTIP0 = result['theta']
    pfu_data = pd.read_csv('pooled_donor_recip corrected 2.csv')

    # strings matched to hamster_id and expected peak time. Provide "TIP" and "donor/recip" identifier to switch between models and peak time constraints
    target_hamster = {'TIP-donor-1': ['TIP', 'donor', 1],
                      'TIP-donor-2': ['TIP', 'donor', 2],
                      'TIP-donor-3': ['TIP', 'donor', 3],
                      'TIP-donor-4': ['TIP', 'donor', 4],
                      'TIP-donor-5': ['TIP', 'donor', 5],
                      'TIP-recip-1': ['TIP', 'recip', 9],
                      'TIP-recip-2': ['TIP', 'recip', 8],
                      'TIP-recip-3': ['TIP', 'recip', 7],
                      'TIP-recip-4': ['TIP', 'recip', 6],
                      'TIP-recip-5': ['TIP', 'recip', 10],
                      'ctrl-donor-1': ['ctrl', 'donor', 1],
                      'ctrl-donor-2': ['ctrl', 'donor', 2],
                      'ctrl-donor-3': ['ctrl', 'donor', 3],
                      'ctrl-donor-4': ['ctrl', 'donor', 4],
                      'ctrl-donor-5': ['ctrl', 'donor', 5],
                      'ctrl-recip-1': ['ctrl', 'recip', 9],
                      'ctrl-recip-2': ['ctrl', 'recip', 10],
                      'ctrl-recip-3': ['ctrl', 'recip', 8],
                      'ctrl-recip-4': ['ctrl', 'recip', 6],
                      'ctrl-recip-5': ['ctrl', 'recip', 7]}

    treat = target_hamster[target_hamster_name][0]
    housing = target_hamster[target_hamster_name][1]
    id = target_hamster[target_hamster_name][2]

    data_x = pfu_data[(pfu_data['hamster_id'] == id) & (pfu_data['treat'] == treat)]['day'].values
    data_y = pfu_data[(pfu_data['hamster_id'] == id) & (pfu_data['treat'] == treat)]['log10_pfu'].values
    mcstat = MCMC.MCMC()
    mcstat.data.add_data_set(x=data_x,
                             y=data_y,
                             user_defined_object=(treat, housing, id, VTIP0_recip))

    pdata = mcstat.data
    sim_times = np.arange(0, 13.5, 0.01)

    if mcstat.data.user_defined_object[0][0] == 'TIP':
        maincolor = '#ff3399'
    else:
        maincolor = '#333399'

    if interval_file is not None:
        tmp = np.genfromtxt(interval_file)
        intervals = {'credible': tmp}
    else:
        intervals = calculate_posterior_intervals(pdata, result)
        np.savetxt(savename+'_VL_intervals.txt', intervals['credible'])

    # Parameters corresponding to MINIMUM error discovered.
    theta_best = np.array([beta, k, pi, delta, c, rho, psi, T0, V0, VTIP0])
    theta_best[0] = result['chain'][np.argmin(result['s2chain']), 0]
    theta_best[3] = result['chain'][np.argmin(result['s2chain']), 1]
    theta_best[4] = result['chain'][np.argmin(result['s2chain']), 2]
    theta_best[8] = result['chain'][np.argmin(result['s2chain']), 3]
    if treat == 'ctrl' or (treat == 'TIP' and housing == 'recip'):
        theta_best[9] = VTIP0_recip
    else:
        theta_best[9] = result['chain'][np.argmin(result['s2chain']), 4]

    # Parameters corresponding to median of the last half of the chain
    theta_median = np.array([beta, k, pi, delta, c, rho, psi, T0, V0, VTIP0])
    halflen = int(len(result['chain'][:, 0])/2)
    theta_median[0] = np.median(result['chain'][halflen:, 0])
    theta_median[3] = np.median(result['chain'][halflen:, 1])
    theta_median[4] = np.median(result['chain'][halflen:, 2])
    theta_median[8] = np.median(result['chain'][halflen:, 3])
    if treat == 'ctrl' or (treat == 'TIP' and housing == 'recip'):
        theta_median[9] = VTIP0_recip
    else:
        theta_median[9] = np.median(result['chain'][halflen:, 4])

    render_prediction_intervals(intervals, mcstat, savename, theta_best, theta_median)

    plt.figure(figsize=(3, 2.5))
    plt.plot(np.arange(0,10000), result['sschain'], color=maincolor)
    plt.gca().set_xlabel('Steps (x1000)', fontsize=22)
    plt.gca().set_ylabel('Error', fontsize=22)
    plt.ylim(-.1, 2.1)
    plt.gca().set_xticks([0, 1000, 5000, 10000])
    plt.gca().set_yticks([0, 1, 2])
    plt.gca().set_xticklabels(labels=['0', '1', '5', '10'], fontsize=18)
    plt.gca().set_yticklabels(labels=['0', '1', '2'], fontsize=18)
    plt.tight_layout()
    plt.savefig(savename+'_error_TossedBurnIn.png', dpi=300)

    print('Rejection rate:', result['total_rejected'])
    plt.close('all')
    # Calculate integrated viral load during cohousing
    if housing == 'donor':
        compute_viral_load_during_cohousing(intervals, savename, pdata, sim_times)

    compute_time_to_reach_LOD(intervals, savename, pdata, sim_times)
    compute_viral_load_during_peak(intervals, savename, pdata, sim_times)
    compute_viral_load_at_start(intervals, savename, pdata, sim_times)

def compute_viral_load_during_peak(intervals, figurepath, data, sim_times):
    #sim_times = np.arange(0, 7.5, 0.01)

    hi = np.percentile(intervals['credible'], q=97.5, axis=0)
    lo = np.percentile(intervals['credible'], q=2.5, axis=0)
    md = np.percentile(intervals['credible'], q=50, axis=0)

    tpeak = sim_times[np.argmax(md)]
    peak_lo = lo[np.argmax(md)]
    peak_md = md[np.argmax(md)]
    peak_hi = hi[np.argmax(md)]
    np.savetxt(figurepath + '_peak_VL.txt', [tpeak, peak_lo, peak_md, peak_hi])

def compute_viral_load_at_start(intervals, figurepath, data, sim_times):
    #sim_times = np.arange(0, 7.5, 0.01)

    hi = np.percentile(intervals['credible'], q=97.5, axis=0)
    lo = np.percentile(intervals['credible'], q=2.5, axis=0)
    md = np.percentile(intervals['credible'], q=50, axis=0)

    start_lo = lo[0]
    start_md = md[0]
    start_hi = hi[0]
    np.savetxt(figurepath + '_initial_VL_from_ODEs.txt', [start_lo, start_md, start_hi])


def compute_time_to_reach_LOD(intervals, figurepath, data, sim_times):
    #sim_times = np.arange(0, 7.5, 0.01)

    hi = np.percentile(intervals['credible'], q=97.5, axis=0)
    lo = np.percentile(intervals['credible'], q=2.5, axis=0)
    md = np.percentile(intervals['credible'], q=50, axis=0)

    # To find the time of viral clearance, we "invert" the problem a bit.
    # Specifically, starting from the END of the timeseries, we ask,
    # what is the first time point above the LOD?
    hi_index = np.argwhere(hi > np.log10(500))
    lo_index = np.argwhere(lo > np.log10(500))
    md_index = np.argwhere(md > np.log10(500))

    if len(hi_index) == 0:
        hi_time = np.nan
    else:
        hi_time = sim_times[hi_index[-1]]
    if len(md_index) == 0:
        md_time = np.nan
    else:
        md_time = sim_times[md_index[-1]]
    if len(lo_index) == 0:
        lo_time = np.nan
    else:
        lo_time = sim_times[lo_index[-1]]

    np.savetxt(figurepath + '_time_to_reach_LOD.txt', [lo_time, md_time, hi_time])


def compute_viral_load_during_cohousing(intervals, figurepath, data, sim_times):
    # Note that this function is only called for donors.
    #sim_times = np.arange(0, 7.5, 0.01)
    matchtimes = np.argwhere((sim_times == 1.5) | \
                             (sim_times == 1.83))
    matchtimes = np.ravel(matchtimes)

    hi = np.percentile(intervals['credible'], q=97.5, axis=0)
    lo = np.percentile(intervals['credible'], q=2.5, axis=0)
    md = np.percentile(intervals['credible'], q=50, axis=0)

    if data.user_defined_object[0][0] == 'TIP':
        maincolor = '#ff3399'
    else:
        maincolor = '#333399'

    # Time interval in hours
    dt = (sim_times[matchtimes[1]] - sim_times[matchtimes[0]])

    # Calculate addition in log10 space, normalize to time interval
    lo_cohousing = np.log10(np.sum(np.power(10, lo[matchtimes[0]:matchtimes[1]])) * dt)
    md_cohousing = np.log10(np.sum(np.power(10, md[matchtimes[0]:matchtimes[1]])) * dt)
    hi_cohousing = np.log10(np.sum(np.power(10, hi[matchtimes[0]:matchtimes[1]])) * dt)
    np.savetxt(figurepath+'_cohousing_VL.txt', [lo_cohousing, md_cohousing, hi_cohousing])


    plt.figure(figsize=(5,3))
    plt.fill_between(sim_times, 0, 1,
                     where=np.logical_and(sim_times>=sim_times[matchtimes[0]],
                                          sim_times<=sim_times[matchtimes[1]]),
                     color='#83C2C9', alpha=1, transform=plt.gca().get_xaxis_transform(),
                     label='Co-housing', linestyle='None', edgecolor='#83C2C9')
    plt.fill_between(sim_times, lo, hi, alpha=0.2,
                     linestyle='None', label='Sim. 95% CI', color=maincolor,
                     edgecolor=maincolor)
    plt.plot(sim_times, md, color=maincolor, label='Sim. Median', linewidth=2)
    plt.plot(data.xdata[0], data.ydata[0][:, 0], 'ko', label='Data', markersize=10)

    plt.xlabel('time past exposure (days)', fontsize=22)
    plt.ylabel('viral load\n (log10 PFU/mL)', fontsize=22)
    plt.gca().set_xticks([0, 2, 4, 6, 8, 10])
    plt.gca().set_yticks([3, 4, 5, 6, 7])
    plt.gca().set_xticklabels(labels=['0', '2', '4', '6', '8', '10'], fontsize=20)
    plt.gca().set_yticklabels(labels=['3', '4', '5', '6', '7'], fontsize=20)
    plt.ylim(np.log10(500), 7.5)
    plt.xlim(0, 10.5)
    plt.tight_layout()
    plt.savefig(figurepath+'_cohousing_VL.png', dpi=300)


    plt.figure(figsize=(5,3))
    plt.fill_between(sim_times, 0, 1,
                     where=np.logical_and(sim_times>=sim_times[matchtimes[0]],
                                          sim_times<=sim_times[matchtimes[1]]),
                     color='#83C2C9', alpha=1, transform=plt.gca().get_xaxis_transform(),
                     label='Co-housing', linestyle='None', edgecolor='#83C2C9')
    plt.fill_between(sim_times, lo, hi, alpha=0.2, linestyle='None', label='Sim. 95% CI', color=maincolor)
    plt.plot(sim_times, md, color=maincolor, label='Sim. Median', linewidth=2)
    plt.plot(data.xdata[0], data.ydata[0][:, 0], 'ko', label='Data', markersize=10)

    plt.xlabel('time past exposure (days)', fontsize=22)
    plt.ylabel('viral load\n (log10 PFU/mL)', fontsize=22)
    plt.gca().set_xticks([0, 2, 4, 6, 8, 10])
    plt.gca().set_yticks([3, 4, 5, 6, 7])
    plt.gca().set_xticklabels(labels=['0', '2', '4', '6', '8', '10'], fontsize=20)
    plt.gca().set_yticklabels(labels=['3', '4', '5', '6', '7'], fontsize=20)
    plt.ylim(np.log10(500), 7.5)
    plt.xlim(0, 10.5)
    plt.tight_layout()
    plt.savefig(figurepath+'_cohousing_VL.pdf', dpi=300)


    plt.figure(figsize=(5,3))
    # This function is already assuming we're looking at donors
    plt.fill_between(sim_times, 0, 1,
                     where=np.logical_and(sim_times>=sim_times[0],
                                          sim_times<=sim_times[-1]),
                     color='#dea25d', alpha=0.15, transform=plt.gca().get_xaxis_transform(),
                     label='Co-housing', linestyle='None')
    plt.fill_between(sim_times, 0, 1,
                     where=np.logical_and(sim_times>=sim_times[matchtimes[0]],
                                          sim_times<=sim_times[matchtimes[1]]),
                     color='#83C2C9', alpha=1, transform=plt.gca().get_xaxis_transform(),
                     label='Co-housing', linestyle='None', edgecolor='#83C2C9')
    plt.fill_between(sim_times, lo, hi, alpha=0.2, linestyle='None', label='Sim. 95% CI', color=maincolor)
    plt.plot(sim_times, md, color=maincolor, label='Sim. Median', linewidth=2)
    plt.plot(data.xdata[0], data.ydata[0][:, 0], 'ko', label='Data', markersize=10)
    plt.xlabel('time past exposure (days)', fontsize=22)
    plt.ylabel('viral load\n (log10 PFU/mL)', fontsize=22)
    plt.gca().set_xticks([0, 2, 4, 6, 8, 10])
    plt.gca().set_yticks([3, 4, 5, 6, 7])
    plt.gca().set_xticklabels(labels=['0', '2', '4', '6', '8', '10'], fontsize=20)
    plt.gca().set_yticklabels(labels=['3', '4', '5', '6', '7'], fontsize=20)
    plt.ylim(np.log10(500), 7.5)
    plt.xlim(0, 10.5)
    plt.tight_layout()
    plt.savefig(figurepath+'_cohousing_VL_shadebackground.pdf', dpi=300)


    plt.figure(figsize=(5,3))
    # This function is already assuming we're looking at donors
    plt.fill_between(sim_times, 0, 1,
                     where=np.logical_and(sim_times>=sim_times[0],
                                          sim_times<=sim_times[-1]),
                     color='#dea25d', alpha=0.15, transform=plt.gca().get_xaxis_transform(),
                     label='Co-housing', linestyle='None')
    plt.fill_between(sim_times, 0, 1,
                     where=np.logical_and(sim_times>=sim_times[matchtimes[0]],
                                          sim_times<=sim_times[matchtimes[1]]),
                     color='#83C2C9', alpha=1, transform=plt.gca().get_xaxis_transform(),
                     label='Co-housing', linestyle='None', edgecolor='#83C2C9')
    plt.fill_between(sim_times, lo, hi, alpha=0.2, linestyle='None', label='Sim. 95% CI', color=maincolor)
    plt.plot(sim_times, md, color=maincolor, label='Sim. Median', linewidth=2)
    plt.plot(data.xdata[0], data.ydata[0][:, 0], 'ko', label='Data', markersize=10)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlabel('time past exposure (days)', fontsize=22)
    plt.ylabel('viral load\n (log10 PFU/mL)', fontsize=22)
    plt.gca().set_xticks([0, 2, 4, 6, 8, 10])
    plt.gca().set_yticks([3, 4, 5, 6, 7])
    plt.gca().set_xticklabels(labels=['0', '2', '4', '6', '8', '10'], fontsize=20)
    plt.gca().set_yticklabels(labels=['3', '4', '5', '6', '7'], fontsize=20)
    plt.ylim(np.log10(500), 7.5)
    plt.xlim(0, 10.5)
    plt.tight_layout()
    plt.savefig(figurepath+'_cohousing_VL_shadebackground.png', dpi=300)


    plt.figure(figsize=(6,3))
    plt.fill_between(sim_times, 0, 1,
                     where=np.logical_and(sim_times>=sim_times[matchtimes[0]],
                                          sim_times<=sim_times[matchtimes[1]]),
                     color='#83C2C9', alpha=1, transform=plt.gca().get_xaxis_transform(),
                     label='Co-housing', linestyle='None', edgecolor='#83C2C9')
    plt.fill_between(sim_times, lo, hi, alpha=0.2, linestyle='None', label='Sim. 95% CI', color=maincolor)
    plt.plot(sim_times, md, color=maincolor, label='Sim. Median', linewidth=2)
    plt.plot(data.xdata[0], data.ydata[0][:, 0], 'ko', label='Data', markersize=10)
    
    plt.xlabel('time past exposure (days)', fontsize=22)
    plt.ylabel('viral load\n (log10 PFU/mL)', fontsize=22)
    plt.gca().set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    plt.gca().set_yticks([3, 4, 5, 6, 7])
    plt.gca().set_xticklabels(labels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'], fontsize=18)
    plt.gca().set_yticklabels(labels=['3', '4', '5', '6', '7'], fontsize=18)
    plt.ylim(np.log10(500), 7.5)
    plt.xlim(0, 7)
    plt.tight_layout()
    plt.savefig(figurepath+'_cohousing_VL_timezoom.png', dpi=300)

# [[MAIN CODE TO EXECUTE]] ======================================
# In the code below, I've preserved the original file naming and organization used.
# Note that the results are split up across two folders (two separate runs of this script):
# 20220105_24 has fitting results from source-CTRL, contact-CTRL, contact-TIP.
# 20220105_24b has fitting results from source-TIP.
# This is because I had to re-simulate the process for source-TIP.

# For running source-CTRL, contact-CTRL, contact-TIP
dirname = '20220105_24_allCtrlTIP_fitOnLogScale_fixPi_recipVTIP0_0_adjPeakPenalty_flat1dtol_don2_rec3'
# For running source-TIP
# dirname = '20220105_24b_allCtrlTIP_fitOnLogScale_fixPi_recipVTIP0_0_adjPeakPenalty_flat1dtol_don2_rec3_resimulateSourceTIP'
VTIP0_recip = 0
for VTIP0_recip in [0]:
    for id in [1, 2, 3, 4, 5]:
        for housing in ['donor', 'recip']:
            for treat in ['TIP', 'ctrl']:
                # Defining which conditions to repeat/not repeat.
                # To exclude source-CTRL, contact-CTRL, contact-TIP
                # if treat == 'ctrl' and VTIP0_recip != 0:
                #     continue
                # if (housing == 'donor' and treat == 'TIP'):
                #     continue

                # To exclude source-TIP
                if not (housing == 'donor' and treat == 'TIP'):
                    continue

                filename = treat+'-'+housing+'-'+str(id)
                hamstercode = treat+'-'+housing+'-'+str(id)

                # Do the initial MCMC fit process. All fitting information is
                # saved in .json files, so you only need to run each fit once.
                # Uncomment fit_model if you want to generate new MCMC fits.
                # fit_model(dirname, filename, hamstercode, VTIP0_recip)

                # Plot the fitting projections on top of the data.
                visualize_result(pathname=dirname + '/' + filename + '.json',
                                 savename=dirname + '/' + filename,
                                 target_hamster_name=hamstercode,
                                 VTIP0_recip=VTIP0_recip,
                                 interval_file=dirname + '/' + filename+'_VL_intervals.txt')
                visualize_result(pathname=dirname + '/' + filename + '.json',
                                 savename=dirname + '/' + filename,
                                 target_hamster_name=hamstercode,
                                 VTIP0_recip=VTIP0_recip)

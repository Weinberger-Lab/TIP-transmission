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

#Compare all parameters in target files
Ctrl_source_json = ['20220105_24_allCtrlTIP_fitOnLogScale_fixPi_recipVTIP0_0_adjPeakPenalty_flat1dtol_don2_rec3/ctrl-donor-1.json',
                    '20220105_24_allCtrlTIP_fitOnLogScale_fixPi_recipVTIP0_0_adjPeakPenalty_flat1dtol_don2_rec3/ctrl-donor-2.json',
                    '20220105_24_allCtrlTIP_fitOnLogScale_fixPi_recipVTIP0_0_adjPeakPenalty_flat1dtol_don2_rec3/ctrl-donor-3.json',
                    '20220105_24_allCtrlTIP_fitOnLogScale_fixPi_recipVTIP0_0_adjPeakPenalty_flat1dtol_don2_rec3/ctrl-donor-4.json',
                    '20220105_24_allCtrlTIP_fitOnLogScale_fixPi_recipVTIP0_0_adjPeakPenalty_flat1dtol_don2_rec3/ctrl-donor-5.json']

TIP_source_json = ['20220105_24b_allCtrlTIP_fitOnLogScale_fixPi_recipVTIP0_0_adjPeakPenalty_flat1dtol_don2_rec3_resimulateSourceTIP/TIP-donor-1.json',
                   '20220105_24b_allCtrlTIP_fitOnLogScale_fixPi_recipVTIP0_0_adjPeakPenalty_flat1dtol_don2_rec3_resimulateSourceTIP/TIP-donor-2.json',
                   '20220105_24b_allCtrlTIP_fitOnLogScale_fixPi_recipVTIP0_0_adjPeakPenalty_flat1dtol_don2_rec3_resimulateSourceTIP/TIP-donor-2.json',
                   '20220105_24b_allCtrlTIP_fitOnLogScale_fixPi_recipVTIP0_0_adjPeakPenalty_flat1dtol_don2_rec3_resimulateSourceTIP/TIP-donor-3.json',
                   '20220105_24b_allCtrlTIP_fitOnLogScale_fixPi_recipVTIP0_0_adjPeakPenalty_flat1dtol_don2_rec3_resimulateSourceTIP/TIP-donor-4.json']

Ctrl_contact_json = ['20220105_24_allCtrlTIP_fitOnLogScale_fixPi_recipVTIP0_0_adjPeakPenalty_flat1dtol_don2_rec3/ctrl-recip-1.json',
                     '20220105_24_allCtrlTIP_fitOnLogScale_fixPi_recipVTIP0_0_adjPeakPenalty_flat1dtol_don2_rec3/ctrl-recip-2.json',
                     '20220105_24_allCtrlTIP_fitOnLogScale_fixPi_recipVTIP0_0_adjPeakPenalty_flat1dtol_don2_rec3/ctrl-recip-3.json',
                     '20220105_24_allCtrlTIP_fitOnLogScale_fixPi_recipVTIP0_0_adjPeakPenalty_flat1dtol_don2_rec3/ctrl-recip-4.json',
                     '20220105_24_allCtrlTIP_fitOnLogScale_fixPi_recipVTIP0_0_adjPeakPenalty_flat1dtol_don2_rec3/ctrl-recip-5.json']

TIP_contact_json = ['20220105_24_allCtrlTIP_fitOnLogScale_fixPi_recipVTIP0_0_adjPeakPenalty_flat1dtol_don2_rec3/TIP-recip-1.json',
                    '20220105_24_allCtrlTIP_fitOnLogScale_fixPi_recipVTIP0_0_adjPeakPenalty_flat1dtol_don2_rec3/TIP-recip-2.json',
                    '20220105_24_allCtrlTIP_fitOnLogScale_fixPi_recipVTIP0_0_adjPeakPenalty_flat1dtol_don2_rec3/TIP-recip-3.json',
                    '20220105_24_allCtrlTIP_fitOnLogScale_fixPi_recipVTIP0_0_adjPeakPenalty_flat1dtol_don2_rec3/TIP-recip-4.json',
                    '20220105_24_allCtrlTIP_fitOnLogScale_fixPi_recipVTIP0_0_adjPeakPenalty_flat1dtol_don2_rec3/TIP-recip-5.json']

result = {'Ctrl-Source-1': structures.ResultsStructure.ResultsStructure.load_json_object(Ctrl_source_json[0]),
          'Ctrl-Source-2': structures.ResultsStructure.ResultsStructure.load_json_object(Ctrl_source_json[1]),
          'Ctrl-Source-3': structures.ResultsStructure.ResultsStructure.load_json_object(Ctrl_source_json[2]),
          'Ctrl-Source-4': structures.ResultsStructure.ResultsStructure.load_json_object(Ctrl_source_json[3]),
          'Ctrl-Source-5': structures.ResultsStructure.ResultsStructure.load_json_object(Ctrl_source_json[4]),
          'TIP-Source-1': structures.ResultsStructure.ResultsStructure.load_json_object(TIP_source_json[0]),
          'TIP-Source-2': structures.ResultsStructure.ResultsStructure.load_json_object(TIP_source_json[1]),
          'TIP-Source-3': structures.ResultsStructure.ResultsStructure.load_json_object(TIP_source_json[2]),
          'TIP-Source-4': structures.ResultsStructure.ResultsStructure.load_json_object(TIP_source_json[3]),
          'TIP-Source-5': structures.ResultsStructure.ResultsStructure.load_json_object(TIP_source_json[4]),
          'Ctrl-Contact-1': structures.ResultsStructure.ResultsStructure.load_json_object(Ctrl_contact_json[0]),
          'Ctrl-Contact-2': structures.ResultsStructure.ResultsStructure.load_json_object(Ctrl_contact_json[1]),
          'Ctrl-Contact-3': structures.ResultsStructure.ResultsStructure.load_json_object(Ctrl_contact_json[2]),
          'Ctrl-Contact-4': structures.ResultsStructure.ResultsStructure.load_json_object(Ctrl_contact_json[3]),
          'Ctrl-Contact-5': structures.ResultsStructure.ResultsStructure.load_json_object(Ctrl_contact_json[4]),
          'TIP-Contact-1': structures.ResultsStructure.ResultsStructure.load_json_object(TIP_contact_json[0]),
          'TIP-Contact-2': structures.ResultsStructure.ResultsStructure.load_json_object(TIP_contact_json[1]),
          'TIP-Contact-3': structures.ResultsStructure.ResultsStructure.load_json_object(TIP_contact_json[2]),
          'TIP-Contact-4': structures.ResultsStructure.ResultsStructure.load_json_object(TIP_contact_json[3]),
          'TIP-Contact-5': structures.ResultsStructure.ResultsStructure.load_json_object(TIP_contact_json[4])}

# Assemble a dataframe containing the last 9000 steps of all..
df = pd.DataFrame()

for key, res in zip(result.keys(), result.items()):

    treat, housing, id = key.split('-')
    # Take parameters. If TIP source need to estimate log10VTIP0 too
    if treat == 'TIP' and housing == 'Source':
        curr_params = np.array(res[1]['chain'])[1000:, 0:5]
        curr_df_entry = pd.DataFrame(curr_params, columns=['beta', 'delta', 'c', 'log10V0', 'log10VTIP0'])
        curr_df_entry["treat"] = treat
        curr_df_entry["housing"] = housing
        curr_df_entry["id"] = id
    else:
        curr_params = np.array(res[1]['chain'])[1000:, 0:4]
        #np.concatenate((curr_params, np.ones((9000, 1))), axis=1)
        curr_df_entry = pd.DataFrame(curr_params, columns=['beta', 'delta', 'c', 'log10V0'])
        curr_df_entry["treat"] = treat
        curr_df_entry["housing"] = housing
        curr_df_entry["id"] = id

    df = df.append(curr_df_entry, ignore_index=True)

# result['chain'] = np.array(result['chain'])

# beta values, split by experimental condition
Ctrl_palette = ['#333399', '#B3B3E6','#7575D1', '#0F0F2E', '#1F1F5C']
TIP_palette = ['#FF3399', '#FFADD6', '#FF70B8', '#3D001E', '#8F0047']

Ctrl_source = df[(df['treat']=='Ctrl') & (df['housing']=='Source')]
TIP_source = df[(df['treat']=='TIP') & (df['housing']=='Source')]
Ctrl_contact = df[(df['treat']=='Ctrl') & (df['housing']=='Contact')]
TIP_contact = df[(df['treat']=='TIP') & (df['housing']=='Contact')]

Ctrl_source_dir = '20220105_24_allCtrlTIP_fitOnLogScale_fixPi_recipVTIP0_0_adjPeakPenalty_flat1dtol_don2_rec3/'
TIP_source_dir = '20220105_24b_allCtrlTIP_fitOnLogScale_fixPi_recipVTIP0_0_adjPeakPenalty_flat1dtol_don2_rec3_resimulateSourceTIP/'
Ctrl_contact_dir = '20220105_24_allCtrlTIP_fitOnLogScale_fixPi_recipVTIP0_0_adjPeakPenalty_flat1dtol_don2_rec3/'
TIP_contact_dir = '20220105_24_allCtrlTIP_fitOnLogScale_fixPi_recipVTIP0_0_adjPeakPenalty_flat1dtol_don2_rec3/'

# CTRL SOURCE
plt.figure(figsize=(4,3))
sns.violinplot(data=Ctrl_source, y="beta", x="id",
               palette=Ctrl_palette, cut=0, scale='width', saturation=1)
plt.ylim(0, 10*10**-5)
plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
plt.tight_layout()
plt.savefig(Ctrl_source_dir+'parameters_Ctrl-Source_beta.png', dpi=300)
plt.figure(figsize=(4,3))
sns.violinplot(data=Ctrl_source, y="delta", x="id",
               palette=Ctrl_palette, cut=0, scale='width', saturation=1)
plt.ylim(-5, 105)
plt.tight_layout()
plt.savefig(Ctrl_source_dir+'parameters_Ctrl-Source_delta.png', dpi=300)
plt.figure(figsize=(4,3))
sns.violinplot(data=Ctrl_source, y="c", x="id",
               palette=Ctrl_palette, cut=0, scale='width', saturation=1)
plt.ylim(-5, 105)
plt.tight_layout()
plt.savefig(Ctrl_source_dir+'parameters_Ctrl-Source_c.png', dpi=300)
plt.figure(figsize=(4,3))
sns.violinplot(data=Ctrl_source, y="log10V0", x="id",
               palette=Ctrl_palette, cut=0, scale='width', saturation=1)
plt.ylim(-0.5, 7.5)
plt.tight_layout()
plt.savefig(Ctrl_source_dir+'parameters_Ctrl-Source_log10V0.png', dpi=300)

# TIP SOURCE
plt.figure(figsize=(4,3))
sns.violinplot(data=TIP_source, y="beta", x="id",
               palette=TIP_palette, cut=0, scale='width', saturation=1)
plt.ylim(0, 10*10**-5)
plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
plt.tight_layout()
plt.savefig(TIP_source_dir+'parameters_TIP-Source_beta.png', dpi=300)
plt.figure(figsize=(4,3))
sns.violinplot(data=TIP_source, y="delta", x="id",
               palette=TIP_palette, cut=0, scale='width', saturation=1)
plt.ylim(-5, 105)
plt.tight_layout()
plt.savefig(TIP_source_dir+'parameters_TIP-Source_delta.png', dpi=300)
plt.figure(figsize=(4,3))
sns.violinplot(data=TIP_source, y="c", x="id",
               palette=TIP_palette, cut=0, scale='width', saturation=1)
plt.ylim(-5, 105)
plt.tight_layout()
plt.savefig(TIP_source_dir+'parameters_TIP-Source_c.png', dpi=300)
plt.figure(figsize=(4,3))
sns.violinplot(data=TIP_source, y="log10V0", x="id",
               palette=TIP_palette, cut=0, scale='width', saturation=1)
plt.ylim(-0.5, 7.5)
plt.tight_layout()
plt.savefig(TIP_source_dir+'parameters_TIP-Source_log10V0.png', dpi=300)
plt.figure(figsize=(4,3))
sns.violinplot(data=TIP_source, y="log10VTIP0", x="id",
               palette=TIP_palette, cut=0, scale='width', saturation=1)
plt.ylim(-0.5, 7.5)
plt.tight_layout()
plt.savefig(TIP_source_dir+'parameters_TIP-Source_log10VTIP0.png', dpi=300)

# CTRL CONTACT
plt.figure(figsize=(4,3))
sns.violinplot(data=Ctrl_contact, y="beta", x="id",
               palette=Ctrl_palette, cut=0, scale='width', saturation=1)
plt.ylim(0, 10*10**-5)
plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
plt.tight_layout()
plt.savefig(Ctrl_contact_dir+'parameters_Ctrl-Contact_beta.png', dpi=300)
plt.figure(figsize=(4,3))
sns.violinplot(data=Ctrl_contact, y="delta", x="id",
               palette=Ctrl_palette, cut=0, scale='width', saturation=1)
plt.ylim(-5, 105)
plt.tight_layout()
plt.savefig(Ctrl_contact_dir+'parameters_Ctrl-Contact_delta.png', dpi=300)
plt.figure(figsize=(4,3))
sns.violinplot(data=Ctrl_contact, y="c", x="id",
               palette=Ctrl_palette, cut=0, scale='width', saturation=1)
plt.ylim(-5, 105)
plt.tight_layout()
plt.savefig(Ctrl_contact_dir+'parameters_Ctrl-Contact_c.png', dpi=300)
plt.figure(figsize=(4,3))
sns.violinplot(data=Ctrl_contact, y="log10V0", x="id",
               palette=Ctrl_palette, cut=0, scale='width', saturation=1)
plt.ylim(-0.5, 7.5)
plt.tight_layout()
plt.savefig(Ctrl_contact_dir+'parameters_Ctrl-Contact_log10V0.png', dpi=300)

# TIP CONTACT
plt.figure(figsize=(4,3))
sns.violinplot(data=TIP_contact, y="beta", x="id",
               palette=TIP_palette, cut=0, scale='width', saturation=1)
plt.ylim(0, 10*10**-5)
plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
plt.tight_layout()
plt.savefig(TIP_contact_dir+'parameters_TIP-Contact_beta.png', dpi=300)
plt.figure(figsize=(4,3))
sns.violinplot(data=TIP_contact, y="delta", x="id",
               palette=TIP_palette, cut=0, scale='width', saturation=1)
plt.ylim(-5, 105)
plt.tight_layout()
plt.savefig(TIP_contact_dir+'parameters_TIP-Contact_delta.png', dpi=300)
plt.figure(figsize=(4,3))
sns.violinplot(data=TIP_contact, y="c", x="id",
               palette=TIP_palette, cut=0, scale='width', saturation=1)
plt.ylim(-5, 105)
plt.tight_layout()
plt.savefig(TIP_contact_dir+'parameters_TIP-Contact_c.png', dpi=300)
plt.figure(figsize=(4,3))
sns.violinplot(data=TIP_contact, y="log10V0", x="id",
               palette=TIP_palette, cut=0, scale='width', saturation=1)
plt.ylim(-0.5, 7.5)
plt.tight_layout()
plt.savefig(TIP_contact_dir+'parameters_TIP-Contact_log10V0.png', dpi=300)
plt.figure(figsize=(4,3))
sns.violinplot(data=TIP_contact, y="log10VTIP0", x="id",
               palette=TIP_palette, cut=0, scale='width', saturation=1)
plt.ylim(-0.5, 7.5)
plt.tight_layout()
plt.savefig(TIP_contact_dir+'parameters_TIP-Contact_log10VTIP0.png', dpi=300)

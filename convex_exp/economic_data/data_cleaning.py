# This file supports the economics_data notebooks

# Download Data
from sklearn import preprocessing
import pandas as pd
import numpy as np

def cash_transfer_data(data_dir):
    # Download data and use only 'Cind' < 10,000 to match original study
    data = pd.read_stata(data_dir)
    data = data[data['Cind'] <10000]

    # Change variables to NA or other
    data['hhhage']=data['hhhage'].replace(['97 y más'],97)
    data['hhhage_clean1']=np.where(data['hhhage']=='no sabe', None, data['hhhage'])
    data['hhhage_cl']=np.where(data['hhhage_clean1']=='nr', None, data['hhhage_clean1'])
    data['hhhspouse_cl']=np.where(data['hhhspouse']=='nr', None, data['hhhspouse'])

    # Change variables to binary
    hhhsex_cond = [(data['hhhsex']=="hombre"),(data['hhhsex']=="mujer"),(data['hhhsex']== 9.0)]
    hhhsex_values = [1,0,pd.NA]
    data['hhhsex_bi'] = np.select(hhhsex_cond,hhhsex_values)

    p16_cond = [(data['p16']=="si"),(data['p16']=="no"),(data['p16']== "nr")]
    p16_values = [1,0,pd.NA]
    data['p16_bi'] = np.select(p16_cond,p16_values)

    hhhalpha_cond = [(data['hhhalpha']=="si"),(data['hhhalpha']=="no"),(data['hhhalpha']== "nr"),(data['hhhalpha']== 'na')]
    hhhalpha_values = [1,0,pd.NA,pd.NA]
    data['hhhalpha_bi'] = np.select(hhhalpha_cond,hhhalpha_values)
    
    # Fill all NA values with mean
    data.fillna(data.mean(), inplace=True)

    # Standardize the independent variables 
    x_data = data[['treatp','treatnp','hhhsex_bi','hectareas','vhhnum','hhhage_cl', 'hhhspouse_cl']]
    scaler_x = preprocessing.StandardScaler().fit(x_data)
    x_data = scaler_x.transform(x_data)
    
    # Combine data into one dataset
    standardized_data = pd.DataFrame(x_data, columns=['treatp','treatnp','hhhsex_bi','hectareas','vhhnum','hhhage_cl', 'hhhspouse_cl'])
    standardized_data['Cind'] = data['Cind']
    standardized_data['t'] = data['t']
    standardized_data = standardized_data.dropna()
    
    return standardized_data


def oregon_medicaid_data(data_descr_dir, data_survey12m_dir):
    # Download data
    data_descr = pd.read_stata(data_descr_dir)
    data_survey12m = pd.read_stata(data_survey12m_dir)
    
    # Dependent Variables (Yihj)
    # Change # of badday to # of goodday out of 30days
    data_survey12m['gooddays_phys_12m'] = 30 - data_survey12m['baddays_phys_12m']
    data_survey12m['gooddays_ment_12m'] = 30 - data_survey12m['baddays_ment_12m']
    data_survey12m['gooddays_tot_12m'] = 30 - data_survey12m['baddays_tot_12m']

    # Create bin for health_gen_12m (not poor)
    data_survey12m['health_gen_bin_fair_12m'] = np.where(data_survey12m['health_gen_12m']=='poor', 0,1)
    depen_var = data_survey12m[["person_id","health_gen_bin_12m","health_gen_bin_fair_12m","health_chg_bin_12m",'gooddays_phys_12m','gooddays_ment_12m','gooddays_tot_12m',"dep_sad_12m"]]

    # Lottery Variable (LOTTERY)
    lottery = data_descr[['person_id','treatment','household_id']]

    # Household size fixed effects, survey wave fixed effects and interaction between the two (Xih)
    fixed_var = data_survey12m[['person_id','hhsize_12m', 'wave_survey12m']]

    # Demographic and Economic covariates (Vih)
    dem_eco_var = data_survey12m[['person_id','employ_hrs_12m', 'edu_12m',"dia_dx_12m", "ast_dx_12m", "hbp_dx_12m", "emp_dx_12m", "dep_dx_12m","ins_any_12m", "ins_ohp_12m", "ins_private_12m", "ins_other_12m", "ins_months_12m" ]]

    return data_survey12m, depen_var, lottery, fixed_var, dem_eco_var
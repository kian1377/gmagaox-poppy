
import numpy as np
import poppy
import astropy.units as u

SURF_UNIT = u.nm
LENGTH_UNIT = u.m
SCREEN_SIZE = int(800*4)
SCREEN_SIZE = 5080
SCREEN_SIZE = 8192

# m1_params = {
#     'psd_parameters': [[]], 
#     'psd_weight': [1],
#     'apply_reflection':True, 
#     'incident_angle':0*u.degree, 
#     'screen_size':SCREEN_SIZE, 
# }

# m2_params = {
#     'psd_parameters': [[]], 
#     'psd_weight': [1],
#     'apply_reflection':True, 
#     'incident_angle':0*u.degree, 
#     'screen_size':SCREEN_SIZE, 
# }

NUM_OPDS = 47
SEEDS = np.arange(1234, 1234+NUM_OPDS)

oap_params = {
    'psd_parameters': [[3.029, 329.3* u.nm**2/(u.m**(3.029-2)), 0.019 * u.m, 0.0001, 0 * u.nm**2 * u.m**2],
                    [-3.103, 1.606e-12 * u.nm**2/(u.m**(-3.103-2)), 16 * u.m, 0.00429,0 * u.nm**2 * u.m**2],
                    [0.8, 0.0001 * u.nm**2/(u.m**(0.8-2)), 0.024 * u.m, 0.00021, 6.01284e-13 * u.nm**2 * u.m**2]], 
    'psd_weight': [1],
    'apply_reflection':True, 
    'screen_size':SCREEN_SIZE, 
}

flat_params = {
    'psd_parameters': [[3.284, 1180 * u.nm**2/(u.m**(3.284-2)), 0.017 * u.m, 0.0225, 0 * u.nm**2 * u.m**2],
                        [1.947, 0.02983 * u.nm**2/(u.m**(1.947-2)), 15 * u.m, 0.00335, 0 * u.nm**2 * u.m**2],
                        [2.827, 44.25 * u.nm**2/(u.m**(2.827-2)), 0.00057 * u.m, 0.000208, 1.27214e-14 * u.nm**2 * u.m**2]], 
    'psd_weight': [1],
    'apply_reflection':True, 
    'screen_size':SCREEN_SIZE, 
}

incident_angles = {
    'm1': 0.0*u.degree,
    'm2': 0.0*u.degree,
    'm3': 32.8*u.degree,
    'Roap1': 15.0*u.degree, 
    'Roap2': 40.0*u.degree, 
    'AOoap1': 20.0*u.degree,  
    'AOoap2': 13.0*u.degree,  
    'AOoap3': 10.0*u.degree,  
    'AOoap4': 20.4*u.degree,  
    'AOoap5': 5.0*u.degree,  
    'AOoap6': 12.0*u.degree,  
    'AOoap7': 15.0*u.degree,  
    'AOoap8': 27.0*u.degree,  
    'AOoap9-1': 25.0*u.degree,  
    'AOoap9-2': 25.0*u.degree,  
    'AOoap9-3': 14.0*u.degree,  
    'km1': 60.0*u.degree, 
    'km2': 30.0*u.degree, 
    'km3': 60.0*u.degree, 
    'pfm1': 47.7*u.degree, 
    'pfm2': 50.4*u.degree, 
    'fm1': 15.0*u.degree, 
    'fm2': 29.4*u.degree, 
    'fm3': 18.2*u.degree, 
    'fm4': 45.0*u.degree, 
    'fm5': 20.6*u.degree, 
    'fm6': 28.6*u.degree, 
    'fm7': 20.2*u.degree, 
    'fm8': 16.3*u.degree, 
    'fm9': 13.0*u.degree, 
    'fm10': 4.0*u.degree, 
    'fm11': 28.0*u.degree, 
    'fm12': 20.0*u.degree, # ? 
    'fm13': 26.0*u.degree, 
    'fm14': 20*u.degree, # ? 
    'fm15': 20*u.degree, # ? 
    'fm16': 20*u.degree, # ? 
    'fm17': 20*u.degree, # ? 
    'fsm': 35.0*u.degree, 
    'woofer':10.0*u.degree,
    'tweeter': 0.1*u.degree,
    'ncpDM': 11.6*u.degree, 
    'fpsm': 28.9*u.degree, 
    'pupilSM': 7.5*u.degree, 
}

wfe_psds = {
    # 'm1':poppy.PowerSpectrumWFE(**m1_params, incident_angle=incident_angles['m1'], seed=SEEDS[0]), 
    # 'm2':poppy.PowerSpectrumWFE(**m2_params, incident_angle=incident_angles['m2'], seed=SEEDS[1]), 
    # 'm3':poppy.PowerSpectrumWFE(**m3_params, incident_angle=incident_angles['m3'], seed=SEEDS[2]), 
    'Roap1':poppy.PowerSpectrumWFE(**oap_params, incident_angle=incident_angles['Roap1'], seed=SEEDS[3]), 
    'Roap2':poppy.PowerSpectrumWFE(**oap_params, incident_angle=incident_angles['Roap2'], seed=SEEDS[4]), 
    'AOoap1':poppy.PowerSpectrumWFE(**oap_params, incident_angle=incident_angles['AOoap1'], seed=SEEDS[5]), 
    'AOoap2':poppy.PowerSpectrumWFE(**oap_params, incident_angle=incident_angles['AOoap2'], seed=SEEDS[6]), 
    'AOoap3':poppy.PowerSpectrumWFE(**oap_params, incident_angle=incident_angles['AOoap3'], seed=SEEDS[7]), 
    'AOoap4':poppy.PowerSpectrumWFE(**oap_params, incident_angle=incident_angles['AOoap4'], seed=SEEDS[8]), 
    'AOoap5':poppy.PowerSpectrumWFE(**oap_params, incident_angle=incident_angles['AOoap5'], seed=SEEDS[9]), 
    'AOoap6':poppy.PowerSpectrumWFE(**oap_params, incident_angle=incident_angles['AOoap6'], seed=SEEDS[10]), 
    'AOoap7':poppy.PowerSpectrumWFE(**oap_params, incident_angle=incident_angles['AOoap7'], seed=SEEDS[11]), 
    'AOoap8':poppy.PowerSpectrumWFE(**oap_params, incident_angle=incident_angles['AOoap8'], seed=SEEDS[12]), 
    'AOoap9-1':poppy.PowerSpectrumWFE(**oap_params, incident_angle=incident_angles['AOoap9-1'], seed=SEEDS[13]), 
    'AOoap9-2':poppy.PowerSpectrumWFE(**oap_params, incident_angle=incident_angles['AOoap9-2'], seed=SEEDS[14]), 
    'AOoap9-3':poppy.PowerSpectrumWFE(**oap_params, incident_angle=incident_angles['AOoap9-3'], seed=SEEDS[15]), 
    'fm1':poppy.PowerSpectrumWFE(**flat_params, incident_angle=incident_angles['fm1'], seed=SEEDS[16]), 
    'fm2':poppy.PowerSpectrumWFE(**flat_params, incident_angle=incident_angles['fm2'], seed=SEEDS[17]), 
    'fm3':poppy.PowerSpectrumWFE(**flat_params, incident_angle=incident_angles['fm3'], seed=SEEDS[18]), 
    'fm4':poppy.PowerSpectrumWFE(**flat_params, incident_angle=incident_angles['fm4'], seed=SEEDS[19]), 
    'fm5':poppy.PowerSpectrumWFE(**flat_params, incident_angle=incident_angles['fm5'], seed=SEEDS[20]), 
    'fm6':poppy.PowerSpectrumWFE(**flat_params, incident_angle=incident_angles['fm6'], seed=SEEDS[21]), 
    'fm7':poppy.PowerSpectrumWFE(**flat_params, incident_angle=incident_angles['fm7'], seed=SEEDS[22]), 
    'fm8':poppy.PowerSpectrumWFE(**flat_params, incident_angle=incident_angles['fm8'], seed=SEEDS[23]), 
    'fm9':poppy.PowerSpectrumWFE(**flat_params, incident_angle=incident_angles['fm9'], seed=SEEDS[24]), 
    'fm10':poppy.PowerSpectrumWFE(**flat_params, incident_angle=incident_angles['fm10'], seed=SEEDS[25]), 
    'fm11':poppy.PowerSpectrumWFE(**flat_params, incident_angle=incident_angles['fm11'], seed=SEEDS[26]), 
    'fm12':poppy.PowerSpectrumWFE(**flat_params, incident_angle=incident_angles['fm12'], seed=SEEDS[27]), 
    'fm13':poppy.PowerSpectrumWFE(**flat_params, incident_angle=incident_angles['fm13'], seed=SEEDS[28]), 
    'fm14':poppy.PowerSpectrumWFE(**flat_params, incident_angle=incident_angles['fm14'], seed=SEEDS[29]), 
    'fm15':poppy.PowerSpectrumWFE(**flat_params, incident_angle=incident_angles['fm15'], seed=SEEDS[30]), 
    'fm16':poppy.PowerSpectrumWFE(**flat_params, incident_angle=incident_angles['fm16'], seed=SEEDS[31]), 
    'fm17':poppy.PowerSpectrumWFE(**flat_params, incident_angle=incident_angles['fm17'], seed=SEEDS[32]), 
    'km1':poppy.PowerSpectrumWFE(**flat_params, incident_angle=incident_angles['km1'], seed=SEEDS[33]), 
    'km2':poppy.PowerSpectrumWFE(**flat_params, incident_angle=incident_angles['km2'], seed=SEEDS[34]), 
    'km3':poppy.PowerSpectrumWFE(**flat_params, incident_angle=incident_angles['km3'], seed=SEEDS[35]), 
    'fpsm':poppy.PowerSpectrumWFE(**flat_params, incident_angle=incident_angles['fpsm'], seed=SEEDS[36]), 
    'pupilSM':poppy.PowerSpectrumWFE(**flat_params, incident_angle=incident_angles['pupilSM'], seed=SEEDS[37]), 
    'pfm1':poppy.PowerSpectrumWFE(**flat_params, incident_angle=incident_angles['pfm1'], seed=SEEDS[38]), 
    'pfm2':poppy.PowerSpectrumWFE(**flat_params, incident_angle=incident_angles['pfm2'], seed=SEEDS[39]), 
    # 'woofer':poppy.PowerSpectrumWFE(**flat_params, incident_angle=incident_angles['woofer'], seed=SEEDS[36]), 
    # 'tweeter':poppy.PowerSpectrumWFE(**flat_params, incident_angle=incident_angles['tweeter'], seed=SEEDS[37]), 
    # 'ncpDM':poppy.PowerSpectrumWFE(**flat_params, incident_angle=incident_angles['ncpDM'], seed=SEEDS[38]), 
    # '':poppy.PowerSpectrumWFE(**_params, seed=SEEDS[]),
}



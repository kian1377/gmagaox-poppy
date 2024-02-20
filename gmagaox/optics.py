import poppy
import astropy.units as u

'''
Notes about the abbreviations used here: 
    ifp = intermediate focal plane
    fm = fold mirror
    km = K-mirror
    Roap = Relay OAP
    AOoap = AO bench OAP
    fpsm = focal plane steering mirror
    pfm = periscope fold mirror
    pupilSM = pupil steering mirror
''' 

'''
These are the corrections (or fudge-factors) calculated using the Fresnel model
in order to fix the issue with focal-planes not being a tight focus or pupil planes
not being reimaged correctly. 
'''

# ifp8157_1_correction = -0.013059196*u.mm
# fsm_correction = 0.75*u.mm
# ifp14_correction = -0*u.mm

# these are distances using Oli's calculations
# distances = {
#     'm1_m2':2.026247641994000E+004*u.mm,
#     'm2_m3':24883.1*u.mm, 
#     'm3_ifp8p157':1209.4*u.mm + ifp8157_1_correction,
#     'ifp8p157_Roap1':130.5*u.mm,
#     'Roap1_fm1':50.5 * u.mm, 
#     'fm1_fsm':80.8 * u.mm + fsm_correction,  
#     'fsm_fm2':40.0 * u.mm, 
#     'fm2_Roap2':190.9 * u.mm, 
#     'Roap2_fm3': 53.1 * u.mm, 
#     'fm3_ifp14': 170.8 * u.mm + ifp14_correction, 
#     'ifp14_km1': 89.9 * u.mm, 
#     'km1_km2': 30.0 * u.mm, 
#     'km2_km3': 30.0 * u.mm, 
#     'km3_fm4': 49.4 * u.mm, 
#     'fm4_AOoap1': 500.4 * u.mm, 
#     'AOoap1_fm5': 448.9 * u.mm, 
#     'fm5_fm6': 283.3 * u.mm, 
#     'fm6_fm7': 138.6 * u.mm, 
#     'fm7_AOoap2': 215.7 * u.mm, 
#     'AOoap2_ifp8157': 407.5 * u.mm, 
#     'ifp8157_fm8': 160.0 * u.mm, 
#     'fm8_AOoap3': 545.7 * u.mm, 
#     'AOoap3_woofer': 634.7 * u.mm, 
#     'woofer_AOoap4': 1264.1 * u.mm, 
#     'AOoap4_fm9': 520.5 * u.mm, 
#     'fm9_ifp15': 778.2 * u.mm, 
#     'ifp15_fm10': 150.0 * u.mm, 
#     'fm10_AOoap5': 945.2 * u.mm, 
#     'AOoap5_tweeter': 1119.7 * u.mm, 
#     'tweeter_AOoap5': 1119.7 * u.mm, 
#     'AOoap5_fm10': 945.2 * u.mm, 
#     'fm10_ifp15': 150.0 * u.mm, 
#     'ifp15_knifeedge': 1.4 * u.mm, 
#     'knifeedge_AOoap6': 178.5 * u.mm, 
#     'AOoap6_fpsm': 179.3 * u.mm, 
#     'fpsm_pfm1': * u.mm, 
#     'pfm1_pfm2': * u.mm, 
#     'pfm2_AOoap7': * u.mm, 
#     'AOoap7_fm11': * u.mm, 
#     'fm11_ifp69': * u.mm, 
#     'ifp69_pupilSM': * u.mm, 
#     'pupilSM_fm12': * u.mm, 
#     'fm12_AOoap8': * u.mm,
#     'AOoap8_fm13': * u.mm,  
#     'fm13_fm14': * u.mm, 
#     'fm14_ncpDM': * u.mm,
#     'ncpDM_AOoap91': * u.mm, # should be a 24mm pupil plane
#     'AOoap91_fm15': * u.mm, 
#     'fm15_ifp34p5': * u.mm, 
#     'ifp34p5_AOoap92': * u.mm, 
#     'AOoap92_lyot': * u.mm, 
#     'lyot_fm16': * u.mm, # should be about a 25.2mm pupil 
#     'fm16_AOoap93' * u.mm, 
#     'AOoap93_fm17': *u.mm, 
#     'fm17_scicamfp34p5': * u.mm, 
#     # '': * u.mm, 
#     # '': * u.mm, 
# }

# using zemax distances
distances = {
    'm1_m2':2.026247641994001E+004*u.mm,
    'm2_m3':2.488307641994001E+004*u.mm,
    'm3_ifp8.157':1.209399999840673E+003* u.mm,
    'ifp8.157_Roap1':1.303819612269508E+002 * u.mm, 
    'Roap1_fm1':5.046576176326926E+001 * u.mm, 
    'fm1_fsm-pp':8.064222541315394E+001 * u.mm, # should be about a 
    'fsm-pp_fm2':4.419473215254402E+001 * u.mm,  
    'fm2_Roap2': 1.866823542617713E+002 * u.mm, 
    'Roap2_fm3': 5.314785939449212E+001 * u.mm, 
    'fm3_ifp14': 1.708024714947751E+002 * u.mm, 
    'ifp14_km1': 9.070354846055852E+001 * u.mm, 
    'km1_km2': 3.000000000000000E+001 * u.mm, 
    'km2_km3': 3.000000000000000E+001 * u.mm, 
    'km3_fm4': 4.944531000001734E+001 * u.mm, 
    'fm4_AOoap1': 5.003924156099456E+002 * u.mm, 
    'AOoap1_fm5': 4.488870780880534E+002 * u.mm, 
    'fm5_ADC-pp': 2.144927190484304E+002 * u.mm + 37.16147000000*u.mm, # should be about a 
    'ADC-pp_fm6': 1.082692247451923E+002 * u.mm - 37.16147000000*u.mm - 17.9667*u.mm,
    'fm6_fm7': 1.385828594115083E+002 * u.mm, 
    'fm7_AOoap2': 2.157380901092838E+002 * u.mm, 
    'AOoap2_ifp8.157': 4.074619813473692E+002 * u.mm, 
    'ifp8.157_fm8': 1.600000142533900E+002 * u.mm, 
    'fm8_AOoap3': 5.457257527700422E+002 * u.mm, 
    'AOoap3_woofer-pp': 6.346731311959666E+002 * u.mm, # should be about a 
    'woofer-pp_AOoap4': 1.264141420946085E+003 * u.mm, 
    'AOoap4_fm9': 5.204624076013715E+002 * u.mm, 
    'fm9_ifp15': 7.781682848657802E+002 * u.mm, 
    'ifp15_fm10': 1.499999982635782E+002 * u.mm, 
    'fm10_AOoap5': 9.451949930353876E+002 * u.mm, 
    'AOoap5_tweeter-pp': 1.119728932177350E+003 * u.mm, # should be about a 
    'tweeter-pp_AOoap5':  * u.mm, 
    'AOoap5_fm10':  * u.mm, 
    'fm10_ifp15':  * u.mm, 
    'ifp15_knifeedge':  * u.mm, 
    'knifeedge_AOoap6':  * u.mm, 
    'AOoap6_fpsm':  * u.mm, 
    'fpsm_pfm1': * u.mm, 
    'pfm1_pfm2': * u.mm, 
    'pfm2_AOoap7': * u.mm, 
    'AOoap7_fm11': * u.mm, 
    'fm11_ifp69': * u.mm, 
    'ifp69_pupilSM': * u.mm, 
    'pupilSM_fm12': * u.mm, 
    'fm12_AOoap8': * u.mm,
    'AOoap8_fm13': * u.mm,  
    'fm13_fm14': * u.mm, 
    'fm14_ncpDM': * u.mm,
    'ncpDM_AOoap9-1': * u.mm, # should be a 24mm pupil plane
    'AOoap91_fm15': * u.mm, 
    'fm15_ifp34.5': * u.mm, 
    'ifp34p5_AOoap9-2': * u.mm, 
    'AOoap92_lyot-pp': * u.mm, 
    'lyot-pp_fm16': * u.mm, # should be about a 25.2mm pupil 
    'fm16_AOoap9-3' * u.mm, 
    'AOoap93_fm17': *u.mm, 
    'fm17_scicamfp34.5': * u.mm, 
    # '': * u.mm, 
    # '': * u.mm, 
}

focal_lengths = {
    'm1':3.599999999999712E+004/2*u.mm,
    'm2':4.163901444921466E+003/2*u.mm,
    # 'Roap1':123.7652225*u.mm, 
    'Roap1': 1.303819612269508E+002*u.mm, 
    # 'Roap1': 131.10799*u.mm,
    # 'Roap2':209.5854548*u.mm, 
    'Roap2': 223.95033*u.mm, 
    # 'AOoap1': 699.7657973 * u.mm, 
    'AOoap1': 700.54127 *u.mm,
    'AOoap2': 407.4637211 * u.mm, 
    # 'AOoap2': 425.4287 * u.mm, 
    # 'AOoap3': 705.7258124 * u.mm, 
    'AOoap3': 705.72577 *u.mm, 
    'AOoap4': 1298.73978 * u.mm, 
    'AOoap5': 1095.194984 * u.mm, 
    'AOoap6':100*u.mm, 
    'AOoap7':100*u.mm, 
    'AOoap8':100*u.mm, 
    'AOoap9-1':100*u.mm, 
    'AOoap9-2':100*u.mm, 
    'AOoap9-3':100*u.mm, 
}

# focal_lengths = {
#     'm1':18000*u.mm,
#     'm2':2081.950722*u.mm,
#     'Roap1':130.3911762*u.mm,
#     'Roap2':315.3732395*u.mm,
#     'AOoap1':768.5710054*u.mm,
#     'AOoap2':423.6816389*u.mm,
#     'AOoap3':722.1402242*u.mm,
#     'AOoap4':1432.004705*u.mm,
#     'AOoap5':1101.478174*u.mm,
#     'AOoap6': 186.0239908*u.mm, 
#     'AOoap7': 718.976155*u.mm, 
#     'AOoap8': 1689.936177*u.mm, 
#     'AOoap9-1': 952.2920583*u.mm, 
#     'AOoap9-2': 966.5768183*u.mm, 
#     'AOoap9-3': 843.2950694*u.mm, 
# }

planes = {
    'm1':poppy.QuadraticLens(f_lens=focal_lengths['m1'], name='m1'), 
    'm2':poppy.QuadraticLens(f_lens=focal_lengths['m2'], name='m2'),
    'm3':poppy.ScalarTransmission(name='m3'), 
    'ifp8.157':poppy.ScalarTransmission(name='ifp8.157'), 
    'Roap1':poppy.QuadraticLens(f_lens=focal_lengths['Roap1'], name='Roap1'),
    'fm1':poppy.ScalarTransmission(name='fm1'),
    'fsm-pp':poppy.ScalarTransmission(name='fsm-pp'),
    'fm2':poppy.ScalarTransmission(name='fm2'),
    'Roap2':poppy.QuadraticLens(f_lens=focal_lengths['Roap2'], name='Roap2'),
    'fm3':poppy.ScalarTransmission(name='fm3'),
    'ifp14':poppy.ScalarTransmission(name='ifp14'),
    'km1':poppy.ScalarTransmission(name='km1'),
    'km2':poppy.ScalarTransmission(name='km2'),
    'km3':poppy.ScalarTransmission(name='km3'),
    'fm4':poppy.ScalarTransmission(name='fm4'),
    'AOoap1':poppy.QuadraticLens(f_lens=focal_lengths['AOoap1'], name='AOoap1'),
    'fm5':poppy.ScalarTransmission(name='fm5'),
    'ADC-pp':poppy.ScalarTransmission(name='ADC-pp'),
    'fm6':poppy.ScalarTransmission(name='fm6'),
    'fm7':poppy.ScalarTransmission(name='fm7'),
    'AOoap2':poppy.QuadraticLens(f_lens=focal_lengths['AOoap2'], name='AOoap2'),
    'ifp8.157':poppy.ScalarTransmission(name='ifp8157'),
    'fm8':poppy.ScalarTransmission(name='fm8'),
    'AOoap3':poppy.QuadraticLens(f_lens=focal_lengths['AOoap3'], name='AOoap3'),
    'woofer-pp':poppy.ScalarTransmission(name='woofer-pp'),
    'AOoap4':poppy.QuadraticLens(f_lens=focal_lengths['AOoap4'], name='AOoap4'),
    'fm9':poppy.ScalarTransmission(name='fm9'),
    'ifp15':poppy.ScalarTransmission(name='ifp15'),
    'fm10':poppy.ScalarTransmission(name='fm10'),
    'AOoap5':poppy.QuadraticLens(f_lens=focal_lengths['AOoap5'], name='AOoap5'),
    'tweeter-pp':poppy.ScalarTransmission(name='tweeter-pp'),
    'knifeedge':poppy.ScalarTransmission(name='knife edge'), 
    'AOoap6':poppy.QuadraticLens(f_lens=focal_lengths['AOoap6'], name='AOoap6'),
    'fpsm':poppy.ScalarTransmission(name='fpsm'), 
    'pfm1':poppy.ScalarTransmission(name='pfm1'), 
    'pfm2':poppy.ScalarTransmission(name='pfm2'), 
    'AOoap7':poppy.QuadraticLens(f_lens=focal_lengths['AOoap7'], name='AOoap7'),
    'fm11':poppy.ScalarTransmission(name='fm11'), 
    'ifp69':poppy.ScalarTransmission(name='ifp69'), 
    'pupilSM':poppy.ScalarTransmission(name='pupilSM'), 
    'fm12':poppy.ScalarTransmission(name='fm12'), 
    'AOoap8':poppy.QuadraticLens(f_lens=focal_lengths['AOoap8'], name='AOoap8'),
    'fm13':poppy.ScalarTransmission(name='fm13'), 
    'fm14':poppy.ScalarTransmission(name='fm14'), 
    'ncpDM':poppy.ScalarTransmission(name='ncpDM'), 
    'AOoap9-1':poppy.QuadraticLens(f_lens=focal_lengths['AOoap9-1'], name='AOoap91'),
    'fm15':poppy.ScalarTransmission(name='fm15'), 
    'ifp34.5':poppy.ScalarTransmission(name='ifp34.5'), 
    'lyot':poppy.ScalarTransmission(name='Lyot Stop'), 
    'AOoap9-2':poppy.QuadraticLens(f_lens=focal_lengths['AOoap9-2'], name='AOoap92'),
    'fm16':poppy.ScalarTransmission(name='fm16'), 
    'AOoap9-3':poppy.QuadraticLens(f_lens=focal_lengths['AOoap9-3'], name='AOoap93'),
    'fm17':poppy.ScalarTransmission(name='fm17'), 
    'scicamfp34.5':poppy.ScalarTransmission(name='scicamfp34.5'), 
}

import numpy as np
import astropy.units as u
from astropy.io import fits
import time
import os
from pathlib import Path
import copy

from . math_module import xp, _scipy, ensure_np_array
from . imshows import *
from . import utils, dm
from . import optics
from . import opds

escpsf_dir = os.path.dirname(__file__)

import poppy
from poppy.poppy_core import PlaneType
pupil = PlaneType.pupil
inter = PlaneType.intermediate
image = PlaneType.image

class MODEL():

    def __init__(self, 
                 wavelength=None,
                 npix=800, 
                 oversample=4,
                 npsf=400, 
                 psf_pixelscale=4.63e-6*u.m/u.pix,
                 psf_pixelscale_lamD=None,
                 Imax_ref=None,
                 det_rotation=0,
                 source_offset=(0,0),
                 use_opds=False,
                 dm_ref=np.zeros((34,34)),
                 PUPILWFE=poppy.ScalarTransmission(name='Pupil WFE Place-holder'), 
                 DM=None, 
                 FPM=poppy.ScalarTransmission(name='FPM Place-holder'), 
                 LYOT=poppy.ScalarTransmission(name='Lyot Stop Place-holder'), 
            ):
        
        self.pupil_diam = 25.4*u.m
        self.wavelength_c = 633*u.nm
        self.wavelength = self.wavelength_c if wavelength is None else wavelength
        
        self.npix = int(npix)
        self.oversample = oversample
        self.N = int(self.npix*self.oversample)
        
        self.as_per_lamD = ((self.wavelength_c/self.pupil_diam)*u.radian).to(u.arcsec)
        # self.um_per_lamD
        self.npsf = npsf
        if psf_pixelscale_lamD is None: # overrides psf_pixelscale this way
            self.psf_pixelscale = psf_pixelscale
            # self.psf_pixelscale_lamD = self.psf_pixelscale.to_value(u.um/u.pix)/self.um_per_lamD
        else:
            self.psf_pixelscale_lamD = psf_pixelscale_lamD
            self.psf_pixelscale = self.psf_pixelscale_lamD/(1/4.9136)
        self.det_rotation = det_rotation
        self.DETECTOR = poppy.Detector(pixelscale=self.psf_pixelscale, fov_pixels=self.npsf, interp_order=3)
        
        self.Imax_ref = Imax_ref
        
        self.use_opds = use_opds

        self.source_offset = source_offset

        self.distances = optics.distances
        self.planes = optics.planes

        pupil_fpath = f'gmagaox/data/gmt_pupil_{self.npix:d}.fits'
        self.PUPIL = poppy.FITSOpticalElement(transmission=pupil_fpath, 
                                              planetype=poppy.poppy_core.PlaneType.pupil,
                                              name='GMT Pupil')
        self.pupil_mask = utils.pad_or_crop(self.PUPIL.amplitude, self.npix)>0

        tweeter_surf_fpath = 'gmagaox/data/tweeter_opd.fits'
        self.tweeter_surf = poppy.FITSOpticalElement(opd=tweeter_surf_fpath, opdunits='meters',
                                                     planetype=poppy.poppy_core.PlaneType.pupil,
                                                     name='Tweeter Surface')
        self.tweeter_surf.opd = 2*self.tweeter_surf.opd 
        
        ncpDM_surf_fpath = 'gmagaox/data/ncp_opd.fits'

        self.ncpDM_surf = poppy.FITSOpticalElement(opd=ncpDM_surf_fpath, opdunits='meters',
                                                   planetype=poppy.poppy_core.PlaneType.pupil,
                                                   name='NCP DM Surface')
        self.ncpDM_surf.opd = 2*self.ncpDM_surf.opd
        
        self.APODIZER = None

        self.ifp8157_1_correction = 0.0*u.mm
        self.fsm_correction = 0.0*u.mm
        self.ifp14_correction = 0.0*u.mm
        self.ADCpupil_correction = 0.0*u.mm
        self.ifp8157_2_correction = 0.0*u.mm
        self.ifp15_correction = 0.0*u.mm
        self.ifp69_correction = 0.0*u.mm
        self.ifp34p5_correction = 0.0*u.mm
        self.scicam_correction = 0.0*u.mm

        self.adc_correction = 0.0*u.mm
        self.woofer_correction = 0.0*u.mm
        self.tweeter_correction = 0.0*u.mm
        self.apodizer_correction = 0.0*u.mm
        self.lyot_correction = 0.0*u.mm
        self.fpsm_correction = 0.0*u.mm

        self.ifp8157_1_correction = 0.01059320*u.mm
        self.ifp14_correction = -0.00011966*u.mm + 0.00000613*u.mm
        self.ifp8157_2_correction = 0.00173228*u.mm + 0.00000039*u.mm
        self.ifp15_correction = 0.10906065*u.mm
        self.ifp15_2_correction = -0.00004823*u.mm
        self.ifp69_correction = -0.01889474*u.mm
        self.ifp34p5_correction = -7.30994261*u.mm + 0.00005939*u.mm
        self.scicam_correction = -0.00109005*u.mm

        self.adc_correction = -70*u.mm
        self.woofer_correction = 75*u.mm
        self.fpsm_correction = -1/2*u.mm
        self.apodizer_correction = -20*u.mm

        self.ideal_coro = False
        self.use_lyot_opd = False
        self.LYOT_WFE = poppy.ScalarTransmission(name='Lyot to Tweeter OPD')
        self.end_at_lyot = False

        self.use_dm_surfaces = False

        self.LYOT = None

        # self.init_dms()
        # self.dm_ref = dm_ref
        # self.set_dm(self.dm_ref)
        
    # useful for parallelization with ray actors
    def getattr(self, attr):
        return getattr(self, attr)
    
    def setattr(self, attr, val):
        setattr(self, attr, val)
    
    # def init_dm(self):
    #     inf_fun, sampling = dm.make_gaussian_inf_fun(act_spacing=300e-6*u.m, sampling=10, coupling=0.15, )
    #     self.DM = dm.DeformableMirror(inf_fun=inf_fun, inf_sampling=sampling,)

    #     self.Nact = self.DM.Nact
    #     self.Nacts = self.DM.Nacts
    #     self.act_spacing = self.DM.act_spacing
    #     self.dm_active_diam = self.DM.active_diam
    #     self.dm_full_diam = self.DM.pupil_diam
        
    #     self.full_stroke = self.DM.full_stroke
    #     self.dm_mask = self.DM.dm_mask
    
    def init_woofer():

        return
    
    def init_tweeter():
        return
    
    def init_ncpDM():
        return

    def init_fosys(self):
        self.psf_pixelscale_lamD = 1/self.oversample

        if self.LYOT is None:
            self.LYOT = self.planes['lyot-pp']
        
        self.fosys = poppy.FresnelOpticalSystem(name='ESC', pupil_diameter=self.pupil_diam, 
                                                npix=self.npix, beam_ratio=1/self.oversample, verbose=True)
        self.fosys.add_optic(self.PUPIL)
        self.fosys.add_optic(self.planes['m1'])
        # if self.use_opds: self.fosys.add_optic(opds.wfe_psds['m1'])
        self.fosys.add_optic(self.planes['m2'], distance=self.distances['m1_m2'])
        # if self.use_opds: self.fosys.add_optic(opds.wfe_psds['m2'])
        self.fosys.add_optic(self.planes['m3'], distance=self.distances['m2_m3'])
        # if self.use_opds: self.fosys.add_optic(opds.wfe_psds['m3'])
        self.fosys.add_optic(self.planes['ifp8.157'], distance=self.distances['m3_ifp8.157'] + self.ifp8157_1_correction)
        self.fosys.add_optic(self.planes['Roap1'], distance=self.distances['ifp8.157_Roap1'])# - self.ifp8157_1_correction)
        if self.use_opds: self.fosys.add_optic(opds.wfe_psds['Roap1'])
        self.fosys.add_optic(self.planes['fm1'], distance=self.distances['Roap1_fm1'])
        if self.use_opds: self.fosys.add_optic(opds.wfe_psds['fm1'])
        self.fosys.add_optic(self.planes['fsm-pp'], distance=self.distances['fm1_fsm-pp'] + self.fsm_correction)
        if self.use_opds: self.fosys.add_optic(opds.wfe_psds['Roap2'])
        self.fosys.add_optic(self.planes['fm2'], distance=self.distances['fsm-pp_fm2'])# - self.fsm_correction)
        if self.use_opds: self.fosys.add_optic(opds.wfe_psds['fm2'])
        self.fosys.add_optic(self.planes['Roap2'], distance=self.distances['fm2_Roap2'])
        if self.use_opds: self.fosys.add_optic(opds.wfe_psds['Roap2'])
        self.fosys.add_optic(self.planes['fm3'], distance=self.distances['Roap2_fm3'])
        if self.use_opds: self.fosys.add_optic(opds.wfe_psds['fm3'])
        self.fosys.add_optic(self.planes['ifp14'], distance=self.distances['fm3_ifp14'] + self.ifp14_correction)
        self.fosys.add_optic(self.planes['km1'], distance=self.distances['ifp14_km1']) #- self.ifp14_correction)
        if self.use_opds: self.fosys.add_optic(opds.wfe_psds['km1'])
        self.fosys.add_optic(self.planes['km2'], distance=self.distances['km1_km2'])
        if self.use_opds: self.fosys.add_optic(opds.wfe_psds['km2'])
        self.fosys.add_optic(self.planes['km3'], distance=self.distances['km2_km3'])
        if self.use_opds: self.fosys.add_optic(opds.wfe_psds['km3'])
        self.fosys.add_optic(self.planes['fm4'], distance=self.distances['km3_fm4'])
        if self.use_opds: self.fosys.add_optic(opds.wfe_psds['fm4'])
        self.fosys.add_optic(self.planes['AOoap1'], distance=self.distances['fm4_AOoap1'])
        if self.use_opds: self.fosys.add_optic(opds.wfe_psds['AOoap1'])
        self.fosys.add_optic(self.planes['fm5'], distance=self.distances['AOoap1_fm5'])
        if self.use_opds: self.fosys.add_optic(opds.wfe_psds['fm5'])
        self.fosys.add_optic(self.planes['ADC-pp'], distance=self.distances['fm5_ADC-pp'] + self.adc_correction)
        self.fosys.add_optic(self.planes['fm6'], distance=self.distances['ADC-pp_fm6'])
        if self.use_opds: self.fosys.add_optic(opds.wfe_psds['fm6'])
        self.fosys.add_optic(self.planes['fm7'], distance=self.distances['fm6_fm7'])
        if self.use_opds: self.fosys.add_optic(opds.wfe_psds['fm7'])
        self.fosys.add_optic(self.planes['AOoap2'], distance=self.distances['fm7_AOoap2'])
        if self.use_opds: self.fosys.add_optic(opds.wfe_psds['AOoap2'])
        self.fosys.add_optic(self.planes['ifp8.157'], distance=self.distances['AOoap2_ifp8.157'] + self.ifp8157_2_correction)
        self.fosys.add_optic(self.planes['fm8'], distance=self.distances['ifp8.157_fm8'])#  - self.ifp8157_2_correction)
        if self.use_opds: self.fosys.add_optic(opds.wfe_psds['fm8'])
        self.fosys.add_optic(self.planes['AOoap3'], distance=self.distances['fm8_AOoap3'])
        if self.use_opds: self.fosys.add_optic(opds.wfe_psds['AOoap3'])
        self.fosys.add_optic(self.planes['woofer-pp'], distance=self.distances['AOoap3_woofer-pp'] + self.woofer_correction)
        # if self.use_opds: self.fosys.add_optic(opds.wfe_psds['woofer'])
        self.fosys.add_optic(self.planes['AOoap4'], distance=self.distances['woofer-pp_AOoap4'])
        if self.use_opds: self.fosys.add_optic(opds.wfe_psds['AOoap4'])
        self.fosys.add_optic(self.planes['fm9'], distance=self.distances['AOoap4_fm9'])
        if self.use_opds: self.fosys.add_optic(opds.wfe_psds['fm9'])
        self.fosys.add_optic(self.planes['ifp15'], distance=self.distances['fm9_ifp15'] + self.ifp15_correction)
        self.fosys.add_optic(self.planes['fm10'], distance=self.distances['ifp15_fm10']) # - self.ifp15_correction)
        if self.use_opds: self.fosys.add_optic(opds.wfe_psds['fm10'])
        self.fosys.add_optic(self.planes['AOoap5'], distance=self.distances['fm10_AOoap5'])
        if self.use_opds: self.fosys.add_optic(opds.wfe_psds['AOoap5'])
        self.fosys.add_optic(self.planes['tweeter-pp'], distance=self.distances['AOoap5_tweeter-pp']+ self.tweeter_correction)
        if self.use_dm_surfaces:
            self.fosys.add_optic(self.tweeter_surf)
        if self.use_lyot_opd:
            self.fosys.add_optic(self.LYOT_WFE)
        self.fosys.add_optic(self.planes['AOoap5'], distance=self.distances['tweeter-pp_AOoap5'])
        if self.use_opds: self.fosys.add_optic(opds.wfe_psds['AOoap5'])
        self.fosys.add_optic(self.planes['fm10'], distance=self.distances['AOoap5_fm10'])
        if self.use_opds: self.fosys.add_optic(opds.wfe_psds['fm10'])
        self.fosys.add_optic(self.planes['ifp15'], distance=self.distances['fm10_ifp15'] + self.ifp15_2_correction)
        self.fosys.add_optic(self.planes['knifeedge'], distance=self.distances['ifp15_knifeedge'])
        self.fosys.add_optic(self.planes['AOoap6'], distance=self.distances['knifeedge_AOoap6'])
        if self.use_opds: self.fosys.add_optic(opds.wfe_psds['AOoap6'])
        self.fosys.add_optic(self.planes['fpsm-pp'], distance=self.distances['AOoap6_fpsm-pp'] + self.fpsm_correction)
        if self.use_opds: self.fosys.add_optic(opds.wfe_psds['fpsm'])
        self.fosys.add_optic(self.planes['pfm1'], distance=self.distances['fpsm-pp_pfm1'])
        if self.use_opds: self.fosys.add_optic(opds.wfe_psds['pfm1'])
        self.fosys.add_optic(self.planes['pfm2'], distance=self.distances['pfm1_pfm2'])
        if self.use_opds: self.fosys.add_optic(opds.wfe_psds['pfm2'])
        self.fosys.add_optic(self.planes['AOoap7'], distance=self.distances['pfm2_AOoap7'])
        if self.use_opds: self.fosys.add_optic(opds.wfe_psds['AOoap7'])
        self.fosys.add_optic(self.planes['fm11'], distance=self.distances['AOoap7_fm11'])
        if self.use_opds: self.fosys.add_optic(opds.wfe_psds['fm11'])
        self.fosys.add_optic(self.planes['ifp69'], distance=self.distances['fm11_ifp69'] + self.ifp69_correction)
        self.fosys.add_optic(self.planes['pupilSM'], distance=self.distances['ifp69_pupilSM'])
        # if self.use_opds: self.fosys.add_optic(opds.wfe_psds['pupilSM'])
        self.fosys.add_optic(self.planes['fm12'], distance=self.distances['pupilSM_fm12'])
        if self.use_opds: self.fosys.add_optic(opds.wfe_psds['fm12'])
        self.fosys.add_optic(self.planes['AOoap8'], distance=self.distances['fm12_AOoap8'])
        if self.use_opds: self.fosys.add_optic(opds.wfe_psds['AOoap8'])
        self.fosys.add_optic(self.planes['fm13'], distance=self.distances['AOoap8_fm13'])
        if self.use_opds: self.fosys.add_optic(opds.wfe_psds['fm13'])
        self.fosys.add_optic(self.planes['fm14'], distance=self.distances['fm13_fm14'])
        if self.use_opds: self.fosys.add_optic(opds.wfe_psds['fm14'])
        self.fosys.add_optic(self.planes['ncpDM'], distance=self.distances['fm14_ncpDM'])
        if self.use_dm_surfaces:
            self.fosys.add_optic(self.ncpDM_surf)
        if self.APODIZER is None: 
            self.fosys.add_optic(self.planes['apodizer'], distance=self.distances['ncpDM_apodizer']+self.apodizer_correction)
        else:
            self.fosys.add_optic(self.APODIZER, distance=self.distances['ncpDM_apodizer']+self.apodizer_correction)
        self.fosys.add_optic(self.planes['AOoap9-1'], distance=self.distances['apodizer_AOoap9-1'])
        if self.use_opds: self.fosys.add_optic(opds.wfe_psds['AOoap9-1'])
        self.fosys.add_optic(self.planes['fm15'], distance=self.distances['AOoap9-1_fm15'])
        if self.use_opds: self.fosys.add_optic(opds.wfe_psds['fm15'])
        self.fosys.add_optic(self.planes['ifp34.5'], distance=self.distances['fm15_ifp34.5'] + self.ifp34p5_correction)
        self.fosys.add_optic(self.planes['AOoap9-2'], distance=self.distances['ifp34.5_AOoap9-2'])
        if self.use_opds: self.fosys.add_optic(opds.wfe_psds['AOoap9-2'])
        self.fosys.add_optic(self.LYOT, distance=self.distances['AOoap9-2_lyot-pp'] + self.lyot_correction)
        if self.end_at_lyot: 
            return
        self.fosys.add_optic(self.planes['fm16'], distance=self.distances['lyot-pp_fm16'])
        if self.use_opds: self.fosys.add_optic(opds.wfe_psds['fm16'])
        self.fosys.add_optic(self.planes['AOoap9-3'], distance=self.distances['fm16_AOoap9-3'])
        if self.use_opds: self.fosys.add_optic(opds.wfe_psds['AOoap9-3'])
        self.fosys.add_optic(self.planes['fm17'], distance=self.distances['AOoap9-3_fm17'])
        if self.use_opds: self.fosys.add_optic(opds.wfe_psds['fm17'])
        self.fosys.add_optic(self.planes['scicamfp34.5'], distance=self.distances['fm17_scicamfp34.5'] + self.scicam_correction)

        return

# scc_lyot_stop = poppy.ArrayOpticalElement(transmission=lyot_ap+scc_ap, pixelscale=wf.pixelscale, name='Lyot/SCC Mask')

# for i,wf in enumerate(wfs):
#     if 'lyot' in wf.location.lower() or 'scc' in wf.location.lower():
#         lyot_ind = i
#         break

    def calc_wfs(self, quiet=False):
        start = time.time()
        if not quiet: print(f'Propagating wavelength {self.wavelength.to(u.nm):.3f}.')
        inwave = poppy.FresnelWavefront(beam_radius=self.pupil_diam/2, wavelength=self.wavelength, npix=self.npix, oversample=self.oversample)
        if self.source_offset[0]>0 or self.source_offset[1]>0:
            inwave.tilt(Xangle=self.source_offset[0]*self.as_per_lamD, Yangle=self.source_offset[1]*self.as_per_lamD)
        self.init_fosys()
        _, wfs = self.fosys.calc_psf(inwave=inwave, normalize='none', return_intermediates=True)
        if not quiet: print(f'PSF calculated in {(time.time()-start):.3f}s')
        
        return wfs
    
    def calc_wf(self, pixelscale=False): 
        self.init_fosys()
        inwave = poppy.FresnelWavefront(beam_radius=self.pupil_diam/2, wavelength=self.wavelength, npix=self.npix, oversample=self.oversample)
        if self.source_offset[0]>0 or self.source_offset[1]>0:
            inwave.tilt(Xangle=self.source_offset[0]*self.as_per_lamD, Yangle=self.source_offset[1]*self.as_per_lamD)
        if self.ideal_coro:
            inwave.wavefront -= utils.pad_or_crop(self.PUPIL.amplitude, self.N)
        _, wfs = self.fosys.calc_psf(inwave=inwave, normalize='none', return_final=True, return_intermediates=False)
        wfarr = wfs[0].wavefront

        if abs(self.det_rotation)>0:
            wfarr = utils.rotate_wf(wfarr, self.det_rotation)

        if self.Imax_ref is not None:
            wfarr = wfarr/np.sqrt(self.Imax_ref)

        if pixelscale:
            return wfarr, wfs[0].pixelscale
        return wfarr
    
    def snap(self):
        im = xp.abs(self.calc_wf())**2
        return im

    



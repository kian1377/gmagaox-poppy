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

escpsf_dir = os.path.dirname(__file__)

import poppy
from poppy.poppy_core import PlaneType
pupil = PlaneType.pupil
inter = PlaneType.intermediate
image = PlaneType.image


class ESC():

    def __init__(self, 
                 wavelength=None,
                 npix=512, 
                 oversample=4,
                 npsf=400, 
                 psf_pixelscale=4.63e-6*u.m/u.pix,
                 psf_pixelscale_lamD=None,
                 fieldstop_diam=2*0.0002275*u.m, # roughly 10lam/D based on 4mm final pupil
                 Imax_ref=None,
                 det_rotation=0,
                 source_offset=(0,0),
                 source_flux=None,
                 use_aps=False,
                 use_synthetic_opds=False,
                 use_measured_opds=False,
                 dm_ref=np.zeros((34,34)),
                 PUPILWFE=poppy.ScalarTransmission(name='Pupil WFE Place-holder'), 
                 DM=None, 
                 FPM=poppy.ScalarTransmission(name='FPM Place-holder'), 
                 LYOT=poppy.ScalarTransmission(name='Lyot Stop Place-holder'), 
            ):
        
        self.pupil_diam = 2.4*u.m
        self.pupil_mask_diam = 2*7.000*u.mm
        self.wavelength_c = 650*u.nm
        self.wavelength = self.wavelength_c if wavelength is None else wavelength
        
        self.npix = int(npix)
        self.oversample = oversample
        self.N = int(self.npix*self.oversample)
        
        self.as_per_lamD = ((self.wavelength_c/self.pupil_mask_diam)*u.radian).to(u.arcsec)

        self.npsf = npsf
        if psf_pixelscale_lamD is None: # overrides psf_pixelscale this way
            self.psf_pixelscale = psf_pixelscale
            self.psf_pixelscale_lamD = (1/4.9136) * self.psf_pixelscale.to_value(u.m/u.pix)/4.63e-6
        else:
            self.psf_pixelscale_lamD = psf_pixelscale_lamD
            self.psf_pixelscale = 4.63e-6*u.m/u.pix / self.psf_pixelscale_lamD/(1/4.9136)
        self.det_rotation = det_rotation
        self.DETECTOR = poppy.Detector(pixelscale=self.psf_pixelscale, fov_pixels=self.npsf, interp_order=3)
        
        self.Imax_ref = Imax_ref
        
        self.use_opds = use_opds
        self.init_opds()
        
        self.init_dm()
        self.dm_ref = dm_ref
        self.set_dm(self.dm_ref)

        self.PUPILMASK = poppy.CircularAperture(radius=self.pupil_mask_diam/2)
        self.PUPILWFE = PUPILWFE
        self.FPM = FPM
        self.LYOT = LYOT
        
    # useful for parallelization with ray actors
    def getattr(self, attr):
        return getattr(self, attr)
    
    def setattr(self, attr, val):
        setattr(self, attr, val)
    
    def init_dm(self):
        print('Initializing default 34x34 DM with 0.15 actuator coupling.')
        inf_fun, sampling = dm.make_gaussian_inf_fun(act_spacing=300e-6*u.m, sampling=10, coupling=0.15, )
        self.DM = dm.DeformableMirror(inf_fun=inf_fun, inf_sampling=sampling,)

        self.Nact = self.DM.Nact
        self.Nacts = self.DM.Nacts
        self.act_spacing = self.DM.act_spacing
        self.dm_active_diam = self.DM.active_diam
        self.dm_full_diam = self.DM.pupil_diam
        
        self.full_stroke = self.DM.full_stroke
        self.dm_mask = self.DM.dm_mask
        
    def reset_dm(self):
        self.set_dm(self.dm_ref)
    
    def zero_dm(self):
        self.set_dm(np.zeros((34,34)))
        
    def set_dm(self, command):
        if command.shape[0]==self.Nacts:
            dm_command = self.DM.map_actuators_to_command(xp.asarray(command))
        else: 
            dm_command = xp.asarray(command)
        self.DM.command = dm_command
        
    def add_dm(self, command):
        if command.shape[0]==self.Nacts:
            dm_command = self.DM.map_actuators_to_command(xp.asarray(command))
        else: 
            dm_command = xp.asarray(command)
        self.DM.command += dm_command
        
    def get_dm(self, only_actuators=False):
        if only_actuators:
            return self.DM.actuators
        else:
            return self.DM.command
    
    def init_inwave(self):
        inwave = poppy.FresnelWavefront(beam_radius=self.pupil_mask_diam/2, wavelength=self.wavelength,
                                        npix=self.npix, oversample=self.oversample)
        
        if self.source_offset[0]>0 or self.source_offset[1]>0:
            inwave.tilt(Xangle=self.source_offset[0]*self.as_per_lamD, Yangle=self.source_offset[1]*self.as_per_lamD)
        
        # imshow2(inwave.amplitude, inwave.phase)

        if self.source_flux is not None:
            # scale input wavefront amplitude by the photon flux of source
            flux_per_pixel = self.source_flux * (inwave.pixelscale*u.pix)**2
            inwave.wavefront *= np.sqrt((flux_per_pixel).value)
            self.normalize = 'none'
        
        return inwave

    def init_opds(self):
        if self.use_measured_opds:
            self.use_opds = True

            '''
            FIXME: Need to add measurements when the optics are acquired
            '''

            raise ValueError('Measured OPDs not available for this model yet.')
        elif self.use_synthetic_opds:
            self.use_opds = True
            seeds = np.arange(0,15)
            
            self.psd_index = 2.8

            rms_vals = (3e-9*np.random.rand(15) + 15e-9)*u.m
            self.oap1_rms = rms_vals[0]
            self.oap2_rms = rms_vals[1]
            self.oap3_rms = rms_vals[2]
            self.oap4_rms = rms_vals[3]
            self.oap5_rms = rms_vals[4]
            self.oap6_rms = rms_vals[5]
            self.oap7_rms = rms_vals[6]
            self.oap8_rms = rms_vals[7]
            self.oap9_rms = rms_vals[8]

            self.lp1_rms = rms_vals[9]
            self.qwp1_rms = rms_vals[10]
            self.qwp2_rms = rms_vals[11]
            self.lp2_rms = rms_vals[12]

            self.flat_rms = rms_vals[13]
            self.fsm_rms = rms_vals[14]

            self.oap1_diam = 25.4*u.mm
            self.oap2_diam = 25.4*u.mm
            self.oap3_diam = 25.4*u.mm
            self.oap4_diam = 25.4*u.mm
            self.oap5_diam = 25.4*u.mm
            self.oap6_diam = 25.4*u.mm
            self.oap7_diam = 25.4*u.mm
            self.oap8_diam = 25.4*u.mm
            self.oap9_diam = 25.4*u.mm

            self.oap1_opd = poppy.StatisticalPSDWFE(name='OAP1 OPD', index=self.psd_index, wfe=self.oap1_rms, radius=self.oap1_diam/2, seed=seeds[0])
            self.oap2_opd = poppy.StatisticalPSDWFE(name='OAP2 OPD', index=self.psd_index, wfe=self.oap2_rms, radius=self.oap2_diam/2, seed=seeds[1])
            self.oap3_opd = poppy.StatisticalPSDWFE(name='OAP3 OPD', index=self.psd_index, wfe=self.oap3_rms, radius=self.oap3_diam/2, seed=seeds[2])
            self.oap4_opd = poppy.StatisticalPSDWFE(name='OAP4 OPD', index=self.psd_index, wfe=self.oap4_rms, radius=self.oap4_diam/2, seed=seeds[3])
            self.oap5_opd = poppy.StatisticalPSDWFE(name='OAP5 OPD', index=self.psd_index, wfe=self.oap5_rms, radius=self.oap5_diam/2, seed=seeds[4])
            self.oap6_opd = poppy.StatisticalPSDWFE(name='OAP6 OPD', index=self.psd_index, wfe=self.oap6_rms, radius=self.oap6_diam/2, seed=seeds[5])
            self.oap7_opd = poppy.StatisticalPSDWFE(name='OAP7 OPD', index=self.psd_index, wfe=self.oap7_rms, radius=self.oap7_diam/2, seed=seeds[6])
            self.oap8_opd = poppy.StatisticalPSDWFE(name='OAP8 OPD', index=self.psd_index, wfe=self.oap8_rms, radius=self.oap8_diam/2, seed=seeds[7])
            self.oap9_opd = poppy.StatisticalPSDWFE(name='OAP9 OPD', index=self.psd_index, wfe=self.oap9_rms, radius=self.oap9_diam/2, seed=seeds[8])
            
            self.lp1_diam = 20*u.mm
            self.qwp1_diam = 20*u.mm
            self.qwp2_diam = 20*u.mm
            self.lp2_diam = 2*6.35*u.mm

            self.lp1_opd = poppy.StatisticalPSDWFE(name='LP1 OPD', index=self.psd_index, wfe=self.lp1_rms, radius=self.lp1_diam/2, seed=seeds[9])
            self.qwp1_opd = poppy.StatisticalPSDWFE(name='QWP1 OPD', index=self.psd_index, wfe=self.qwp1_rms, radius=self.qwp1_diam/2, seed=seeds[10])
            self.qwp2_opd = poppy.StatisticalPSDWFE(name='QWP2 OPD', index=self.psd_index, wfe=self.qwp2_rms, radius=self.qwp2_diam/2, seed=seeds[11])
            self.lp2_opd = poppy.StatisticalPSDWFE(name='LP2 OPD', index=self.psd_index, wfe=self.lp2_rms, radius=self.lp2_diam/2, seed=seeds[12])

            self.flat_diam = 25.4*u.mm
            self.fsm_diam = 25.4*u.mm

            self.fold_flat_1_opd = poppy.StatisticalPSDWFE(name='Flat OPD', index=self.psd_index, wfe=self.flat_rms, radius=self.flat_diam/2, seed=seeds[13])
            self.fsm_opd = poppy.StatisticalPSDWFE(name='FSM OPD', index=self.psd_index, wfe=self.fsm_rms, radius=self.fsm_diam/2, seed=seeds[14])
            
            print('Model using synthetic OPD data')
        else:
            self.use_opds = False
            print('No OPD data implemented into model.')

    def init_fosys(self):
        


        self.fosys = poppy.FresnelOpticalSystem(name='ESC', pupil_diameter=self.pupil_diam, 
                                                npix=self.npix, beam_ratio=1/self.oversample, verbose=True)

# scc_lyot_stop = poppy.ArrayOpticalElement(transmission=lyot_ap+scc_ap, pixelscale=wf.pixelscale, name='Lyot/SCC Mask')

# for i,wf in enumerate(wfs):
#     if 'lyot' in wf.location.lower() or 'scc' in wf.location.lower():
#         lyot_ind = i
#         break


    def calc_wfs(self, quiet=False):
        start = time.time()
        if not quiet: print(f'Propagating wavelength {self.wavelength.to(u.nm):.3f}.')
        fosys = self.init_fosys()
        inwave = self.init_inwave()
        _, wfs = fosys.calc_psf(inwave=inwave, normalize=self.norm, return_intermediates=True)
        if not quiet: print(f'PSF calculated in {(time.time()-start):.3f}s')
        
        return wfs
    
    def calc_wf(self, plot=False, vmin=None, grid=False): 
        inwave = self.init_inwave()
        _, wfs = self.fosys.calc_psf(inwave=inwave, normalize=self.norm, return_final=True, return_intermediates=False)
        wfarr = wfs[0].wavefront

        if abs(self.det_rotation)>0:
            wfarr = utils.rotate_wf(wfarr, self.det_rotation)

        if self.Imax_ref is not None:
            wfarr = wfarr/np.sqrt(self.Imax_ref)

        return wfarr
    
    def snap(self, plot=False, vmin=None, grid=False):
        im = xp.abs(self.fosys.calc_psf())**2
        return im



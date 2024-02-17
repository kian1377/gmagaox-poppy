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
        self.wavelength_c = 650*u.nm
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
        if self.use_opds: self.init_opds()

        self.source_offset = source_offset

        self.distances = optics.distances
        self.planes = optics.planes

        self.PUPIL = poppy.FITSOpticalElement(transmission='gmagaox/data/gmt_pupil_800.fits', 
                                              planetype=poppy.poppy_core.PlaneType.pupil,
                                              name='GMT Pupil')

        # self.init_dm()
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
        inwave = poppy.FresnelWavefront(beam_radius=self.pupil_diam/2, wavelength=self.wavelength,
                                        npix=self.npix, oversample=self.oversample)
        
        if self.source_offset[0]>0 or self.source_offset[1]>0:
            inwave.tilt(Xangle=self.source_offset[0]*self.as_per_lamD, Yangle=self.source_offset[1]*self.as_per_lamD)
        
        return inwave

    def init_fosys(self):

        self.fosys = poppy.FresnelOpticalSystem(name='ESC', pupil_diameter=self.pupil_diam, 
                                                npix=self.npix, beam_ratio=1/self.oversample, verbose=True)
        self.fosys.add_optic(self.PUPIL)
        self.fosys.add_optic(self.planes['m1'])
        self.fosys.add_optic(self.planes['m2'], distance=self.distances['m1_m2'])
        self.fosys.add_optic(self.planes['m3'], distance=self.distances['m2_m3'])
        self.fosys.add_optic(self.planes['ifp8157'], distance=self.distances['m3_ifp8157'])
        self.fosys.add_optic(self.planes['Roap1'], distance=self.distances['ifp8157_Roap1'])
        self.fosys.add_optic(self.planes['fm1'], distance=self.distances['Roap1_fm1'])
        self.fosys.add_optic(self.planes['fsm'], distance=self.distances['fm1_fsm'])
        self.fosys.add_optic(self.planes['fm2'], distance=self.distances['fsm_fm2'])
        self.fosys.add_optic(self.planes['Roap2'], distance=self.distances['fm2_Roap2'])
        self.fosys.add_optic(self.planes['fm3'], distance=self.distances['Roap2_fm3'])
        self.fosys.add_optic(self.planes['ifp14'], distance=self.distances['fm3_ifp14'])
        # self.fosys.add_optic(self.planes['km1'], distance=self.distances['ifp14_km1'])
        # self.fosys.add_optic(self.planes['km2'], distance=self.distances['km1_km2'])
        # self.fosys.add_optic(self.planes['km3'], distance=self.distances['km2_km3'])
        # self.fosys.add_optic(self.planes['fm4'], distance=self.distances['km3_fm4'])
        # self.fosys.add_optic(self.planes['AOoap1'], distance=self.distances['fm4_AOoap1'])
        # self.fosys.add_optic(self.planes['fm5'], distance=self.distances['AOoap1_fm5'])
        # self.fosys.add_optic(self.planes['fm6'], distance=self.distances['fm5_fm6'])
        # self.fosys.add_optic(self.planes['fm7'], distance=self.distances['fm6_fm7'])
        # self.fosys.add_optic(self.planes['AOoap2'], distance=self.distances['fm7_AOoap2'])
        # self.fosys.add_optic(self.planes['ifp8157'], distance=self.distances['AOoap2_ifp8157'])
        # self.fosys.add_optic(self.planes['fm8'], distance=self.distances['ifp8157_fm8'])
        # self.fosys.add_optic(self.planes['AOoap3'], distance=self.distances['fm8_AOoap3'])
        # self.fosys.add_optic(self.planes['woofer'], distance=self.distances['AOoap3_woofer'])
        # self.fosys.add_optic(self.planes['AOoap4'], distance=self.distances['woofer_AOoap4'])
        # self.fosys.add_optic(self.planes['fm9'], distance=self.distances['AOoap4_fm9'])
        # self.fosys.add_optic(self.planes['ifp15'], distance=self.distances['fm9_ifp15'])
        # self.fosys.add_optic(self.planes['fm10'], distance=self.distances['ifp15_fm10'])
        # self.fosys.add_optic(self.planes['AOoap5'], distance=self.distances['fm10_AOoap5'])
        # self.fosys.add_optic(self.planes['tweeter'], distance=self.distances['AOoap5_tweeter'])

        return

# scc_lyot_stop = poppy.ArrayOpticalElement(transmission=lyot_ap+scc_ap, pixelscale=wf.pixelscale, name='Lyot/SCC Mask')

# for i,wf in enumerate(wfs):
#     if 'lyot' in wf.location.lower() or 'scc' in wf.location.lower():
#         lyot_ind = i
#         break

    def calc_wfs(self, quiet=False):
        start = time.time()
        if not quiet: print(f'Propagating wavelength {self.wavelength.to(u.nm):.3f}.')
        self.init_fosys()
        inwave = self.init_inwave()
        _, wfs = self.fosys.calc_psf(inwave=inwave, return_intermediates=True)
        if not quiet: print(f'PSF calculated in {(time.time()-start):.3f}s')
        
        return wfs
    
    def calc_wf(self): 
        inwave = self.init_inwave()
        _, wfs = self.fosys.calc_psf(inwave=inwave, normalize=self.norm, return_final=True, return_intermediates=False)
        wfarr = wfs[0].wavefront

        if abs(self.det_rotation)>0:
            wfarr = utils.rotate_wf(wfarr, self.det_rotation)

        if self.Imax_ref is not None:
            wfarr = wfarr/np.sqrt(self.Imax_ref)

        return wfarr
    
    def snap(self):
        im = xp.abs(self.calc_wf())**2
        return im



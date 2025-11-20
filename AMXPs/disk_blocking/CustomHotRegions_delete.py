import xpsi

class CustomHotRegions(xpsi.HotRegions):
    def __init__(self, hotregions):
        self.objects = hotregions
        # self._disk_blocking = self.objects[0]._disk_blocking
        super(CustomHotRegions, self).__init__(hotregions)


    def integrate(self, st, energies, threads,
                  hot_atmosphere, elsewhere_atmosphere, atm_ext_else, R_in = 1e6):
        """ Integrate over the photospheric radiation field.

        Calls the CellMesh integrator, with or without exploitation of
        azimuthal invariance of the radiation field.

        :param st: Instance of :class:`~.Spacetime.Spacetime`.

        :param energies:
            A one-dimensional :class:`numpy.ndarray` of energies in keV.

        :param int threads:
            Number of ``OpenMP`` threads for pulse integration.

        """
        
 
        
        
        if not isinstance(energies, tuple):
            energies = [energies] * len(self)

        signals = []
        for obj, E in zip(self._objects, energies):
            # if self._disk_blocking:
            signals.append(obj.integrate(st, 
                                         E,
                                         threads,
                                         hot_atmosphere,
                                         elsewhere_atmosphere,
                                         atm_ext_else,
                                         R_in))
            # elif not self._disk_blocking:
            #     signals.append(obj.integrate(st, 
            #                                  E,
            #                                  threads,
            #                                  hot_atmosphere,
            #                                  elsewhere_atmosphere,
            #                                  atm_ext_else))
        return tuple(signals)

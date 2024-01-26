import numpy as np

from pyparrm import PARRM


class PARRMArtifactRejection:
    def __init__(
        self,
        data: np.array,
        sampling_freq: float,
        artefact_freq: float,
        verbose=False,
    ):
        """Initialization of the Period-based Artefact Reconstruction and Removal Method
        For further motivation see the original paper:
        PARRM, Dastin-val Rijn et al 2021, Cell Report Methods,
        https://doi.org/10.1016/j.crmeth.2021.100010

        And the documentation of the original PyPARRM package: https://pyparrm.readthedocs.io/en/stable/

        In brief, the stimulation artifact frequency needs to be known, and the method requires
        to compute the filter which is then applied to the whole data duration.

        Parameters
        ----------
        data : np.array
            _description_
        sampling_freq : float
            _description_
        artefact_freq : float
            _description_
        verbose : bool, optional
            _description_, by default False
        """
        self.data = data
        self.sampling_freq = sampling_freq
        self.artefact_freq = artefact_freq
        self.verbose = verbose

        self.parrm = PARRM(
            data=data,
            sampling_freq=sampling_freq,
            artefact_freq=artefact_freq,
            verbose=False,
        )

    def filter_data(self):
        self.parrm.find_period()
        self.parrm.create_filter(
            filter_direction="both",
        )
        filtered_data = self.parrm.filter_data()

        return filtered_data

class StandardDeviationArtifactAnnot:

    def __init__(self, data: np.array, std_deviations_reject: float, ) -> None:
        
        self.data = data
        self.std = np.std(self.data, axis=1)
        np.where(self.std > std_deviations_reject, 1, 0)
        
    def apply_annotation(self):
        


    
    def apply_annotation(self):
        


        

# from ptychozoon.enhance import VSPIFluorescenceEnhancingAlgorithm
import h5py
import scipy.io as sio
from ptychozoon.enhance import ElementMap, FluorescenceDataset, Product

# def read_mda_file():
#     pass

if __name__ == "__main__":
    # get relevant xrf data
    file_path = "/net/micdata/data1/bnp/2023-1/Isaure_fly0085_for_SH/img.dat/bnp_fly0085.mda.h5"
    with h5py.File(file_path, "r") as F:
        channel_names = F["/MAPS/channel_names"][()]
        element_maps = []
        for channel in channel_names:
            element_maps += [
                ElementMap(channel, F["/MAPS/XRF_Analyzed/NNLS/Counts_Per_Sec"][()])
            ]
    flourescence_in = FluorescenceDataset(element_maps)

    # get ptycho data
    file_path = "/net/micdata/data1/bnp/2023-1/Isaure_fly0085_for_SH/results/ML_recon/fly085/roi0_Ndp256/MLs_L1_p10_g1000_Ndp64_pc50_noModelCon_bg0.1_vp5_vi_mm/Niter1000.mat"
    d = sio.loadmat(file_path)
    print(d.keys())
    # Product(probe_positions=)

            

    # file_path = Path("/net/micdata/data1/bnp/2023-1/Isaure_fly0085_for_SH/mda/bnp_fly0085.mda")
    # mda_file = MDAFile.read(file_path)
    # flourescence_result = VSPIFluorescenceEnhancingAlgorithm().enhance(flourescence_in, ptycho_in)


# Maye also want to use coutns data from one of these. Not sure what is correct.
# ROI = "/MAPS/XRF_Analyzed/ROI/Counts_Per_Sec"
# NNLS = "/MAPS/XRF_Analyzed/NNLS/Counts_Per_Sec"
# MATRIX = " /MAPS/XRF_Analyzed/Fitted/Counts_Per_Sec"
# LEGACY_ROI = "/MAPS/XRF_roi"
# LEGACY_MATRIX = "/MAPS/XRF_roi"

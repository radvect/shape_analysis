import os 
import numpy as np 
import matplotlib.pyplot as plt
from multiprocessing import Pool
import cv2
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import gaussian_filter
import glob
import ot
from scipy.interpolate import CubicSpline
SIGMA = 1
from collections import defaultdict
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.draw import polygon, disk
from scipy.ndimage import distance_transform_edt
from skimage.measure import label
from scipy.ndimage import shift
from skimage.color import label2rgb

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from scipy.fft import fft, fftfreq
from scipy.interpolate import interp1d
from scipy.spatial.distance import euclidean

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FFMpegWriter
import matplotlib.cm as cm

import umap
reducer = umap.UMAP()

##########################DELETE THEN!!!!!!!!!!!!!#########################################################################################################
######################################################
######################################################
######################################################
############################################################################################################
######################################################
############################################################################################################
############################################################################################################
######################################################
from geomstats.geometry.discrete_curves import ElasticMetric
from geomstats.geometry.discrete_curves import DiscreteCurvesStartingAtOrigin

from geomstats.geometry.pre_shape import PreShapeSpace
import matplotlib.pyplot as plt

import geomstats.backend as gs


import h5py
import geomstats.backend as gs
import numpy as np 
#from numba import jit, njit, prange
import scipy.stats as stats
from scipy.integrate import simpson
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm



from geomstats.geometry.discrete_curves import (
    DiscreteCurvesStartingAtOrigin,
    DynamicProgrammingAligner,
    ReparametrizationBundle,
    RotationBundle,
    ElasticMetric
)


from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import LineCollection

def del_arr_elements(arr, indices):
    """
    Delete elements in indices from array arr
    """

    # Sort the indices in reverse order to avoid index shifting during deletion
    indices.sort(reverse=True)

    # Iterate over each index in the list of indices
    for index in indices:
        del arr[index]
    return arr





# def exhaustive_align(curve, ref_curve, k_sampling_points, rescale=True, dynamic=False, reparameterization=True):
#     """ 
#     Quotient out
#         - translation (move curve to start at the origin) 
#         - rescaling (normalize to have length one)
#         - rotation (try different starting points, during alignment)
#         - reparametrization (resampling in the discrete case, during alignment)
    
#     :param bool rescale: quotient out rescaling or not 
#     :param bool dynamic: Use dynamic aligner or not 
#     :param bool reparamterization: quotient out rotation only rather than rotation and reparameterization

#     """
    
#     curves_r2 = DiscreteCurvesStartingAtOrigin(
#         ambient_dim=2, k_sampling_points=k_sampling_points, equip=False
#     )

#     if dynamic:
#         print("Use dynamic programming aligner")
#         curves_r2.fiber_bundle = ReparametrizationBundle(curves_r2)
#         curves_r2.fiber_bundle.aligner = DynamicProgrammingAligner()

#     # Quotient out translation
#     print("Quotienting out translation")
#     curve = curves_r2.projection(curve)
#     ref_curve = curves_r2.projection(ref_curve)

#     # Quotient out rescaling
#     if rescale:
#         print("Quotienting out rescaling")
#         curve = curves_r2.normalize(curve)
#         ref_curve = curves_r2.normalize(ref_curve)

#     # Quotient out rotation and reparamterization
#     curves_r2.equip_with_metric(ElasticMetric)
#     if not reparameterization:
#         print("Quotienting out rotation")
#         curves_r2.equip_with_group_action("rotations")
#     else:
#         print("Quotienting out rotation and reparamterization")
#         curves_r2.equip_with_group_action("rotations and reparametrizations")
        
#     curves_r2.equip_with_quotient_structure()
#     aligned_curve = curves_r2.fiber_bundle.align(curve, ref_curve)
#     return aligned_curve




def rotation_align(curve, base_curve, k_sampling_points):
    """Align curve to base_curve to minimize the L² distance by \
        trying different start points.

    Returns
    -------
    aligned_curve : discrete curve
    """
    nb_sampling = len(curve)
    distances = gs.zeros(nb_sampling)
    base_curve = gs.array(base_curve)

    # Rotation is done after projection, so the origin is removed
    total_space = DiscreteCurvesStartingAtOrigin(k_sampling_points=k_sampling_points-1)
    total_space.fiber_bundle = RotationBundle(total_space)

    for shift in range(nb_sampling):
        reparametrized = [curve[(i + shift) % nb_sampling] for i in range(nb_sampling)]
        aligned = total_space.fiber_bundle.align(
            point=gs.array(reparametrized), base_point=base_curve
        )
        distances[shift] = np.linalg.norm(
            gs.array(aligned) - gs.array(base_curve)
        )
    shift_min = gs.argmin(distances)
    reparametrized_min = [
        curve[(i + shift_min) % nb_sampling] for i in range(nb_sampling)
    ]
    aligned_curve = total_space.fiber_bundle.align(
        point=gs.array(reparametrized_min), base_point=base_curve
    )
    return aligned_curve


def align(point, base_point, rescale, rotation, reparameterization, k_sampling_points):
    """
    Align point and base_point via quotienting out translation, rescaling, rotation and reparameterization
    """

    total_space = DiscreteCurvesStartingAtOrigin(k_sampling_points=k_sampling_points)
   
    
    # Quotient out translation 
    point = total_space.projection(point) 
    point = point - gs.mean(point, axis=0)

    base_point = total_space.projection(base_point)
    base_point = base_point - gs.mean(base_point, axis=0)

    # Quotient out rescaling
    if rescale:
        point = total_space.normalize(point) 
        base_point = total_space.normalize(base_point)
    
    # Quotient out rotation
    if rotation:
        point = rotation_align(point, base_point, k_sampling_points)

    # Quotient out reparameterization
    if reparameterization:
        aligner = DynamicProgrammingAligner(total_space)
        total_space.fiber_bundle = ReparametrizationBundle(total_space, aligner=aligner)
        point = total_space.fiber_bundle.align(point, base_point)
    return point

def project_on_kendall_space(curve,PRESHAPE_SPACE= None):
    if PRESHAPE_SPACE is None:
        PRESHAPE_SPACE = PreShapeSpace(ambient_dim=2, k_landmarks=len(curve))
    projected_curve = PRESHAPE_SPACE.projection(curve)
    return projected_curve

def interpolate(curve, nb_points):
    """Interpolate a discrete curve with nb_points from a discrete curve.

    Returns
    -------
    interpolation : discrete curve with nb_points points
    """
    old_length = curve.shape[0]
    interpolation = gs.zeros((nb_points, 2))
    incr = old_length / nb_points
    pos = 0
    for i in range(nb_points):
        index = int(gs.floor(pos))
        interpolation[i] = curve[index] + (pos - index) * (
            curve[(index + 1) % old_length] - curve[index]
        )
        pos += incr
    return interpolation

def preprocess(curve, tol=1e-10):
    """Preprocess curve to ensure that there are no consecutive duplicate points.

    Returns
    -------
    curve : discrete curve
    """

    dist = curve[1:] - curve[:-1]
    dist_norm = np.sqrt(np.sum(np.square(dist), axis=1))

    if np.any( dist_norm < tol ):
        for i in range(len(curve)-1):
            if np.sqrt(np.sum(np.square(curve[i+1] - curve[i]), axis=0)) < tol:
                curve[i+1] = (curve[i] + curve[i+2]) / 2

    return curve
#######################################################################################################################333

def l2_fourier(x, y, x1, y1):

    return euclidean(y, y1)





def fft_transformation(x_coords, y_coords, f_max):
        x_coords = (x_coords - x_coords[0]) / (x_coords[-1] - x_coords[0])
        x_fourier  = x_coords
        y_fourier = y_coords

        x_uniform = np.linspace(x_fourier[0], x_fourier[-1], 256)
        interp_func = interp1d(x_fourier, y_fourier, kind='cubic')

        y_uniform = interp_func(x_uniform)

        n = len(y_uniform)
        timestep = x_uniform[1] - x_uniform[0]  

        yf = fft(y_uniform)
        xf = fftfreq(n, d=timestep)

        mask = xf >= 0  
        mask &= xf <= f_max

  #      plt.figure(figsize=(12, 5))

 #       plt.stem(xf[mask], np.abs(yf[mask]))

#        plt.show()
        return xf[mask], 2*np.abs(yf[mask])/n/5



def height_interpolation(x_coords, y_coords, number_of_points):
        x_coords = (x_coords - x_coords[0]) / (x_coords[-1] - x_coords[0])
        x_uniform = np.linspace(x_coords[0], x_coords[-1], number_of_points)

        interp_func = interp1d(x_coords, y_coords, kind='cubic')

        y_uniform = interp_func(x_uniform)

        return y_uniform



def op_index(cell_shape_1, cell_shape_2, index_from):

    N1 = len(cell_shape_1)
    N2 = len(cell_shape_2)
    a = np.ones(N1) / N1  
    b = np.ones(N2) / N2 

    #print(cell_shape_1[:, None, :])


    # M = np.zeros((N1, N2))
    # for i in range(N2):
    #     for j in range(N1):
    #         M[j,i] = np.sqrt((cell_shape_1[j][0]-cell_shape_2[i][0])**2+ (cell_shape_1[j][1]-cell_shape_2[i][1])**2)


    M = np.linalg.norm(cell_shape_1[:, None, :] - cell_shape_2[None, :, :], axis=-1)**2
    T = ot.emd(a, b, M)
    
    mapping = np.argmax(T, axis=1)

    #return
    return mapping[index_from]




def conformal_representation(cell_folder ,
                             ref_disk_radius = 50,
                             ref_iter_no = 3, 
                             lambda_flow = 200, 
                             curvature_threshold = 0.1):
    
    import glob
    #import skimage.io as skio 
    import pylab as plt 
    #import skimage.transform as sktform 
    
    import unwrap2D.unwrap2D as unwrap2D_fns
    import unwrap3D.Mesh.meshtools as unwrap3D_meshtools
    #import unwrap3D.Segmentation.segmentation as unwrap3D_segmentation
    
    
    # https://github.com/clementsoubrier/shape_analysis/blob/main/construct_dataset.py


    # ref_disk_radius = 50
    # ref_iter_no = 3 # 3 use this and 
    # lambda_flow = 200 # 5e1  use this to tune the extent of smoothing of base shape, the less the smoother 
    # curvature_threshold = 0.1 # only apply flow iterations if np.mean(np.abs(curvature)) > threshold i.e. there are potential protrusions! 
    
    
    """
    Iterate over cells. 
    """

    print(cell_folder)

    times = np.sort(os.listdir(cell_folder)) # perform nature sort
    print(times)
    times_int = np.hstack([int(tt.split('frame_')[1]) for tt in times])
    print(times_int)


    natsort = np.argsort(times_int)

    times_int = times_int[natsort]
    times = times[natsort]
    
    all_outlines = []
    all_outlines_cMCF = []
    all_outlines_cMCF_topography = []
    all_outlines_curvature = []
    fin_times = []
    fin_centr = []

    
    for ttt in np.arange(len(times))[:]:
        
        outline_ttt_file = os.path.join(cell_folder, times[ttt], 'outline.npy')
        centroid_ttt_file = os.path.join(cell_folder, times[ttt], 'centroid.npy')
        time_ttt_file = os.path.join(cell_folder, times[ttt], 'time.npy')
        
        outline = np.load(outline_ttt_file)
        all_outlines.append(outline)
        fin_times.append(np.load(time_ttt_file))
        centroid = np.load(centroid_ttt_file)
        fin_centr.append(centroid)
        # evolve the contour. 
        outline_input = outline.copy()
        _,_, outline_input_curvature = unwrap2D_fns.curvature_splines(outline_input[:,1], 
                                                                outline_input[:,0],
                                                                k=4,
                                                                error=0.1)
        all_outlines_curvature.append(outline_input_curvature)
        outline_input_curvature_norm = np.nanmean(np.linalg.norm(outline_input-centroid[None,:], axis=-1), axis=0)/2. * outline_input_curvature

        # print(outline_input_curvature_norm.min(),outline_input_curvature_norm.max(), np.nanmean(np.abs(outline_input_curvature_norm)))
                    
        # =============================================================================
        #            Find reference
        # =============================================================================
        contour_evolve = unwrap3D_meshtools.conformalized_mean_line_flow( np.hstack([np.ones(len(outline_input))[:,None], 
                                                                    outline_input]), 
                                                            E=None, 
                                                            close_contour=True, 
                                                            fixed_boundary = False, 
                                                            lambda_flow=lambda_flow, # change this to control the degree of smoothing... 
                                                            niters=20, # hm... increasing this helps find ref.  
                                                            topography_edge_fix=False, 
                                                            conformalize=True)
        contour_evolve = contour_evolve[:,1:]
        '''
        # mask_contour = unwrap2D_fns.pixelize_contours(outline, 
        #                                               imshape=None,
        #                                               padsize=50)
        # H_normal, sdf_vol_normal, sdf_vol = unwrap3D_segmentation.mean_curvature_binary(mask_contour, 
        #                                                                         smooth=1, 
        #                                                                         mask=False, 
        #                                                                         smooth_gradient=3, eps=1e-12)
        
        # contour_evolve = unwrap2D_fns.parametric_line_flow_2D(outline_input,
        #                                                         external_img_gradient=sdf_vol_normal.transpose(1,2,0), 
        #                                                         E=None, 
        #                                                         close_contour=True, 
        #                                                         fixed_boundary = False, 
        #                                                         lambda_flow=5e1, # adjusts the balance.  (decrease this to be more similar to original cell shape ), increase to be more like version 1
        #                                                         step_size=0.25, # adjusts the spacing between curves. 
        #                                                         niters=20, 
        #                                                         conformalize=False,
        #                                                         eps=1e-12)
        '''
        # register the contours to original 
        contour_evolve_register = unwrap2D_fns.register_contour_array(contour_evolve, ref_id=0, tform='Similarity')
        
        # errs = np.linalg.norm(contour_evolve-contour_evolve[...,0][...,None], axis=1).mean(axis=0)
        errs_register = np.linalg.norm(contour_evolve_register-contour_evolve_register[...,0][...,None], axis=1).mean(axis=0)
        '''
        # errs_register = np.nanmean(np.cumsum(np.linalg.norm(np.diff(contour_evolve, axis=-1), axis=1), axis=-1), axis=0)
        
        # inds = unwrap3D_meshtools.find_all_curvature_cutoff_index(errs_register, winsize=5,
        #                                                           min_peak_height=0)
        # # inds_original = unwrap3D_meshtools.find_all_curvature_cutoff_index(errs, winsize=5)
        # if len(inds) == 0:
        #     ind = 0
        # else:
        #     ind = inds[0] + 1 
        # # ind = inds_original[0] #+ 1 
        '''
        if np.nanmean(np.abs(outline_input_curvature_norm)) > curvature_threshold:
            ind = ref_iter_no
        else:
            ind = 1 
        
        # plt.figure(figsize=(10,10))
        # plt.plot(outline[:,1], 
        #           outline[:,0], 'k-', lw=10)
        # plt.plot(contour_evolve_register[:,1,ind], 
        #             contour_evolve_register[:,0,ind], 'g-', lw=10)
        # plt.show()
    
    
        """
        map to circle 
        """
        mask_contour = unwrap2D_fns.pixelize_contours(outline, 
                                                    imshape=None,
                                                    padsize=50)
        
        contour_ref = contour_evolve_register[...,ind].copy()
        contour_ref_protrusion_d, sign_d = unwrap2D_fns.measure_signed_distance(contour_ref, 
                                                                        contours=contour_evolve_register.transpose(2,0,1)[:ind+1], 
                                                                        imshape=mask_contour.shape)
        
        all_outlines_cMCF.append(contour_ref)
        
        topography_coords_disk, disk_coords = unwrap2D_fns.map_protrusion_topography_to_circle(contour_ref, 
                                                                                        contour_ref_protrusion_d, 
                                                                                        circle_R=ref_disk_radius, 
                                                                                        close_contour=True)
    
        
        all_outlines_cMCF_topography.append(topography_coords_disk)
        
        
    return all_outlines, all_outlines_cMCF, all_outlines_cMCF_topography, all_outlines_curvature, disk_coords, np.array(fin_centr), np.array(fin_times)



def max_protusion_plot(direct):
    
    all_cell_folders = [os.path.join(direct, ff) for ff in os.listdir(direct)]
    for cell_folder_ii in np.arange(204,205):#len(all_cell_folders)):
        all_outlines, _, all_outlines_cMCF_topography, all_outlines_curvature, _, _, fin_times = conformal_representation(all_cell_folders[cell_folder_ii])
     
        plt.figure()
        plt.title(f'Shape evolution, cell {cell_folder_ii}')
        ax = plt.axes(projection='3d')
        tot_len = len(all_outlines_cMCF_topography) #10
        colors = plt.cm.viridis(np.linspace(0,1,tot_len))
        max_top = []
        max_curv = []
        print(tot_len)
        for jj in np.arange(tot_len) :
            topography_coords_disk = all_outlines_cMCF_topography[jj]
            outline = all_outlines[jj]
            time = fin_times[jj]
            curv = all_outlines_curvature[jj]
            index = np.argmax(topography_coords_disk [:,0]**2+topography_coords_disk [:,1]**2)
            index_curvature = np.argmin(curv)                                                                                                                                                                                                                                                                           

            ax.plot3D(outline[:,0],outline[:,1], time*np.ones(len(outline[:,1])), c=colors[jj] )
            max_top.append([outline[index][0], outline[index][1], time])
            max_curv.append([outline[index_curvature][0], outline[index_curvature][1], time])
            
        max_top = np.array(max_top)
        max_curv = np.array(max_curv)   
        
        ax.scatter(max_top[:,0], 
                        max_top[:,1], max_top[:,2], c='k', label='max topo height')
        ax.scatter(max_curv[:,0], 
                        max_curv[:,1], max_curv[:,2], c='r', label='max curvature')
        ax.legend()
        
            
        plt.figure()
        plt.title(f'Topographical representation, cell {cell_folder_ii}')
        ax = plt.axes(projection='3d')
        tot_len = len(all_outlines_cMCF_topography) #10
        colors = plt.cm.viridis(np.linspace(0,1,tot_len))
        max_top = []
        max_curv = []
        
        for jj in np.arange(tot_len) :
            topography_coords_disk = all_outlines_cMCF_topography[jj]
            outline = all_outlines[jj]
            time = fin_times[jj]
            curv = all_outlines_curvature[jj]
            index = np.argmax(topography_coords_disk [:,0]**2+topography_coords_disk [:,1]**2)
            index_curvature = np.argmin(curv)

            ax.plot3D(topography_coords_disk[:,0],topography_coords_disk[:,1], time*np.ones(len(outline[:,1])), c=colors[jj] )
            max_top.append([topography_coords_disk[index][0], topography_coords_disk[index][1], time])
            max_curv.append([topography_coords_disk[index_curvature][0], topography_coords_disk[index_curvature][1], time])
            
        max_top = np.array(max_top)
        max_curv = np.array(max_curv)   
        
        ax.scatter(max_top[:,0], 
                        max_top[:,1], max_top[:,2], c='k', label='max topo height')
        ax.scatter(max_curv[:,0], 
                        max_curv[:,1], max_curv[:,2], c='r', label='max curvature')
        ax.legend()
        plt.show()
            

def max_protusion_plot(direct, time_events_h5="time_events_90.h5"):
    all_cell_folders = [
        os.path.join(direct, ff) 
        for ff in sorted(os.listdir(direct), key=lambda x: int(x.split('_')[1]) if '_' in x and x.split('_')[1].isdigit() else 0)
    ]
    print(all_cell_folders)
    
    for cell_folder_ii in np.arange(86,87):
        all_outlines, _, all_outlines_cMCF_topography, all_outlines_curvature, _, _, fin_times = conformal_representation(all_cell_folders[cell_folder_ii])

        with h5py.File(time_events_h5, 'r') as f:
            track_data = f[f'/track_{cell_folder_ii+1}'][:]  
            event_indices = track_data[0, :].astype(int) - 1
            interval_types = track_data[2, :]
            print(track_data[0:3])
            
        interval_colors = {
            0: "brown",        # Immobile
            1: "blue",         # Confined Diffusion
            2: "cyan",         # Free Diffusion
            3: "magenta",      # Directed Diffusion
            "unclassified": "black"  # Unclassified
        }
        
        fig, axes = plt.subplots(1, 2, subplot_kw={'projection': '3d'}, figsize=(12, 6))  # Два субплота
        fig.suptitle(f'Cell {cell_folder_ii+1} Analysis', fontsize=16)
        
        # ====== Первый субплот: Shape Evolution ======
        ax1 = axes[0]
        ax1.set_title('Shape Evolution')
        max_top = []
        max_curv = []
        time_interval = np.arange(48,51)
        for jj in time_interval:#(1,len(all_outlines)):
            topography_coords_disk = all_outlines_cMCF_topography[jj]
            outline = all_outlines[jj]
            time = fin_times[jj]
            curv = all_outlines_curvature[jj]
            
            index = np.argmax(topography_coords_disk[:, 0]**2 + topography_coords_disk[:, 1]**2)
            index_curvature = np.argmin(curv)
            
            current_type = None
            for start_idx, interval_type in enumerate(interval_types):
                if jj >= event_indices[start_idx] and (start_idx + 1 == len(event_indices) or jj < event_indices[start_idx + 1]):
                    current_type = interval_type
                    break
            interval_type = int(current_type) if not np.isnan(current_type) else "unclassified"
            color = interval_colors.get(interval_type, "black")
            
            ax1.plot3D(outline[:, 0], outline[:, 1], time * np.ones(len(outline[:, 1])), color=color, linewidth=1)
            max_top.append([outline[index][0], outline[index][1], time])
            max_curv.append([outline[index_curvature][0], outline[index_curvature][1], time])
        
        max_top = np.array(max_top)
        max_curv = np.array(max_curv)
        ax1.scatter(max_top[:, 0], max_top[:, 1], max_top[:, 2], c='k', label='max topo height', s=50)
        ax1.scatter(max_curv[:, 0], max_curv[:, 1], max_curv[:, 2], c='r', label='max curvature', s=50)
        
        ax2 = axes[1]
        ax2.set_title('Topographical Representation')
        max_top = []
        max_curv = []
        for jj in time_interval:
            topography_coords_disk = all_outlines_cMCF_topography[jj]
            outline = all_outlines[jj]
            time = fin_times[jj]
            curv = all_outlines_curvature[jj]
            
            index = np.argmax(topography_coords_disk[:, 0]**2 + topography_coords_disk[:, 1]**2)
            index_curvature = np.argmin(curv)
            
            current_type = None
            for start_idx, interval_type in enumerate(interval_types):
                if jj >= event_indices[start_idx] and (start_idx + 1 == len(event_indices) or jj < event_indices[start_idx + 1]):
                    current_type = interval_type
                    break
            interval_type = int(current_type) if not np.isnan(current_type) else "unclassified"
            color = interval_colors.get(interval_type, "black")
            
            ax2.plot3D(topography_coords_disk[:, 0], topography_coords_disk[:, 1], time * np.ones(len(outline[:, 1])), color=color, linewidth=1)
            max_top.append([topography_coords_disk[index][0], topography_coords_disk[index][1], time])
            max_curv.append([topography_coords_disk[index_curvature][0], topography_coords_disk[index_curvature][1], time])
        
        max_top = np.array(max_top)
        max_curv = np.array(max_curv)
        ax2.scatter(max_top[:, 0], max_top[:, 1], max_top[:, 2], c='k', label='max topo height', s=50)
        ax2.scatter(max_curv[:, 0], max_curv[:, 1], max_curv[:, 2], c='r', label='max curvature', s=50)
        
        # Настройка общих осей для обоих субплотов
        for ax in [ax1, ax2]:
            ax.legend(loc="upper right", fontsize=8)
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            ax.set_zlabel('Time')
            ax.view_init(elev=90, azim=-90)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.savefig(f"combined_cell_{cell_folder_ii+1}_1.png")
        plt.show()


def _stats_max_protusions(cell_folder, smoothing = False, smooth_time = 2):
    all_outlines, _, all_outlines_cMCF_topography, all_outlines_curvature, _, fin_centr, fin_times = conformal_representation(cell_folder)
    tot_len = len(all_outlines)
    vec_topo = []
    vec_curv = []
    val_curv = []
    for jj in np.arange(tot_len)[1:]:
        
        topography_coords_disk = all_outlines_cMCF_topography[jj]
        outline = all_outlines[jj]
        curv = all_outlines_curvature[jj]
        centr = fin_centr[jj]
        
        index = np.argmax(topography_coords_disk [:,0]**2+topography_coords_disk [:,1]**2)
        index_curv = np.argmin(curv)
        
        vec_topo.append(outline[index] - centr)
        vec_curv.append(outline[index_curv] - centr)
        val_curv.append(curv[index_curv])
        
    displacement = fin_centr[1:]- fin_centr[:-1]
    velocity = np.zeros(np.shape(displacement))
    if smoothing:
        
        for jj, val_jj in enumerate(displacement):
            mask = np.logical_and(fin_times[1:]>= fin_times[jj] - smooth_time, fin_times[1:]<= fin_times[jj] + smooth_time)
            if np.sum(mask) >1 :
                time_var = fin_times[1:] - fin_times[:-1]
                velocity[jj] = (np.sum(displacement[mask])) / (np.sum(time_var[mask]))
            else:
                velocity[jj] = val_jj / (fin_times[jj+1] - fin_times[jj])
            
    else :
        velocity[:,0] = displacement[:,0] / (fin_times[1:]-fin_times[:-1])
        velocity[:,1] = displacement[:,1] / (fin_times[1:]-fin_times[:-1])
        

    return velocity, vec_topo, vec_curv, val_curv

def stats_max_protusions(direct):
    from multiprocessing import Pool
    res_velo = []
    res_topo = [] 
    res_curv = []
    val_curv = []
    all_cell_folders = [os.path.join(direct, ff) for ff in os.listdir(direct)]
    
    
    with Pool(processes=8) as pool:
            for _velo, _topo, _curv, _vcurv in pool.imap_unordered(_stats_max_protusions, all_cell_folders):
                val_curv.extend(_vcurv)
                res_velo.extend(_velo)
                res_topo.extend(_topo)
                res_curv.extend(_curv)
            
    res_curv = np.array(res_curv)
    res_velo = np.array(res_velo)
    res_topo = np.array(res_topo)
    val_curv = np.array(val_curv)
    
    plt.figure()
    plt.xlabel('Max curvature')
    plt.ylabel('Velocity norm')
    vel_norm = (res_velo[:,0]**2+res_velo[:,1]**2)**0.5
    plt.scatter(-val_curv, vel_norm)
    
    plt.figure()
    bin_num = 20
    bottom = 400

    theta_topo = np.arctan2(res_topo[:,0], res_topo[:,1])
    theta_velo = np.arctan2(res_velo[:,0], res_velo[:,1])
    theta = (theta_topo - theta_velo ) % (2 * np.pi)
    
    hist = np.histogram(theta, bins=bin_num, range=(0,2 * np.pi))
    width = (2*np.pi) / bin_num

    ax = plt.subplot(111, polar=True)
    ax.set_title('Angle of max protusion with the displacement, topology')
    bars = ax.bar((hist[1][:-1]+hist[1][1:])/2, hist[0], width=width, bottom=bottom)
    max_val = np.max(hist[0])
    for r, bar in zip(hist[0], bars):
        bar.set_facecolor(plt.cm.jet(r / max_val))
        bar.set_alpha(0.8)
        
    
    plt.figure()
    bin_num = 20
    bottom = 400

    theta_curv = np.arctan2(res_curv[:,0], res_curv[:,1])
    theta_velo = np.arctan2(res_velo[:,0], res_velo[:,1])
    theta = (theta_curv - theta_velo) % (2 * np.pi)
    
    hist = np.histogram(theta, bins=bin_num, range=(0,2 * np.pi))
    width = (2*np.pi) / bin_num

    ax = plt.subplot(111, polar=True)
    ax.set_title('Angle of max protusion with the displacement, curvature')
    bars = ax.bar((hist[1][:-1]+hist[1][1:])/2, hist[0], width=width, bottom=bottom)

    # Use custom colors and opacity
    max_val = np.max(hist[0])
    for r, bar in zip(hist[0], bars):
        bar.set_facecolor(plt.cm.jet(r / max_val))
        bar.set_alpha(0.8)

    
    plt.show()


def stats_max_protusions_2(direct):
    all_cell_folders = [os.path.join(direct, ff) for ff in os.listdir(direct)]

    bin_num = 36
    bins = np.linspace(0, 180, bin_num + 1)
    bin_values = np.zeros(bin_num)
    bin_number_of = np.zeros(bin_num)

    with Pool(processes=8) as pool:
        for all_outlines, _, all_outlines_cMCF_topography, _, disk_coords, fin_centr, fin_times in pool.imap_unordered(conformal_representation, all_cell_folders):
             index_from = 0
             for t in range(1, len(all_outlines_cMCF_topography)-1):
                disp = fin_centr[t] - fin_centr[t-1]
                dt = fin_times[t] - fin_times[t-1]
                velocity = disp / dt
                velocity = gaussian_filter1d(velocity, SIGMA)
                theta_vel = np.arctan2(velocity[0], velocity[1])
                index_from = op_index(all_outlines[t-1], all_outlines[t], index_from)
                theta_vel = np.arctan2(all_outlines[t][index_from][0], all_outlines[t][index_from][1])

                theta_topo = np.arctan2(all_outlines_cMCF_topography[t][:, 0], all_outlines_cMCF_topography[t][:, 1])
                theta_diff = np.unwrap(theta_topo - theta_vel)
                theta = np.degrees(theta_diff)
                theta = (theta + 180) % 360 - 180  
                theta = np.abs(theta) 

                print(len(all_outlines_cMCF_topography))
                print(len(disk_coords))
                topo_height = np.sqrt((all_outlines_cMCF_topography[t][:, 0])**2 + (all_outlines_cMCF_topography[t][:, 1])**2) -50


                for i in range(len(theta)):
                    index = np.digitize(theta[i], bins) - 1 
                    index = max(0, min(index, bin_num - 1))  
                    bin_values[index] += topo_height[i]
                    bin_number_of[index] += 1


    for i in range(bin_num):
        if bin_number_of[i] > 0:
            bin_values[i] /= bin_number_of[i]


    plt.figure(figsize=(10, 6))
    plt.xlabel('Polar angle (degrees)')
    plt.ylabel('Average Topological height')
    plt.bar(bins[:-1], bin_values, width=np.diff(bins), edgecolor='black', align='edge')

    plt.xlim(0, 180)
    plt.title(f'Histogram of Topological height vs Polar angle')
    plt.savefig("topo_2.png")
    plt.show()

def stats_max_protusions_360(direct):
    all_cell_folders = [os.path.join(direct, ff) for ff in os.listdir(direct)]

    bin_num = 72 
    bins = np.linspace(0, 360, bin_num + 1)
    bin_values = np.zeros(bin_num)
    bin_number_of = np.zeros(bin_num)

    with Pool(processes=8) as pool:
        for all_outlines, _, all_outlines_cMCF_topography, _, disk_coords, fin_centr, fin_times in pool.imap_unordered(conformal_representation, all_cell_folders):
            index_from = 0
            for t in range(1, len(all_outlines_cMCF_topography)):
                disp = fin_centr[t] - fin_centr[t-1]
                dt = fin_times[t] - fin_times[t-1]
                velocity = disp / dt
                sigma = 2 
                # velocity = gaussian_filter1d(velocity, SIGMA)
                # theta_vel = np.arctan2(velocity[0], velocity[1])
                index_from = op_index(all_outlines[t-1], all_outlines[t], index_from)
                theta_vel = np.arctan2(all_outlines[t][index_from][0], all_outlines[t][index_from][1])
                theta_topo = np.arctan2(all_outlines_cMCF_topography[t][:, 0], all_outlines_cMCF_topography[t][:, 1])
                theta_diff = np.unwrap(theta_topo - theta_vel)
                theta = np.degrees(theta_diff)
                theta = (theta + 360) % 360  

                topo_height = np.sqrt((all_outlines_cMCF_topography[t][:, 0])**2 + (all_outlines_cMCF_topography[t][:, 1])**2) -50

                for i in range(len(theta)):
                    index = np.digitize(theta[i], bins) - 1  
                    index = max(0, min(index, bin_num - 1))  
                    bin_values[index] += topo_height[i]
                    bin_number_of[index] += 1

    for i in range(bin_num):
        if bin_number_of[i] > 0:
            bin_values[i] /= bin_number_of[i]

    plt.figure(figsize=(10, 6))
    plt.xlabel('Polar angle (degrees)')
    plt.ylabel('Average Topological height')
    plt.bar(bins[:-1], bin_values, width=np.diff(bins), edgecolor='black', align='edge')

    plt.xlim(0, 360)
    plt.xticks(np.arange(0, 361, 45))  
    plt.title('Histogram of Topological height vs Polar angle (0-360 degrees)')
    plt.savefig("topo_360.png")
    plt.show()
def stats_single_cell_protusions(direct, cell_number, time_int):

    folder_path = os.path.join(direct, f"cell_{cell_number}")

    # with Pool(processes=8) as pool:
    #     for _velo, _topo, _curv, _vcurv in pool.imap_unordered(_stats_max_protusions, folder_path):
    #         val_curv.extend(_vcurv)
    #         res_velo.extend(_velo)
    #         res_topo.extend(_topo)
    #         res_curv.extend(_curv)

    all_outlines, _, all_outlines_cMCF_topography, all_outlines_curvature, disk_coords, fin_centr, fin_times = conformal_representation(folder_path)


    print("disk _", disk_coords.shape)
    print("disk _", disk_coords)
    print("disk [0]",disk_coords[0].shape)
    print("disk [0]",disk_coords[0])


    for t in range(time_int[0], time_int[1]):
        disp = fin_centr[t] - fin_centr[t-1]
        dt = fin_times[t]-fin_times[t-1]
        
        index_curvature_maximum =  np.argmax(all_outlines_curvature[t])
        theta_curv_max = np.arctan2(all_outlines_cMCF_topography[t][index_curvature_maximum,0], all_outlines_cMCF_topography[t][index_curvature_maximum,1])

        index_curvature_minimum =  np.argmin(all_outlines_curvature[t])
        theta_curv_min = np.arctan2(all_outlines_cMCF_topography[t][index_curvature_minimum,0], all_outlines_cMCF_topography[t][index_curvature_minimum,1])

        velocity = disp/dt

        theta_vel = np.arctan2(velocity[0], velocity[1])
        vel_angle = np.degrees(theta_vel % (2 * np.pi))
        
        plt.figure(figsize=(10, 6))
        plt.xlabel('Polar angle (degrees)')
        plt.ylabel('Topological height')
        
        theta_topo = np.arctan2(all_outlines_cMCF_topography[t][:, 0], all_outlines_cMCF_topography[t][:, 1])
        theta = np.degrees(theta_topo % (2 * np.pi))
        topo_height = np.sqrt((all_outlines_cMCF_topography[t][:, 0])**2 + (all_outlines_cMCF_topography[t][:, 1])**2) -50
        bin_num = 36
        bins = np.linspace(0, 360, bin_num + 1)
        
        plt.hist(theta, bins=bins, weights=topo_height, edgecolor='black')
        
        plt.axvline(vel_angle, color='red', linestyle='dashed', linewidth=2, label='Velocity direction')
        plt.axvline(theta_curv_max, color='yellow', linestyle='dashed', linewidth=2, label='Curvature max')
        plt.axvline(theta_curv_min, color='green', linestyle='dashed', linewidth=2, label='Curvature min')
        plt.legend()
        
        plt.xlim(0, 360)
        plt.title(f'Histogram of Topological height vs Polar angle of {cell_number} in {t}')
        plt.show()




def stats_single_cell_protusions_wrt_vel(direct, cell_number, time_int):
    folder_path = os.path.join(direct, f"cell_{cell_number}")
    frame_path = os.path.join(os.getcwd())
    
    
    all_outlines, _, all_outlines_cMCF_topography, all_outlines_curvature, disk_coords, fin_centr, fin_times = conformal_representation(folder_path)
    

    frames_dir = os.path.join(frame_path, "frames")
    os.makedirs(frames_dir, exist_ok=True) 
    if(len(time_int)==2):
        init  = time_int[0]
        final = time_int[1]
    elif(len(time_int)==1):
        init  = time_int[0]
        final = len(all_outlines_cMCF_topography)
    elif(len(time_int)==0):
        init  = 1
        final = len(all_outlines_cMCF_topography)

    index_from = 0

    for t in range(init, final):
        disp = fin_centr[t] - fin_centr[t-1]
        dt = fin_times[t]-fin_times[t-1]
        
        index_curvature_maximum = np.argmax(all_outlines_curvature[t])
        theta_curv_max = np.arctan2(all_outlines_cMCF_topography[t][index_curvature_maximum,0],
                                     all_outlines_cMCF_topography[t][index_curvature_maximum,1])

        index_curvature_minimum = np.argmin(all_outlines_curvature[t])
        theta_curv_min = np.arctan2(all_outlines_cMCF_topography[t][index_curvature_minimum,0],
                                     all_outlines_cMCF_topography[t][index_curvature_minimum,1])

        #velocity = disp/dt
        sigma = 2  
        index_from = op_index(all_outlines[t-1], all_outlines[t], index_from)
        theta_vel = np.arctan2(all_outlines[t][index_from][0], all_outlines[t][index_from][1])
        
        
        #velocity = gaussian_filter1d(velocity, SIGMA)

        #theta_vel = np.arctan2(velocity[0], velocity[1])
        
        plt.figure(figsize=(10, 6))
        plt.xlabel('Polar angle (degrees)')
        plt.ylabel('Topological height')
        
        theta_topo = np.arctan2(all_outlines_cMCF_topography[t][:, 0], all_outlines_cMCF_topography[t][:, 1])
        theta = np.degrees((theta_topo - theta_vel) % (2 * np.pi))
        
        topo_height = np.sqrt((all_outlines_cMCF_topography[t][:, 0])**2 + (all_outlines_cMCF_topography[t][:, 1])**2) -50
        
        bin_num = 36
        bins = np.linspace(0, 360, bin_num + 1)
        
        plt.hist(theta, bins=bins, weights=topo_height, edgecolor='black')
        
        plt.axvline(0, color='red', linestyle='dashed', linewidth=2, label='Velocity direction')
        plt.axvline(np.degrees((theta_curv_max - theta_vel) % (2 * np.pi)), color='yellow', linestyle='dashed', linewidth=2, label='Curvature max')
        plt.axvline(np.degrees((theta_curv_min - theta_vel) % (2 * np.pi)), color='green', linestyle='dashed', linewidth=2, label='Curvature min')
        plt.legend()
        
        plt.xlim(0, 360)
        plt.title(f'Histogram of Topological height vs Polar angle of {cell_number} in {t}')
        
        frame_path = os.path.join(frames_dir, f'frame_{t:03d}.png')
        plt.savefig(frame_path)
        plt.close()
    
    create_video(frames_dir, os.path.join(os.getcwd(), "video.mp4"))
    clean_frames(frames_dir)

def create_video(frames_dir, output_path, fps=2):
    frame_files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    if not frame_files:
        print("No frames found for video.")
        return
    
    first_frame = cv2.imread(frame_files[0])
    height, width, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame_file in frame_files:
        frame = cv2.imread(frame_file)
        video_writer.write(frame)
    
    video_writer.release()
    print(f"Video saved to {output_path}")

def clean_frames(frames_dir):
    frame_files = glob.glob(os.path.join(frames_dir, "frame_*.png"))
    for frame_file in frame_files:
        os.remove(frame_file)
    os.rmdir(frames_dir)
    print("Frames cleaned up.")

def create_outline_video(direct, output_file="cell_evolution.mp4"):
    all_cell_folders = [
        os.path.join(direct, ff) 
        for ff in sorted(os.listdir(direct), key=lambda x: int(x.split('_')[1]) if '_' in x and x.split('_')[1].isdigit() else 0)
    ]
    print(all_cell_folders)

    cell_folder_ii = 86

    all_outlines, _, all_outlines_cMCF_topography, all_outlines_curvature, _, _, fin_times = conformal_representation(all_cell_folders[cell_folder_ii])


    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    writer = FFMpegWriter(fps=1, metadata={"title": "Cell Evolution", "artist": "Pavel"})

    with writer.saving(fig, output_file, dpi=100):
        for jj in range(len(all_outlines)):
          
            ax1.cla()
            ax2.cla()

     
            ax1.set_title("Shape Evolution")
            ax2.set_title("Topographical Representation")

     
            outline = all_outlines[jj]
            topography_coords_disk = all_outlines_cMCF_topography[jj]
            curvatures = all_outlines_curvature[jj]
            time = fin_times[jj]


            norm = plt.Normalize(vmin=np.min(curvatures), vmax=np.max(curvatures))
            colors = cm.viridis(norm(curvatures))
                        
            for i in range(len(outline) - 1):
                ax1.plot3D(
                    [outline[i, 0], outline[i + 1, 0]],
                    [outline[i, 1], outline[i + 1, 1]],
                    [time, time],
                    color=colors[i],
                    linewidth=1
                )
            for i in range(len(topography_coords_disk) - 1):

                ax2.plot3D(
                    [topography_coords_disk[i, 0], topography_coords_disk[i + 1, 0]],
                    [topography_coords_disk[i, 1], topography_coords_disk[i + 1, 1]],
                    [time, time],
                    color=colors[i],
                    linewidth=1
                )

            for ax in [ax1, ax2]:
                ax.set_xlabel('X Coordinate')
                ax.set_ylabel('Y Coordinate')
                ax.set_zlabel('Time')
                ax.view_init(elev=90, azim=-90)

            writer.grab_frame()

    plt.close()


def stats_single_cell_360(direct, cell_number, time_int):
    folder_path = os.path.join(direct, f"cell_{cell_number}")

    bin_num = 72  
    bins = np.linspace(0, 360, bin_num + 1)
    bin_values = np.zeros(bin_num)
    bin_number_of = np.zeros(bin_num)
    all_outlines, _, all_outlines_cMCF_topography, all_outlines_curvature, disk_coords, fin_centr, fin_times = conformal_representation(folder_path)
    if(len(time_int)==2):
        init  = time_int[0]
        final = time_int[1]
    elif(len(time_int)==1):
        init  = time_int[0]
        final = len(all_outlines_cMCF_topography)
    elif(len(time_int)==0):
        init  = 1
        final = len(all_outlines_cMCF_topography)

    print(len(all_outlines_cMCF_topography))
    print(len(fin_centr))
    index_from = 0
    for t in range(init, final):
        disp = fin_centr[t] - fin_centr[t-1]
        dt = fin_times[t] - fin_times[t-1]
        velocity = disp / dt
        sigma = 2  
        #velocity = gaussian_filter1d(velocity, SIGMA)
        
        #theta_vel = np.arctan2(velocity[0], velocity[1])
        index_from = op_index(all_outlines[t-1], all_outlines[t], index_from)
        theta_vel = np.arctan2(all_outlines[t][index_from][0], all_outlines[t][index_from][1])
        print(f"index {index_from}, {theta_vel}, coors {all_outlines[t][index_from][0]} , {all_outlines[t][index_from][1]}")
        
        theta_topo = np.arctan2(all_outlines_cMCF_topography[t][:, 0], all_outlines_cMCF_topography[t][:, 1])
        theta_diff = np.unwrap(theta_topo - theta_vel)
        theta = np.degrees(theta_diff)
        #print(theta)
        theta = (theta + 360) % 360  

        topo_height = np.sqrt((all_outlines_cMCF_topography[t][:, 0])**2 + (all_outlines_cMCF_topography[t][:, 1])**2) -50

        for i in range(len(theta)):
            index = np.digitize(theta[i], bins) - 1 
            index = max(0, min(index, bin_num - 1))  
            bin_values[index] += topo_height[i]
            bin_number_of[index] += 1


    for i in range(bin_num):
        if bin_number_of[i] > 0:
            bin_values[i] /= bin_number_of[i]

    plt.figure(figsize=(10, 6))
    plt.xlabel('Polar angle (degrees)')
    plt.ylabel('Average Topological height')
    plt.bar(bins[:-1], bin_values, width=np.diff(bins), edgecolor='black', align='edge')

    plt.xlim(0, 360)
    plt.xticks(np.arange(0, 361, 45))  
    plt.title('Histogram of Topological height vs Polar angle (0-360 degrees)')
    plt.savefig("topo_360.png")
    plt.show()





def get_shift(all_outlines, centr, fin_times):
    shifts = []
    for jj in range(len(all_outlines)):
        if jj == 0:
            shifts.append([0, 0])
        else:
            dt = fin_times[jj] - fin_times[jj - 1]
            #dt = 0.1 #заглушка
            shift = [
                (centr[jj][0] - centr[jj - 1][0]) / dt,
                (centr[jj][1] - centr[jj - 1][1]) / dt
            ]
            shifts.append(shift)
    print(shifts)
    shifts = np.array(shifts, dtype=float)

    shifts = gaussian_filter1d(np.array(shifts), sigma=1, axis=0)
    return shifts
def smooth_topo(topography_coords_disk):
    topo_extended = np.concatenate([
                topography_coords_disk,
                topography_coords_disk,
                topography_coords_disk
            ], axis=0)

    smoothed_extended = gaussian_filter1d(topo_extended, sigma=SIGMA, axis=0)

    n = len(topography_coords_disk)
    topography_coords_disk = smoothed_extended[n:2*n]

    return topography_coords_disk

def save_circle_watershed_evolution_to_pdf(direct, cell_number, pdf_path):

    critical_curvature = 0#0.0454# 0.06
    critical_height_protrusion = 0#6.1495# 4.
    critical_height_intrusion =  0#5.0797


    all_cell_folders = [
        os.path.join(direct, ff) 
        for ff in sorted(os.listdir(direct), key=lambda x: int(x.split('_')[1]) if '_' in x and x.split('_')[1].isdigit() else 0)
    ]

    image_size = 300
    #print(all_cell_folders)
    all_outlines, _, all_outlines_cMCF_topography, all_outlines_curvature, disk_coors, centr, fin_times = conformal_representation(all_cell_folders[cell_number])


    shifts = get_shift(all_outlines, centr, fin_times)

    with PdfPages(pdf_path) as pdf:
        index_from = 0
        for jj in range(1):#(len(all_outlines)):
            shift = shifts[jj]
            outline = all_outlines[jj]
            topography_coords_disk = all_outlines_cMCF_topography[jj]
            curvatures = all_outlines_curvature[jj]
            max_curv = max(curvatures)
            time = fin_times[jj]
            ####PICTURES CONFIGURATION####################################################################
            fig = plt.figure(figsize=(24, 6))
            ax1 = fig.add_subplot(141)
            ax2 = fig.add_subplot(142)
            ax3 = fig.add_subplot(143)
            ax4 = fig.add_subplot(144)
            #############################################################################################



            ####smoothing topological map
            topography_coords_disk = smooth_topo(topography_coords_disk)
            ####OP index for alignment
            index_from = op_index(all_outlines[jj - 1], all_outlines[jj], index_from) if jj > 0 else 0         
            theta = np.arctan2(topography_coords_disk[index_from][0], topography_coords_disk[index_from][1])

            ###############################################
            circle_radius_new = [50, 0]
            for i in range(2):
                #outline repicturing
                min_val = np.min(outline[:, i])
                max_val = np.max(outline[:, i])
                relative_margin = 0.05  # 5% от размера
                margin = relative_margin * image_size
                scale = image_size - 2 * margin
                outline[:, i] = (outline[:, i] - min_val) / (max_val - min_val) * scale + margin

                #topo repicturing
                relative_margin = 0.15
                margin = relative_margin * image_size
                scale = image_size - 2 * margin
                min_val = np.min(disk_coors[:, i])
                max_val = np.max(disk_coors[:, i])
                disk_coors[:, i] = (disk_coors[:, i] - min_val) / (max_val - min_val) * scale + margin
                
                
                shift[i] = shift[i] / (max_val - min_val) * scale
                topography_coords_disk[:, i] = (topography_coords_disk[:, i] - min_val) / (max_val - min_val) * scale + margin
                circle_radius_new[i] = (circle_radius_new[i] - min_val) / (max_val - min_val) * scale + margin
                circle_center_new = (image_size/2, image_size/2)
                
            circle_radius_new = np.sqrt((circle_radius_new[0]-circle_center_new[0])**2 + (circle_radius_new[1]-circle_center_new[1])**2)
            print(circle_radius_new)

            theta_zero_index = np.arctan2(topography_coords_disk[index_from][0]-circle_center_new[0], topography_coords_disk[index_from][1]-circle_center_new[1])
            theta_topo = np.arctan2(topography_coords_disk[:, 0]-circle_center_new[0], topography_coords_disk[:, 1]-circle_center_new[1])
            theta_3 = np.degrees((theta_topo - theta_zero_index) % (2 * np.pi))


            topo_height = np.sqrt((topography_coords_disk[:, 0]-circle_center_new[0])**2 + (topography_coords_disk[:, 1]-circle_center_new[1])**2) - circle_radius_new
            print(topo_height)
            theta_3, topo_height = zip(*sorted(zip(theta_3, topo_height)))
            topo_height = np.array(topo_height)


            

            #ax3.plot(theta_3, topo_height, 'b-', linewidth=1, label="Topological Map")
            ax3.scatter(theta_3, topo_height, label="Topological Map")
            

            ################FOURIER

            xf, yf = fft_transformation(theta_3, topo_height, 30)
            plt.figure(figsize=(12, 5))



            ax4.stem(xf, yf)


            #####################################

            image_size_tuple = (300, 300)
            # topography_coords_disk[:, 0] = (topography_coords_disk[:, 0] + image_size) / 2
            # topography_coords_disk[:, 1] = (topography_coords_disk[:, 1] + image_size) / 2

            center = np.array([image_size / 2, image_size / 2])
            print(theta)
            R = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta),  np.cos(theta)]
            ])
            rr, cc = polygon(topography_coords_disk[:, 1], topography_coords_disk[:, 0], shape=image_size_tuple)
            binary_image = np.zeros(image_size_tuple, dtype=np.uint8)
            binary_image[rr, cc] = 1
            
            # ax1.imshow(binary_image)
            topography_coords_disk = (topography_coords_disk - center) @ R.T + center

            rr, cc = polygon(topography_coords_disk[:, 1], topography_coords_disk[:, 0], shape=image_size_tuple)
            binary_image = np.zeros(image_size_tuple, dtype=np.uint8)
            binary_image[rr, cc] = 1

            
            # ax2.imshow(binary_image)
            # pdf.savefig(fig)
            # plt.close(fig)
            # return 
        

            #center_topo = np.mean(topography_coords_disk, axis=0)  # (x, y)
    
            
            circle_mask = np.zeros_like(binary_image)
            print(circle_center_new)
            rr1, cc1 = disk(circle_center_new, circle_radius_new, shape=image_size_tuple)
            circle_mask[rr1, cc1] = 1

            protrusions = np.logical_and(binary_image, np.logical_not(circle_mask))
            intrusions = np.logical_and(np.logical_not(binary_image), circle_mask)

            distance_map_protrusions = gaussian_filter(distance_transform_edt(protrusions), sigma=1.0)
            distance_map_intrusions = gaussian_filter(distance_transform_edt(intrusions), sigma=1.0)

            local_maxi_protrusions = peak_local_max(distance_map_protrusions, footprint=np.ones((3, 3)), labels=protrusions, min_distance=10)
            local_maxi_intrusions = peak_local_max(distance_map_intrusions, footprint=np.ones((3, 3)), labels=intrusions, min_distance=10)

            mask_protrusions = np.zeros(distance_map_protrusions.shape, dtype=bool)
            mask_protrusions[tuple(local_maxi_protrusions.T)] = True
            markers_protrusions, _ = ndi.label(mask_protrusions)

            mask_intrusions = np.zeros(distance_map_intrusions.shape, dtype=bool)
            mask_intrusions[tuple(local_maxi_intrusions.T)] = True
            markers_intrusions, _ = ndi.label(mask_intrusions)

            labels_protrusions = watershed(-distance_map_protrusions, markers_protrusions, mask=protrusions)
            labels_intrusions = watershed(-distance_map_intrusions, markers_intrusions, mask=intrusions)

            protrusion_indices = []
            intrusion_indices = []
            color_by_index = {}
            rgb_border = np.zeros((image_size_tuple[0], image_size_tuple[1], 3))

            for idx, coord in enumerate(topography_coords_disk):
                x, y = np.round(coord).astype(int)
                if 0 <= y < image_size_tuple[0] and 0 <= x < image_size_tuple[1]:
                    if labels_protrusions[y, x] > 0:
                        rgb_border[y, x] = [1, 0, 0]
                        protrusion_indices.append(idx)
                        color_by_index[idx] = (1.0, 0.0, 0.0)
                    elif labels_intrusions[y, x] > 0:
                        rgb_border[y, x] = [0, 1, 0]
                        intrusion_indices.append(idx)
                        color_by_index[idx] = (0.0, 1.0, 0.0)
                    else:
                        color_by_index[idx] = (1.0, 1.0, 1.0)
            print(color_by_index)
            length = len(color_by_index)
            # stride = 10
            # changed = True
            # 
            

            # while changed:
            #     changed = False
            #     color_copy = color_by_index.copy()

            #     for i in range(length):
            #         #print(jj)
            #         #print("Keys in color_copy:", list(color_copy.keys()))
            #         if color_copy[i] == (1.0, 1.0, 1.0):
            #             prev_idx = (i - 1) % length
            #             prev_color = color_copy[prev_idx]

            #             for offset in range(1, stride + 1):
            #                 forward_idx = (i + offset) % length
            #                 forward_color = color_copy[forward_idx]

            #                 if forward_color != (1.0, 1.0, 1.0):
            #                     if forward_color == prev_color:
            #                         for z in range(offset):
            #                             idx_to_paint = (i + z) % length
            #                             if color_by_index[idx_to_paint] == (1.0, 1.0, 1.0):
            #                                 color_by_index[idx_to_paint] = prev_color
            #                                 changed = True
            #                     break


            # stride = 5
            # length = len(color_by_index)
            # color_copy = color_by_index.copy()

            # i = 0
            # while i < length:
            #     current_color = color_copy[i]
                
            #     if current_color in [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]: 
            #         segment_length = 0

                   
            #         while segment_length < length:
            #             forward_idx = (i + segment_length) % length
            #             forward_color = color_copy[forward_idx]
            #             if forward_color == current_color:
            #                 segment_length += 1
            #             else:
            #                 break

            #         if segment_length < stride:
            #             for j in range(segment_length):
            #                 idx_to_white = (i + j) % length
            #                 color_copy[idx_to_white] = (1.0, 1.0, 1.0)

            #         i += segment_length
            #     else:
            #         i += 1
            # color_by_index = color_copy

            # print(color_by_index)
            criteria_index = {}

            for idx, coord in enumerate(topography_coords_disk):
                x, y = coord 



                r = (np.sqrt((x - center[0])**2 + (y - center[1])**2) - circle_radius_new)*2
                curvature = curvatures[idx]

                if r >= critical_height_protrusion and np.abs(curvature) > np.abs(critical_curvature):
                    characteristic = "P"
                elif r <= -critical_height_intrusion and np.abs(curvature) > np.abs(critical_curvature):
                    characteristic = "I"
                else:
                    characteristic = "N"

                criteria_index[idx] = characteristic

            print(f"save ")
            print("".join(criteria_index.values()))
            print()

            length = len(color_by_index)
            color_copy = color_by_index.copy()
            i = 0

            while i < length:
                current_color = color_copy[i]


                if current_color in [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]:
                    segment_indices = []
                    segment_has_important_criteria = False
                    segment_length = 0

                    while segment_length < length:
                        idx = (i + segment_length) % length
                        if color_copy[idx] == current_color:
                            segment_indices.append(idx)
                            if criteria_index.get(idx) in ("P", "I"):
                                segment_has_important_criteria = True
                            segment_length += 1
                        else:
                            break

                    if not segment_has_important_criteria:
                        for idx_to_white in segment_indices:
                            color_copy[idx_to_white] = (1.0, 1.0, 1.0)

                    i += segment_length
                else:
                    i += 1

            color_by_index = color_copy


            rgb_border = np.ones((image_size_tuple[0], image_size_tuple[1], 3))
            rgb1 = np.ones((image_size_tuple[0], image_size_tuple[1], 3))

            for idx, color in color_by_index.items():
                x, y = np.round(topography_coords_disk[idx]).astype(int)
                if 0 <= y < image_size_tuple[0] and 0 <= x < image_size_tuple[1]:
                    if color == (1.0, 1.0, 1.0):
                        color = (0.5, 0.5, 0.5)
                    rgb_border[y, x] = color
                    

            for idx, coord in enumerate(disk_coors):
                x, y = np.round(coord).astype(int)
                if 0 <= y < image_size_tuple[0] and 0 <= x < image_size_tuple[1]:
                    rgb_border[y, x] = (0, 0, 0)

            for idx, color in color_by_index.items():
                x, y = np.round(outline[idx]).astype(int)
                if color == (1.0, 1.0, 1.0):
                    color = (0.0, 0.0, 0.0)
                if 0 <= y < image_size_tuple[0] and 0 <= x < image_size_tuple[1]:
                    rgb1[y, x] = color
                    

            center = np.mean(outline, axis=0)
            if shift[0] == 0.0 and shift[1] == 0.0:
                ax1.scatter(center[0], center[1], color='blue')
            else:
                ax1.arrow(center[0], center[1], shift[0] * 1.5, shift[1] * 1.5,
                          head_width=1, head_length=1, fc='blue', ec='blue', linewidth=1.5)

            ax1.imshow(rgb1)
            ax2.imshow(rgb_border)

            ax2.scatter(topography_coords_disk[index_from, 0], topography_coords_disk[index_from, 1], color='blue', s=20, zorder=10)
            ax1.scatter(outline[index_from, 0], outline[index_from, 1], color='blue', s=20, zorder=10)

            num_protrusions = 0
            num_intrusions = 0
            colors = [color_by_index[i] for i in range(length)]
            for i in range(length):
                current_color = colors[i]
                prev_color = colors[(i - 1) % length]
                if current_color == (1.0, 0.0, 0.0) and prev_color != (1.0, 0.0, 0.0):
                    num_protrusions += 1
                if current_color == (0.0, 1.0, 0.0) and prev_color != (0.0, 1.0, 0.0):
                    num_intrusions += 1

            textstr = f'Protrusions: {num_protrusions}\nIntrusions: {num_intrusions}'

            text_curv = f"Max curvature: {max_curv}"
            ################
            ax1.set_title(f"Shape Evolution (t = {time:.2f})")
            ax2.set_title("Topographical Representation")
            ax2.text(0.95, 0.05, textstr, transform=ax2.transAxes, fontsize=10,
                     verticalalignment='bottom', horizontalalignment='right',
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
            print(text_curv)
            ax1.text(0.95, 0.05, text_curv, transform=ax1.transAxes, fontsize=10,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

            for ax in [ax1, ax2]:
                ax.tick_params(colors='black')
                for spine in ax.spines.values():
                    spine.set_color('black')

            ax3.set_title("Topographical height")
            ax4.set_title("FFT frequencies analysis")
            
            ax3.set_xlim(0, 360)
            ax3.set_xticks(np.arange(0, 361, 45))
            ax3.axhline(y=0.0, color='g', linestyle='-')
            ax3.set_xlabel('Polar angle (degrees)')
            ax3.set_ylabel('Topological height')
            ax3.set_xlabel('Polar angle (degrees)')
            ax3.set_ylabel('Topological height')
            ax4.set_xlabel('Frequency FFT')
            ax4.set_ylim(0, 1)

            fig.patch.set_facecolor("white")
            ######################################3
            pdf.savefig(fig)
            plt.close(fig)

    print(f"PDF saved: {pdf_path}")
def save_circle_watershed_evolution_to_pdf_steps(direct, cell_number, pdf_path):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from skimage.draw import polygon, disk
    from skimage.segmentation import watershed
    from scipy.ndimage import distance_transform_edt, gaussian_filter
    from skimage.feature import peak_local_max
    import scipy.ndimage as ndi

    critical_curvature = 0.0454
    critical_height_protrusion = 6.1495
    critical_height_intrusion = 5.0797

    all_cell_folders = [
        os.path.join(direct, ff)
        for ff in sorted(os.listdir(direct), key=lambda x: int(x.split('_')[1]) if '_' in x and x.split('_')[1].isdigit() else 0)
    ]

    image_size = 100
    all_outlines, _, all_outlines_cMCF_topography, all_outlines_curvature, disk_coors, centr, fin_times = conformal_representation(all_cell_folders[cell_number])
    disk_coors[:, 0] = (disk_coors[:, 0] + image_size) / 2
    disk_coors[:, 1] = (disk_coors[:, 1] + image_size) / 2

    shifts = get_shift(all_outlines, centr, fin_times)

    with PdfPages(pdf_path) as pdf:
        jj = 0
        index_from = 0
        shift = shifts[jj]
        outline = all_outlines[jj]
        topography_coords_disk = smooth_topo(all_outlines_cMCF_topography[jj])
        curvatures = all_outlines_curvature[jj]
        time = fin_times[jj]
        fig, ((ax1,ax2,ax3,ax4), (ax5,ax6,ax7,ax8)) = plt.subplots(2, 4, figsize=(28, 10))

        for i in range(2):
            min_val = np.min(outline[:, i])
            max_val = np.max(outline[:, i])
            margin = 5
            scale = image_size - 2 * margin
            outline[:, i] = (outline[:, i] - min_val) / (max_val - min_val) * scale + margin
            shift[i] = shift[i] / (max_val - min_val) * scale
        image_size_tuple = (image_size, image_size)
        index_from = op_index(all_outlines[jj - 1], all_outlines[jj], index_from) if jj > 0 else 0         
        theta = np.arctan2(topography_coords_disk[index_from][0], topography_coords_disk[index_from][1]) 
        topography_coords_disk[:, 0] = (topography_coords_disk[:, 0] + image_size) / 2
        topography_coords_disk[:, 1] = (topography_coords_disk[:, 1] + image_size) / 2
        center = np.array([image_size / 2, image_size / 2])
 
        print(theta)
        
        # rr, cc = polygon(topography_coords_disk[:, 1], topography_coords_disk[:, 0], shape=image_size_tuple)
        # binary_image = np.zeros(image_size_tuple, dtype=np.uint8)
        # binary_image[rr, cc] = 1
        from scipy.ndimage import rotate
        # ax1.imshow(binary_image)
        # # R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        # # topography_coords_disk = (topography_coords_disk - center) @ R.T + center

        # binary_image = rotate(binary_image, angle=np.degrees(theta), reshape=False, order=0)
        # rr, cc = polygon(topography_coords_disk[:, 1], topography_coords_disk[:, 0], shape=image_size_tuple)
        # # binary_image = np.zeros(image_size_tuple, dtype=np.uint8)
        # # binary_image[rr, cc] = 1
        # ax2.imshow(binary_image)

        # theta = 3
        # # R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        # # topography_coords_disk = (topography_coords_disk - center) @ R.T + center

        # binary_image = rotate(binary_image, angle=np.degrees(theta), reshape=False, order=0)

        # rr, cc = polygon(topography_coords_disk[:, 1], topography_coords_disk[:, 0], shape=image_size_tuple)
        # # binary_image = np.zeros(image_size_tuple, dtype=np.uint8)
        # # binary_image[rr, cc] = 1
        # ax3.imshow(binary_image)


        # theta = 0.6
        # # R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        # # topography_coords_disk = (topography_coords_disk - center) @ R.T + center
        # binary_image = rotate(binary_image, angle=np.degrees(theta), reshape=False, order=0)
        
        # rr, cc = polygon(topography_coords_disk[:, 1], topography_coords_disk[:, 0], shape=image_size_tuple)
        # # binary_image = np.zeros(image_size_tuple, dtype=np.uint8)
        # # binary_image[rr, cc] = 1
        
        # ax4.imshow(binary_image)
        from skimage.draw import polygon2mask
        mask = polygon2mask(image_size_tuple, topography_coords_disk).astype(np.uint8)

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))
        theta1, theta2, theta3 = 0.3, 3, 0.6 
        # Оригинал
        ax1.imshow(mask, cmap='gray')
        ax1.set_title("Исходная фигура")
        ax1.axis('off')

        # Поворот 1
        binary_image = rotate(mask, angle=np.degrees(theta1), reshape=False, order=0)
        ax2.imshow(binary_image, cmap='gray')
        ax2.set_title(f"Поворот {theta1:.2f} рад")
        ax2.axis('off')

        # Поворот 2
        binary_image = rotate(binary_image, angle=np.degrees(theta2), reshape=False, order=0)
        ax3.imshow(binary_image, cmap='gray')
        ax3.set_title(f"+ ещё {theta2:.2f} рад")
        ax3.axis('off')

        # Поворот 3
        binary_image = rotate(binary_image, angle=np.degrees(theta3), reshape=False, order=0)
        ax4.imshow(binary_image, cmap='gray')
        ax4.set_title(f"+ ещё {theta3:.2f} рад")
        ax4.axis('off')


        pdf.savefig(fig)
        plt.close(fig)
        
        return 
        circle_center = (image_size // 2, image_size // 2)
        circle_radius = 25
        circle_mask = np.zeros_like(binary_image)
        rr1, cc1 = disk(circle_center, circle_radius, shape=image_size_tuple)
        circle_mask[rr1, cc1] = 1

        protrusions = np.logical_and(binary_image, np.logical_not(circle_mask))
        intrusions = np.logical_and(np.logical_not(binary_image), circle_mask)

        distance_map_protrusions = gaussian_filter(distance_transform_edt(protrusions), sigma=1.0)
        distance_map_intrusions = gaussian_filter(distance_transform_edt(intrusions), sigma=1.0)

        local_maxi_protrusions = peak_local_max(distance_map_protrusions, footprint=np.ones((3, 3)), labels=protrusions, min_distance=10)
        local_maxi_intrusions = peak_local_max(distance_map_intrusions, footprint=np.ones((3, 3)), labels=intrusions, min_distance=10)

        mask_protrusions = np.zeros(distance_map_protrusions.shape, dtype=bool)
        mask_protrusions[tuple(local_maxi_protrusions.T)] = True
        markers_protrusions, _ = ndi.label(mask_protrusions)

        mask_intrusions = np.zeros(distance_map_intrusions.shape, dtype=bool)
        mask_intrusions[tuple(local_maxi_intrusions.T)] = True
        markers_intrusions, _ = ndi.label(mask_intrusions)

        labels_protrusions = watershed(-distance_map_protrusions, markers_protrusions, mask=protrusions)
        labels_intrusions = watershed(-distance_map_intrusions, markers_intrusions, mask=intrusions)

        color_by_index = {}
        for idx, coord in enumerate(topography_coords_disk):
            x, y = np.round(coord).astype(int)
            if 0 <= y < image_size and 0 <= x < image_size:
                if labels_protrusions[y, x] > 0:
                    color_by_index[idx] = (1.0, 0.0, 0.0)  # Red
                elif labels_intrusions[y, x] > 0:
                    color_by_index[idx] = (0.0, 1.0, 0.0)  # Green
                else:
                    color_by_index[idx] = (1.0, 1.0, 1.0)  # White

        # save copy before any processing
        original_color_map = color_by_index.copy()

        # Step 1: propagation of color between same-colored endpoints
        stride = 10
        changed = True
        while changed:
            changed = False
            color_copy = color_by_index.copy()
            for i in range(len(color_copy)):
                if color_copy[i] == (1.0, 1.0, 1.0):
                    prev_color = color_copy[(i - 1) % len(color_copy)]
                    for offset in range(1, stride + 1):
                        fwd_idx = (i + offset) % len(color_copy)
                        if color_copy[fwd_idx] != (1.0, 1.0, 1.0):
                            if color_copy[fwd_idx] == prev_color:
                                for z in range(offset):
                                    to_paint = (i + z) % len(color_copy)
                                    if color_by_index[to_paint] == (1.0, 1.0, 1.0):
                                        color_by_index[to_paint] = prev_color
                                        changed = True
                            break

        # Step 2: remove short segments
        stride = 5
        color_copy = color_by_index.copy()
        i = 0
        while i < len(color_copy):
            cur_color = color_copy[i]
            if cur_color in [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]:
                seg_len = 0
                while seg_len < len(color_copy):
                    fwd_idx = (i + seg_len) % len(color_copy)
                    if color_copy[fwd_idx] == cur_color:
                        seg_len += 1
                    else:
                        break
                if seg_len < stride:
                    for j in range(seg_len):
                        color_copy[(i + j) % len(color_copy)] = (1.0, 1.0, 1.0)
                i += seg_len
            else:
                i += 1
        color_by_index = color_copy

        # Step 3: curvature + height-based criteria
        criteria_index = {}
        for idx, coord in enumerate(topography_coords_disk):
            x, y = coord
            r = (np.sqrt((x - center[0])**2 + (y - center[1])**2) - 25) * 2
            curvature = curvatures[idx]
            if r >= critical_height_protrusion and abs(curvature) > abs(critical_curvature):
                criteria_index[idx] = "P"
            elif r <= -critical_height_intrusion and abs(curvature) > abs(critical_curvature):
                criteria_index[idx] = "I"
            else:
                criteria_index[idx] = "N"

        # Remove segments without meaningful P/I points
        color_copy = color_by_index.copy()
        i = 0
        while i < len(color_copy):
            cur_color = color_copy[i]
            if cur_color in [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]:
                seg_indices = []
                has_important = False
                seg_len = 0
                while seg_len < len(color_copy):
                    idx = (i + seg_len) % len(color_copy)
                    if color_copy[idx] == cur_color:
                        seg_indices.append(idx)
                        if criteria_index.get(idx) in ("P", "I"):
                            has_important = True
                        seg_len += 1
                    else:
                        break
                if not has_important:
                    for idx_to_w in seg_indices:
                        color_copy[idx_to_w] = (1.0, 1.0, 1.0)
                i += seg_len
            else:
                i += 1
        color_by_index = color_copy
        def plot_outline(outline, size):
            img = np.ones(size)
            for x, y in np.round(outline).astype(int):
                if 0 <= y < size[0] and 0 <= x < size[1]:
                    img[y, x] = 0
            return img

        def make_rgb(color_map, coords, size):
            rgb = np.ones((size, size, 3))
            for idx, color in color_map.items():
                x, y = np.round(coords[idx]).astype(int)
                if 0 <= y < size and 0 <= x < size:
                    if color == (1.0, 1.0, 1.0):
                        color = (0.5, 0.5, 0.5)
                    rgb[y, x] = color
            return rgb

        # Визуализация 8 шагов
        steps = [
            ("1. Outline", lambda: plot_outline(outline, image_size_tuple)),
            ("2. Binary mask", lambda: binary_image),
            ("3. Circle mask", lambda: circle_mask),
            ("4. P/I masks", lambda: protrusions.astype(int) + 2 * intrusions.astype(int)),
            ("5. Watershed", lambda: (labels_protrusions > 0).astype(int) + 2 * (labels_intrusions > 0).astype(int)),
            ("6. Raw paint", lambda: make_rgb(original_color_map, topography_coords_disk, image_size)),
            ("7. After propagation+cut", lambda: make_rgb(color_copy, topography_coords_disk, image_size)),
            ("8. After criteria", lambda: make_rgb(color_by_index, topography_coords_disk, image_size)),
        ]

        fig, axes = plt.subplots(2, 4, figsize=(28, 10))
        for ax, (title, image_func) in zip(axes.flatten(), steps):
            img = image_func()
            if img.ndim == 2:
                ax.imshow(img, cmap='gray')
            else:
                ax.imshow(img)
            ax.set_title(title)
            ax.axis('off')

        fig.suptitle(f"Cell {cell_number}, t = {time:.2f}", fontsize=18)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    print("FINAL COLOR_BY_INDEX:")
    for idx in sorted(color_by_index.keys()):
        print(idx, color_by_index[idx])
    print(f"PDF saved: {pdf_path}")


def collect_protrusion_intrusion_stats_by_motion_type(direct, time_events_path):
    image_size = 100
    critical_curvature = 0.06
    critical_height =4

    all_cell_folders = [
        os.path.join(direct, ff)
        for ff in sorted(os.listdir(direct), key=lambda x: int(x.split('_')[1]) if '_' in x and x.split('_')[1].isdigit() else 0)
    ]

    stats_by_type = defaultdict(list)

    with h5py.File(time_events_path, 'r') as h5_file:
        for cell_number, cell_path in enumerate(all_cell_folders):
            print(f"Processing cell {cell_number+1}: {cell_path}")
            # if (cell_path!= '/home/pavel/cell_morphology/cells/cell_87'):
            #     continue
            all_outlines, _, all_topo, all_curvatures, disk_coors, centr, fin_times = conformal_representation(cell_path)

            disk_coors[:, 0] = (disk_coors[:, 0] + image_size) / 2
            disk_coors[:, 1] = (disk_coors[:, 1] + image_size) / 2

            shifts = get_shift(all_outlines, centr, fin_times)

            group = f"/track_{cell_number + 1}"
            if group not in h5_file:
                print(f"Missing track: {group}")
                continue

            motion_data = h5_file[group][:]
            event_indices = motion_data[0, :].astype(int) - 1
            interval_types = motion_data[2, :]

            print(f"event_indices {event_indices}")
            print(f"interval_types {interval_types}")
            index_from = 0
            for jj in range(len(all_outlines)):
                motion_type = "unclassified"
                for k in range(len(interval_types)):
                    raw_type = interval_types[k]
                    if k < len(event_indices) - 1:
                        start = event_indices[k] if k == 0 else event_indices[k] + 1
                        end = event_indices[k + 1]
                    else:
                        start = event_indices[k] if k == 0 else event_indices[k] + 1
                        end = len(all_outlines) - 1
                    if start <= jj <= end:
                        motion_type = int(raw_type) + 1 if not np.isnan(raw_type) else "unclassified"
                        break

                shift = shifts[jj]
                outline = all_outlines[jj]
                topo = smooth_topo(all_topo[jj])
                curvatures = all_curvatures[jj]
                time = fin_times[jj]
                print(f"t ={time},  motion_type = {motion_type}")
                index_from = op_index(all_outlines[jj - 1], outline, index_from) if jj > 0 else 0
                theta = np.arctan2(topo[index_from][0], topo[index_from][1])

                for i in range(2):
                    min_val = np.min(outline[:, i])
                    max_val = np.max(outline[:, i])
                    margin = 5
                    scale = image_size - 2 * margin
                    outline[:, i] = (outline[:, i] - min_val) / (max_val - min_val) * scale + margin
                    shift[i] = shift[i] / (max_val - min_val) * scale

                topo[:, 0] = (topo[:, 0] + image_size) / 2
                topo[:, 1] = (topo[:, 1] + image_size) / 2
                center = np.array([image_size / 2, image_size / 2])
                R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                topo = (topo - center) @ R.T + center

                rr, cc = polygon(topo[:, 1], topo[:, 0], shape=(image_size, image_size))
                binary_image = np.zeros((image_size, image_size), dtype=np.uint8)
                binary_image[rr, cc] = 1
                image_size_tuple = (100, 100)
                circle_center_new = ((0 + image_size_tuple[1]) / 2, (0 + image_size_tuple[0]) / 2)
                circle_radius_new = 25#50 / 2
                circle_mask = np.zeros_like(binary_image)
                rr1, cc1 = disk(circle_center_new, circle_radius_new, shape=image_size_tuple)
                circle_mask[rr1, cc1] = 1

                protrusions = np.logical_and(binary_image, np.logical_not(circle_mask))
                intrusions = np.logical_and(np.logical_not(binary_image), circle_mask)

                distance_map_protrusions = gaussian_filter(distance_transform_edt(protrusions), sigma=1.0)
                distance_map_intrusions = gaussian_filter(distance_transform_edt(intrusions), sigma=1.0)

                local_maxi_protrusions = peak_local_max(distance_map_protrusions, footprint=np.ones((3, 3)), labels=protrusions, min_distance=10)
                local_maxi_intrusions = peak_local_max(distance_map_intrusions, footprint=np.ones((3, 3)), labels=intrusions, min_distance=10)

                mask_protrusions = np.zeros(distance_map_protrusions.shape, dtype=bool)
                mask_protrusions[tuple(local_maxi_protrusions.T)] = True
                markers_protrusions, _ = ndi.label(mask_protrusions)

                mask_intrusions = np.zeros(distance_map_intrusions.shape, dtype=bool)
                mask_intrusions[tuple(local_maxi_intrusions.T)] = True
                markers_intrusions, _ = ndi.label(mask_intrusions)

                labels_protrusions = watershed(-distance_map_protrusions, markers_protrusions, mask=protrusions)
                labels_intrusions = watershed(-distance_map_intrusions, markers_intrusions, mask=intrusions)

                topo = np.clip(topo, 0, image_size - 1)
                color_by_index = {}
                for idx, coord in enumerate(topo):
                    x, y = np.round(coord).astype(int)
                    if 0 <= y < image_size and 0 <= x < image_size:
                        if labels_protrusions[y, x] > 0:
                            color_by_index[idx] = (1.0, 0.0, 0.0)
                        elif labels_intrusions[y, x] > 0:
                            color_by_index[idx] = (0.0, 1.0, 0.0)
                        else:
                            color_by_index[idx] = (1.0, 1.0, 1.0)

                stride = 10
                changed = True
                length = len(color_by_index)

                while changed:
                    changed = False
                    color_copy = color_by_index.copy()

                    for i in range(length):
                        if color_copy[i] == (1.0, 1.0, 1.0):
                            prev_idx = (i - 1) % length
                            prev_color = color_copy[prev_idx]

                            for offset in range(1, stride + 1):
                                forward_idx = (i + offset) % length
                                forward_color = color_copy[forward_idx]

                                if forward_color != (1.0, 1.0, 1.0):
                                    if forward_color == prev_color:
                                        for z in range(offset):
                                            idx_to_paint = (i + z) % length
                                            if color_by_index[idx_to_paint] == (1.0, 1.0, 1.0):
                                                color_by_index[idx_to_paint] = prev_color
                                                changed = True
                                    break


                stride = 5
                length = len(color_by_index)
                color_copy = color_by_index.copy()

                i = 0
                while i < length:
                    current_color = color_copy[i]
                    
                    if current_color in [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]: 
                        segment_length = 0

                    
                        while segment_length < length:
                            forward_idx = (i + segment_length) % length
                            forward_color = color_copy[forward_idx]
                            if forward_color == current_color:
                                segment_length += 1
                            else:
                                break

                        if segment_length < stride:
                            for j in range(segment_length):
                                idx_to_white = (i + j) % length
                                color_copy[idx_to_white] = (1.0, 1.0, 1.0)

                        i += segment_length
                    else:
                        i += 1
                color_by_index = color_copy


                criteria_index = {}

                for idx, coord in enumerate(topo):
                    x, y = coord
                    r = (np.sqrt((x - center[0])**2 + (y - center[1])**2) - 25) * 2
                    curvature = curvatures[idx]

                    if r >= critical_height and np.abs(curvature) > np.abs(critical_curvature):
                        characteristic = "P"
                    elif r <= -critical_height and np.abs(curvature) > np.abs(critical_curvature):
                        characteristic = "I"
                    else:
                        characteristic = "N"

                    criteria_index[idx] = characteristic

                length = len(color_by_index)
                color_copy = color_by_index.copy()
                i = 0

                while i < length:
                    current_color = color_copy[i]

                    if current_color in [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]:
                        segment_indices = []
                        segment_has_important_criteria = False
                        segment_length = 0

                        while segment_length < length:
                            idx = (i + segment_length) % length
                            if color_copy[idx] == current_color:
                                segment_indices.append(idx)
                                if criteria_index.get(idx) in ("P", "I"):
                                    segment_has_important_criteria = True
                                segment_length += 1
                            else:
                                break

                        if not segment_has_important_criteria:
                            for idx_to_white in segment_indices:
                                color_copy[idx_to_white] = (1.0, 1.0, 1.0)

                        i += segment_length
                    else:
                        i += 1

                color_by_index = color_copy

                num_prot = 0
                num_intr = 0
                colors = [color_by_index[i] for i in range(len(color_by_index))]
                #print(colors)
                
                for i in range(len(colors)):
                    cur_color = colors[i]
                    prev_color = colors[(i - 1) % len(colors)]
                    if cur_color == (1.0, 0.0, 0.0) and prev_color != cur_color:
                        num_prot += 1
                    if cur_color == (0.0, 1.0, 0.0) and prev_color != cur_color:
                        num_intr += 1
                #print(num_prot)
                #print(num_intr)

                stats_by_type[motion_type].append((num_prot, num_intr))
              

    # Вывод статистики и расчет стандартного отклонения
    label_map = {
        1: "Immobile",
        2: "Confined Diffusion",
        3: "Free Diffusion",
        4: "Directed Diffusion",
        "unclassified": "Unclassified"
    }

    print("\n📊 Average protrusions/intrusions per frame by motion type:\n")
    keys = [1, 2, 3, 4, "unclassified"]
    motion_types = []
    protrusions_vals = []
    print(protrusions_vals)
    intrusions_vals = []
    print(intrusions_vals)
    std_protrusions_vals = []
    std_intrusions_vals = []
    
    for key in keys:
        values = np.array(stats_by_type.get(key, []))
        if len(values) > 0:
            mean_prot = np.mean(values[:, 0])
            mean_intr = np.mean(values[:, 1])
            std_prot = np.std(values[:, 0])  # Среднеквадратичное отклонение для протрузий
            std_intr = np.std(values[:, 1])  # Среднеквадратичное отклонение для интрузий
        else:
            mean_prot = 0
            mean_intr = 0
            std_prot = 0
            std_intr = 0
        
        label = label_map.get(key, str(key))
        print(f"{label}:")
        print(f"  Avg. Protrusions: {mean_prot:.2f} ± {std_prot:.2f}")
        print(f"  Avg. Intrusions : {mean_intr:.2f} ± {std_intr:.2f}\n")

        # Подготовка данных для графика
        motion_types.append(label)
        protrusions_vals.append(mean_prot)
        intrusions_vals.append(mean_intr)
        std_protrusions_vals.append(std_prot)
        std_intrusions_vals.append(std_intr)

    # Построение графика с точками и вертикальными отрезками
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(motion_types))

    # Центральные точки для среднего значения
    ax.scatter(x, protrusions_vals, color='cyan', label='Mean Protrusions', zorder=5)
    ax.scatter(x, intrusions_vals, color='magenta', label='Mean Intrusions', zorder=5)
    protrusions_vals = np.array(protrusions_vals)
    std_protrusions_vals = np.array(std_protrusions_vals)
    intrusions_vals = np.array(intrusions_vals)
    std_intrusions_vals = np.array(std_intrusions_vals)

    # Вертикальные отрезки для стандартного отклонения
    ax.vlines(x, protrusions_vals - std_protrusions_vals, protrusions_vals + std_protrusions_vals, color='cyan', linewidth=2)
    ax.vlines(x, intrusions_vals - std_intrusions_vals, intrusions_vals + std_intrusions_vals, color='magenta', linewidth=2, linestyles = "dashed")

    ax.set_xticks(x)
    ax.set_xticklabels(motion_types)
    ax.set_xlabel('Motion Type')
    ax.set_ylabel('Number of Protrusion and Intrusions')
    ax.set_title('Average Protrusions and Intrusions')

    ax.legend()
    
     
    from scipy import stats

    # Сбор данных для "Free Diffusion" и "Directed Diffusion"
    free_diffusion_protrusions = []
    directed_diffusion_protrusions = []
    free_diffusion_intrusions = []
    directed_diffusion_intrusions = []

    # Пройдем по всем ключам и соберем значения для t-теста
    for key in [3, 4]:  # 3 - Free Diffusion, 4 - Directed Diffusion
        values = np.array(stats_by_type.get(key, []))
        if len(values) > 0:
            if key == 3:  # Free Diffusion
                free_diffusion_protrusions.extend(values[:, 0])
                free_diffusion_intrusions.extend(values[:, 1])
            elif key == 4:  # Directed Diffusion
                directed_diffusion_protrusions.extend(values[:, 0])
                directed_diffusion_intrusions.extend(values[:, 1])

    # Применение t-теста
    t_stat_prot, p_val_prot = stats.ttest_ind(free_diffusion_protrusions, directed_diffusion_protrusions)
    t_stat_intr, p_val_intr = stats.ttest_ind(free_diffusion_intrusions, directed_diffusion_intrusions)

    # Вывод результатов t-теста
    print("\n📊 T-test results between Free Diffusion and Directed Diffusion:")
    print(f"T-statistic for Protrusions: {t_stat_prot}, p-value: {p_val_prot}")
    print(f"T-statistic for Intrusions: {t_stat_intr}, p-value: {p_val_intr}")
    
    
    plt.tight_layout()
    plt.show()



# def plot_shape_and_topo_3D(direct, cell_number):
#     all_cell_folders = [
#         os.path.join(direct, ff) 
#         for ff in sorted(os.listdir(direct), key=lambda x: int(x.split('_')[1]) if '_' in x and x.split('_')[1].isdigit() else 0)
#     ]

#     all_outlines, _, all_topo, _, _, _, fin_times = conformal_representation(all_cell_folders[cell_number])

#     fig = plt.figure(figsize=(16, 8))
#     ax1 = fig.add_subplot(121, projection='3d')  # Outline
#     ax2 = fig.add_subplot(122, projection='3d')  # Topography
#     fig.suptitle(f"Cell {cell_number}: 3D Shape and Topographical Evolution", fontsize=16)

#     image_size = 200

#     for jj in range(len(all_outlines)):
#         outline = all_outlines[jj]
#         topo = all_topo[jj]
#         time = fin_times[jj]

#         # Масштабируем outline
#         for i in range(2):
#             min_val = np.min(outline[:, i])
#             max_val = np.max(outline[:, i])
#             outline[:, i] = (outline[:, i] - min_val) / (max_val - min_val) * (image_size - 1)

#         # Масштабируем topography_coords_disk
#         topo[:, 0] = (topo[:, 0] + image_size) / 2
#         topo[:, 1] = (topo[:, 1] + image_size) / 2

#         # Сегментация для topography
#         rr, cc = polygon(topo[:, 1], topo[:, 0], shape=(image_size, image_size))
#         binary_image = np.zeros((image_size, image_size), dtype=np.uint8)
#         binary_image[rr, cc] = 1

#         circle_center_new = ((0 + image_size_tuple[1]) / 2, (0 + image_size_tuple[0]) / 2)
#         circle_radius_new = 50 / 2
#         circle_mask = np.zeros_like(binary_image)
#         rr1, cc1 = disk(circle_center_new, circle_radius_new, shape=image_size_tuple)
#         circle_mask[rr1, cc1] = 1


#         protrusions = np.logical_and(binary_image, np.logical_not(circle_mask))
#         intrusions = np.logical_and(np.logical_not(binary_image), circle_mask)

#         dist_prot = gaussian_filter(distance_transform_edt(protrusions), sigma=1.0)
#         dist_intr = gaussian_filter(distance_transform_edt(intrusions), sigma=1.0)

#         max_prot = peak_local_max(dist_prot, footprint=np.ones((3, 3)), labels=protrusions, min_distance=10)
#         max_intr = peak_local_max(dist_intr, footprint=np.ones((3, 3)), labels=intrusions, min_distance=10)

#         mask_prot = np.zeros_like(dist_prot, dtype=bool)
#         mask_prot[tuple(max_prot.T)] = True
#         markers_prot, _ = ndi.label(mask_prot)

#         mask_intr = np.zeros_like(dist_intr, dtype=bool)
#         mask_intr[tuple(max_intr.T)] = True
#         markers_intr, _ = ndi.label(mask_intr)

#         labels_prot = watershed(-dist_prot, markers_prot, mask=protrusions)
#         labels_intr = watershed(-dist_intr, markers_intr, mask=intrusions)

#         # Построим color_by_index
#         color_by_index = {}
#         for idx, coord in enumerate(topo):
#             x, y = np.round(coord).astype(int)
#             if 0 <= y < image_size and 0 <= x < image_size:
#                 if labels_prot[y, x] > 0:
#                     color_by_index[idx] = (1.0, 0.0, 0.0)
#                 elif labels_intr[y, x] > 0:
#                     color_by_index[idx] = (0.0, 1.0, 0.0)
#                 else:
#                     color_by_index[idx] = (0.0, 0.0, 0.0)

#         # === Outline view (ax1)
#         z = np.full(len(outline), time)
#         for idx, (x, y) in enumerate(outline):
#             color = color_by_index.get(idx, (1.0, 1.0, 1.0))  # белый — если нет цвета
#             ax1.plot([x], [y], [z[idx]], marker='o', markersize=2, color=color)

#         # === Topographical view (ax2)
#         for idx, (x, y) in enumerate(topo):
#             color = color_by_index.get(idx, (1.0, 1.0, 1.0))
#             ax2.plot([x], [y], [time], marker='o', markersize=2, color=color)

#     # Настройки графиков
#     for ax, label in zip([ax1, ax2], ["Outline Representation", "Topographical Representation"]):
#         ax.set_xlabel("X")
#         ax.set_ylabel("Y")
#         ax.set_zlabel("Time")
#         ax.set_title(label)
    
#     for ax, label in zip([ax3], ["Outline Representation", "Topographical Representation"]):
#         ax.set_xlabel("X")
#         ax.set_ylabel("Y")
#         ax.set_zlabel("Time")
#         ax.set_title(label)

#     plt.tight_layout()
#     plt.show()



def fft_spare_l2_matrix(direct, cell_number):
    all_cell_folders = [
        os.path.join(direct, ff) 
        for ff in sorted(os.listdir(direct), key=lambda x: int(x.split('_')[1]) if '_' in x and x.split('_')[1].isdigit() else 0)
    ]

    all_outlines, _, all_outlines_cMCF_topography, all_outlines_curvature, disk_coors, centr, fin_times = conformal_representation(all_cell_folders[cell_number])
    num = len(np.sort(os.listdir(all_cell_folders[cell_number])))
    
    num = len(all_outlines)
    l2_matrix = np.zeros((num, num))
    
    
    fft_spectra = []
    for jj in range(num):
        topography_coords_disk = smooth_topo(all_outlines_cMCF_topography[jj])

        index_from = op_index(all_outlines[jj - 1], all_outlines[jj], index_from) if jj > 0 else 0
        theta_zero_index = np.arctan2(topography_coords_disk[index_from][0], topography_coords_disk[index_from][1])
        theta_topo = np.arctan2(topography_coords_disk[:, 0], topography_coords_disk[:, 1])
        theta_3 = np.degrees((theta_topo - theta_zero_index) % (2 * np.pi))

        topo_height = np.sqrt((topography_coords_disk[:, 0])**2 + (topography_coords_disk[:, 1])**2) - 50
        theta_3, topo_height = zip(*sorted(zip(theta_3, topo_height)))
        topo_height = np.array(topo_height)

        xf, yf = fft_transformation(theta_3, topo_height, 30)
        fft_spectra.append((xf, yf))

  
    for i in range(num):
        for j in range(i, num):
            xf_i, yf_i = fft_spectra[i]
            xf_j, yf_j = fft_spectra[j]
            norm = l2_fourier(xf_i, yf_i, xf_j, yf_j)
            l2_matrix[i, j] = norm
            l2_matrix[j, i] = norm  

    
    plt.figure(figsize=(8, 6))
    #im = plt.imshow(l2_matrix, cmap='viridis', origin='upper')
    embedding = reducer.fit_transform(l2_matrix)
    import seaborn as sns
    sc = plt.scatter(
        embedding[:, 0],
        embedding[:, 1],c=fin_times)
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('UMAP projection ', fontsize=24)
    plt.colorbar(sc, label='Time')
    plt.show()

def fft_spare_l2_matrix_whole_dataset(direct, migration_types_to_keep=["Free Diffusion", "Confined Diffusion","Immobile", "Directed Diffusion", "Unclassified"]):

    label_map = {
        1: "Immobile",
        2: "Confined Diffusion",
        3: "Free Diffusion",
        4: "Directed Diffusion",
        "unclassified": "Unclassified"
    }

    all_cell_folders = [
        os.path.join(direct, ff)
        for ff in sorted(os.listdir(direct), key=lambda x: int(x.split('_')[1]) if '_' in x and x.split('_')[1].isdigit() else 0)
    ]

    fft_spectra = []
    migration_type = []
    theta_3_array = []
    topo_array = []
    included_indices = []

    time_events_path = "time_events_95.h5"
    with h5py.File(time_events_path, 'r') as h5_file:
        idx_counter = 0
        for cell_number in range(len(all_cell_folders)):
            cell_path = all_cell_folders[cell_number]
            all_outlines, _, all_topo, _, _, _, fin_times = conformal_representation(cell_path)
            num = len(all_outlines)

            group = f"/track_{cell_number + 1}"
            if group not in h5_file:
                print(f"Skipping {group} (no data)")
                continue

            motion_data = h5_file[group][:]
            event_indices = motion_data[0, :].astype(int) - 1
            interval_types = motion_data[2, :]

            for jj in range(num):
                motion_type = "unclassified"
                for k in range(len(interval_types)):
                    raw_type = interval_types[k]
                    if k < len(event_indices) - 1:
                        start = event_indices[k] if k == 0 else event_indices[k] + 1
                        end = event_indices[k + 1]
                    else:
                        start = event_indices[k] if k == 0 else event_indices[k] + 1
                        end = num - 1
                    if start <= jj <= end:
                        motion_type = int(raw_type) + 1 if not np.isnan(raw_type) else "unclassified"
                        break

                label = label_map[motion_type]
                if label not in migration_types_to_keep:
                    idx_counter += 1
                    continue

                topography_coords_disk = smooth_topo(all_topo[jj])
                index_from = op_index(all_outlines[jj - 1], all_outlines[jj], 0) if jj > 0 else 0
                theta_zero_index = np.arctan2(topography_coords_disk[index_from][0], topography_coords_disk[index_from][1])
                theta_topo = np.arctan2(topography_coords_disk[:, 0], topography_coords_disk[:, 1])
                theta_3 = np.degrees((theta_topo - theta_zero_index) % (2 * np.pi))

                topo_height = np.sqrt((topography_coords_disk[:, 0])**2 + (topography_coords_disk[:, 1])**2) - 50
                theta_3, topo_height = zip(*sorted(zip(theta_3, topo_height)))
                topo_height = np.array(topo_height)

                theta_3_array.append(theta_3)
                topo_array.append(topo_height)
                migration_type.append(label)
                included_indices.append(idx_counter)
                idx_counter += 1

    total = len(theta_3_array)
    l2_matrix = np.zeros((total, total))

    for i in range(total):
        topo_i = height_interpolation(theta_3_array[i], topo_array[i], number_of_points = 150)
        for j in range(i, total):
            topo_j = height_interpolation(theta_3_array[j], topo_array[j], number_of_points = 150)
            n_points = len(topo_j)

            min_norm = np.inf
            for shift in range(n_points):
                shifted_topo = np.roll(topo_j, shift)
                norm = euclidean(shifted_topo, topo_i)
                if norm < min_norm:
                    min_norm = norm

            l2_matrix[i, j] = min_norm
            l2_matrix[j, i] = min_norm

    np.save("l2_matrix_new.npy", l2_matrix)

    unique_types = sorted(set(migration_type))
    type_to_index = {label: idx for idx, label in enumerate(unique_types)}
    index_labels = [type_to_index[label] for label in migration_type]

    reducer = umap.UMAP(metric="precomputed", random_state=42)
    embedding = reducer.fit_transform(l2_matrix)

    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10(index_labels)
    sc = plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=colors,
        s=10
    )
    plt.title("UMAP Projection Colored by Migration Type", fontsize=16)
    plt.gca().set_aspect('equal', 'datalim')

    legend_handles = [
        mpatches.Patch(color=plt.cm.tab10(type_to_index[label]), label=label)
        for label in unique_types
    ]
    plt.legend(handles=legend_handles, title="Migration Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"fourier_matrix_filtered_umap_new.png")

def fft_spare_l2_matrix_whole_dataset_cutted(direct, l2_matrix_path, migration_types_to_keep=["Free Diffusion", "Confined Diffusion"]):
    label_map = {
        1: "Immobile",
        2: "Confined Diffusion",
        3: "Free Diffusion",
        4: "Directed Diffusion",
        "unclassified": "Unclassified"
    }


    l2_matrix = np.load(l2_matrix_path)

    all_cell_folders = [
        os.path.join(direct, ff)
        for ff in sorted(os.listdir(direct), key=lambda x: int(x.split('_')[1]) if '_' in x and x.split('_')[1].isdigit() else 0)
    ]

    migration_type = []

    time_events_path = "time_events_95.h5"
    with h5py.File(time_events_path, 'r') as h5_file:
        for cell_number in range(len(all_cell_folders)):
            cell_path = all_cell_folders[cell_number]
            num = len(np.sort(os.listdir(all_cell_folders[cell_number])))
    

            group = f"/track_{cell_number + 1}"
            if group not in h5_file:
                print(f"Skipping {group} (no data)")
                continue

            motion_data = h5_file[group][:]
            event_indices = motion_data[0, :].astype(int) - 1
            interval_types = motion_data[2, :]

            for jj in range(num):
                # Определение типа миграции для jj
                motion_type = "unclassified"
                for k in range(len(interval_types)):
                    raw_type = interval_types[k]
                    if k < len(event_indices) - 1:
                        start = event_indices[k] if k == 0 else event_indices[k] + 1
                        end = event_indices[k + 1]
                    else:
                        start = event_indices[k] if k == 0 else event_indices[k] + 1
                        end = num - 1
                    if start <= jj <= end:
                        motion_type = int(raw_type) + 1 if not np.isnan(raw_type) else "unclassified"
                        break
                
                # Фильтрация только для "Free Diffusion" и "Directed Diffusion"
                if label_map[motion_type] not in migration_types_to_keep:
                    continue


                migration_type.append(label_map[motion_type])

    # Фильтрация: только для "Free Diffusion" и "Directed Diffusion"
    unique_types = sorted(set(migration_type))
    type_to_index = {label: idx for idx, label in enumerate(unique_types)}
    index_labels = [type_to_index[label] for label in migration_type]

    # Индексы для нужных типов миграции
    indices_to_keep = [i for i, label in enumerate(migration_type) if label in migration_types_to_keep]

    # Вырезаем из матрицы L2 только нужные строки и столбцы
    filtered_l2_matrix = l2_matrix[np.ix_(indices_to_keep, indices_to_keep)]

    # UMAP проекция
    reducer = umap.UMAP(metric="precomputed", random_state=42)
    embedding = reducer.fit_transform(filtered_l2_matrix)

    # Визуализация
    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10([type_to_index[migration_type[i]] for i in indices_to_keep])
    sc = plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=colors,
        s=10
    )
    plt.title("UMAP Projection Colored by Migration Type", fontsize=16)
    plt.gca().set_aspect('equal', 'datalim')

    # Легенда
    legend_handles = [
        mpatches.Patch(color=plt.cm.tab10(type_to_index[label]), label=label)
        for label in migration_types_to_keep
    ]
    plt.legend(handles=legend_handles, title="Migration Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"fourier_matrix_filtered_free_confined.png")

def fft_spare_l2_matrix_whole_dataset_num_protrusions(direct, l2_matrix_path):

    l2_matrix = np.load(l2_matrix_path)

    all_cell_folders = [
        os.path.join(direct, ff)
        for ff in sorted(os.listdir(direct), key=lambda x: int(x.split('_')[1]) if '_' in x and x.split('_')[1].isdigit() else 0)
    ]

    migration_type = []

    time_events_path = "time_events_95.h5"
    with h5py.File(time_events_path, 'r') as h5_file:
        for cell_number in range(len(all_cell_folders)):
            cell_path = all_cell_folders[cell_number]
            num = len(np.sort(os.listdir(all_cell_folders[cell_number])))
    

            group = f"/track_{cell_number + 1}"
            if group not in h5_file:
                print(f"Skipping {group} (no data)")
                continue

            motion_data = h5_file[group][:]
            event_indices = motion_data[0, :].astype(int) - 1
            interval_types = motion_data[2, :]

            for jj in range(num):
                # Определение типа миграции для jj
                motion_type = "unclassified"
                for k in range(len(interval_types)):
                    raw_type = interval_types[k]
                    if k < len(event_indices) - 1:
                        start = event_indices[k] if k == 0 else event_indices[k] + 1
                        end = event_indices[k + 1]
                    else:
                        start = event_indices[k] if k == 0 else event_indices[k] + 1
                        end = num - 1
                    if start <= jj <= end:
                        motion_type = int(raw_type) + 1 if not np.isnan(raw_type) else "unclassified"
                        break
                
 

    unique_types = sorted(set(migration_type))
    type_to_index = {label: idx for idx, label in enumerate(unique_types)}
    index_labels = [type_to_index[label] for label in migration_type]


    filtered_l2_matrix = l2_matrix[np.ix_(indices_to_keep, indices_to_keep)]

    # UMAP проекция
    reducer = umap.UMAP(metric="precomputed", random_state=42)
    embedding = reducer.fit_transform(filtered_l2_matrix)

    # Визуализация
    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10([type_to_index[migration_type[i]] for i in indices_to_keep])
    sc = plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=colors,
        s=10
    )
    plt.title("UMAP Projection Colored by Migration Type", fontsize=16)
    plt.gca().set_aspect('equal', 'datalim')

    # Легенда
    legend_handles = [
        mpatches.Patch(color=plt.cm.tab10(type_to_index[label]), label=label)
        for label in migration_types_to_keep
    ]
    plt.legend(handles=legend_handles, title="Migration Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"fourier_matrix_filtered_free_confined.png")

def fft_spare_l2_matrix_whole_dataset_riemann(direct):
    
    label_map = {
        1: "Immobile",
        2: "Confined Diffusion",
        3: "Free Diffusion",
        4: "Directed Diffusion",
        "unclassified": "Unclassified"
    }

    all_cell_folders = [
        os.path.join(direct, ff)
        for ff in sorted(os.listdir(direct), key=lambda x: int(x.split('_')[1]) if '_' in x and x.split('_')[1].isdigit() else 0)
    ]

    migration_type = []
    all_outlines_list = []
    time_events_path = "time_events_95.h5"
    with h5py.File(time_events_path, 'r') as h5_file:
        for cell_number in range(1, len(all_cell_folders)):#(len(all_cell_folders)):
            
            cell_path = os.path.join(direct, f"cell_{cell_number+1}")
            number_of_frames = sum(
                os.path.isdir(os.path.join(cell_path, entry)) for entry in os.listdir(cell_path)
            )
            #num = len(np.sort(os.listdir(all_cell_folders[cell_number])))

            group = f"/track_{cell_number + 1}"
            if group not in h5_file:
                print(f"Skipping {group} (no data)")
                continue

            motion_data = h5_file[group][:]
            event_indices = motion_data[0, :].astype(int) - 1
            interval_types = motion_data[2, :]

            for jj in range(number_of_frames):
                frame_path = os.path.join(cell_path, f"frame_{jj+1}")
               
                border_cell = np.load(os.path.join(frame_path, "outline.npy"))
                
                num = number_of_frames
                # Определение типа миграции для jj
                motion_type = "unclassified"
                for k in range(len(interval_types)):
                    raw_type = interval_types[k]
                    if k < len(event_indices) - 1:
                        start = event_indices[k] if k == 0 else event_indices[k] + 1
                        end = event_indices[k + 1]
                    else:
                        start = event_indices[k] if k == 0 else event_indices[k] + 1
                        end = num - 1
                    if start <= jj <= end:
                        motion_type = int(raw_type) + 1 if not np.isnan(raw_type) else "unclassified"
                        break


                all_outlines_list.append(border_cell)
                print(all_outlines_list)
                migration_type.append(label_map[motion_type])

    total = len(all_outlines_list)
    
    l2_matrix = np.zeros((total, total))
    CURVES_SPACE_ELASTIC = DiscreteCurvesStartingAtOrigin(
        ambient_dim=2, k_sampling_points=1000, equip=False
    )
    CURVES_SPACE_ELASTIC.equip_with_metric(ElasticMetric, a=1, b=0.5)
    for i in range(total):
        for j in range(i, total):
            all_outlines_i = all_outlines_list[i]
            all_outlines_j= all_outlines_list[j]
            
            cell_interpolation_i = interpolate(all_outlines_i, 1000)
            cell_preprocess_i = preprocess(cell_interpolation_i)
            cell_interpolation_j = interpolate(all_outlines_j, 1000)
            cell_preprocess_j = preprocess(cell_interpolation_j)
            

            aligned_border = align(
                cell_preprocess_i, cell_preprocess_j, rescale=True, rotation=False, reparameterization=True, k_sampling_points=1000
            )
            
            iter_distance = CURVES_SPACE_ELASTIC.metric.dist(
                CURVES_SPACE_ELASTIC.projection(cell_preprocess_i), 
                CURVES_SPACE_ELASTIC.projection(cell_preprocess_j)
            )

            l2_matrix[i, j] = iter_distance
            l2_matrix[j, i] = iter_distance

    np.save("l2_matrix_riemann.npy", l2_matrix)
    unique_types = sorted(set(migration_type))
    type_to_index = {label: idx for idx, label in enumerate(unique_types)}
    index_labels = [type_to_index[label] for label in migration_type]

    # UMAP проекция
    reducer = umap.UMAP(metric="precomputed", random_state=42)
    embedding = reducer.fit_transform(l2_matrix)

    # Визуализация
    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10(index_labels)
    sc = plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=colors,
        s=10
    )
    plt.title("UMAP Projection Colored by Migration Type Riemann", fontsize=16)
    plt.gca().set_aspect('equal', 'datalim')

    # Легенда
    legend_handles = [
        mpatches.Patch(color=plt.cm.tab10(type_to_index[label]), label=label)
        for label in unique_types
    ]
    plt.legend(handles=legend_handles, title="Migration Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"fourier_matrix_free_directed_riemann.png")

def collect_statistics(direct):
    all_cell_folders = [
        os.path.join(direct, ff)
        for ff in sorted(os.listdir(direct), key=lambda x: int(x.split('_')[1]) if '_' in x and x.split('_')[1].isdigit() else 0)
    ]

    def collect_protrusions_height(topo_height):
        result, n, i = [], len(topo_height), 0
        while i < n:
            if topo_height[i] <= 0:
                i += 1
                continue
            vals, start = [], i
            while topo_height[i % n] > 0:
                vals.append(topo_height[i % n])
                i += 1
                if i - start >= n: break
            if vals: result.append(max(vals))
        return result

    def collect_intrusions_height(topo_height):
        result, n, i = [], len(topo_height), 0
        while i < n:
            if topo_height[i] >= 0:
                i += 1
                continue
            vals, start = [], i
            while topo_height[i % n] < 0:
                vals.append(abs(topo_height[i % n]))
                i += 1
                if i - start >= n: break
            if vals: result.append(max(vals))
        return result

    def collect_protrusions_width(topo_height):
        result, n, i = [], len(topo_height), 0
        while i < n:
            if topo_height[i] <= 0:
                i += 1
                continue
            length, start = 0, i
            while topo_height[i % n] > 0:
                length += 1
                i += 1
                if i - start >= n: break
            result.append(length)
        return result

    def collect_intrusions_width(topo_height):
        result, n, i = [], len(topo_height), 0
        while i < n:
            if topo_height[i] >= 0:
                i += 1
                continue
            length, start = 0, i
            while topo_height[i % n] < 0:
                length += 1
                i += 1
                if i - start >= n: break
            result.append(length)
        return result

    def print_box_stats(data, label):
        fig, ax = plt.subplots()
        bp = ax.boxplot(data, patch_artist=True)


        box_path = bp['boxes'][0].get_path().vertices
        q1 = box_path[0][1]  # нижняя граница
        q3 = box_path[2][1]  # верхняя граница

        median = bp['medians'][0].get_ydata()[0]
        whisker_low = bp['whiskers'][0].get_ydata()[1]
        whisker_high = bp['whiskers'][1].get_ydata()[1]

        print(f"{label}:")
        print(f"  Q1 = {q1:.4f}")
        print(f"  Median = {median:.4f}")
        print(f"  Q3 = {q3:.4f}")
        print(f"  Lower whisker = {whisker_low:.4f}")
        print(f"  Upper whisker = {whisker_high:.4f}")
        print("")
        plt.close(fig)

    # Общие списки
    all_prot_h = []
    all_prot_w = []
    all_intr_h = []
    all_intr_w = []
    all_curvatures = []

    for folder in all_cell_folders:
        all_outlines, _, all_topo, all_curvs, _, _, fin_times = conformal_representation(folder)
        index_from = 0

        for jj in range(len(all_outlines)):
            topo = all_topo[jj]
            curv = all_curvs[jj]

            index_from = op_index(all_outlines[jj - 1], all_outlines[jj], index_from) if jj > 0 else 0
            theta0 = np.arctan2(topo[index_from][0], topo[index_from][1])
            theta = np.arctan2(topo[:, 0], topo[:, 1])
            theta_deg = np.degrees((theta - theta0) % (2 * np.pi))

            topo_height = np.sqrt(topo[:, 0] ** 2 + topo[:, 1] ** 2) - 50

            theta_deg, topo_height = zip(*sorted(zip(theta_deg, topo_height)))
            theta_deg = np.array(theta_deg)
            topo_height = np.array(topo_height)

            if theta_deg[0] != 0 or theta_deg[-1] != 360:
                theta_deg = np.append(theta_deg, 360)
                topo_height = np.append(topo_height, topo_height[0])

            spline = CubicSpline(theta_deg, topo_height, bc_type='periodic')
            x_uniform = np.linspace(0, 360, 1000)
            topo_height = spline(x_uniform)

            all_prot_h.extend(collect_protrusions_height(topo_height))
            all_intr_h.extend(collect_intrusions_height(topo_height))
            all_prot_w.extend(collect_protrusions_width(topo_height))
            all_intr_w.extend(collect_intrusions_width(topo_height))
            all_curvatures.extend(np.abs(curv))

    # Построение боксплотов и вывод статистики
    fig, axes = plt.subplots(1, 5, figsize=(25, 6))

    axes[0].boxplot(all_prot_h)
    axes[0].set_title("Protrusion height")
    print_box_stats(all_prot_h, "Protrusion height")

    axes[1].boxplot(all_prot_w)
    axes[1].set_title("Protrusion width, number of points of 1000")
    print_box_stats(all_prot_w, "Protrusion width")

    axes[2].boxplot(all_intr_h)
    axes[2].set_title("Intrusion height")
    print_box_stats(all_intr_h, "Intrusion height")

    axes[3].boxplot(all_intr_w)
    axes[3].set_title("Intrusion width, number of points of 1000")
    print_box_stats(all_intr_w, "Intrusion width")

    axes[4].boxplot(all_curvatures)
    axes[4].set_title("Abs Curvature")
    axes[4].set_ylim(0, 0.05)
    print_box_stats(all_curvatures, "Abs Curvature")

    for ax in axes:
        ax.grid(True)

    fig.tight_layout()
    plt.savefig("statistics_cells.png", dpi=300)
    plt.close(fig)

def generate_umap_pdf_for_all_cells(direct, h5_path, output_pdf_path):
    plots_per_page = 12
    rows, cols = 4, 3

    strict_color_map = {
        "Immobile": "#b10000",
        "Confined Diffusion": "#6600cc",
        "Free Diffusion": "#00e5ff",
        "Directed Diffusion": "#ff00ff",
        "Unclassified": "#000000"
    }

    label_map = {
        1: "Immobile",
        2: "Confined Diffusion",
        3: "Free Diffusion",
        4: "Directed Diffusion",
        "unclassified": "Unclassified"
    }

    with PdfPages(output_pdf_path) as pdf, h5py.File(h5_path, 'r') as h5_file:
        all_cell_folders = [
            os.path.join(direct, f"cell_{i}") for i in range(1, 205)
        ]
        total_plots = len(all_cell_folders)
        
        for page_start in range(0, total_plots, plots_per_page):
            fig, axes = plt.subplots(rows, cols, figsize=(8.27, 11.69))
            axes = axes.flatten()

            for i in range(plots_per_page):
                plot_index = page_start + i
                ax = axes[i]

                if plot_index >= total_plots:
                    ax.axis('off')
                    continue

                cell_path = all_cell_folders[plot_index]
                cell_number = plot_index + 1
                group = f"/track_{cell_number}"

                if not os.path.exists(cell_path) or group not in h5_file:
                    ax.axis('off')
                    continue

                all_outlines, _, all_topo, _, _, _, fin_times = conformal_representation(cell_path)
                num = len(all_outlines)

                motion_data = h5_file[group][:]
                event_indices = motion_data[0, :].astype(int) - 1
                interval_types = motion_data[2, :]

                fft_spectra = []
                migration_type = []

                for jj in range(num):
                    motion_type = "unclassified"
                    for k in range(len(interval_types)):
                        raw_type = interval_types[k]
                        if k < len(event_indices) - 1:
                            start = event_indices[k] if k == 0 else event_indices[k] + 1
                            end = event_indices[k + 1]
                        else:
                            start = event_indices[k] if k == 0 else event_indices[k] + 1
                            end = num - 1
                        if start <= jj <= end:
                            motion_type = int(raw_type) + 1 if not np.isnan(raw_type) else "unclassified"
                            break
                    motion_label = label_map[motion_type]
                    # if motion_label not in ["Free Diffusion", "Directed Diffusion"]:
                    #     continue

                    topography_coords_disk = smooth_topo(all_topo[jj])
                    index_from = op_index(all_outlines[jj - 1], all_outlines[jj], 0) if jj > 0 else 0
                    theta_zero_index = np.arctan2(topography_coords_disk[index_from][0], topography_coords_disk[index_from][1])
                    theta_topo = np.arctan2(topography_coords_disk[:, 0], topography_coords_disk[:, 1])
                    theta_3 = np.degrees((theta_topo - theta_zero_index) % (2 * np.pi))
                    topo_height = np.sqrt((topography_coords_disk[:, 0])**2 + (topography_coords_disk[:, 1])**2) - 50
                    theta_3, topo_height = zip(*sorted(zip(theta_3, topo_height)))
                    topo_height = np.array(topo_height)

                    xf, yf = fft_transformation(theta_3, topo_height, 30)
                    fft_spectra.append((xf, yf))
                    migration_type.append(motion_label)

                if not fft_spectra:
                    ax.axis('off')
                    continue

                total = len(fft_spectra)
                l2_matrix = np.zeros((total, total))
                for m in range(total):
                    for n in range(m, total):
                        xf_i, yf_i = fft_spectra[m]
                        xf_j, yf_j = fft_spectra[n]
                        norm = l2_fourier(xf_i, yf_i, xf_j, yf_j)
                        l2_matrix[m, n] = norm
                        l2_matrix[n, m] = norm

                unique_types = sorted(set(migration_type))
                type_to_index = {label: idx for idx, label in enumerate(unique_types)}
                index_labels = [type_to_index[label] for label in migration_type]

                reducer = umap.UMAP(metric="precomputed", random_state=42)
                embedding = reducer.fit_transform(l2_matrix)

                color_list = [strict_color_map[label] for label in migration_type]
                ax.scatter(embedding[:, 0], embedding[:, 1], c=color_list, s=10)
                ax.set_title(f"Cell {cell_number}", fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_aspect('equal', 'datalim')

                if i == 0:
                    legend_handles = [
                        Line2D([0], [0], color=color, lw=6, label=label)
                        for label, color in strict_color_map.items()
                    ]
                    ax.legend(handles=legend_handles, title="Migration Type", fontsize=6, title_fontsize=7)

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)





def count_protrusions_and_intrusions_full(topography_coords_disk, curvatures,
                                           outline=None, centr=None, fin_times=None,
                                           image_size=100,
                                           critical_curvature=0.0454,
                                           critical_height_protrusion=6.1495,
                                           critical_height_intrusion=5.0797,
                                           circle_radius=25,
                                           stride_fill=10,
                                           stride_filter=5,
                                           frame_index=0):
    # Step 1: Smooth topo
    topography_coords_disk = smooth_topo(topography_coords_disk)

    # Step 2: Determine index_from for angular alignment
    index_from = 0
    if frame_index > 0 and outline is not None and centr is not None and fin_times is not None:
        index_from = op_index(outline[frame_index - 1], outline[frame_index], 0)

    theta = np.arctan2(topography_coords_disk[index_from][0], topography_coords_disk[index_from][1])
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])

    # Step 3: Shift and rotate into standard frame
    coords = (topography_coords_disk + image_size) / 2
    center = np.array([image_size / 2, image_size / 2])
    coords = (coords - center) @ R.T + center

    rr, cc = polygon(coords[:, 1], coords[:, 0], shape=(image_size, image_size))
    binary_image = np.zeros((image_size, image_size), dtype=np.uint8)
    binary_image[rr, cc] = 1

    circle_mask = np.zeros_like(binary_image)
    rr1, cc1 = disk((image_size // 2, image_size // 2), circle_radius, shape=binary_image.shape)
    circle_mask[rr1, cc1] = 1

    protrusions = np.logical_and(binary_image, ~circle_mask)
    intrusions = np.logical_and(~binary_image, circle_mask)

    def segment_region(region_mask, label_color, height_sign, height_thresh):
        distance_map = gaussian_filter(distance_transform_edt(region_mask), sigma=1.0)
        local_maxi = peak_local_max(distance_map, footprint=np.ones((3, 3)), labels=region_mask, min_distance=10)
        mask = np.zeros(distance_map.shape, dtype=bool)
        mask[tuple(local_maxi.T)] = True
        markers, _ = ndi.label(mask)
        labels = watershed(-distance_map, markers, mask=region_mask)

        color_by_index = {}
        for idx, coord in enumerate(coords):
            x, y = np.round(coord).astype(int)
            if 0 <= y < image_size and 0 <= x < image_size:
                if labels[y, x] > 0:
                    color_by_index[idx] = label_color
                else:
                    color_by_index[idx] = (1.0, 1.0, 1.0)
            else:
                color_by_index[idx] = (1.0, 1.0, 1.0)

        length = len(color_by_index)

        # Fill small gaps
        changed = True
        while changed:
            changed = False
            color_copy = color_by_index.copy()
            for i in range(length):
                if color_copy[i] == (1.0, 1.0, 1.0):
                    prev_color = color_copy[(i - 1) % length]
                    for offset in range(1, stride_fill + 1):
                        forward_idx = (i + offset) % length
                        forward_color = color_copy[forward_idx]
                        if forward_color != (1.0, 1.0, 1.0):
                            if forward_color == prev_color:
                                for z in range(offset):
                                    idx_to_paint = (i + z) % length
                                    if color_by_index[idx_to_paint] == (1.0, 1.0, 1.0):
                                        color_by_index[idx_to_paint] = prev_color
                                        changed = True
                            break

        # Remove short segments
        color_copy = color_by_index.copy()
        i = 0
        while i < length:
            current_color = color_copy[i]
            if current_color == label_color:
                segment_length = 0
                while segment_length < length and color_copy[(i + segment_length) % length] == label_color:
                    segment_length += 1
                if segment_length < stride_filter:
                    for j in range(segment_length):
                        color_copy[(i + j) % length] = (1.0, 1.0, 1.0)
                i += segment_length
            else:
                i += 1
        color_by_index = color_copy

        # Assign criteria based on height and curvature
        criteria_index = {}
        for idx, coord in enumerate(coords):
            r = (np.linalg.norm(coord - center) - circle_radius) * 2
            curvature = curvatures[idx]
            if height_sign * r >= height_thresh and abs(curvature) > abs(critical_curvature):
                criteria_index[idx] = True
            else:
                criteria_index[idx] = False

        # Filter segments without qualifying criteria
        color_copy = color_by_index.copy()
        i = 0
        while i < length:
            current_color = color_copy[i]
            if current_color == label_color:
                segment_indices = []
                has_important = False
                segment_length = 0
                while segment_length < length:
                    idx = (i + segment_length) % length
                    if color_copy[idx] == current_color:
                        segment_indices.append(idx)
                        if criteria_index.get(idx):
                            has_important = True
                        segment_length += 1
                    else:
                        break
                if not has_important:
                    for idx_to_white in segment_indices:
                        color_copy[idx_to_white] = (1.0, 1.0, 1.0)
                i += segment_length
            else:
                i += 1

        # Final count
        color_by_index = color_copy
        colors = [color_by_index[i] for i in range(length)]
        count = 0
        for i in range(length):
            if colors[i] == label_color and colors[(i - 1) % length] != label_color:
                count += 1
        return count

    num_protrusions = segment_region(protrusions, (1.0, 0.0, 0.0), 1, critical_height_protrusion)
    num_intrusions = segment_region(intrusions, (0.0, 1.0, 0.0), -1, critical_height_intrusion)

    return num_protrusions, num_intrusions

def compute_and_save_all_protrusions_and_intrusions(data_dir, output_path):
    cell_dirs = sorted(
        [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))],
        key=lambda x: int(os.path.basename(x).split('_')[1])
    )

    with h5py.File(output_path, 'w') as h5file:
        for cell_idx, cell_path in enumerate(cell_dirs, start=1):
            if(cell_idx!=204):
                continue
            group = h5file.create_group(f'cell_{cell_idx}')
            all_outlines, _, all_topo, all_curvatures, _, centr, fin_times = conformal_representation(cell_path)
            for frame_idx, (topo_coords, curvs) in enumerate(zip(all_topo, all_curvatures)):
                n_protrusions, n_intrusions = count_protrusions_and_intrusions_full(
                    topo_coords, curvs,
                    outline=all_outlines,
                    centr=centr,
                    fin_times=fin_times,
                    frame_index=frame_idx
                )
                frame_group = group.create_group(f'frame_{frame_idx + 1}')
                frame_group.create_dataset('protrusions', data=n_protrusions)
                frame_group.create_dataset('intrusions', data=n_intrusions)

    print(f"Saved qualified protrusion and intrusion counts to {output_path}")

def generate_umap_pdf_with_trajectories(direct, h5_path, output_pdf_path):
    import os
    import numpy as np
    import h5py
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D
    import umap

    plots_per_page = 6
    cols= 4#3
    rows = 3#(plots_per_page * 2 + cols - 1) // cols

    strict_color_map = {
        "Immobile": "#b10000",
        "Confined Diffusion": "#6600cc",
        "Free Diffusion": "#00e5ff",
        "Directed Diffusion": "#ff00ff",
        "Unclassified": "#000000"
    }

    label_map = {
        1: "Immobile",
        2: "Confined Diffusion",
        3: "Free Diffusion",
        4: "Directed Diffusion",
        "unclassified": "Unclassified"
    }

    with PdfPages(output_pdf_path) as pdf, h5py.File(h5_path, 'r') as h5_file:
        all_cell_folders = [os.path.join(direct, f"cell_{i}") for i in range(1, 205)]
        total_plots = len(all_cell_folders)

        for page_start in range(0, total_plots, plots_per_page):
            fig, axes = plt.subplots(rows, cols, figsize=(11.69, 8.27)) #fig, axes = plt.subplots(rows, cols, figsize = (cols * 2, rows * 2.5))  # A4 landscape
            axes = axes.flatten()

            for i in range(plots_per_page):
                plot_index = page_start + i
                if plot_index >= total_plots:
                    axes[2 * i].axis('off')
                    axes[2 * i + 1].axis('off')
                    continue

                ax_umap = axes[2 * i]
                ax_traj = axes[2 * i + 1]

                cell_path = all_cell_folders[plot_index]
                cell_number = plot_index + 1
                group = f"/track_{cell_number}"

                if not os.path.exists(cell_path) or group not in h5_file:
                    ax_umap.axis('off')
                    ax_traj.axis('off')
                    continue

                all_outlines, _, all_topo, _, _, centr, fin_times = conformal_representation(cell_path)
                num = len(all_outlines)

                motion_data = h5_file[group][:]
                event_indices = motion_data[0, :].astype(int) - 1
                interval_types = motion_data[2, :]

                fft_spectra = []
                migration_type = []

                for jj in range(num):
                    motion_type = "unclassified"
                    for k in range(len(interval_types)):
                        raw_type = interval_types[k]
                        if k < len(event_indices) - 1:
                            start = event_indices[k] if k == 0 else event_indices[k] + 1
                            end = event_indices[k + 1]
                        else:
                            start = event_indices[k] if k == 0 else event_indices[k] + 1
                            end = num - 1
                        if start <= jj <= end:
                            motion_type = int(raw_type) + 1 if not np.isnan(raw_type) else "unclassified"
                            break
                    motion_label = label_map[motion_type]

                    topography_coords_disk = smooth_topo(all_topo[jj])
                    index_from = op_index(all_outlines[jj - 1], all_outlines[jj], 0) if jj > 0 else 0
                    theta_zero_index = np.arctan2(topography_coords_disk[index_from][0], topography_coords_disk[index_from][1])
                    theta_topo = np.arctan2(topography_coords_disk[:, 0], topography_coords_disk[:, 1])
                    theta_3 = np.degrees((theta_topo - theta_zero_index) % (2 * np.pi))
                    topo_height = np.sqrt((topography_coords_disk[:, 0])**2 + (topography_coords_disk[:, 1])**2) - 50
                    theta_3, topo_height = zip(*sorted(zip(theta_3, topo_height)))
                    topo_height = np.array(topo_height)

                    xf, yf = fft_transformation(theta_3, topo_height, 30)
                    fft_spectra.append((xf, yf))
                    migration_type.append(motion_label)

                if not fft_spectra:
                    ax_umap.axis('off')
                    ax_traj.axis('off')
                    continue

                total = len(fft_spectra)
                l2_matrix = np.zeros((total, total))
                for m in range(total):
                    for n in range(m, total):
                        xf_i, yf_i = fft_spectra[m]
                        xf_j, yf_j = fft_spectra[n]
                        norm = l2_fourier(xf_i, yf_i, xf_j, yf_j)
                        l2_matrix[m, n] = norm
                        l2_matrix[n, m] = norm

                reducer = umap.UMAP(metric="precomputed", random_state=42)
                embedding = reducer.fit_transform(l2_matrix)
                color_list = [strict_color_map[label] for label in migration_type]

                ax_umap.scatter(embedding[:, 0], embedding[:, 1], c=color_list, s=10)
                ax_umap.set_title(f"Cell {cell_number}", fontsize=8)
                ax_umap.set_xticks([])
                ax_umap.set_yticks([])
                ax_umap.set_aspect('equal', 'datalim')

                if i == 0:
                    legend_handles = [
                        Line2D([0], [0], color=color, lw=6, label=label)
                        for label, color in strict_color_map.items()
                    ]
                    ax_umap.legend(handles=legend_handles, title="Migration Type", fontsize=6, title_fontsize=7)

                # Trajectory plot
                centroids = np.array(centr)
                if centroids.ndim == 2 and centroids.shape[0] >= 2:
                    x_coords = centroids[:, 0]
                    y_coords = centroids[:, 1]
                    time_steps = np.array(fin_times)

                    scatter = ax_traj.scatter(
                        x_coords[1:], y_coords[1:],
                        c=time_steps[1:],
                        cmap='plasma',
                        marker='o',
                        edgecolor='k',
                        s=40,
                        alpha=0.7
                    )
                    ax_traj.scatter(
                        x_coords[0], y_coords[0],
                        c='black',
                        marker='o',
                        edgecolor='k',
                        s=50,
                        alpha=0.9,
                        label='Start'
                    )
                    ax_traj.plot(x_coords, y_coords, linestyle='-', color='gray', alpha=0.5)
                    ax_traj.set_title("Trajectory", fontsize=8)
                    ax_traj.set_xlabel("X", fontsize=6)
                    ax_traj.set_ylabel("Y", fontsize=6)
                    ax_traj.tick_params(axis='both', which='major', labelsize=6)

                    cbar = fig.colorbar(scatter, ax=ax_traj, orientation='vertical', fraction=0.046, pad=0.04)
                    cbar.set_label("Time", rotation=270, labelpad=8, fontsize=6)
                    cbar.ax.tick_params(labelsize=6)
                else:
                    ax_traj.axis('off')

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)



if __name__ == '__main__':
    glob_folder = '/home/pavel/cell_morphology/cells/'
    sergio_folder = "/home/pavel/Downloads/Code/kp0.2/cells"
    protrusion_height = 0.0
    protrusion_width = 1
    eps = 10
    #plot_3d_cell_outlines(glob_folder, 87)
    #glob_folder = '/home/pavel/Downloads/cells_v2_filtered/cells'
    #max_protusion_plot(glob_folder)
    #create_outline_video(glob_folder)
    #stats_max_protusions_2(glob_folder)
    #stats_max_protusions_360(glob_folder)
    #stats_single_cell_360(glob_folder, 87,[60,])
    #curve_protrusions(glob_folder, 86, 1, protrusion_height, protrusion_width, eps)
    #circle_watershed_border(glob_folder, 41, 5)
    #stats_single_cell_protusions_wrt_vel(glob_folder, 86, [1, ])
    #stats_single_cell_protusions_wrt_vel(glob_folder, 87, [1,100])

    #plot_shape_and_topo_3D(glob_folder, 86)
        #77, 24, 20, 193, 194, 195, 201, 
    #for i in [77, 24, 20, 193, 194, 195, 201, 202,203, 86]:
    #    save_circle_watershed_evolution_to_pdf(glob_folder, i, f"cell_{i+1}_pdf_new.pdf")
    save_circle_watershed_evolution_to_pdf_steps(glob_folder, 203, f"cell_{203+1}_pdf_steps.pdf")
    save_circle_watershed_evolution_to_pdf(glob_folder, 203, f"cell_{203+1}_pdf.pdf")
    #collect_statistics(glob_folder)
    #generate_umap_pdf_for_all_cells(glob_folder, "time_events_95.h5", "umap_all_cells.pdf")
    #fft_spare_l2_matrix(glob_folder,86)
    #collect_protrusion_intrusion_stats_by_motion_type(glob_folder, "time_events_95.h5")

    #generate_umap_pdf_with_trajectories(glob_folder, "time_events_95.h5", "umap_all_cells.pdf")
    #fft_spare_l2_matrix_whole_dataset(glob_folder)
    
    #compute_and_save_all_protrusions_and_intrusions(glob_folder, "number_protrusions_dataset.h5")
    #fft_spare_l2_matrix_whole_dataset_cutted(glob_folder, 'l2_matrix.npy')
    #fft_spare_l2_matrix_whole_dataset_riemann(glob_folder)
import os 
import numpy as np 
import matplotlib.pyplot as plt

def conformal_representation(cell_folder ,
                             ref_disk_radius = 50,
                             ref_iter_no = 3, 
                             lambda_flow = 200, 
                             curvature_threshold = 0.1):
    
    import glob
    import skimage.io as skio 
    import pylab as plt 
    import skimage.transform as sktform 
    
    import unwrap2D.unwrap2D as unwrap2D_fns
    import unwrap3D.Mesh.meshtools as unwrap3D_meshtools
    import unwrap3D.Segmentation.segmentation as unwrap3D_segmentation
    
    
    # https://github.com/clementsoubrier/shape_analysis/blob/main/construct_dataset.py


    # ref_disk_radius = 50
    # ref_iter_no = 3 # 3 use this and 
    # lambda_flow = 200 # 5e1  use this to tune the extent of smoothing of base shape, the less the smoother 
    # curvature_threshold = 0.1 # only apply flow iterations if np.mean(np.abs(curvature)) > threshold i.e. there are potential protrusions! 
    
    
    """
    Iterate over cells. 
    """


    times = np.sort(os.listdir(cell_folder)) # perform nature sort
    times_int = np.hstack([int(tt.split('frame_')[1]) for tt in times])
    natsort = np.argsort(times_int)
    
    times_int = times_int[natsort]
    times = times[natsort]
    
    all_outlines = []
    all_outlines_cMCF = []
    all_outlines_cMCF_topography = []
    all_outlines_curvature = []
    fin_times = []
    
    for ttt in np.arange(len(times))[:]:
        
        outline_ttt_file = os.path.join(cell_folder, times[ttt], 'outline.npy')
        centroid_ttt_file = os.path.join(cell_folder, times[ttt], 'centroid.npy')
        time_ttt_file = os.path.join(cell_folder, times[ttt], 'time.npy')
        
        outline = np.load(outline_ttt_file)
        all_outlines.append(outline)
        fin_times.append(np.load(time_ttt_file))
        
        centroid = np.load(centroid_ttt_file)
        
        # evolve the contour. 
        outline_input = outline.copy()
        _,_, outline_input_curvature = unwrap2D_fns.curvature_splines(outline_input[:,1], 
                                                                outline_input[:,0],
                                                                k=4,
                                                                error=0.1)
        all_outlines_curvature.append(outline_input_curvature)
        outline_input_curvature_norm = np.nanmean(np.linalg.norm(outline_input-centroid[None,:], axis=-1), axis=0)/2. * outline_input_curvature

        print(outline_input_curvature_norm.min(),outline_input_curvature_norm.max(), np.nanmean(np.abs(outline_input_curvature_norm)))
                    
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
        
        
    return all_outlines, all_outlines_cMCF, all_outlines_cMCF_topography, all_outlines_curvature, disk_coords, fin_times






def max_protusion_plot(all_outlines, all_outlines_cMCF_topography, all_outlines_curvature, fin_times):
    plt.figure()
    ax = plt.axes(projection='3d')
    tot_len = len(all_outlines_cMCF_topography) #10
    colors = plt.cm.viridis(np.linspace(0,1,tot_len))

    for jj in np.arange(tot_len) :
        topography_coords_disk = all_outlines_cMCF_topography[jj]
        outline = all_outlines[jj]
        time = fin_times[jj]
        curv = all_outlines_curvature[jj]
        index = np.argmax(topography_coords_disk [:,0]**2+topography_coords_disk [:,1]**2)
        index_curvature = np.argmin(curv)

        ax.plot3D(outline[:,0],outline[:,1], time*np.ones(len(outline[:,1])), c=colors[jj] )
        # ax.scatter(outline[index,0], 
        #             outline[index,1], time, c='k')
        ax.scatter(outline[index_curvature,0], 
                    outline[index_curvature,1], time, c='r')
        
    plt.figure()
    ax = plt.axes(projection='3d')
    tot_len = len(all_outlines_cMCF_topography) #10
    colors = plt.cm.viridis(np.linspace(0,1,tot_len))

    for jj in np.arange(tot_len) :
        topography_coords_disk = all_outlines_cMCF_topography[jj]
        outline = all_outlines[jj]
        time = fin_times[jj]
        curv = all_outlines_curvature[jj]
        index = np.argmax(topography_coords_disk [:,0]**2+topography_coords_disk [:,1]**2)
        index_curvature = np.argmin(curv)

        ax.plot3D(topography_coords_disk[:,0],topography_coords_disk[:,1], time*np.ones(len(outline[:,1])), c=colors[jj] )
    #     ax.scatter(topography_coords_disk[index,0], 
    #                 topography_coords_disk[index,1], time, c='k')
    #     ax.scatter(topography_coords_disk[index_curvature,0], 
    #                 topography_coords_disk[index_curvature,1], time, c='r')
    plt.show()
            








if __name__ == '__main__':
    glob_folder = 'cells'
    all_cell_folders = [os.path.join(glob_folder, ff) for ff in os.listdir(glob_folder)]
    for cell_folder_ii in np.arange(len(all_cell_folders)):
        outlines, _, outlines_topo, curvature, _, times = conformal_representation(all_cell_folders[cell_folder_ii])
        print(curvature)
        max_protusion_plot(outlines, outlines_topo, curvature, times)
        

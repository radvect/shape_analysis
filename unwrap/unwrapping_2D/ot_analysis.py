import os
import numpy as np
import h5py
import umap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from scipy.spatial import distance_matrix
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans, DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import MDS


import ot
from ot.gromov import gromov_wasserstein

STRICT_COLOR_MAP = {
    "Immobile": "#b10000",
    "Confined Diffusion": "#6600cc",
    "Free Diffusion": "#00e5ff",
    "Directed Diffusion": "#ff00ff",
    "Unclassified": "#000000",
}
LABEL_MAP = {
    1: "Immobile",
    2: "Confined Diffusion",
    3: "Free Diffusion",
    4: "Directed Diffusion",
    "unclassified": "Unclassified",
}

def get_outlines_centr_times(cell_path):
    times = np.sort(os.listdir(cell_path))
    times_int = np.hstack([int(tt.split('frame_')[1]) for tt in times])
    

    natsort = np.argsort(times_int)

    times_int = times_int[natsort]
    times = times[natsort]
    all_outlines = []
    fin_times = []
    fin_centr = []
    for ttt in np.arange(len(times))[:]:
        
        outline_ttt_file = os.path.join(cell_path, times[ttt], 'outline.npy')
        centroid_ttt_file = os.path.join(cell_path, times[ttt], 'centroid.npy')
        time_ttt_file = os.path.join(cell_path, times[ttt], 'time.npy')
        
        outline = np.load(outline_ttt_file)
        all_outlines.append(outline)
        fin_times.append(np.load(time_ttt_file))
        centroid = np.load(centroid_ttt_file)
        fin_centr.append(centroid)

    return  all_outlines, fin_centr, fin_times


def compute_Wass_distance(Out1,Out2):
      D12=distance_matrix(Out1,Out2,p=2)
      n1=Out1.shape[0]
      n2=Out2.shape[0]
      a=1/n1*np.ones(n1)
      b=1/n2*np.ones(n2)
      Wdist=ot.emd2(a, b, D12)
      return Wdist

def interpolate(curve, nb_points):
    """Interpolate a discrete curve with nb_points from a discrete curve.

    Returns
    -------
    interpolation : discrete curve with nb_points points
    """
    old_length = curve.shape[0]
    interpolation = np.zeros((nb_points, 2))
    incr = old_length / nb_points
    pos = 0
    for i in range(nb_points):
        index = int(np.floor(pos))
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


def generate_umap_pdf_with_trajectories_OT_modes(
    direct: str,
    h5_path: str,
    output_pdf_path: str,
    plots_per_page: int = 6,
    cols: int = 4,
    rows: int = 3,
    k_sampling_points: int = 200,
    random_state: int = 42,
):
    with PdfPages(output_pdf_path) as pdf, h5py.File(h5_path, "r") as h5_file:
        all_cell_folders = [os.path.join(direct, f"cell_{i}") for i in range(1, 122)]
        total_cells = len(all_cell_folders)

        for page_start in range(0, total_cells, plots_per_page):
            fig, axes = plt.subplots(rows, cols, figsize=(11.69, 8.27)) 
            axes = axes.flatten()
            
            put_legend_once = True

            for i in range(plots_per_page):
                plot_index = page_start + i

                umap_ax_idx = 2 * i
                traj_ax_idx = 2 * i + 1


                ax_umap = axes[umap_ax_idx]
                ax_traj = axes[traj_ax_idx]

                if plot_index >= total_cells:
                    ax_umap.axis('off')
                    ax_traj.axis('off')
                    continue

                cell_path = all_cell_folders[plot_index]
                cell_number = plot_index + 1

                group = f"/track_{cell_number}"

                print(f"cell number #{cell_number} is being processed")
                (
                    all_outlines,   
                    centr,         
                    fin_times,      
                ) = get_outlines_centr_times(cell_path)


                T = fin_times
                print(T)
                if len(T) < 2:
                    ax_umap.axis('off')
                    ax_traj.axis('off')
                    continue

                labels = ["Unclassified"] * len(T)
                if group in h5_file:
                    motion_data = h5_file[group][:] 
                    first_three_rows = motion_data[:3]
                    event_indices = motion_data[0, :].astype(int) - 1
                    event_indices = np.append(event_indices, int(first_three_rows[1 ,-1]) - 1)
                    
                    interval_types = motion_data[2, :]
                    for index_label,jj in enumerate(T):
                        motion_type = "unclassified"
                        K = len(interval_types)
                        for k in range(K):
                            raw_type = interval_types[k]
                            if k < len(interval_types) - 1:
                                print(k)
                                print(len(interval_types)-1)
                                start = event_indices[k] if k == 0 else event_indices[k] + 1
                                end = event_indices[k + 1]
                            else:
                                start = event_indices[k] if k == 0 else event_indices[k] + 1
                                end = T[-1] 

                            if start <= jj <= end:
                                motion_type = int(raw_type) + 1 if not np.isnan(raw_type) else "unclassified"
                                break
                            # if(jj>=end)
                        # print("start ", start)
                        # print("jj ", jj)
                        # print("end ", end)
                        #print(motion_type)
                        labels[index_label] = LABEL_MAP.get(motion_type, "Unclassified")
                        #print(labels)
                # print(event_indices)
                # print(interval_types)
                print(labels)
                colors = [STRICT_COLOR_MAP.get(lbl, STRICT_COLOR_MAP["Unclassified"]) for lbl in labels]
                preproc = []
                for outline in all_outlines:
                    interp = interpolate(outline, k_sampling_points)
                    preproc.append(preprocess(interp))

                l2_matrix = np.zeros((len(T), len(T)), dtype=float)
                for a in range(len(T)):
                    Xa = preproc[a] - np.mean(preproc[a], axis=0, keepdims=True)
                    for b in range(a + 1, len(T)):
                        Xb = preproc[b] - np.mean(preproc[b], axis=0, keepdims=True)
                        dist = compute_Wass_distance(Xa, Xb)   
                        l2_matrix[a, b] = dist
                        l2_matrix[b, a] = dist


      
                reducer = umap.UMAP(metric="precomputed", random_state=random_state)
                embedding = reducer.fit_transform(l2_matrix)

                ax_umap.scatter(embedding[:, 0], embedding[:, 1], c=colors, s=10)
                ax_umap.set_title(f"Cell {cell_number}", fontsize=8)
                ax_umap.set_xticks([])
                ax_umap.set_yticks([])
                ax_umap.set_aspect('equal', 'datalim')

                if put_legend_once:
                    legend_handles = [
                        Line2D([0], [0], color=color, lw=6, label=label)
                        for label, color in STRICT_COLOR_MAP.items()
                    ]
                    ax_umap.legend(handles=legend_handles, title="Migration Type", fontsize=6, title_fontsize=7)
                    put_legend_once = False

                centroids = np.array(centr)
                times = np.array(fin_times)
                if centroids.ndim == 2 and centroids.shape[0] >= 2:
                    x_coords = centroids[:, 0]
                    y_coords = centroids[:, 1]
                    sc = ax_traj.scatter(
                        x_coords[1:], y_coords[1:],
                        c=times[1:],
                        cmap='plasma',
                        marker='o',
                        edgecolor='k',
                        s=40,
                        alpha=0.7
                    )
                    ax_traj.scatter(
                        x_coords[0], y_coords[0],
                        c='black', marker='o', edgecolor='k', s=50, alpha=0.9, label='Start'
                    )
                    ax_traj.plot(x_coords, y_coords, linestyle='-', color='gray', alpha=0.5)
                    ax_traj.set_title("Trajectory", fontsize=8)
                    ax_traj.set_xlabel("X", fontsize=6)
                    ax_traj.set_ylabel("Y", fontsize=6)
                    ax_traj.tick_params(axis='both', which='major', labelsize=6)

                    cbar = fig.colorbar(sc, ax=ax_traj, orientation='vertical', fraction=0.046, pad=0.04)
                    cbar.set_label("Time", rotation=270, labelpad=8, fontsize=6)
                    cbar.ax.tick_params(labelsize=6)
                else:
                    ax_traj.axis('off')

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Saved to: {output_pdf_path}")



def generate_umap_pdf_with_trajectories_OT_coloring(
    direct: str,
    output_pdf_path: str,
    plots_per_page: int = 6,
    cols: int = 4,
    rows: int = 3,
    k_sampling_points: int = 200,
    random_state: int = 42,
):

    with PdfPages(output_pdf_path) as pdf:
        all_cell_folders = [os.path.join(direct, f"cell_{i}") for i in range(1, 122)]
        total_cells = len(all_cell_folders)

        for page_start in range(0, total_cells, plots_per_page):
            fig, axes = plt.subplots(rows, cols, figsize=(11.69, 8.27))  
            axes = axes.flatten()

            put_time_cbar_once = True

            for i in range(plots_per_page):
                plot_index = page_start + i
                umap_ax_idx = 2 * i
                traj_ax_idx = 2 * i + 1
                if umap_ax_idx >= len(axes) or traj_ax_idx >= len(axes):
                    break

                ax_umap = axes[umap_ax_idx]
                ax_traj = axes[traj_ax_idx]

                if plot_index >= total_cells:
                    ax_umap.axis('off'); ax_traj.axis('off'); continue

                cell_path = all_cell_folders[plot_index]
                cell_number = plot_index + 1
                group = f"/track_{cell_number}"

                if not os.path.isdir(cell_path):
                    ax_umap.axis('off'); ax_traj.axis('off'); continue

                (
                    all_outlines,   
                    centr,         
                    fin_times,      
                ) = get_outlines_centr_times(cell_path)

                T = len(all_outlines)
                if T < 2:
                    ax_umap.axis('off'); ax_traj.axis('off'); continue

                preproc = []
                for outline in all_outlines:
                    interp = interpolate(outline, k_sampling_points)
                    preproc.append(preprocess(interp))

                l2_matrix = np.zeros((T, T), dtype=float)
                centered = [p - np.mean(p, axis=0, keepdims=True) for p in preproc]

                for a in range(T):
                    for b in range(a + 1, T):
                        dist = compute_Wass_distance(centered[a], centered[b])
                        l2_matrix[a, b] = dist
                        l2_matrix[b, a] = dist


                reducer = umap.UMAP(metric="precomputed", random_state=random_state)
                embedding = reducer.fit_transform(l2_matrix)

                times = np.asarray(fin_times)
                tmin, tmax = np.nanmin(times), np.nanmax(times)
                import matplotlib as mpl
                norm = mpl.colors.Normalize(vmin=tmin, vmax=tmax)
                cmap = 'plasma'

                sc_umap = ax_umap.scatter(embedding[:, 0], embedding[:, 1], c=times, cmap=cmap, norm=norm, s=10)
                ax_umap.set_title(f"Cell {cell_number}", fontsize=8)
                ax_umap.set_xticks([]); ax_umap.set_yticks([])
                ax_umap.set_aspect('equal', 'datalim')

                if put_time_cbar_once:
                    cbar_u = fig.colorbar(sc_umap, ax=ax_umap, orientation='vertical', fraction=0.046, pad=0.04)
                    cbar_u.set_label("Time", rotation=270, labelpad=8, fontsize=6)
                    cbar_u.ax.tick_params(labelsize=6)
                    put_time_cbar_once = False

                centroids = np.array(centr)
                if centroids.ndim == 2 and centroids.shape[0] >= 2:
                    x_coords = centroids[:, 0]
                    y_coords = centroids[:, 1]
                    sc_tr = ax_traj.scatter(
                        x_coords[1:], y_coords[1:],
                        c=times[1:], cmap=cmap, norm=norm,
                        marker='o', edgecolor='k', s=40, alpha=0.7
                    )
                    ax_traj.scatter(
                        x_coords[0], y_coords[0],
                        c='black', marker='o', edgecolor='k', s=50, alpha=0.9, label='Start'
                    )
                    ax_traj.plot(x_coords, y_coords, linestyle='-', color='gray', alpha=0.5)
                    ax_traj.set_title("Trajectory", fontsize=8)
                    ax_traj.set_xlabel("X", fontsize=6); ax_traj.set_ylabel("Y", fontsize=6)
                    ax_traj.tick_params(axis='both', which='major', labelsize=6)

                    cbar_t = fig.colorbar(sc_tr, ax=ax_traj, orientation='vertical', fraction=0.046, pad=0.04)
                    cbar_t.set_label("Time", rotation=270, labelpad=8, fontsize=6)
                    cbar_t.ax.tick_params(labelsize=6)
                else:
                    ax_traj.axis('off')

            for k in range(2 * plots_per_page, rows * cols):
                axes[k].axis('off')

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)





def generate_umap_pdf_with_trajectories_OT_clustering(
    direct: str,
    output_pdf_path: str,
    plots_per_page: int = 6,
    cols: int = 4,
    rows: int = 3,
    k_sampling_points: int = 200,
    random_state: int = 42,
):

    with PdfPages(output_pdf_path) as pdf:
        all_cell_folders = [os.path.join(direct, f"cell_{i}") for i in range(1, 122)]
        total_cells = len(all_cell_folders)

        for page_start in range(0, total_cells, plots_per_page):
            fig, axes = plt.subplots(rows, cols, figsize=(11.69, 8.27))  
            axes = axes.flatten()

            put_time_cbar_once = True

            for i in range(plots_per_page):
                plot_index = page_start + i
                umap_ax_idx = 2 * i
                traj_ax_idx = 2 * i + 1
                if umap_ax_idx >= len(axes) or traj_ax_idx >= len(axes):
                    break

                ax_umap = axes[umap_ax_idx]
                ax_traj = axes[traj_ax_idx]

                if plot_index >= total_cells:
                    ax_umap.axis('off'); ax_traj.axis('off'); continue

                cell_path = all_cell_folders[plot_index]
                cell_number = plot_index + 1
                group = f"/track_{cell_number}"
                # if(cell_number!=1):
                #     continue
                if not os.path.isdir(cell_path):
                    ax_umap.axis('off'); ax_traj.axis('off'); continue
                (
                    all_outlines,   
                    centr,         
                    fin_times,      
                ) = get_outlines_centr_times(cell_path)
                centroids = np.array(centr)
                print(cell_path)
                T = len(all_outlines)
                if T < 2:
                    ax_umap.axis('off'); ax_traj.axis('off'); continue

                preproc = []
                for outline in all_outlines:
                    interp = interpolate(outline, k_sampling_points)
                    preproc.append(preprocess(interp))

                l2_matrix = np.zeros((T, T), dtype=float)
                centered = [p - np.mean(p, axis=0, keepdims=True) for p in preproc]

                for a in range(T):
                    for b in range(a + 1, T):
                        dist = compute_Wass_distance(centered[a], centered[b])
                        dist = dist#/(fin_times[b]-fin_times[a])
                        l2_matrix[a, b] = dist
                        l2_matrix[b, a] = dist


                reducer = umap.UMAP(metric="precomputed", random_state=random_state)
                embedding = reducer.fit_transform(l2_matrix)

                D_embed = np.linalg.norm(
                    embedding[:, None, :] - embedding[None, :, :],
                    axis=2
                )
                np.fill_diagonal(D_embed, 0.0)
                D_traj = np.linalg.norm(
                    centroids[:, None, :] - centroids[None, :, :],
                    axis=2
                )
                np.fill_diagonal(D_traj, 0.0)

                      
                
                D_embed_sq = D_embed ** 2
                D_traj_sq  = D_traj ** 2
                
                p = np.ones(len(D_embed_sq)) / len(D_embed_sq)
                q = np.ones(len(D_traj_sq)) / len(D_traj_sq)  
                
                gw_plan = gromov_wasserstein(D_embed_sq, D_traj_sq, p, q, 'square_loss')
                gw_plan = gw_plan / gw_plan.sum(axis=0, keepdims=True)

                cmap = 'plasma'
                
                
                clusterer = DBSCAN(eps=1)
                cluster_labels = clusterer.fit_predict(embedding)

                transferred_labels = (gw_plan.T @ cluster_labels).round().astype(int)

                sc_umap = ax_umap.scatter(embedding[:, 0], embedding[:, 1], c=cluster_labels, cmap=cmap,  s=10)
                ax_umap.set_title(f"Cell {cell_number}", fontsize=8)
                ax_umap.set_aspect('equal', 'datalim')
                ax_umap.tick_params(axis='both', which='major', labelsize=6)

                
                if centroids.ndim == 2 and centroids.shape[0] >= 2:
                    x_coords = centroids[:, 0]
                    y_coords = centroids[:, 1]
                    sc_tr = ax_traj.scatter(
                        x_coords[1:], y_coords[1:],
                        c=transferred_labels[1:], cmap=cmap,
                        marker='o', edgecolor='k', s=40, alpha=0.7
                    )
                    ax_traj.scatter(
                        x_coords[0], y_coords[0],
                        c=transferred_labels[0], marker='o', edgecolor='k', s=50, alpha=0.9, label='Start'
                    )
                    ax_traj.plot(x_coords, y_coords, linestyle='-', color='gray', alpha=0.5)
                    ax_traj.set_title("Trajectory", fontsize=8)
                    ax_traj.set_xlabel("X", fontsize=6); ax_traj.set_ylabel("Y", fontsize=6)
                    ax_traj.tick_params(axis='both', which='major', labelsize=6)

                    # cbar_t = fig.colorbar(sc_tr, ax=ax_traj, orientation='vertical', fraction=0.046, pad=0.04)
                    # cbar_t.set_label("Time", rotation=270, labelpad=8, fontsize=6)
                    # cbar_t.ax.tick_params(labelsize=6)
                else:
                    ax_traj.axis('off')

            for k in range(2 * plots_per_page, rows * cols):
                axes[k].axis('off')

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)



def generate_umap_pdf_with_trajectories_OT_clustering_no_gromov(
    direct: str,
    output_pdf_path: str,
    plots_per_page: int = 6,
    cols: int = 4,
    rows: int = 3,
    k_sampling_points: int = 200,
    random_state: int = 42,
):

    with PdfPages(output_pdf_path) as pdf:
        all_cell_folders = [os.path.join(direct, f"cell_{i}") for i in range(1, 122)]
        total_cells = len(all_cell_folders)

        for page_start in range(0, total_cells, plots_per_page):
            fig, axes = plt.subplots(rows, cols, figsize=(11.69, 8.27))  
            axes = axes.flatten()

            put_time_cbar_once = True

            for i in range(plots_per_page):
                plot_index = page_start + i
                umap_ax_idx = 2 * i
                traj_ax_idx = 2 * i + 1
                if umap_ax_idx >= len(axes) or traj_ax_idx >= len(axes):
                    break

                ax_umap = axes[umap_ax_idx]
                ax_traj = axes[traj_ax_idx]

                if plot_index >= total_cells:
                    ax_umap.axis('off'); ax_traj.axis('off'); continue

                cell_path = all_cell_folders[plot_index]
                cell_number = plot_index + 1
                group = f"/track_{cell_number}"
                # if(cell_number!=1):
                #     continue
                if not os.path.isdir(cell_path):
                    ax_umap.axis('off'); ax_traj.axis('off'); continue
                (
                    all_outlines,   
                    centr,         
                    fin_times,      
                ) = get_outlines_centr_times(cell_path)
                centroids = np.array(centr)
                print(cell_path)
                T = len(all_outlines)
                if T < 2:
                    ax_umap.axis('off'); ax_traj.axis('off'); continue

                preproc = []
                for outline in all_outlines:
                    interp = interpolate(outline, k_sampling_points)
                    preproc.append(preprocess(interp))

                l2_matrix = np.zeros((T, T), dtype=float)
                centered = [p - np.mean(p, axis=0, keepdims=True) for p in preproc]

                for a in range(T):
                    for b in range(a + 1, T):
                        dist = compute_Wass_distance(centered[a], centered[b])
                        dist = dist#/(fin_times[b]-fin_times[a])
                        l2_matrix[a, b] = dist
                        l2_matrix[b, a] = dist


                reducer = umap.UMAP(metric="precomputed", random_state=random_state)
                embedding = reducer.fit_transform(l2_matrix)

                # D_embed = np.linalg.norm(
                #     embedding[:, None, :] - embedding[None, :, :],
                #     axis=2
                # )
                # np.fill_diagonal(D_embed, 0.0)
                # D_traj = np.linalg.norm(
                #     centroids[:, None, :] - centroids[None, :, :],
                #     axis=2
                # )
                # np.fill_diagonal(D_traj, 0.0)

                      
                
                # D_embed_sq = D_embed ** 2
                # D_traj_sq  = D_traj ** 2
                
                # p = np.ones(len(D_embed_sq)) / len(D_embed_sq)
                # q = np.ones(len(D_traj_sq)) / len(D_traj_sq)  
                
                # gw_plan = gromov_wasserstein(D_embed_sq, D_traj_sq, p, q, 'square_loss')
                # gw_plan = gw_plan / gw_plan.sum(axis=0, keepdims=True)

                cmap = 'plasma'
                from matplotlib.colors import Normalize

                clusterer = DBSCAN(eps=1)
                cluster_labels = clusterer.fit_predict(embedding)
                # vmin, vmax = cluster_labels.min(), cluster_labels.max()
                # norm = Normalize(vmin=vmin, vmax=vmax)
                transferred_labels = cluster_labels.copy()

                sc_umap = ax_umap.scatter(embedding[:, 0], embedding[:, 1], c=cluster_labels, cmap=cmap,  s=10)
                ax_umap.set_title(f"Cell {cell_number}", fontsize=8)
                ax_umap.set_aspect('equal', 'datalim')
                ax_umap.tick_params(axis='both', which='major', labelsize=6)

                
                if centroids.ndim == 2 and centroids.shape[0] >= 2:
                    x_coords = centroids[:, 0]
                    y_coords = centroids[:, 1]
                    sc_tr = ax_traj.scatter(
                        x_coords[1:], y_coords[1:],
                        c=transferred_labels[1:], cmap=cmap,
                        marker='o', edgecolor='k', s=40, alpha=0.7,
                    )
                    ax_traj.scatter(
                        x_coords[0], y_coords[0],
                        c=transferred_labels[0], marker='o', edgecolor='k', s=50, alpha=0.9, label='Start',cmap=cmap,
                    )
                    ax_traj.plot(x_coords, y_coords, linestyle='-', color='gray', alpha=0.5)
                    ax_traj.set_title("Trajectory", fontsize=8)
                    ax_traj.set_xlabel("X", fontsize=6); ax_traj.set_ylabel("Y", fontsize=6)
                    ax_traj.tick_params(axis='both', which='major', labelsize=6)

                    # cbar_t = fig.colorbar(sc_tr, ax=ax_traj, orientation='vertical', fraction=0.046, pad=0.04)
                    # cbar_t.set_label("Time", rotation=270, labelpad=8, fontsize=6)
                    # cbar_t.ax.tick_params(labelsize=6)
                else:
                    ax_traj.axis('off')

            for k in range(2 * plots_per_page, rows * cols):
                axes[k].axis('off')

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

def generate_trajectory_ot_clustering(
    direct: str,
    output_pdf_path: str,
    plots_per_page: int = 6,
    cols: int = 4,
    rows: int = 3,
    k_sampling_points: int = 200,
):

    with PdfPages(output_pdf_path) as pdf:
        all_cell_folders = [os.path.join(direct, f"cell_{i}") for i in range(1, 122)]
        total_cells = len(all_cell_folders)

        for page_start in range(0, total_cells, plots_per_page):

            fig, axes = plt.subplots(rows, cols, figsize=(11.69, 8.27))
            axes = axes.flatten()

            for i in range(plots_per_page):

                plot_index = page_start + i
                mat_ax_idx  = 2 * i       # раньше был UMAP
                traj_ax_idx = 2 * i + 1   # раньше траектория

                if mat_ax_idx >= len(axes) or traj_ax_idx >= len(axes):
                    break

                ax_mat  = axes[mat_ax_idx]
                ax_traj = axes[traj_ax_idx]

                if plot_index >= total_cells:
                    ax_mat.axis('off')
                    ax_traj.axis('off')
                    continue

                cell_path = all_cell_folders[plot_index]
                cell_number = plot_index + 1

                if not os.path.isdir(cell_path):
                    ax_mat.axis('off')
                    ax_traj.axis('off')
                    continue

                (
                    all_outlines,
                    centr,
                    fin_times,
                ) = get_outlines_centr_times(cell_path)

                centroids = np.array(centr)
                T = len(all_outlines)

                if T < 2:
                    ax_mat.axis('off')
                    ax_traj.axis('off')
                    continue

                # --- контуры ---
                preproc = []
                for outline in all_outlines:
                    interp = interpolate(outline, k_sampling_points)
                    preproc.append(preprocess(interp))

                # --- матрица расстояний ---
                l2_matrix = np.zeros((T, T), dtype=float)
                centered = [p - np.mean(p, axis=0, keepdims=True) for p in preproc]

                for a in range(T):
                    for b in range(a + 1, T):
                        dist = compute_Wass_distance(centered[a], centered[b])
                        l2_matrix[a, b] = dist
                        l2_matrix[b, a] = dist

                print(f"Cell {cell_number} L2 matrix:\n", l2_matrix)

                # --- heatmap l2_matrix ---
                im = ax_mat.imshow(l2_matrix, cmap="viridis", origin="lower")
                ax_mat.set_title(f"Cell {cell_number} – L2 Matrix", fontsize=8)
                ax_mat.set_xlabel("Frame", fontsize=6)
                ax_mat.set_ylabel("Frame", fontsize=6)
                ax_mat.tick_params(axis='both', which='major', labelsize=6)
                fig.colorbar(im, ax=ax_mat, fraction=0.046, pad=0.04)

                # --- кластеризация ---
                clustering = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=4.0,    # нормальный порог
                    metric="precomputed",
                    linkage="average",
                ).fit(l2_matrix)

                labels = clustering.labels_
                cmap = "plasma"

                # --- траектория ---
                if centroids.ndim == 2 and centroids.shape[0] >= 2:
                    x = centroids[:, 0]
                    y = centroids[:, 1]

                    ax_traj.scatter(
                        x[1:], y[1:], c=labels[1:], cmap=cmap,
                        s=40, edgecolor='k', alpha=0.7
                    )

                    ax_traj.scatter(
                        x[0], y[0], c=labels[0], cmap=cmap,
                        s=50, edgecolor='k', label="Start", alpha=0.9
                    )

                    ax_traj.plot(x, y, '-',
                        color='gray', alpha=0.5
                    )

                    ax_traj.set_title(f"Cell {cell_number} – Trajectory", fontsize=8)
                    ax_traj.set_xlabel("X", fontsize=6)
                    ax_traj.set_ylabel("Y", fontsize=6)
                    ax_traj.tick_params(axis='both', which='major', labelsize=6)

                else:
                    ax_traj.axis('off')

            # закрываем пустые оси
            for k in range(2 * plots_per_page, rows * cols):
                axes[k].axis('off')

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def generate_trajectory_ot_clustering(
    direct: str,
    output_pdf_path: str,
    plots_per_page: int = 6,
    cols: int = 4,
    rows: int = 3,
    k_sampling_points: int = 200,
):

    with PdfPages(output_pdf_path) as pdf:
        all_cell_folders = [os.path.join(direct, f"cell_{i}") for i in range(1, 122)]
        total_cells = len(all_cell_folders)

        for page_start in range(0, total_cells, plots_per_page):

            fig, axes = plt.subplots(rows, cols, figsize=(11.69, 8.27))
            axes = axes.flatten()

            for i in range(plots_per_page):

                plot_index = page_start + i
                mat_ax_idx  = 2 * i       # раньше был UMAP
                traj_ax_idx = 2 * i + 1   # раньше траектория

                if mat_ax_idx >= len(axes) or traj_ax_idx >= len(axes):
                    break

                ax_mat  = axes[mat_ax_idx]
                ax_traj = axes[traj_ax_idx]

                if plot_index >= total_cells:
                    ax_mat.axis('off')
                    ax_traj.axis('off')
                    continue

                cell_path = all_cell_folders[plot_index]
                cell_number = plot_index + 1

                if not os.path.isdir(cell_path):
                    ax_mat.axis('off')
                    ax_traj.axis('off')
                    continue

                (
                    all_outlines,
                    centr,
                    fin_times,
                ) = get_outlines_centr_times(cell_path)

                centroids = np.array(centr)
                T = len(all_outlines)

                if T < 2:
                    ax_mat.axis('off')
                    ax_traj.axis('off')
                    continue

                # --- контуры ---
                preproc = []
                for outline in all_outlines:
                    interp = interpolate(outline, k_sampling_points)
                    preproc.append(preprocess(interp))

                # --- матрица расстояний ---
                l2_matrix = np.zeros((T, T), dtype=float)
                centered = [p - np.mean(p, axis=0, keepdims=True) for p in preproc]

                for a in range(T):
                    for b in range(a + 1, T):
                        dist = compute_Wass_distance(centered[a], centered[b])
                        l2_matrix[a, b] = dist
                        l2_matrix[b, a] = dist

                print(f"Cell {cell_number} L2 matrix:\n", l2_matrix)

                # --- heatmap l2_matrix ---
                im = ax_mat.imshow(l2_matrix, cmap="viridis", origin="lower")
                ax_mat.set_title(f"Cell {cell_number} – L2 Matrix", fontsize=8)
                ax_mat.set_xlabel("Frame", fontsize=6)
                ax_mat.set_ylabel("Frame", fontsize=6)
                ax_mat.tick_params(axis='both', which='major', labelsize=6)
                fig.colorbar(im, ax=ax_mat, fraction=0.046, pad=0.04)

                # --- кластеризация ---
                clustering = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=4.0,    # нормальный порог
                    metric="precomputed",
                    linkage="average",
                ).fit(l2_matrix)

                labels = clustering.labels_
                cmap = "plasma"

                # --- траектория ---
                if centroids.ndim == 2 and centroids.shape[0] >= 2:
                    x = centroids[:, 0]
                    y = centroids[:, 1]

                    ax_traj.scatter(
                        x[1:], y[1:], c=labels[1:], cmap=cmap,
                        s=40, edgecolor='k', alpha=0.7
                    )

                    ax_traj.scatter(
                        x[0], y[0], c=labels[0], cmap=cmap,
                        s=50, edgecolor='k', label="Start", alpha=0.9
                    )

                    ax_traj.plot(x, y, '-',
                        color='gray', alpha=0.5
                    )

                    ax_traj.set_title(f"Cell {cell_number} – Trajectory", fontsize=8)
                    ax_traj.set_xlabel("X", fontsize=6)
                    ax_traj.set_ylabel("Y", fontsize=6)
                    ax_traj.tick_params(axis='both', which='major', labelsize=6)

                else:
                    ax_traj.axis('off')

            # закрываем пустые оси
            for k in range(2 * plots_per_page, rows * cols):
                axes[k].axis('off')

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

def _cluster_kmeans_with_silhouette(embedding, random_state=0, k_min=2, k_max=6):
    """
    Подбираем k по максимуму silhouette score в диапазоне [k_min, k_max].
    Возвращаем labels, k, sil_score.
    """
    n_samples = embedding.shape[0]
    # Бессмысленно пробовать k >= n_samples
    max_k = min(k_max, n_samples - 1)
    if max_k < k_min:
        # Слишком мало точек, чтобы кластеризовать
        labels = np.zeros(n_samples, dtype=int)
        return labels, 1, np.nan

    best_k = None
    best_score = -np.inf
    best_labels = None

    for k in range(k_min, max_k + 1):
        kmeans = KMeans(
            n_clusters=k,
            n_init=10,
            random_state=random_state,
        )
        labels = kmeans.fit_predict(embedding)

        # silhouette требует хотя бы 2 разных кластера
        if len(np.unique(labels)) < 2:
            continue

        score = silhouette_score(embedding, labels)

        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels

    if best_labels is None:
        # не удалось получить ≥2 кластера — считаем всё одним кластером
        labels = np.zeros(n_samples, dtype=int)
        return labels, 1, np.nan

    return best_labels, best_k, best_score


def _process_single_cell(args):
    """
    Хелпер для multiprocessing: считает всё для одной клетки.
    Никаких pyplot, только вычисления.
    """
    (
        plot_index,
        cell_path,
        k_sampling_points,
        random_state,
    ) = args

    cell_number = plot_index + 1

    if not os.path.isdir(cell_path):
        return {
            "index": plot_index,
            "valid": False,
        }

    (
        all_outlines,
        centr,
        fin_times,
    ) = get_outlines_centr_times(cell_path)

    centroids = np.array(centr)
    T = len(all_outlines)

    if T < 2:
        return {
            "index": plot_index,
            "valid": False,
        }

    # ---- предобработка контуров ----
    preproc = []
    for outline in all_outlines:
        interp = interpolate(outline, k_sampling_points)
        preproc.append(preprocess(interp))

    # ---- матрица расстояний (Wass / твоя метрика) ----
    l2_matrix = np.zeros((T, T), dtype=float)
    centered = [p - np.mean(p, axis=0, keepdims=True) for p in preproc]

    for a in range(T):
        for b in range(a + 1, T):
            dist = compute_Wass_distance(centered[a], centered[b])
            # если захочешь делить на разницу во времени:
            # dist = dist / max(fin_times[b] - fin_times[a], 1e-8)
            l2_matrix[a, b] = dist
            l2_matrix[b, a] = dist

    # ---- MDS ----
    mds = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=random_state,
        n_init=4,
        max_iter=300,
        n_jobs=1,  # важно: внутри процесса не параллелим ещё раз
    )
    embedding = mds.fit_transform(l2_matrix)

    # немного нормализуем
    embedding -= embedding.mean(axis=0)
    std = embedding.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    embedding /= std

    # ---- k-means + silhouette ----
    cluster_labels, best_k, sil_score = _cluster_kmeans_with_silhouette(
        embedding,
        random_state=random_state,
        k_min=2,
        k_max=6,  # можешь поменять диапазон
    )

    return {
        "index": plot_index,
        "valid": True,
        "cell_number": cell_number,
        "embedding": embedding,
        "centroids": centroids,
        "cluster_labels": cluster_labels,
        "best_k": best_k,
        "silhouette": sil_score,
    }


def generate_mds_pdf_with_trajectories(
    direct: str,
    output_pdf_path: str,
    plots_per_page: int = 6,
    cols: int = 4,
    rows: int = 3,
    k_sampling_points: int = 200,
    random_state: int = 42,
):
    all_cell_folders = [os.path.join(direct, f"cell_{i}") for i in range(1, 122)]
    total_cells = len(all_cell_folders)

    # ---------- ШАГ 1: параллельно считаем всё для каждой клетки ---------- #
    args_list = [
        (plot_index, all_cell_folders[plot_index], k_sampling_points, random_state)
        for plot_index in range(total_cells)
    ]

    results = [None] * total_cells
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with ctx.Pool() as pool:
        for res in pool.imap_unordered(_process_single_cell, args_list):
            idx = res["index"]
            results[idx] = res

    # ---------- ШАГ 2: только рисуем (без тяжёлых вычислений) ---------- #
    cmap = "plasma"

    with PdfPages(output_pdf_path) as pdf:
        for page_start in range(0, total_cells, plots_per_page):
            fig, axes = plt.subplots(rows, cols, figsize=(11.69, 8.27))
            axes = axes.flatten()

            for i in range(plots_per_page):
                plot_index = page_start + i
                umap_ax_idx = 2 * i
                traj_ax_idx = 2 * i + 1

                if umap_ax_idx >= len(axes) or traj_ax_idx >= len(axes):
                    break

                ax_mds = axes[umap_ax_idx]
                ax_traj = axes[traj_ax_idx]

                if plot_index >= total_cells:
                    ax_mds.axis("off")
                    ax_traj.axis("off")
                    continue

                res = results[plot_index]
                if res is None or not res.get("valid", False):
                    ax_mds.axis("off")
                    ax_traj.axis("off")
                    continue

                embedding = res["embedding"]
                centroids = res["centroids"]
                cluster_labels = res["cluster_labels"]
                cell_number = res["cell_number"]
                best_k = res["best_k"]
                sil_score = res["silhouette"]

                # ---- MDS-embedding с раскраской по k-means ----
                sc_mds = ax_mds.scatter(
                    embedding[:, 0],
                    embedding[:, 1],
                    c=cluster_labels,
                    cmap=cmap,
                    s=10,
                )

                if np.isfinite(sil_score):
                    ax_mds.set_title(
                        f"Cell {cell_number}\n"
                        f"k-means (k={best_k}), silhouette = {sil_score:.2f}",
                        fontsize=8,
                    )
                else:
                    ax_mds.set_title(
                        f"Cell {cell_number}\n"
                        f"k-means (k={best_k}), silhouette = N/A",
                        fontsize=8,
                    )

                ax_mds.set_aspect("equal", "datalim")
                ax_mds.tick_params(axis="both", which="major", labelsize=6)

                # ---- траектория центроидов ----
                if centroids.ndim == 2 and centroids.shape[0] >= 2:
                    x_coords = centroids[:, 0]
                    y_coords = centroids[:, 1]

                    ax_traj.scatter(
                        x_coords[:],
                        y_coords[:],
                        c=cluster_labels[:],
                        cmap=cmap,
                        marker="o",
                        edgecolor="k",
                        s=40,
                        alpha=0.7,
                    )
                    ax_traj.plot(
                        x_coords,
                        y_coords,
                        linestyle="-",
                        color="gray",
                        alpha=0.5,
                    )

                    ax_traj.set_title("Trajectory", fontsize=8)
                    ax_traj.set_xlabel("X", fontsize=6)
                    ax_traj.set_ylabel("Y", fontsize=6)
                    ax_traj.tick_params(axis="both", which="major", labelsize=6)
                else:
                    ax_traj.axis("off")

            # выключаем лишние оси, если остались
            for k in range(2 * plots_per_page, rows * cols):
                axes[k].axis("off")

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

def generate_mds_pdf_with_trajectories_filtration(
    direct: str,
    output_pdf_path: str,
    plots_per_page: int = 6,
    cols: int = 4,
    rows: int = 3,
    k_sampling_points: int = 200,
    random_state: int = 42,
    filter_clusters: bool = True,
    allowed_k: list[int] | None = None,
):
    """
    Строит PDF с:
      - MDS-embedding (слева, раскрашенное по k-means-кластерам),
      - физической траекторией центроидов (справа, с теми же цветами).

    Параметры фильтрации:
        filter_clusters:
            Если True — отрезаем клетки, где best_k < 2.
        allowed_k:
            Если список значений (например [2,3]) —
            в PDF попадут только клетки с best_k из этого списка.
            Если None — не фильтруем по конкретному k.
    """
    all_cell_folders = [os.path.join(direct, f"cell_{i}") for i in range(1, 122)]
    total_cells = len(all_cell_folders)

    # ---------- ШАГ 1: параллельно считаем всё для каждой клетки ---------- #
    args_list = [
        (plot_index, all_cell_folders[plot_index], k_sampling_points, random_state)
        for plot_index in range(total_cells)
    ]

    results = [None] * total_cells
    import multiprocessing as mp
    ctx = mp.get_context("spawn")  # для надёжности, особенно под Mac/Windows
    with ctx.Pool() as pool:
        for res in pool.imap_unordered(_process_single_cell, args_list):
            idx = res["index"]
            results[idx] = res

    # ---------- ШАГ 2: рисуем в PDF ---------- #
    cmap = "plasma"

    with PdfPages(output_pdf_path) as pdf:
        for page_start in range(0, total_cells, plots_per_page):
            fig, axes = plt.subplots(rows, cols, figsize=(11.69, 8.27))
            axes = axes.flatten()

            for i in range(plots_per_page):
                plot_index = page_start + i
                umap_ax_idx = 2 * i
                traj_ax_idx = 2 * i + 1

                if umap_ax_idx >= len(axes) or traj_ax_idx >= len(axes):
                    break

                ax_mds = axes[umap_ax_idx]
                ax_traj = axes[traj_ax_idx]

                if plot_index >= total_cells:
                    ax_mds.axis("off")
                    ax_traj.axis("off")
                    continue

                res = results[plot_index]
                if res is None or not res.get("valid", False):
                    ax_mds.axis("off")
                    ax_traj.axis("off")
                    continue

                embedding = res["embedding"]
                centroids = res["centroids"]
                cluster_labels = res["cluster_labels"]
                cell_number = res["cell_number"]
                best_k = res["best_k"]
                sil_score = res["silhouette"]

                # --- ФИЛЬТРАЦИЯ ПО ЧИСЛУ КЛАСТЕРОВ ---
                if filter_clusters and (best_k is None or best_k < 2):
                    ax_mds.axis("off")
                    ax_traj.axis("off")
                    continue

                if allowed_k is not None and best_k not in allowed_k:
                    ax_mds.axis("off")
                    ax_traj.axis("off")
                    continue
                # -------------------------------------

                # ---- MDS-embedding ----
                ax_mds.scatter(
                    embedding[:, 0],
                    embedding[:, 1],
                    c=cluster_labels,
                    cmap=cmap,
                    s=10,
                )

                if np.isfinite(sil_score):
                    ax_mds.set_title(
                        f"Cell {cell_number}\n"
                        f"k={best_k}, silhouette={sil_score:.2f}",
                        fontsize=8,
                    )
                else:
                    ax_mds.set_title(
                        f"Cell {cell_number}\n"
                        f"k={best_k}, silhouette=N/A",
                        fontsize=8,
                    )

                ax_mds.set_aspect("equal", "datalim")
                ax_mds.tick_params(axis="both", which="major", labelsize=6)

                # ---- Траектория центроидов ----
                if centroids.ndim == 2 and centroids.shape[0] >= 2:
                    x_coords = centroids[:, 0]
                    y_coords = centroids[:, 1]

                    ax_traj.scatter(
                        x_coords[:],
                        y_coords[:],
                        c=cluster_labels[:],
                        cmap=cmap,
                        marker="o",
                        edgecolor="k",
                        s=40,
                        alpha=0.7,
                    )
                    ax_traj.plot(
                        x_coords,
                        y_coords,
                        linestyle="-",
                        color="gray",
                        alpha=0.5,
                    )

                    ax_traj.set_title("Trajectory", fontsize=8)
                    ax_traj.set_xlabel("X", fontsize=6)
                    ax_traj.set_ylabel("Y", fontsize=6)
                    ax_traj.tick_params(axis="both", which="major", labelsize=6)
                else:
                    ax_traj.axis("off")

            # Гасим оставшиеся пустые оси, если они есть
            for k in range(2 * plots_per_page, rows * cols):
                axes[k].axis("off")

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
glob_filtered_folder = '/home/pavel/cell_morphology/filtered_data/cells_filtered/'
def main():
    glob_filtered_folder = '/home/pavel/cell_morphology/filtered_data/cells_filtered/'
    #generate_umap_pdf_with_trajectories_OT_modes(glob_filtered_folder, "/home/pavel/cell_morphology/filtered_data/time_events_filtered.h5", "umap_all_cells_OT.pdf")
    #generate_umap_pdf_with_trajectories_OT(glob_filtered_folder, "umap_all_cells_OT_coloring.pdf")
    #generate_umap_pdf_with_trajectories_OT_clustering(glob_filtered_folder, "umap_all_cells_OT_clustering.pdf")
    #generate_umap_pdf_with_trajectories_OT_clustering_no_gromov(glob_filtered_folder, "umap_all_cells_OT_clustering_no_gromov.pdf")
    #generate_trajectory_ot_clustering(glob_filtered_folder, "clustering_distance_matrix.pdf")
    #generate_mds_pdf_with_trajectories(glob_filtered_folder, "clustering_mds_trajectories.pdf")
    generate_mds_pdf_with_trajectories_filtration(
        glob_filtered_folder,
        "clustering_mds_trajectories.pdf",
        allowed_k=list(range(3, 20)),
    )



if __name__ == "__main__":
    main()
from protusion_stats import conformal_representation
import os
import numpy as np
from scipy.ndimage import gaussian_filter1d
from numba import njit
from enum import IntEnum, auto
import matplotlib.pyplot as plt
import ot
from matplotlib.backends.backend_pdf import PdfPages
SIGMA = 2
from matplotlib.patches import ConnectionPatch


@njit
def ang_gap_deg(a, b):
    d = np.abs(a - b)
    return np.minimum(d, 360.0 - d)

@njit
def first_filter_extrema(xs, extrema, values, classif, min_width):
    n = len(extrema)
    keep = np.zeros(n, dtype=np.bool_)
    if n == 0:
        return extrema, values, classif

    for i in range(n):
        this_type = classif[i]
        # ищем ближайших слева/справа ТОГО ЖЕ ТИПА
        left = -1
        for k in range(i-1, -1, -1):
            if classif[k] == this_type:
                left = k
                break
        right = -1
        for k in range(i+1, n):
            if classif[k] == this_type:
                right = k
                break

        ok = False
        if left != -1:
            ok = ok or (ang_gap_deg(xs[extrema[i]], xs[extrema[left]]) > min_width)
        if right != -1:
            ok = ok or (ang_gap_deg(xs[extrema[i]], xs[extrema[right]]) > min_width)

        # если по обе стороны такого же типа нет (крайний) — оставляем
        if left == -1 and right == -1:
            ok = True

        keep[i] = ok

    return extrema[keep], values[keep], classif[keep]
@njit
def second_filter_extrema(xs,ys,extrema, values, classif, min_depth):           #excluding flat extrema
    extr_len=len(extrema)
    bool_arr=np.zeros(extr_len,dtype=np.bool_)
    if extr_len<=2:
        return extrema, values, classif
    bool_arr[0],bool_arr[-1]=True,True
    for i in range(1,extr_len-1):               
        a_orth=np.array([[values[i-1]-values[i+1],1000*xs[extrema[i+1]]-1000*xs[extrema[i-1]]]])  #vector joining two minima or two maxima
        len_pts=extrema[i+1]-extrema[i-1]
        vect_list=np.ones((2,len_pts))*np.array([[1000*xs[extrema[i-1]]],[values[i-1]]])
        vect_list[0,:]-=xs[extrema[i-1]:extrema[i+1]]
        vect_list[1,:]-=ys[extrema[i-1]:extrema[i+1]]
        proj=np.abs(a_orth@vect_list)           
        bool_arr[i]=np.max(proj)/np.linalg.norm(a_orth)>min_depth   # distance to the line between the optima
    return extrema[bool_arr], values[bool_arr], classif[bool_arr]


@njit
def alternating_feature(extrema, values, classif):
    extr_len=len(extrema)
    bool_arr=np.ones(extr_len,dtype=np.bool_)
    if extr_len<=1:
        return extrema, values, classif
    for i in range(extr_len-1):               #excluding twice the same feature
        if classif[i]==classif[i+1]:
            bool_arr[i]=False
            match classif[i]:
                case Feature.PEAK:
                    j=i+(values[i+1]>values[i])
                    classif[i+1]=classif[j]
                    values[i+1]=values[j]
                    extrema[i+1]=extrema[j]
                case Feature.TROUGH:
                    j=i+(values[i+1]<values[i])
                    classif[i+1]=classif[j]
                    values[i+1]=values[j]
                    extrema[i+1]=extrema[j]
    return extrema[bool_arr], values[bool_arr], classif[bool_arr]



def circ_extrema(xs_deg, ys):
    """
    Периодический поиск экстремумов по углу.
    xs_deg — монотонно возрастающие углы в градусах [0..360), ys — значения (например, Δr).
    Возвращает индексы локальных экстремумов в диапазоне 0..N-1.
    """
    xs = np.asarray(xs_deg, float)
    ys = np.asarray(ys, float)
    N = len(xs)
    if N < 3:
        return np.array([], dtype=int)

    # добавим соседние точки, учитывая периодичность по углу
    xs_ext = np.r_[xs[-1] - 360.0, xs, xs[0] + 360.0]
    ys_ext = np.r_[ys[-1],         ys, ys[0]]

    der = (ys_ext[1:] - ys_ext[:-1]) / (xs_ext[1:] - xs_ext[:-1])
    is_pos = der >= 0
    sign_change = np.logical_xor(is_pos[1:], is_pos[:-1])
    intern = 1 + np.flatnonzero(sign_change)   # индексы в xs_ext

    # оставляем только центральные, соответствующие исходному xs
    mask = (intern >= 1) & (intern <= N)
    return (intern[mask] - 1).astype(int)      # обратно к 0..N-1
    




def resample_extrema(intern_extrema, xs, ys, min_width,min_depth):
    extrema = []
    values = []
    classif = []
    for i in intern_extrema:
        N   = len(xs)            
        #im1 = (i - 1) % N
        #ip1 = (i + 1) % N
        #a   = (ys[ip1] - 2.0*ys[i] + ys[im1]) / 2.0
        y = ys[i]
        extrema.append(i)
        values.append(y)
        #print("a ", a)
        if y < 0:
            classif.append(Feature.TROUGH)
        else:
            classif.append(Feature.PEAK)
    
    if len(extrema)<=2:
        return extrema, values, classif
    #print("extrema0 ",extrema)
    #print("classif0 ", classif)
    
    argsort = sorted(range(len(extrema)), key=extrema.__getitem__)
    extrema = np.array([extrema[i] for i in argsort])
    values = np.array([values[i] for i in argsort])
    classif = np.array([classif[i] for i in argsort])

    extrema, values, classif = first_filter_extrema(xs,extrema, values, classif,min_width)

    #print("extrema1 ",extrema)
    #print("classif1 ", classif)
    extrema, values, classif = alternating_feature(extrema, values, classif)

    #print("extrema2 ",extrema)
    #print("classif2 ", classif)
    extrema, values, classif = second_filter_extrema(xs,ys,extrema, values, classif, min_depth)

    #print("extrema3 ",extrema)
    #print("classif3 ", classif)

    return alternating_feature(extrema, values, classif)        


def dedupe_boundary(xs_deg, ys, peaks, troughs, min_width_deg):
    """
    Если первый и последний экстремум одного типа ближе min_width_deg
    по модульной угловой метрике, оставляем тот, у кого |ys| больше.
    """

    def mod_gap(a, b):
        d = abs(a - b)
        return min(d, 360.0 - d)

    def dedupe_one(arr):
        arr = np.asarray(arr, dtype=int)
        if arr.size < 2:
            return arr

        first, last = arr[0], arr[-1]
        if mod_gap(xs_deg[first], xs_deg[last]) < min_width_deg:
            # оставляем "сильнейший" по |ys|
            if abs(ys[first]) >= abs(ys[last]):
                arr = arr[:-1]
            else:
                arr = arr[1:]
        return arr

    peaks = dedupe_one(peaks)
    troughs = dedupe_one(troughs)
    return peaks, troughs

class Feature(IntEnum):
    PEAK = auto()
    TROUGH = auto()

def find_peaks_troughs(
    xs,
    ys,
    smooth_std,
    min_width,
    min_depth,
):



    xs_smooth = gaussian_filter1d(xs, smooth_std, mode="wrap")
    ys_smooth = gaussian_filter1d(ys, smooth_std, mode="wrap")
    der = (ys_smooth[1:] - ys_smooth[:-1]) / (xs_smooth[1:] - xs_smooth[:-1])

   

    intern_extrema = circ_extrema(xs_smooth, ys_smooth)
    extrema, _, classif = resample_extrema(intern_extrema,xs_smooth, ys_smooth, min_width,min_depth)
    

    peaks = []
    troughs = []

    #print("extrema ",extrema)
    #print("classif ", classif)
    for i in range(len(extrema)):
        match classif[i]:
            case Feature.PEAK:
                peaks.append(extrema[i])
            case Feature.TROUGH:
                troughs.append(extrema[i])
            case _:
                raise ValueError(
                    f"Unknown feature {classif[i]}, feature should "
                    f"be a {Feature.PEAK} or a {Feature.TROUGH}."
                )
    peaks = np.array(peaks)
    troughs = np.array(troughs)
    
    
    return xs, ys, peaks, troughs

def filter_by_height_curvature(height_sorted,
                               curvature_sorted,
                               peaks, troughs,
                               H_prot=None,H_int=None,  K_MIN=None):

    peaks = np.asarray(peaks, dtype=int)
    troughs = np.asarray(troughs, dtype=int)

    if H_prot is not None and H_int is not None:
        mask_p_h = np.abs(height_sorted[peaks])   >= H_prot if peaks.size   else np.array([], bool)
        #print(height_sorted)
        mask_t_h = height_sorted[troughs] <= -H_int if troughs.size else np.array([], bool)
        peaks    = peaks[mask_p_h]
        troughs  = troughs[mask_t_h]
    # print(curvature_sorted)
    if K_MIN is not None and curvature_sorted is not None:
        mask_p_k = np.abs(curvature_sorted[peaks])   >= K_MIN if peaks.size   else np.array([], bool)
        mask_t_k = np.abs(curvature_sorted[troughs]) >= K_MIN if troughs.size else np.array([], bool)
        peaks    = peaks[mask_p_k]
        troughs  = troughs[mask_t_k]

    return peaks, troughs

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
    #print(shifts)
    shifts = np.array(shifts, dtype=float)

    shifts = gaussian_filter1d(np.array(shifts), sigma=1, axis=0)
    return shifts

def op_index(cell_shape_1, cell_shape_2, index_from):


    N1 = len(cell_shape_1)
    N2 = len(cell_shape_2)
    a = np.ones(N1) / N1
    b = np.ones(N2) / N2

    M = np.linalg.norm(cell_shape_1[:, None, :] - cell_shape_2[None, :, :], axis=-1) ** 2
    T = ot.emd(a, b, M)

    mapping = np.argmax(T, axis=1)  # для каждого i в prev -> j в curr

    # index_from может быть int или np.ndarray
    if np.isscalar(index_from):
        return int(mapping[int(index_from)])
    else:
        index_from = np.asarray(index_from, dtype=int).ravel()
        return mapping[index_from]

def single_cell_tracking(direct, cell_number):
    all_cell_folders = [
        os.path.join(direct, ff)
        for ff in sorted(
            os.listdir(direct),
            key=lambda x: int(x.split("_")[1]) if "_" in x and x.split("_")[1].isdigit() else 0,
        )
    ]
    derivative_cmcf = []

    (
        all_outlines,
        _,
        all_outlines_cMCF_topography,
        all_outlines_curvature,
        disk_coors,
        centr,
        fin_times,
    ) = conformal_representation(all_cell_folders[cell_number])

    all_outlines_cMCF_topography = [
    smooth_topo(t.copy()) for t in all_outlines_cMCF_topography]

    disk_center = np.mean(disk_coors, axis=0)
    disk_radius = np.mean(np.linalg.norm(disk_coors - disk_center, axis=1))
    shifts = get_shift(all_outlines, centr, fin_times)
    index_from = 0 
    cell = []
    for jj, (
            outline,
            topo_coords,
            curvatures,
            shift,
            time,
        ) in enumerate(
            zip(
                all_outlines,
                all_outlines_cMCF_topography,
                all_outlines_curvature,
                shifts,
                fin_times,
            )
        ):

        topo_coords = all_outlines_cMCF_topography[jj]
        prev_topo = all_outlines_cMCF_topography[jj - 1].copy()

        # обновляем якорь только начиная со 2-го кадра
        if jj > 0:
            prev_topo = all_outlines_cMCF_topography[jj - 1].copy()
            old = index_from
            index_from = op_index(prev_topo, topo_coords, index_from)

            # DEBUG
            if jj in [8, 9, 10]:  # выбери свои кадры
                n = len(topo_coords)
                d = (index_from - old) % n
                d = min(d, n - d)
                print(f"[anchor] frame {jj}: old={old} -> new={index_from} | jump={d} / {n}")

        # дальше — ровно как было: строим theta_deg относительно index_from, сортируем и т.д.
        theta_ref = np.arctan2(
            topo_coords[index_from][0] - disk_center[0],
            topo_coords[index_from][1] - disk_center[1],
        )
        theta_raw = np.arctan2(
            topo_coords[:, 0] - disk_center[0],
            topo_coords[:, 1] - disk_center[1],
        )
        theta_deg = np.degrees((theta_raw - theta_ref) % (2 * np.pi))

        topo_height = np.linalg.norm(topo_coords - disk_center, axis=1) - disk_radius

        sort_idx = np.argsort(theta_deg)
        theta_sorted  = theta_deg[sort_idx]
        height_sorted = topo_height[sort_idx]
        print("peaks before")
        xs, ys, peaks, troughs = find_peaks_troughs(theta_sorted, height_sorted, 2, 15, 1)
        if(int(time)==9):
            print(ys)
            print(peaks)
        print("end")
        peaks, troughs = dedupe_boundary(theta_sorted, height_sorted, peaks, troughs, min_width_deg=15.0)

        curv_sorted = curvatures[sort_idx] if curvatures is not None else None

        max_height = np.max(height_sorted)
        min_height = np.min(height_sorted)

        H_prot = 4
        H_int  = 1.2 * np.abs(0.85 * min_height)
        K_MIN  = 0.001
        # if(int(time)==9):
        #     print(peaks)
        peaks, troughs = filter_by_height_curvature(
            height_sorted, curv_sorted, peaks, troughs,
            H_prot, H_int, K_MIN
        )

        peaks   = np.asarray(peaks, dtype=int) if len(peaks) else np.asarray([], dtype=int)
        troughs = np.asarray(troughs, dtype=int) if len(troughs) else np.asarray([], dtype=int)
        # if(int(time)==9):
        #     print(peaks)
        frame_dict = {
            "xs": np.asarray(theta_sorted, dtype=np.float64),
            "ys": np.asarray(height_sorted, dtype=np.float64),
            "curv": np.asarray(curv_sorted, dtype=np.float64) if curv_sorted is not None else None,
            "peaks": peaks,
            "troughs": troughs,
            "order_idx": sort_idx.astype(int),
            "timestamp": float(time),
        }
        # print("frame_dict")
        # if(int(time)==9):
        #     print(frame_dict)

        cell.append(frame_dict)

        
    return cell, all_outlines, all_outlines_cMCF_topography



def create_peaks_troughs_list(cell):
    """
    Формирует список всех экстремумов (протрузий и интрузий) из cell.

    Возвращает np.ndarray формы (N, 8):
    [global_id, frame_id, type, theta_deg, dr, time, outline_idx, lineage_id]

    где:
      global_id    — уникальный ID этой точки (int)
      frame_id     — номер кадра
      type         — 1 = протрузия (peak), 0 = впадина (trough)
      theta_deg    — угол (в градусах) в отсортированной параметризации
      dr           — высота (радиальная аномалия)
      time         — timestamp кадра
      outline_idx  — индекс точки на исходном контуре all_outlines[frame_id]
      lineage_id   — изначально -1, потом будет заполнен алгоритмом родословных
    """
    pnt_list = []
    gid = 0  # глобальный счётчик точек

    for frame_id, frame_data in enumerate(cell):

        time = frame_data["timestamp"]
        order_idx = frame_data["order_idx"]  # мэппинг sorted -> индекс точки на контуре

        # --- протрузии (peaks) ---
        for local_idx in frame_data["peaks"]:
            outline_idx = int(order_idx[local_idx])
            theta = frame_data["xs"][local_idx]
            dr    = frame_data["ys"][local_idx]
            curv_val = frame_data["curv"][local_idx]

            pnt_list.append([
                gid,         # global_id
                frame_id,    # frame_id
                1,           # type = 1 (peak)
                theta,       # theta_deg
                dr,          # dr
                time,        # time
                outline_idx, # outline_idx
                -1,           # lineage_id (будет заполнено позже)
                curv_val
            ])
            gid += 1

        # --- впадины (troughs) ---
        for local_idx in frame_data["troughs"]:
            outline_idx = int(order_idx[local_idx])
            theta = frame_data["xs"][local_idx]
            dr    = frame_data["ys"][local_idx]

            pnt_list.append([
                gid,         # global_id
                frame_id,    # frame_id
                0,           # type = 0 (trough)
                theta,       # theta_deg
                dr,          # dr
                time,        # time
                outline_idx, # outline_idx
                -1           # lineage_id
            ])
            gid += 1

    return np.array(pnt_list, dtype=float)



@njit
def ang_mod_dist_deg(a, b):
    d = np.abs(a - b)
    return np.minimum(d, 360.0 - d)

@njit
def pnt_link_matrix(pnt_list, max_xdrift, max_ydrift, max_time):
    matlen = len(pnt_list)
    time_mat = np.zeros((matlen, matlen), dtype=np.int32)
    dist_mat = np.zeros((matlen, matlen))
    for i in range(matlen-1):
        for j in range(i+1,matlen):
            time=np.abs(pnt_list[i][5]-pnt_list[j][5]) 
            dist_x= ang_mod_dist_deg(pnt_list[i][3],pnt_list[j][3])
            
            cond =  dist_x < max_xdrift \
                and np.abs(pnt_list[i][4]-pnt_list[j][4]) < max_ydrift \
                and 0 < time < max_time \
                and pnt_list[i][2] == pnt_list[j][2]
            if cond :
                time_mat[j,i] = time 
                dist_mat[j,i] = dist_x
            
    return time_mat, dist_mat





def final_pnt_link_matrix(new_pnt_list, pnt_ROI, final_max_xdrift, max_ydrift, max_time):
    matlen = len(new_pnt_list)
    time_mat = np.zeros((matlen, matlen), dtype=np.int32)
    dist_mat = np.zeros((matlen, matlen), dtype=float)

    DEBUG = True

    for i in range(matlen - 1):
        for j in range(i + 1, matlen):
            ti = new_pnt_list[i, 5]
            tj = new_pnt_list[j, 5]


            # циклический dθ
            d = abs(new_pnt_list[i, 3] - new_pnt_list[j, 3])
            dist_x = min(d, 360 - d)

            # НЕ циклический dR / амплитуда
            dist_y = abs(new_pnt_list[i, 4] - new_pnt_list[j, 4])

            same_type = (new_pnt_list[i, 2] == new_pnt_list[j, 2])

            lid_i = int(new_pnt_list[i, 7])
            lid_j = int(new_pnt_list[j, 7])
            
     

            e_i = end_of_lineage(i, new_pnt_list)
            e_j = end_of_lineage(j, new_pnt_list)

            # --- направление 1: i -> j ---
            # print(ti)
            # print(tj)

            time_ij = tj - ti
            # print(time_ij)
            cond_ij = (
                dist_x < final_max_xdrift
                and dist_y < max_ydrift
                and 0 < time_ij < max_time
                and same_type
                and (e_i == 1 and e_j == -1)
            )

            # --- направление 2: j -> i ---
            time_ji = ti - tj
            cond_ji = (
                dist_x < final_max_xdrift
                and dist_y < max_ydrift
                and 0 < time_ji < max_time
                and same_type
                and (e_j == 1 and e_i == -1)
            )

            if DEBUG and lid_i != -1 and lid_j != -1 and lid_i != lid_j:
                print(
                    f"[PAIR] lids {lid_i}–{lid_j} | "
                    f"frames {new_pnt_list[i,1]:.0f}–{new_pnt_list[j,1]:.0f} | "
                    f"time_ij={time_ij:.1f}, time_ji={time_ji:.1f}, "
                    f"dθ={dist_x:.2f}, dR={dist_y:.2f}, "
                    f"type={'OK' if same_type else 'DIFF'}, "
                    f"end_i={e_i}, end_j={e_j}, "
                    f"cond_ij={cond_ij}, cond_ji={cond_ji}"
                )

            # Записываем ориентированные ребра в матрицу
            if cond_ij:
                time_mat[j, i] = int(time_ij)
                dist_mat[j, i] = dist_x

            if cond_ji:
                time_mat[i, j] = int(time_ji)
                dist_mat[i, j] = dist_x

    return time_mat, dist_mat


# def end_of_lineage(i, new_pnt_list, pnt_ROI):
#     lineage = int(new_pnt_list[i,-1])
#     if lineage == -1:
#         return 0
#     print("test ROI", pnt_ROI)
#     if new_pnt_list[i,0] == pnt_ROI[lineage][0]:
#         return -1
#     print("nontest")
#     return 1
    

def end_of_lineage(i, new_pnt_list, lid_col=7, frame_col=1):
    lid = int(new_pnt_list[i, lid_col])
    if lid == -1:
        return 0

    same = new_pnt_list[:, lid_col].astype(int) == lid
    frames = new_pnt_list[same, frame_col]

    if frames.size == 0:
        return 0

    fi = new_pnt_list[i, frame_col]
    fmin = frames.min()
    fmax = frames.max()

    if fmin == fmax:
        return 0

    if fi == fmin:
        return -1   # start
    if fi == fmax:
        return 1    # end
    return 0

def update_list(pnt_list, time_mat, dist_mat):
    index = 0
    generations = int(np.max(pnt_list[:,1]))
    last_gen = find_gen(pnt_list,generations)
    for elem in last_gen:
        pnt_list[int(elem[0]),7] = index
        index+=1
    for gen_num in range(generations-1,-1,-1):
        index = first_update_pnt_list(pnt_list, gen_num, time_mat, dist_mat,index)
    return pnt_list
def fill_gaps(pnt_list, cell, all_outlines, outlines_CMF, max_dtheta_deg):
    """
    Fill missing frames INSIDE each lineage (lid) by inserting synthetic points.

    Rules:
      - Never create duplicates inside a lineage for same (lid, frame, typ).
      - Never create global duplicates for same (frame, typ, outline_idx).
        If best candidate is occupied -> try next best candidate in window.
      - Curvature is taken SAME for peaks and troughs: curv_arr[k] if exists else nan.
    """
    if pnt_list is None or len(pnt_list) == 0:
        return np.asarray(pnt_list)

    pnt_list = np.asarray(pnt_list)

    # require at least 8 cols: [gid, frame, typ, theta, dr, time, outline_idx, lid]
    if pnt_list.ndim != 2 or pnt_list.shape[1] < 8:
        raise ValueError(f"pnt_list must be 2D with >=8 cols, got shape {pnt_list.shape}")

    # ensure 9 cols (curv at col8)
    want_cols = 9
    if pnt_list.shape[1] < want_cols:
        tmp = np.zeros((pnt_list.shape[0], want_cols), dtype=float)
        tmp[:, :pnt_list.shape[1]] = pnt_list.astype(float, copy=False)
        pnt = tmp
    else:
        pnt = pnt_list.astype(float, copy=False)

    result_rows = [row.copy() for row in pnt]

    # guards
    seen_lft = set()  # (lid, frame, typ)
    seen_fto = set()  # (frame, typ, outline_idx)

    for row in result_rows:
        fr  = int(row[1])
        typ = int(row[2])
        out = int(row[6])
        lid = int(row[7])
        seen_fto.add((fr, typ, out))
        if lid != -1:
            seen_lft.add((lid, fr, typ))

    def pick_free_k(xs, ys, order_idx, theta0, typ, frame_f):
        """
        Choose best candidate k in window around theta0.
        If (frame_f, typ, out_idx) occupied -> try next best.
        Return k or None.
        """
        dtheta = np.abs(xs - theta0)
        dtheta = np.minimum(dtheta, 360.0 - dtheta)
        idxs = np.flatnonzero(dtheta < max_dtheta_deg)
        if idxs.size == 0:
            return None

        # sort by "goodness"
        if typ == 1:  # peak -> max dr
            idxs = idxs[np.argsort(ys[idxs])[::-1]]
        else:         # trough -> min dr
            idxs = idxs[np.argsort(ys[idxs])]

        for k in idxs:
            out_idx = int(order_idx[k])
            if (frame_f, typ, out_idx) not in seen_fto:
                return int(k)
        return None

    lids = np.unique(pnt[:, 7].astype(int))
    for lid in lids:
        if lid == -1:
            continue

        lineage_mask = (pnt[:, 7].astype(int) == lid)
        lineage_points = pnt[lineage_mask]
        if lineage_points.shape[0] < 2:
            continue

        lineage_points = lineage_points[np.argsort(lineage_points[:, 1].astype(int))]

        for a, b in zip(lineage_points[:-1], lineage_points[1:]):
            frame_a = int(a[1])
            frame_b = int(b[1])
            if(lid==7):
                print("frame_0a_", frame_a)
                print("frame_0b_", frame_b)
            #print(f"[fill_gaps] lid={lid} pair: {frame_a} -> {frame_b} (gap={frame_b-frame_a-1}) typ={int(a[2])}")
            if frame_b <= frame_a + 1:
                continue

            typ = int(a[2])      # keep type fixed for gap fill
            idx_a = int(a[6])    # outline idx at frame_a
            if(lid==7):
                print("frame_1a_", frame_a)
                print("frame_1b_", frame_b)   
            for f in range(frame_a + 1, frame_b):
                if(lid==7):
                    print("frame_a_", frame_a)
                    print("frame_b_", frame_b)
                # already exists in this lineage
                if (lid, f, typ) in seen_lft:
                    continue

                # propagate outline index from frame_a to f
                curr_idx = idx_a
                ok = True
                for step_frame in range(frame_a + 1, f + 1):
                    if step_frame - 1 < 0 or step_frame >= len(outlines_CMF):
                        ok = False
                        break
                    if(lid==7):
                        print("prev_idx_",curr_idx)
                    curr_idx = op_index(outlines_CMF[step_frame - 1],
                                        outlines_CMF[step_frame],
                                        curr_idx)
                    if(lid==7):
                        print("curr_idx_", curr_idx)
                if not ok:
                    continue

                idx_prop = int(curr_idx)
                
                # access frame data
                if f < 0 or f >= len(cell):
                    continue
                frame_dict = cell[f]

                order_idx = np.asarray(frame_dict["order_idx"], dtype=int)
                xs = np.asarray(frame_dict["xs"], dtype=float)
                ys = np.asarray(frame_dict["ys"], dtype=float)
                #print(f"  [f={f}] idx_a={idx_a} -> idx_prop={idx_prop} | order_idx len={len(frame_dict['order_idx'])}")
                # map propagated idx to sorted profile location
                where0 = np.where(order_idx == idx_prop)[0]
                if where0.size == 0:
                    #print(f"  [f={f}] FAIL: idx_prop={idx_prop} not in order_idx")
                    continue
                
                k0 = int(where0[0])
                theta0 = float(xs[k0])
                if(lid==7):
                    print("k0_", k0)
                # pick best FREE candidate in theta window
                k = pick_free_k(xs, ys, order_idx, theta0, typ, f)
                if k is None:
                    continue

                out_idx = int(order_idx[k])
                if(lid==7):
                    print("outputindex_",out_idx)

                theta = float(xs[k])
                dr = float(ys[k])
                tstamp = float(frame_dict.get("timestamp", np.nan))

                # curvature: same rule for peaks & troughs
                curv_val = np.nan
                if "curv" in frame_dict and frame_dict["curv"] is not None:
                    curv_arr = np.asarray(frame_dict["curv"], dtype=float)
                    if 0 <= k < len(curv_arr):
                        curv_val = float(curv_arr[k])

                synthetic_row = np.array(
                    [-1, f, typ, theta, dr, tstamp, out_idx, lid, curv_val],
                    dtype=float
                )

                # append + update guards
                result_rows.append(synthetic_row)
                seen_fto.add((f, typ, out_idx))
                seen_lft.add((lid, f, typ))

    return np.vstack(result_rows)

def find_gen(pnt_list, generations):
    mask = pnt_list[:,1] == generations
    new_list = pnt_list[mask]
    return new_list[np.argsort(new_list[:, 3])]



def first_update_pnt_list(pnt_list, gen_num, time_mat, dist_mat, index): # create ROIs for elements that are the closest to eachother in both directions
    gen = find_gen(pnt_list, gen_num)
    for elem in gen:
        
        feature_number = int(elem[0])
        compar_ind, closest_ind_compar = closest_neighbour(feature_number, time_mat, dist_mat)
        
        if closest_ind_compar == feature_number:
            if pnt_list[compar_ind, 7] == -1:
                pnt_list[compar_ind, 7] = index
                index += 1
            pnt_list[feature_number, 7] = pnt_list[compar_ind, 7] 
                    
    return index



def closest_neighbour(feature_number, time_mat, dist_mat):
    compar_ind, closest_ind_compar = -1, -1
    mask = time_mat[:,feature_number]>0
    if np.any(mask):
        min_time = np.min(time_mat[:,feature_number][mask])
        tent_ind = np.nonzero(time_mat[:,feature_number] == min_time)[0]
        closest_pt = np.argmin(dist_mat[:,feature_number][tent_ind])
        
        compar_ind = tent_ind[closest_pt]   # index of closest point linked with elem
        compar_mask = np.ravel(time_mat[compar_ind,:]>0)
        
        if np.any(compar_mask):
            compar_line = np.ravel(time_mat[compar_ind,:])
            compar_min_time = np.min(compar_line[compar_mask])
            compar_tent_ind = np.nonzero(compar_line  == compar_min_time)[0]
            compar_closest_pt = np.argmin(np.ravel(dist_mat[compar_ind,:])[compar_tent_ind])
            closest_ind_compar = compar_tent_ind[compar_closest_pt]
    return compar_ind, closest_ind_compar  # closest point to feature_number, closest point to closest_ind

def reconstruct_ROI(pnt_list):
    ROI_dict = {}
    print(pnt_list[:, 7])
    for lid in np.unique(pnt_list[:, 7]):
        if lid == -1:
            continue
        idxs = np.nonzero(pnt_list[:, 7] == lid)[0]
        idxs = idxs[np.argsort(pnt_list[idxs, 1])]
        ROI_dict[lid] = idxs
    return ROI_dict
def dedupe_by_lid_frame_type(pnt_list):
    """
    Enforce uniqueness of points within each (lid, frame, type).
    Keep:
      - peaks (type=1): max dr
      - troughs (type=0): min dr
    Assumes columns:
      col1=frame_id, col2=type, col4=dr, col7=lid
    """
    if pnt_list is None or len(pnt_list) == 0:
        return pnt_list

    p = np.asarray(pnt_list, dtype=float)

    # key = (lid, frame, type)
    lids  = p[:, 7].astype(int)
    frs   = p[:, 1].astype(int)
    typs  = p[:, 2].astype(int)
    dr    = p[:, 4].astype(float)

    keys = np.stack([lids, frs, typs], axis=1)

    # group indices by key using np.unique
    uniq_keys, inv = np.unique(keys, axis=0, return_inverse=True)

    keep = np.zeros(len(p), dtype=bool)

    for g in range(len(uniq_keys)):
        idxs = np.where(inv == g)[0]
        if idxs.size == 1:
            keep[idxs[0]] = True
            continue

        typ = int(uniq_keys[g, 2])
        if typ == 1:
            best = idxs[np.argmax(dr[idxs])]
        else:
            best = idxs[np.argmin(dr[idxs])]
        keep[best] = True

    return p[keep]



def update_list_non_crossing(new_pnt_list, new_gid, pnt_list, new_time_mat, new_dist_mat, final_max_xdrift):
    possible_link = []
    for feature_number in range(len(new_pnt_list)):
        compar_ind, closest_ind_compar = closest_neighbour(feature_number, new_time_mat, new_dist_mat)
        if closest_ind_compar == feature_number:
            gA = int(new_gid[compar_ind])         # <-- вместо col0
            gB = int(new_gid[feature_number])     # <-- вместо col0
            distAB = new_dist_mat[compar_ind, feature_number]

            if gA < 0 or gB < 0:
                continue

            possible_link.append([gA, gB, distAB])
    print("\n=== DEBUG: possible_link BEFORE drift filter ===")
    print(possible_link)

    if len(possible_link) >= 1 :
        possible_link = np.array(possible_link)
        mask_to_erase = np.ones(len(possible_link), dtype=bool)
        
        for elem in possible_link[:,0]:
            arg_mask=np.logical_or(possible_link[:,0]==elem,possible_link[:,1]==elem)
            args = np.nonzero(arg_mask)[0]
            if sum(possible_link[args,2]) > final_max_xdrift:
                mask_to_erase[args] = False
                
        possible_link = possible_link[mask_to_erase]
        if len(possible_link) >= 1:
            possible_link = possible_link[np.argsort(possible_link[:,-1])]
            pnt_list = glue_non_crossing(pnt_list, possible_link)
    
    return pnt_list
def merge_overlapping_lineages(
    pnt_list: np.ndarray,
    *,
    theta_thr: float = 12.0,   # degrees
    dr_thr: float = 0.35,      # dr units
    min_overlap: int = 3,      # number of shared frames
    same_type_only: bool = True,
    choose_keep: str = "longer",  # "longer" | "min_lid"
):
    """
    Merge two lineages if they overlap in time and are close in (theta, dr).

    Conditions for merge (for common frames):
      - count(common_frames) >= min_overlap
      - median(|Δθ|) <= theta_thr
      - max(|Δθ|)    <= 2*theta_thr
      - median(|Δdr|) <= dr_thr
      - max(|Δdr|)    <= 2*dr_thr

    Columns assumed:
      frame_id -> col1
      type     -> col2
      theta    -> col3
      dr       -> col4
      lid      -> col7
    """
    if pnt_list.size == 0:
        return pnt_list

    lids = np.unique(pnt_list[:, 7].astype(int))
    lids = [lid for lid in lids if lid != -1]
    if len(lids) <= 1:
        return pnt_list

    # collect per-lid info
    per = {}
    for lid in lids:
        rows = np.flatnonzero(pnt_list[:, 7].astype(int) == lid)
        frames = pnt_list[rows, 1].astype(int)
        per[lid] = {
            "rows": rows,
            "frame_to_row": {int(f): int(r) for f, r in zip(frames, rows)},
        }

    # union–find
    parent = {lid: lid for lid in lids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if choose_keep == "min_lid":
            keep, drop = (ra, rb) if ra < rb else (rb, ra)
        else:  # "longer"
            la = len(per[ra]["rows"])
            lb = len(per[rb]["rows"])
            if la > lb:
                keep, drop = ra, rb
            elif lb > la:
                keep, drop = rb, ra
            else:
                keep, drop = (ra, rb) if ra < rb else (rb, ra)
        parent[drop] = keep

    lids_sorted = sorted(lids)
    for i in range(len(lids_sorted)):
        for j in range(i + 1, len(lids_sorted)):
            a = find(lids_sorted[i])
            b = find(lids_sorted[j])
            if a == b:
                continue

            fa = set(per[a]["frame_to_row"].keys())
            fb = set(per[b]["frame_to_row"].keys())
            common = sorted(fa & fb)
            if len(common) < min_overlap:
                continue

            dtheta = []
            ddr = []

            for f in common:
                ra = per[a]["frame_to_row"][f]
                rb = per[b]["frame_to_row"][f]

                if same_type_only:
                    if int(pnt_list[ra, 2]) != int(pnt_list[rb, 2]):
                        dtheta = None
                        break

                dtheta.append(ang_gap_deg(
                    float(pnt_list[ra, 3]),
                    float(pnt_list[rb, 3])
                ))
                ddr.append(abs(
                    float(pnt_list[ra, 4]) -
                    float(pnt_list[rb, 4])
                ))

            if dtheta is None:
                continue

            dtheta = np.array(dtheta)
            ddr = np.array(ddr)

            if (
                np.median(dtheta) <= theta_thr and
                np.max(dtheta)    <= 2 * theta_thr and
                np.median(ddr)    <= dr_thr and
                np.max(ddr)       <= 2 * dr_thr
            ):
                union(a, b)

    # relabel lids
    out = pnt_list.copy()
    out_lids = out[:, 7].astype(int)
    for lid in lids:
        out_lids[out_lids == lid] = find(lid)
    out[:, 7] = out_lids.astype(float)

    return out
def glue_non_crossing(pnt_list, possible_link):
    for elem in possible_link:
        left_feature = []
        right_feature = []
        gA, gB, dist = int(elem[0]), int(elem[1]), float(elem[2])

        print("\n=== TRY GLUE ===")
        print(f"  candidate global {gA} <--> {gB}, dist={dist:.3f}")
        print(f"  gA: frame={pnt_list[gA,1]}, theta={pnt_list[gA,3]:.2f}, lid={pnt_list[gA,7]}")
        print(f"  gB: frame={pnt_list[gB,1]}, theta={pnt_list[gB,3]:.2f}, lid={pnt_list[gB,7]}")

        for i in [0,1]:
            first_ind = int(elem[i])
            if pnt_list[first_ind,7] == -1:
                xposition = pnt_list[first_ind,3]
                gen = pnt_list[first_ind,1]
                for point in pnt_list[pnt_list[:,1] == gen]:
                    if point[3] < xposition and point[-1] != -1:
                        left_feature.append(int(point[-1]))
                    if point[3] > xposition and point[-1] != -1:
                        right_feature.append(int(point[-1]))
            else:
                val = pnt_list[first_ind,7]
                for subelem in np.nonzero(pnt_list[:,7] == val)[0]:
                    gen = pnt_list[subelem,1]
                    xposition = pnt_list[subelem,3]
                    for point in pnt_list[pnt_list[:,1] == gen]:
                        if point[3] < xposition and point[-1] != -1:
                            left_feature.append(int(point[-1]))
                        if point[3] > xposition and point[-1] != -1:
                            right_feature.append(int(point[-1]))
        
        left_set  = set(left_feature)
        right_set = set(right_feature)
        inter     = left_set & right_set

        print(f"  left_feature lids:  {sorted(left_set)}")
        print(f"  right_feature lids: {sorted(right_set)}")
        print(f"  intersection:       {sorted(inter)}")

        gluing = True
        if inter:
            print("  -> REJECT: crossing detected")
            gluing = False
            
        if gluing:
            print("  -> ACCEPT: glue lineages")
            old_index = pnt_list[int(elem[1]),7]
            new_index = pnt_list[int(elem[0]),7]
            
            if old_index == new_index == -1:
                ind = np.max(pnt_list[:,7])+1
                pnt_list[int(elem[1]),7] = ind
                pnt_list[int(elem[0]),7] = ind
            elif old_index == -1:
                pnt_list[int(elem[1]),7] = pnt_list[int(elem[0]),7]
            elif new_index == -1:
                pnt_list[int(elem[0]),7] = pnt_list[int(elem[1]),7]
            else :
                mask = pnt_list[:,7] == old_index
                pnt_list[:,7][mask] = new_index 
            
    return pnt_list






def _save_lineage_debug(
    pnt_list,
    saving_dic: str,
    step_idx: int,
    step_name: str,
    *,
    pdf=None,
    show=False,
    dpi=200,
):
    """
    Сохраняет отладочную картинку lineage + снапшот pnt_list.
    """
    if pnt_list is None or len(pnt_list) == 0:
        # всё равно сохраним метку, что пусто
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.set_title(f"step {step_idx:02d}: {step_name} (EMPTY)")
        ax.set_xlabel("frame_id")
        ax.set_ylabel("θ (deg)")
        fig.tight_layout()
        png_path = os.path.join(saving_dic, f"step_{step_idx:02d}_{step_name}_EMPTY.png")
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
        if pdf is not None:
            pdf.savefig(fig, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        return

    # Снапшот массива
    snap_path = os.path.join(saving_dic, f"step_{step_idx:02d}_{step_name}.npz")
    np.savez_compressed(snap_path, pnt_list=pnt_list)

    # Попробуем понять колонки безопасно
    # Ожидается: frame_id = col 1, theta = col 3, lineage_id = col 7, color = last col
    frame = pnt_list[:, 1]
    theta = pnt_list[:, 3]
    color = pnt_list[:, 7]
    lid   = pnt_list[:, 7] if pnt_list.shape[1] > 7 else None

    fig, ax = plt.subplots(figsize=(6, 3))
    sc = ax.scatter(frame, theta, c=color, cmap="tab20", s=8)

    # Заголовок с быстрой статистикой
    n = len(pnt_list)
    if lid is not None:
        lids, counts = np.unique(lid, return_counts=True)
        # часто "-1" — особая метка, уберём её из счёта “нормальных” линий
        normal_mask = lids != -1
        n_lines = int(np.sum(normal_mask))
        max_len = int(counts[normal_mask].max()) if np.any(normal_mask) else int(counts.max())
        ax.set_title(f"step {step_idx:02d}: {step_name} | points={n} | lines={n_lines} | max_len={max_len}")
    else:
        ax.set_title(f"step {step_idx:02d}: {step_name} | points={n}")

    ax.set_xlabel("frame_id")
    ax.set_ylabel("θ (deg)")
    fig.tight_layout()

    png_path = os.path.join(saving_dic, f"step_{step_idx:02d}_{step_name}.png")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    if pdf is not None:
        pdf.savefig(fig, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)




def peak_troughs_lineage(cell, all_outlines, outlines_CMF, roi_dir, *, debug=True, debug_pdf=True, debug_show=False):
    max_time         = 10
    first_max_xdrift = 45
    max_ydrift       = 3000
    final_max_xdrift = 60
    min_len          = 3

    saving_dic = roi_dir
    os.makedirs(saving_dic, exist_ok=True)

    # (опционально) чистим только debug-файлы, чтобы не удалять важное
    # но ты хотел “чисто”, оставляю твой вариант:
    for file in os.listdir(saving_dic):
        os.remove(os.path.join(saving_dic, file))

    pnt_list_path = os.path.join(saving_dic, "peaks_troughs_list.npz")
    pnt_ROI_path  = os.path.join(saving_dic, "peaks_troughs_ROI_list.npz")

    pnt_list = create_peaks_troughs_list(cell)
    if len(pnt_list) < 1:
        return

    # PDF для шагов (если включено)
    pdf = None
    if debug and debug_pdf:
        from matplotlib.backends.backend_pdf import PdfPages
        pdf = PdfPages(os.path.join(saving_dic, "debug_lineage_steps.pdf"))

    step = 0
    if debug:
        _save_lineage_debug(pnt_list, saving_dic, step, "00_init", pdf=pdf, show=debug_show)

    # 1) первая линковка
    step += 1
    time_mat, dist_mat = pnt_link_matrix(pnt_list, first_max_xdrift, max_ydrift, max_time,)

    update_list(pnt_list, time_mat, dist_mat)
    if debug:
        _save_lineage_debug(pnt_list, saving_dic, step, "01_after_first_link", pdf=pdf, show=debug_show)
    # # 2) залатываем дырки
    step += 1
    #pnt_list = fill_gaps(pnt_list, cell, all_outlines, outlines_CMF,first_max_xdrift)
    if debug:
        _save_lineage_debug(pnt_list, saving_dic, step, "02_after_fill_gaps", pdf=pdf, show=debug_show)

    # 3) первый final-проход
    step += 1

    pnt_ROI = reconstruct_ROI(pnt_list)

    mask = pnt_list[:, 7] == -1
    for key in pnt_ROI:
        mask[pnt_ROI[key][0]]  = True
        mask[pnt_ROI[key][-1]] = True
    new_gid = np.flatnonzero(mask)
    new_pnt_list = pnt_list[mask]



    if debug:
        _save_lineage_debug(new_pnt_list, saving_dic, step, "03a_new_pnt_list_for_final1", pdf=pdf, show=debug_show)

    new_time_mat, new_dist_mat = final_pnt_link_matrix(
        new_pnt_list, pnt_ROI,
        final_max_xdrift, max_ydrift, max_time
    )
    pnt_list = update_list_non_crossing(
        new_pnt_list, new_gid, pnt_list,
        new_time_mat, new_dist_mat,
        final_max_xdrift,
    )

    # pnt_list = merge_overlapping_lineages(pnt_list,
    #     theta_thr=10,      # начни с 8–12
    #     dr_thr=2000,
    #     min_overlap=3
    # # )
    # pnt_list = dedupe_by_lid_frame_type(pnt_list)




    if debug:
        _save_lineage_debug(pnt_list, saving_dic, step, "03b_after_final1_update", pdf=pdf, show=debug_show)

    # ещё раз залатываем
    step += 1
    #pnt_list = fill_gaps(pnt_list, cell, all_outlines,outlines_CMF,first_max_xdrift)
    if debug:
        _save_lineage_debug(pnt_list, saving_dic, step, "04_after_fill_gaps2", pdf=pdf, show=debug_show)

    # 4) первый фильтр по длине
    step += 1
    pnt_ROI = reconstruct_ROI(pnt_list)

    keep_lids = [lid for lid in pnt_ROI if len(pnt_ROI[lid]) >= min_len]
    #pnt_list = np.array([row for row in pnt_list if int(row[7]) in keep_lids])
    pnt_ROI  = {lid: idxs for lid, idxs in pnt_ROI.items() if lid in keep_lids}
    if debug:
        _save_lineage_debug(pnt_list, saving_dic, step, "05_after_minlen_filter1", pdf=pdf, show=debug_show)

    # 5) второй final-проход ПОСЛЕ отсечения коротких
    step += 1
    pnt_ROI = reconstruct_ROI(pnt_list)
    mask = pnt_list[:, 7] == -1
    for key in pnt_ROI:
        mask[pnt_ROI[key][0]]  = True
        mask[pnt_ROI[key][-1]] = True

    #mistakes happens when we switch from pnt_list to new_pnt_list - check filters
    # print("point list ", pnt_list)
    
    # print("new point list ", new_pnt_list)

    new_pnt_list = pnt_list[mask]
    for lid, idxs in pnt_ROI.items():
        frames = pnt_list[idxs, 1]
        print(f"lid {lid}: start={frames[0]}, end={frames[-1]}, len={len(idxs)}")

    if debug:
        _save_lineage_debug(new_pnt_list, saving_dic, step, "06a_new_pnt_list_for_final2", pdf=pdf, show=debug_show)

    new_time_mat, new_dist_mat = final_pnt_link_matrix(
        new_pnt_list, pnt_ROI,
        final_max_xdrift, max_ydrift, max_time
    )

    new_gid = np.flatnonzero(mask)
    new_pnt_list = pnt_list[mask]
    pnt_list = update_list_non_crossing(
        new_pnt_list,new_gid, pnt_list,
        new_time_mat, new_dist_mat,
        final_max_xdrift,
    )
    # pnt_list = merge_overlapping_lineages(pnt_list,
    #     theta_thr=10,      # начни с 8–12
    #     dr_thr=2000,
    #     min_overlap=3
    # )
    # pnt_list = dedupe_by_lid_frame_type(pnt_list)

    if debug:
        _save_lineage_debug(pnt_list, saving_dic, step, "06b_after_final2_update", pdf=pdf, show=debug_show)
    # 6) ещё раз fill_gaps
    step += 1
    #pnt_list = fill_gaps(pnt_list, cell, all_outlines,outlines_CMF,first_max_xdrift)
    # pnt_list = dedupe_by_lid_frame_type(pnt_list)

    if debug:
        _save_lineage_debug(pnt_list, saving_dic, step, "07_after_fill_gaps3", pdf=pdf, show=debug_show)

    # 7) финальный ROI и финальный min_len
    step += 1
    pnt_ROI = reconstruct_ROI(pnt_list)
    keep_lids = [lid for lid in pnt_ROI if len(pnt_ROI[lid]) >= min_len]
    pnt_list = np.array([row for row in pnt_list if int(row[7]) in keep_lids])
    pnt_ROI  = {lid: idxs for lid, idxs in pnt_ROI.items() if lid in keep_lids}
    if debug:
        _save_lineage_debug(pnt_list, saving_dic, step, "08_after_minlen_filter2_final", pdf=pdf, show=debug_show)
    # keys = pnt_list[:, [1,2,6]].astype(int)
    # _, cnt = np.unique(keys, axis=0, return_counts=True)
    # print("max duplicates:", cnt.max())
    # print("n duplicated rows:", np.sum(cnt > 1))

    # финальная картинка (как у тебя)
    plt.figure(figsize=(6,3))
    plt.scatter(pnt_list[:,1], pnt_list[:,3], c=pnt_list[:,7], cmap='tab20', s=8)
    plt.xlabel("frame_id"); plt.ylabel("θ (deg)")
    plt.title("Lineages")
    if debug_show:
        plt.show()
    else:
        plt.close()

    # сохраняем финальные артефакты
    np.savez_compressed(pnt_list_path, pnt_list=pnt_list)
    np.savez_compressed(pnt_ROI_path, pnt_ROI=pnt_ROI)

    if pdf is not None:
        pdf.close()



def polar_pdf_from_cell(
    cell: list,
    output_pdf: str,
    base_radius=50,            # скаляр или массив длины len(cell); это твой disk_radius
    dpi: int = 150,
    marker_size: int = 120,
    linewidth: float = 1.2,
    title_prefix: str = "Frame",
):
    """
    Делает один PDF: на каждой странице полярный график r(θ), где r = base_radius + Δr.
    Кривая и маркеры берутся ТОЛЬКО из cell[i], т.е. из тех же рядов, где пики/впадины были найдены.
    Никакого сопоставления с pnt_list — чтобы ничего не «поехало».

    cell[i]: {
        'xs': θ_sorted (deg),
        'ys': height_sorted (Δr),
        'peaks': np.ndarray[int],
        'troughs': np.ndarray[int],
        'timestamp': float
    }
    """

    # базовый радиус по кадрам
    if np.isscalar(base_radius):
        base_per_frame = None
    else:
        base_per_frame = np.asarray(base_radius, dtype=float)
        assert len(base_per_frame) == len(cell), "base_radius как массив должен совпадать с len(cell)."

    os.makedirs(os.path.dirname(output_pdf) or ".", exist_ok=True)
    with PdfPages(output_pdf) as pdf:
        for i, frame in enumerate(cell):
            theta_deg = np.asarray(frame["xs"], dtype=float)      # θ в градусах, уже отсортирован
            d_r       = np.asarray(frame["ys"], dtype=float)      # Δr
            peaks     = np.asarray(frame["peaks"], dtype=int)
            troughs   = np.asarray(frame["troughs"], dtype=int)

            theta_sorted_rad = np.radians(theta_deg)
            rad_sorted = 50.0 + d_r  # тот же радиус, что на левом графике (база + высота)
        #     ax2.plot(theta_sorted_rad, rad_sorted, lw=1)


            # theta = np.radians(theta_deg % 360.0)
            # R0 = float(base_radius if base_per_frame is None else base_per_frame[i])
            # r  = R0 + d_r

            # # закрыть кривую (без разрыва)
            # theta_c = np.r_[theta, theta[0]]
            # r_c     = np.r_[r,     r[0]]

            fig = plt.figure(figsize=(6, 6), dpi=dpi)
            ax = fig.add_subplot(111, projection="polar")

            ax.plot(theta_sorted_rad, rad_sorted, lw=linewidth)

            rspan = float(np.max(rad_sorted) - np.min(rad_sorted) + 1e-6)
            dtheta = np.radians(0.1)
            dr = 0.1 * rspan
            if peaks.size:
                ax.scatter(theta_sorted_rad[peaks], rad_sorted[peaks],
                           s=marker_size, marker="^",
                           facecolors="none", edgecolors="tab:red",
                           linewidths=1.6, zorder=3, label="peaks")
                for idx in peaks:
                    ax.text(theta_sorted_rad[idx] + dtheta, rad_sorted[idx] + dr, str(int(idx)),
                            fontsize=4, ha="left", va="bottom", zorder=4)

            if troughs.size:
                ax.scatter(theta_sorted_rad[troughs], rad_sorted[troughs],
                           s=marker_size, marker="v",
                           facecolors="none", edgecolors="tab:orange",
                           linewidths=1.6, zorder=3, label="troughs")
                for idx in troughs:
                    ax.text(theta_sorted_rad[idx] + dtheta, rad_sorted[idx] - dr, str(int(idx)),
                            fontsize=4, ha="left", va="top", zorder=4)




            t = frame.get("timestamp", np.nan)
            t_str = f"{t:.3f}" if np.isfinite(t) else "—"
            ax.set_title(f"{title_prefix} {i} | t={t_str} | peaks={peaks.size}, troughs={troughs.size}")
            if peaks.size + troughs.size:
                ax.legend(loc="lower right", fontsize=8, framealpha=0.4)

            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"[ok] Polar PDF saved: {output_pdf}")

def polar_pdf_from_pntlist(
    cell,                  # что ты уже строишь в single_cell_tracking
    outlines,              # список контуров [frame] -> (N,2)
    pnt_npz_path,          # файл с pnt_list (после fill_gaps + фин. апдейтов)
    output_pdf,
    base_radius=50,
    dpi=150,
    marker_size=120,
    linewidth=1.2,
    title_prefix="Frame",
):


    dat = np.load(pnt_npz_path, allow_pickle=True)
    pnt_list = dat["pnt_list"] if "pnt_list" in dat else dat[list(dat.keys())[0]]
    pnt_list = np.asarray(pnt_list, dtype=float)

    # Разбираем pnt_list по кадрам. Формат строки pnt_list:
    # [global_id, frame_id, type(1=peak,0=trough), theta_deg, dr, time, outline_idx, lineage_id]
    by_frame = {}
    for row in pnt_list:
        gid, fno, typ, theta, dr, t, out_idx, lid,_ = row
        fno = int(fno); typ = int(typ); out_idx = int(out_idx); lid = int(lid)
        by_frame.setdefault(fno, []).append((gid, typ, theta, dr, t, out_idx, lid))

    # --- КОЛОРИНГ КАК В peak_troughs_lineage ---
    # Используем cmap='tab20' и нормируем по всем lid >= 0
    cmap = plt.get_cmap("tab20")
    lids_all = pnt_list[:, 7]
    valid = lids_all >= 0
    if np.any(valid):
        vmin = lids_all[valid].min()
        vmax = lids_all[valid].max()
    else:
        vmin, vmax = 0, 1  # fallback, если вдруг всё -1

    def color_for(lid):
        if lid < 0 or vmax == vmin:
            return (0, 0, 0, 1)
        # как в scatter(c=pnt_list[:,-1], cmap='tab20'): линейная нормировка lid
        alpha = (lid - vmin) / float(vmax - vmin)
        return cmap(alpha)

    os.makedirs(os.path.dirname(output_pdf) or ".", exist_ok=True)
    with PdfPages(output_pdf) as pdf:
        for i, frame in enumerate(cell):
            theta_deg = np.asarray(frame["xs"], dtype=float)    # уже отсортированные углы
            d_r       = np.asarray(frame["ys"], dtype=float)    # высоты (Δr)
            order_idx = np.asarray(frame["order_idx"], dtype=int)

            theta_rad  = np.radians(theta_deg)
            rad_sorted = float(base_radius) + d_r

            # Собираем точки этого кадра из pnt_list:
            rows = by_frame.get(i, [])

            # Разделим по типам, индексация через order_idx == outline_idx
            peak_idx_local, peak_colors = [], []
            trough_idx_local, trough_colors = [], []

            for (gid, typ, theta, dr, t, out_idx, lid) in rows:
                where = np.where(order_idx == out_idx)[0]
                if where.size == 0:
                    continue
                k = int(where[0])

                c = color_for(lid)
                if typ == 1:
                    peak_idx_local.append(k)
                    peak_colors.append(c)
                else:
                    trough_idx_local.append(k)
                    trough_colors.append(c)

            peak_idx_local   = np.array(peak_idx_local, dtype=int) if peak_idx_local else np.array([], dtype=int)
            trough_idx_local = np.array(trough_idx_local, dtype=int) if trough_idx_local else np.array([], dtype=int)
            peak_colors      = np.array(peak_colors) if peak_colors else np.empty((0,4))
            trough_colors    = np.array(trough_colors) if trough_colors else np.empty((0,4))

            # --- РИСУЕМ ---
            fig = plt.figure(figsize=(15, 4), dpi=dpi)

            # 1) Контур (декартово)
            ax1 = fig.add_subplot(1, 3, 1)
            x_outline, y_outline = outlines[i][:,0], outlines[i][:,1]
            ax1.plot(x_outline, y_outline, lw=linewidth)
            ax1.set_aspect("equal", adjustable="box")
            ax1.set_title("Outline")
            ax1.set_xlabel("x")
            ax1.set_ylabel("y")

            # 2) Полярная топография
            ax2 = fig.add_subplot(1, 3, 2, projection="polar")
            ax2.plot(theta_rad, rad_sorted, lw=linewidth)
            ax2.set_title("Topographical polar representation")

            # 3) θ–height
            ax3 = fig.add_subplot(1, 3, 3)
            ax3.plot(theta_deg, rad_sorted, lw=1)
            ax3.set_xlabel("θ (deg)")
            ax3.set_ylabel("topo height")
            ax3.set_title("Topographical θ–height")

            # Маркеры протрузий (по pnt_list)
            if peak_idx_local.size:
                ax1.scatter(
                    x_outline[order_idx[peak_idx_local]],
                    y_outline[order_idx[peak_idx_local]],
                    s=25,
                    marker="^",
                    facecolors="none",
                    edgecolors=peak_colors,
                    linewidths=1.6,
                    zorder=3,
                    label="peaks",
                )
                ax2.scatter(
                    theta_rad[peak_idx_local],
                    rad_sorted[peak_idx_local],
                    s=marker_size,
                    marker="^",
                    facecolors="none",
                    edgecolors=peak_colors,
                    linewidths=1.6,
                    zorder=3,
                    label="peaks",
                )
                ax3.scatter(
                    theta_deg[peak_idx_local],
                    rad_sorted[peak_idx_local],
                    s=25,
                    marker="^",
                    facecolors="none",
                    edgecolors=peak_colors,
                    linewidths=1.6,
                    zorder=3,
                    label="peaks",
                )

            # Маркеры интрузий (по pnt_list)
            if trough_idx_local.size:
                ax1.scatter(
                    x_outline[order_idx[trough_idx_local]],
                    y_outline[order_idx[trough_idx_local]],
                    s=25,
                    marker="v",
                    facecolors="none",
                    edgecolors=trough_colors,
                    linewidths=1.6,
                    zorder=3,
                    label="troughs",
                )
                ax2.scatter(
                    theta_rad[trough_idx_local],
                    rad_sorted[trough_idx_local],
                    s=marker_size,
                    marker="v",
                    facecolors="none",
                    edgecolors=trough_colors,
                    linewidths=1.6,
                    zorder=3,
                    label="troughs",
                )
                ax3.scatter(
                    theta_deg[trough_idx_local],
                    rad_sorted[trough_idx_local],
                    s=25,
                    marker="v",
                    facecolors="none",
                    edgecolors=trough_colors,
                    linewidths=1.6,
                    zorder=3,
                    label="troughs",
                )

            # Заголовок и легенда
            t = frame.get("timestamp", np.nan)
            t_str = f"{t:.3f}" if np.isfinite(t) else "—"
            fig.suptitle(f"{title_prefix} {i} | t={t_str}", y=1.02, fontsize=10)

            if peak_idx_local.size + trough_idx_local.size:
                ax2.legend(loc="lower right", fontsize=8, framealpha=0.4)

            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"[ok] PDF saved: {output_pdf}")




def generate_lineages_count_vs_frame_pdf_for_dataset(
    dataset_dir: str,
    roi_root_dir: str,
    output_pdf_path: str,
    min_len: int = 10,
):
    """
    Генерирует график: (кол-во активных lineages) vs (frame_id) для каждой cell_* из dataset_dir.
    Теперь проходит по ВСЕМ кадрам от 0 до max_frame (в пределах данных клетки),
    чтобы отображались кадры с 0 lineages.

    Ожидается формат pnt_list:
      frame_id   = pnt_list[:, 1]
      lineage_id = pnt_list[:, 7]
    """
    cell_folders = sorted(
        [f for f in os.listdir(dataset_dir) if f.startswith("cell_")],
        key=lambda x: int(x.split("_")[1])
    )

    os.makedirs(roi_root_dir, exist_ok=True)

    with PdfPages(output_pdf_path) as pdf:
        total_cells = len(cell_folders)
        print(f"[INFO] Found {total_cells} cells")

        for ci, cell_name in enumerate(cell_folders, start=1):
            print(f"\n=== {cell_name} ({ci}/{total_cells}) ===")

            roi_dir   = os.path.join(roi_root_dir, cell_name, "roi_data")
            os.makedirs(roi_dir, exist_ok=True)
            pnt_path  = os.path.join(roi_dir, "peaks_troughs_list.npz")

            # --- если нет pnt_list.npz, генерим ---
            if not os.path.exists(pnt_path):
                try:
                    cell, outlines,outlines_CMF = single_cell_tracking(dataset_dir, ci - 1)
                    peak_troughs_lineage(cell, outlines, outlines_CMF,roi_dir)
                    print(f"[ok] generated pnt_list for {cell_name}")
                except Exception as e:
                    print(f"[skip] generation failed for {cell_name}: {e}")
                    continue

            # --- загрузка ---
            try:
                dat = np.load(pnt_path, allow_pickle=True)
                pnt_list = dat["pnt_list"] if "pnt_list" in dat else dat[list(dat.keys())[0]]
                pnt_list = np.asarray(pnt_list, dtype=float)
            except Exception as e:
                print(f"[skip] cannot load {pnt_path}: {e}")
                continue

            if pnt_list.size == 0:
                print(f"[skip] empty pnt_list for {cell_name}")
                continue

            # --- сохраним max_frame ДО фильтрации (иначе потеряешь хвосты, где все lineages короткие) ---
            all_frame_ids = pnt_list[:, 1].astype(int)
            max_frame = int(all_frame_ids.max())
            min_frame = 0  # как ты просил: от нуля

            # --- фильтрация lineages по min_len ---
            lids, counts = np.unique(pnt_list[:, 7], return_counts=True)
            long_ids = lids[counts >= min_len]
            pnt_list = pnt_list[np.isin(pnt_list[:, 7], long_ids)]

            # если после фильтра пусто — всё равно рисуем нули по кадрам
            if pnt_list.size == 0:
                frames = np.arange(min_frame, max_frame + 1, dtype=int)
                n_lineages_per_frame = np.zeros_like(frames, dtype=int)
            else:
                frame_ids = pnt_list[:, 1].astype(int)
                lineage_ids = pnt_list[:, 7].astype(int)

                frames = np.arange(min_frame, max_frame + 1, dtype=int)
                n_lineages_per_frame = np.zeros_like(frames, dtype=int)

                # считаем уникальные lineage_id внутри каждого frame_id (только для тех fr, что реально встречаются)
                for i, fr in enumerate(frames):
                    mask = frame_ids == fr
                    if np.any(mask):
                        n_lineages_per_frame[i] = np.unique(lineage_ids[mask]).size
                    else:
                        n_lineages_per_frame[i] = 0

            # --- построение ---
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.plot(frames, n_lineages_per_frame, marker="o", markersize=3, linewidth=1)

            ax.set_xlabel("frame_id")
            ax.set_ylabel("number of active lineages")
            ax.set_title(f"{cell_name} | lineages per frame (only lineages with len ≥ {min_len})")
            ax.grid(True, alpha=0.3)

            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        print(f"\n[OK] Saved all lineage-count plots to {output_pdf_path}")



def _angular_diff_deg(a: float, b: float) -> float:
    """
    Минимальная разность углов (в градусах) на окружности.
    Возвращает значение в [-180, 180].
    """
    d = (a - b + 180.0) % 360.0 - 180.0
    return d



# generate_lineages_pdf_highlight_best_point_each_frame(
#      dataset_dir="/home/pavel/cell_morphology/filtered_data/cells_filtered",
#      roi_root_dir="/home/pavel/shape_analysis/roi_data_all",
#      output_pdf_path="/home/pavel/shape_analysis/lineages_dataset_curvature_filtered_best_curv_height_velocity.pdf",
#      min_len=10)


# generate_lineages_count_vs_frame_pdf_for_dataset(
#     dataset_dir="/home/pavel/cell_morphology/filtered_data/cells_filtered",
#     roi_root_dir="/home/pavel/shape_analysis/roi_data_all",
#     output_pdf_path="/home/pavel/shape_analysis/lineage_amount_per_time.pdf",
#     min_len=10)


# accumulate_protrusion_statistics(
#     dataset_dir="/home/pavel/cell_morphology/filtered_data/cells_filtered",
#     roi_root_dir="/home/pavel/shape_analysis/roi_data_all",
#     output_png_path = "statistics_protrusion_frames.png",
# )




# glob_filtered_folder = '/home/pavel/cell_morphology/filtered_data/cells_filtered/'

# #peak_troughs_lineage(single_cell_tracking(glob_filtered_folder, 28)[0],single_cell_tracking(glob_filtered_folder, 28)[1], '/home/pavel/shape_analysis/roi_data')
# cell, outlines,outlines_CMF = single_cell_tracking(glob_filtered_folder, 120)
# peak_troughs_lineage(cell, outlines, outlines_CMF,'/home/pavel/shape_analysis/roi_data')
# npz_path= "/home/pavel/shape_analysis/roi_data/peaks_troughs_list.npz"

# #polar_pdf_from_cell(cell,"test1.pdf")

# polar_pdf_from_pntlist(cell, outlines,npz_path, "cell_121_tracking.pdf")
# #polar_pdf_from_lineages(npz_path, "test_2_ver_3.pdf")
#single_cell_tracking(glob_filtered_folder, 1)

#

#data = np.load('/home/pavel/shape_analysis/roi_data/peaks_troughs_ROI_list.npz', allow_pickle = True)
#print(data["arr_0"])
#print(data["arr_0"].shape)



def protrusions_per_frame_from_pnt_list(pnt_list, n_frames=None):
    """
    pnt_list: np.ndarray with columns:
        [global_id, frame_id, type(1=peak,0=trough), theta_deg, dr, time, outline_idx, lineage_id]
    n_frames: if known externally; otherwise inferred as max frame_id + 1

    Returns: counts_per_frame: (n_frames,) int
    """
    if pnt_list.size == 0:
        return np.zeros(0, dtype=int)

    # ensure 2D
    pnt_list = np.asarray(pnt_list, dtype=float)
    frame_ids = pnt_list[:, 1].astype(int)
    types = pnt_list[:, 2].astype(int)

    if n_frames is None:
        n_frames = frame_ids.max() + 1

    counts = np.zeros(n_frames, dtype=int)

    # считаем только протрузии (type == 1)
    mask_peaks = (types == 1)
    peak_frames = frame_ids[mask_peaks]

    for f in peak_frames:
        if 0 <= f < n_frames:
            counts[f] += 1

    return counts


def build_protrusion_counts_array(
    dataset_root,
    n_cells=121,
    pnt_filename="peaks_troughs_list.npz",
    output_path="protrusion_counts_per_cell.npy",
):
    """
    dataset_root/cell_{i}/.../{pnt_filename} должен содержать pnt_list
    (формат как после peak_troughs_lineage).
    """

    result = np.empty(n_cells, dtype=object)

    for cell_idx in range(1, n_cells + 1):
        # подстрой путь при необходимости:
        # например, если pnt лежит прямо в cell_{i}: os.path.join(dataset_root, f"cell_{cell_idx}", pnt_filename)
        # или если в ROI-папке: os.path.join(dataset_root, f"cell_{cell_idx}", "ROI", pnt_filename)
        pnt_path = os.path.join(dataset_root, f"cell_{cell_idx}","roi_data", pnt_filename)
        print(pnt_path)
        if not os.path.exists(pnt_path):
            result[cell_idx - 1] = np.zeros(0, dtype=int)
            continue

        dat = np.load(pnt_path, allow_pickle=True)

        pnt_list = dat["pnt_list"] if "pnt_list" in dat else dat[list(dat.keys())[0]]
        pnt_list = np.asarray(pnt_list, dtype=float)
        
        counts = protrusions_per_frame_from_pnt_list(pnt_list)
        
        result[cell_idx - 1] = counts

    np.save(output_path, result)
    print(f"[ok] saved protrusion counts to: {output_path}")
    print(f"shape: {result.shape}, example cell[0] shape: {result[0].shape if n_cells > 0 else None}")


#build_protrusion_counts_array("/home/pavel/shape_analysis/roi_data_all", n_cells=121)



def polar_pdf_from_lineages(
    pnt_npz_path: str,
    output_pdf: str,
    base_radius: float = 50.0,
    dpi: int = 150,
    marker_size: int = 120,
    linewidth: float = 1.2,
    title_prefix: str = "Frame",
):
    """Рисуем строго по lineage: в каждом кадре берем все точки из pnt_list для этого кадра,
    цвет = lineage_id, тип маркера = пик/впадина; соединяем соседние кадры одной линии.
    """
    dat = np.load(pnt_npz_path, allow_pickle=True)
    pnt_list = dat["pnt_list"] if "pnt_list" in dat else dat[list(dat.keys())[0]]
    pnt_list = np.asarray(pnt_list, dtype=float)

    # Разбираем поля
    gid      = pnt_list[:, 0].astype(int)
    fno      = pnt_list[:, 1].astype(int)
    is_peak  = pnt_list[:, 2].astype(int)          # 1=peak, 0=trough
    th_deg   = pnt_list[:, 3].astype(float)
    dr_val   = pnt_list[:, 4].astype(float)
    lid      = pnt_list[:, 7].astype(int)

    # Кадры
    frames = np.unique(fno)
    # Группировка по линиям (исключаем -1)
    valid_mask = lid != -1
    lids = np.unique(lid[valid_mask])

    cmap = plt.get_cmap("tab20")
    def color_for(l):
        return cmap((l % 20)/20.0) if l >= 0 else (0,0,0,1)

    # Подготовим индекс по кадрам (для быстрой выборки)
    frame_idx = {fr: np.where(fno == fr)[0] for fr in frames}

    # Для линий подготовим отсортированные по времени последовательности
    seq_by_lid = {}
    for L in lids:
        idx = np.where(lid == L)[0]
        if idx.size == 0:
            continue
        order = np.argsort(fno[idx])
        seq_by_lid[L] = idx[order]

    with PdfPages(output_pdf) as pdf:
        for fr in frames:
            fig = plt.figure(figsize=(10, 4), dpi=dpi)

            # === Полярный ===
            ax_polar = fig.add_subplot(1, 2, 1, projection="polar")
            # === θ–height (для контроля непрерывности) ===
            ax_th = fig.add_subplot(1, 2, 2)

            # Рисуем маркеры текущего кадра по линиям
            if fr in frame_idx:
                idx_fr = frame_idx[fr]
                for idx in idx_fr:
                    L   = lid[idx]
                    th  = th_deg[idx]
                    dr  = dr_val[idx]
                    typ = is_peak[idx]
                    col = color_for(L)

                    r = base_radius + dr
                    th_rad = np.radians(th)

                    mk = "^" if typ == 1 else "v"
                    ax_polar.scatter(th_rad, r, s=marker_size, marker=mk,
                                     facecolors="none", edgecolors=col,
                                     linewidths=1.6, zorder=3)

                    ax_th.scatter(th, r, s=20, marker=mk,
                                  facecolors="none", edgecolors=col,
                                  linewidths=1.2, zorder=3)

            # Соединяем соседние кадры каждой линии (если точка есть и в fr, и в fr+1)
            next_fr = fr + 1
            if next_fr in frame_idx:
                for L, idx_seq in seq_by_lid.items():
                    # ищем точку линии в fr
                    i_now  = idx_seq[np.searchsorted(fno[idx_seq], fr)]
                    if fno[i_now] != fr:
                        # нет точной точки в этом кадре
                        continue
                    # ищем следующую точку линии в следующем кадре (может быть синтетическая из fill_gaps)
                    pos_next = np.searchsorted(fno[idx_seq], next_fr)
                    if pos_next >= len(idx_seq) or fno[idx_seq][pos_next] != next_fr:
                        continue
                    i_nxt = idx_seq[pos_next]

                    col = color_for(L)
                    th1, r1 = th_deg[i_now], base_radius + dr_val[i_now]
                    th2, r2 = th_deg[i_nxt], base_radius + dr_val[i_nxt]

                    # линии на полярном
                    ax_polar.plot([np.radians(th1), np.radians(th2)], [r1, r2],
                                  lw=1.0, color=col, alpha=0.8)
                    # линии на θ–height
                    ax_th.plot([th1, th2], [r1, r2], lw=1.0, color=col, alpha=0.8)

            # Подписи/оси
            ax_polar.set_title(f"Polar (frame {fr})")
            ax_th.set_xlabel("θ (deg)")
            ax_th.set_ylabel("base_radius + Δr")
            ax_th.set_title("θ–height (by lineages)")
            fig.suptitle(f"{title_prefix} {fr}", y=1.02)

            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"[ok] Polar (by lineages) PDF saved: {output_pdf}")

def generate_lineages_pdf_for_dataset(
    dataset_dir: str,
    roi_root_dir: str,
    output_pdf_path: str,
    min_len: int = 10,
):
    """
    Генерирует lineage-графики для всех cell_* из dataset_dir.
    Ничего не записывает внутрь dataset_dir.
    Все roi_data и pnt_list сохраняются в roi_root_dir/cell_<id>/.
    Все графики объединяются в один PDF (output_pdf_path).
    """
    cell_folders = sorted(
        [f for f in os.listdir(dataset_dir) if f.startswith("cell_")],
        key=lambda x: int(x.split("_")[1])
    )

    os.makedirs(roi_root_dir, exist_ok=True)

    with PdfPages(output_pdf_path) as pdf:
        total_cells = len(cell_folders)
        print(f"[INFO] Found {total_cells} cells")

        for ci, cell_name in enumerate(cell_folders, start=1):
            print(f"\n=== {cell_name} ({ci}/{total_cells}) ===")

            cell_path = os.path.join(dataset_dir, cell_name)
            roi_dir   = os.path.join(roi_root_dir, cell_name, "roi_data")
            os.makedirs(roi_dir, exist_ok=True)
            pnt_path  = os.path.join(roi_dir, "peaks_troughs_list.npz")

            # --- если нет pnt_list.npz, генерим ---
            if not os.path.exists(pnt_path):
                try:
                    cell, outlines,outlines_CMF = single_cell_tracking(dataset_dir, ci - 1)
                    peak_troughs_lineage(cell, outlines, outlines_CMF,roi_dir)
                    print(f"[ok] generated pnt_list for {cell_name}")
                except Exception as e:
                    print(f"[skip] generation failed for {cell_name}: {e}")
                    continue

            # --- загрузка ---
            try:
                dat = np.load(pnt_path, allow_pickle=True)
                pnt_list = dat["pnt_list"] if "pnt_list" in dat else dat[list(dat.keys())[0]]
                pnt_list = np.asarray(pnt_list, dtype=float)
            except Exception as e:
                print(f"[skip] cannot load {pnt_path}: {e}")
                continue

            if pnt_list.size == 0:
                print(f"[skip] empty pnt_list for {cell_name}")
                continue

            lids, counts = np.unique(pnt_list[:, 7], return_counts=True)
            long_ids = lids[counts >= min_len]
            mask = np.isin(pnt_list[:, 7], long_ids)
            pnt_list = pnt_list[mask]

            if pnt_list.size == 0:
                print(f"[skip] no long lineages (≥{min_len}) for {cell_name}")
                continue

            # --- построение ---
            fig, ax = plt.subplots(figsize=(6, 3))
            sc = ax.scatter(pnt_list[:, 1], pnt_list[:, 3],
                            c=pnt_list[:, 7], cmap='tab20', s=8)
            ax.set_xlabel("frame_id")
            ax.set_ylabel("θ (deg)")
            ax.set_title(f"{cell_name} | {len(long_ids)} lineages (≥{min_len})")
            ax.set_ylim(0, 360)
            ax.set_yticks(np.arange(0,360,12))
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        print(f"\n[OK] Saved all lineage plots to {output_pdf_path}")
def accumulate_protrusion_statistics_2modes(
    dataset_dir: str,
    roi_root_dir: str,
    output_png_path: str,
    min_len: int = 10,
    max_protrusions_bin: int = 8,
):
    """
    ИТОГО 6 картинок:

    ОБЩИЕ (all cells):
      1) <stem>_overall_meanstd_speed_vs_k.png     (точки mean±std)
      2) <stem>_overall_meanstd_dtheta_vs_k.png    (точки mean±std)
      3) <stem>_overall_frames_hist_k.png          (ГИСТОГРАММА #frames vs k)

    ПО МОДАМ (messy/straight), 2 панели одна под другой:
      4) output_png_path                            (ГИСТОГРАММЫ #frames vs k, 2 панели)
      5) <stem>_meanstd_speed_vs_k_2modes.png        (точки mean±std, 2 панели)
      6) <stem>_meanstd_dtheta_vs_k_2modes.png       (точки mean±std, 2 панели)

    k(frame) = число активных lineages в кадре (после фильтрации lineage по min_len)
    speed, theta_vel из conformal_representation:
      |v| относится к кадру t=1..T-1
      |Δθ| относится к кадру t=2..T-1

    ВАЖНО: cell_idx берём из имени папки cell_XXX -> XXX (чтобы не съезжало на поднаборах).
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter1d

    def cell_idx_from_name(cell_name: str) -> int:
        return int(cell_name.split("_")[1])

    def last_nonzero_idx(arr: np.ndarray) -> int:
        nz = np.where(arr > 0)[0]
        return int(nz.max()) if nz.size > 0 else 0

    def mean_std_per_k(values_by_k):
        mean = np.full(max_protrusions_bin + 1, np.nan, float)
        std  = np.full(max_protrusions_bin + 1, np.nan, float)
        n    = np.zeros(max_protrusions_bin + 1, int)
        for k in range(max_protrusions_bin + 1):
            vals = np.asarray(values_by_k[k], float)
            n[k] = vals.size
            if vals.size > 0:
                mean[k] = float(vals.mean())
                std[k]  = float(vals.std())
        return mean, std, n

    # ------------ folders ------------
    cell_folders_all = sorted(
        [f for f in os.listdir(dataset_dir) if f.startswith("cell_")],
        key=lambda x: int(x.split("_")[1])
    )
    os.makedirs(roi_root_dir, exist_ok=True)

    ind_straight = np.array([
        0, 1, 2, 4, 5, 6, 7, 11, 12, 15, 17, 23, 24, 26,
        41, 50, 65, 66, 67, 77, 78, 80, 85, 86
    ])
    ind_messy = np.array([
        13, 20, 21, 28, 44, 49, 53, 63, 87, 91, 100, 101,
        102, 103, 105, 106, 107, 109, 110, 111, 112, 113,
        116, 120
    ])

    straight_folders = [cell_folders_all[i] for i in ind_straight]
    messy_folders    = [cell_folders_all[i] for i in ind_messy]
    messy_set = set(messy_folders)
    straight_set = set(straight_folders)

    def mode_of_cell(cell_name: str) -> str:
        if cell_name in messy_set:
            return "messy"
        if cell_name in straight_set:
            return "straight"
        return "other"

    # ------------ accumulators ------------
    frames_k_overall  = np.zeros(max_protrusions_bin + 1, dtype=int)
    frames_k_messy    = np.zeros(max_protrusions_bin + 1, dtype=int)
    frames_k_straight = np.zeros(max_protrusions_bin + 1, dtype=int)

    speed_by_k_overall  = [[] for _ in range(max_protrusions_bin + 1)]
    speed_by_k_messy    = [[] for _ in range(max_protrusions_bin + 1)]
    speed_by_k_straight = [[] for _ in range(max_protrusions_bin + 1)]

    dtheta_by_k_overall  = [[] for _ in range(max_protrusions_bin + 1)]
    dtheta_by_k_messy    = [[] for _ in range(max_protrusions_bin + 1)]
    dtheta_by_k_straight = [[] for _ in range(max_protrusions_bin + 1)]

    def append_mode_list(mode: str, overall_list, messy_list, straight_list, k: int, value: float):
        kk = min(int(k), max_protrusions_bin)
        overall_list[kk].append(value)
        if mode == "messy":
            messy_list[kk].append(value)
        elif mode == "straight":
            straight_list[kk].append(value)

    # ------------ main loop over ALL cells once ------------
    total_cells = len(cell_folders_all)
    for ii, cell_name in enumerate(cell_folders_all, start=1):
        mode = mode_of_cell(cell_name)
        print(f"\n=== {cell_name} ({ii}/{total_cells}) mode={mode} ===")

        cell_path = os.path.join(dataset_dir, cell_name)
        roi_dir   = os.path.join(roi_root_dir, cell_name, "roi_data")
        os.makedirs(roi_dir, exist_ok=True)
        pnt_path  = os.path.join(roi_dir, "peaks_troughs_list.npz")

        # generate if missing
        if not os.path.exists(pnt_path):
            try:
                idx_true = cell_idx_from_name(cell_name)
                cell, outlines, outlines_CMF = single_cell_tracking(dataset_dir, idx_true)
                peak_troughs_lineage(cell, outlines, outlines_CMF, roi_dir)
                print(f"[ok] generated pnt_list for {cell_name}")
            except Exception as e:
                print(f"[skip] generation failed for {cell_name}: {e}")
                continue

        # load pnt_list
        try:
            dat = np.load(pnt_path, allow_pickle=True)
            pnt_list = dat["pnt_list"] if "pnt_list" in dat else dat[list(dat.keys())[0]]
            pnt_list = np.asarray(pnt_list, dtype=float)
        except Exception as e:
            print(f"[skip] cannot load {pnt_path}: {e}")
            continue

        if pnt_list.size == 0:
            print(f"[skip] empty pnt_list for {cell_name}")
            continue

        all_frame_ids = pnt_list[:, 1].astype(int)
        max_frame = int(all_frame_ids.max())
        frames_full = np.arange(0, max_frame + 1, dtype=int)

        # filter lineages by min_len
        lids, counts = np.unique(pnt_list[:, 7], return_counts=True)
        long_ids = lids[counts >= min_len]
        pnt_filt = pnt_list[np.isin(pnt_list[:, 7], long_ids)]

        # k_per_frame
        k_per_frame = np.zeros(max_frame + 1, dtype=int)
        if pnt_filt.size > 0:
            frame_ids   = pnt_filt[:, 1].astype(int)
            lineage_ids = pnt_filt[:, 7].astype(int)
            for fr in np.unique(frame_ids):
                if 0 <= fr <= max_frame:
                    k_per_frame[fr] = np.unique(lineage_ids[frame_ids == fr]).size

        # update frames histogram (include zeros)
        for fr in frames_full:
            k = int(k_per_frame[fr])
            kk = min(k, max_protrusions_bin)
            frames_k_overall[kk] += 1
            if mode == "messy":
                frames_k_messy[kk] += 1
            elif mode == "straight":
                frames_k_straight[kk] += 1

        # speed/theta from conformal_representation
        theta_vel = None
        frames_vel = None
        speed_mag = None
        try:
            (
                _all_outlines,
                _,
                _all_outlines_cMCF_topography,
                _all_outlines_curvature,
                _disk_coors,
                centr,
                fin_times,
            ) = conformal_representation(cell_path)

            centr = np.asarray(centr, dtype=float)
            if centr.ndim == 2 and centr.shape[0] >= 2:
                v = np.diff(centr, axis=0)
                speed_mag = np.linalg.norm(v, axis=1)  # len T-1
                theta_vel = (np.degrees(np.arctan2(v[:, 1], v[:, 0])) % 360.0)
                theta_vel = gaussian_filter1d(theta_vel, 1, mode="nearest")
                frames_vel = np.arange(1, len(centr), dtype=int)  # frames 1..T-1
        except Exception as e:
            print(f"[warn] cannot compute velocity for {cell_name}: {e}")

        # aggregate speed by k (frames 1..T-1)
        if frames_vel is not None and speed_mag is not None:
            for idx, fr in enumerate(frames_vel):
                if 0 <= fr <= max_frame:
                    k = int(k_per_frame[fr])
                    append_mode_list(mode, speed_by_k_overall, speed_by_k_messy, speed_by_k_straight, k, float(speed_mag[idx]))

            # aggregate dtheta by k (frames 2..T-1)
            if theta_vel is not None and len(theta_vel) >= 2:
                for i in range(1, len(theta_vel)):  # i=1 -> frame 2
                    fr = i + 1
                    if 0 <= fr <= max_frame:
                        dtheta = abs(_angular_diff_deg(theta_vel[i], theta_vel[i - 1]))
                        k = int(k_per_frame[fr])
                        append_mode_list(mode, dtheta_by_k_overall, dtheta_by_k_messy, dtheta_by_k_straight, k, float(dtheta))

    # ------------ compute mean±std arrays ------------
    sp_o_mean, sp_o_std, sp_o_n = mean_std_per_k(speed_by_k_overall)
    dt_o_mean, dt_o_std, dt_o_n = mean_std_per_k(dtheta_by_k_overall)

    sp_m_mean, sp_m_std, sp_m_n = mean_std_per_k(speed_by_k_messy)
    sp_s_mean, sp_s_std, sp_s_n = mean_std_per_k(speed_by_k_straight)

    dt_m_mean, dt_m_std, dt_m_n = mean_std_per_k(dtheta_by_k_messy)
    dt_s_mean, dt_s_std, dt_s_n = mean_std_per_k(dtheta_by_k_straight)

    # ------------ output paths ------------
    stem, _ = os.path.splitext(output_png_path)

    out_overall_speed  = f"{stem}_overall_meanstd_speed_vs_k.png"
    out_overall_dtheta = f"{stem}_overall_meanstd_dtheta_vs_k.png"
    out_overall_hist   = f"{stem}_overall_frames_hist_k.png"

    out_speed_2modes   = f"{stem}_meanstd_speed_vs_k_2modes.png"
    out_dtheta_2modes  = f"{stem}_meanstd_dtheta_vs_k_2modes.png"

    ks = np.arange(max_protrusions_bin + 1)

    # ------------ (3) overall frames histogram ------------
    kmax_o = last_nonzero_idx(frames_k_overall)
    ks_hist = np.arange(0, kmax_o + 1)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(ks_hist, frames_k_overall[:kmax_o + 1])
    ax.set_xlabel("number of protrusions (k)")
    ax.set_ylabel("number of frames")
    ax.set_title(f"Overall: frames count vs k (min_len ≥ {min_len})")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_overall_hist, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ------------ (1) overall speed mean±std vs k (errorbar) ------------
    valid = np.where(~np.isnan(sp_o_mean))[0]
    kmax = int(valid.max()) if valid.size > 0 else 0
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.errorbar(ks[:kmax + 1], sp_o_mean[:kmax + 1], yerr=sp_o_std[:kmax + 1], fmt="o", capsize=4)
    ax.set_xlabel("number of protrusions (k)")
    ax.set_ylabel("mean |v|  ± std")
    ax.set_title(f"Overall: speed vs k (min_len ≥ {min_len})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_overall_speed, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ------------ (2) overall dtheta mean±std vs k (errorbar) ------------
    valid = np.where(~np.isnan(dt_o_mean))[0]
    kmax = int(valid.max()) if valid.size > 0 else 0
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.errorbar(ks[:kmax + 1], dt_o_mean[:kmax + 1], yerr=dt_o_std[:kmax + 1], fmt="o", capsize=4)
    ax.set_xlabel("number of protrusions (k)")
    ax.set_ylabel("mean |Δθ| (deg) ± std")
    ax.set_title(f"Overall: |Δθ| vs k (min_len ≥ {min_len})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_overall_dtheta, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ------------ (4) 2-panel frames hist by mode -> output_png_path ------------
    kmax_m = last_nonzero_idx(frames_k_messy)
    kmax_s = last_nonzero_idx(frames_k_straight)
    kmax = max(kmax_m, kmax_s)
    ks_hist = np.arange(0, kmax + 1)

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    ax0.bar(ks_hist, frames_k_messy[:kmax + 1])
    ax0.set_title(f"Messy: frames count vs k (min_len ≥ {min_len})")
    ax0.set_ylabel("# frames")
    ax0.grid(True, axis="y", alpha=0.3)

    ax1.bar(ks_hist, frames_k_straight[:kmax + 1])
    ax1.set_title(f"Straight: frames count vs k (min_len ≥ {min_len})")
    ax1.set_xlabel("number of protrusions (k)")
    ax1.set_ylabel("# frames")
    ax1.grid(True, axis="y", alpha=0.3)

    fig.savefig(output_png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ------------ (5) 2-panel speed mean±std vs k by mode (errorbar) ------------
    valid_m = np.where(~np.isnan(sp_m_mean))[0]
    valid_s = np.where(~np.isnan(sp_s_mean))[0]
    kmax = int(max(valid_m.max() if valid_m.size else 0, valid_s.max() if valid_s.size else 0))

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    ax0.errorbar(ks[:kmax + 1], sp_m_mean[:kmax + 1], yerr=sp_m_std[:kmax + 1], fmt="o", capsize=4)
    ax0.set_title(f"Messy: speed vs k (mean ± std, min_len ≥ {min_len})")
    ax0.set_ylabel("mean |v| ± std")
    ax0.grid(True, alpha=0.3)

    ax1.errorbar(ks[:kmax + 1], sp_s_mean[:kmax + 1], yerr=sp_s_std[:kmax + 1], fmt="o", capsize=4)
    ax1.set_title(f"Straight: speed vs k (mean ± std, min_len ≥ {min_len})")
    ax1.set_xlabel("number of protrusions (k)")
    ax1.set_ylabel("mean |v| ± std")
    ax1.grid(True, alpha=0.3)

    fig.savefig(out_speed_2modes, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ------------ (6) 2-panel dtheta mean±std vs k by mode (errorbar) ------------
    valid_m = np.where(~np.isnan(dt_m_mean))[0]
    valid_s = np.where(~np.isnan(dt_s_mean))[0]
    kmax = int(max(valid_m.max() if valid_m.size else 0, valid_s.max() if valid_s.size else 0))

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    ax0.errorbar(ks[:kmax + 1], dt_m_mean[:kmax + 1], yerr=dt_m_std[:kmax + 1], fmt="o", capsize=4)
    ax0.set_title(f"Messy: |Δθ| vs k (mean ± std, min_len ≥ {min_len})")
    ax0.set_ylabel("mean |Δθ| ± std")
    ax0.grid(True, alpha=0.3)

    ax1.errorbar(ks[:kmax + 1], dt_s_mean[:kmax + 1], yerr=dt_s_std[:kmax + 1], fmt="o", capsize=4)
    ax1.set_title(f"Straight: |Δθ| vs k (mean ± std, min_len ≥ {min_len})")
    ax1.set_xlabel("number of protrusions (k)")
    ax1.set_ylabel("mean |Δθ| ± std")
    ax1.grid(True, alpha=0.3)

    fig.savefig(out_dtheta_2modes, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[OK] Saved overall mean±std speed plot to {out_overall_speed}")
    print(f"[OK] Saved overall mean±std dtheta plot to {out_overall_dtheta}")
    print(f"[OK] Saved overall frames hist to {out_overall_hist}")
    print(f"[OK] Saved 2-mode frames hist to {output_png_path}")
    print(f"[OK] Saved 2-mode mean±std speed plot to {out_speed_2modes}")
    print(f"[OK] Saved 2-mode mean±std dtheta plot to {out_dtheta_2modes}")
def accumulate_protrusion_statistics_ot_2modes(
    dataset_dir: str,
    roi_root_dir: str,
    output_png_path: str,
    min_len: int = 10,
    max_protrusions_bin: int = 5,
    times_path: str = "/home/pavel/Desktop/ot_res_08_10/times.npy",
    riemann_path: str = "/home/pavel/Desktop/ot_res_08_10/ot_distances.npy",
):
    """
    По образу и подобию твоей accumulate_protrusion_statistics_2modes (speed/dtheta),
    но вместо speed/dtheta считаем OT-rate = OT_distance / Δt.

    ИТОГО 4 картинки:

    ОБЩИЕ (all cells):
      1) <stem>_overall_ot_meanstd_vs_k.png     (точки mean±std)
      2) <stem>_overall_frames_hist_k.png       (ГИСТОГРАММА #frames vs k)

    ПО МОДАМ (messy/straight), 2 панели одна под другой:
      3) <stem>_ot_meanstd_vs_k_2modes.png      (точки mean±std, 2 панели)
      4) output_png_path                         (ГИСТОГРАММЫ #frames vs k, 2 панели)

    k(frame) = число активных lineages в кадре (после фильтрации lineage по min_len),
    из peaks_troughs_list.npz: frame_id=pnt_list[:,1], lineage_id=pnt_list[:,7]

    OT-rate[idx] = riemann[cell][idx] / (times[cell][idx] - times[cell][idx-1]), idx=1..T-1
    и относится к кадру fr = idx.

    ВАЖНО:
      - индекс в times/riemann берём из имени папки cell_XXX -> XXX (без ci-1!)
      - OT и k синхронизируем по кадрам: idx_max = min(T-1, max_frame)
      - в последний бин сваливаем хвост: kk = min(k, max_protrusions_bin)
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    # ---------- load OT arrays once ----------
    times = np.load(times_path, allow_pickle=True)
    riemann = np.load(riemann_path, allow_pickle=True)

    def get_times(cell: int, frame: int) -> float:
        return float(times[cell][frame])

    def get_riemann_dist(cell: int, frame: int) -> float:
        return float(riemann[cell][frame])

    def cell_idx_from_name(cell_name: str) -> int:
        return int(cell_name.split("_")[1])

    def last_nonzero_idx(arr: np.ndarray) -> int:
        nz = np.where(arr > 0)[0]
        return int(nz.max()) if nz.size > 0 else 0

    def mean_std_per_k(values_by_k):
        mean = np.full(max_protrusions_bin + 1, np.nan, float)
        std  = np.full(max_protrusions_bin + 1, np.nan, float)
        n    = np.zeros(max_protrusions_bin + 1, int)
        for k in range(max_protrusions_bin + 1):
            vals = np.asarray(values_by_k[k], float)
            n[k] = vals.size
            if vals.size > 0:
                mean[k] = float(vals.mean())
                std[k]  = float(vals.std())
        return mean, std, n

    # ------------ folders ------------
    cell_folders_all = sorted(
        [f for f in os.listdir(dataset_dir) if f.startswith("cell_")],
        key=lambda x: int(x.split("_")[1])
    )
    os.makedirs(roi_root_dir, exist_ok=True)

    ind_straight = np.array([
        0, 1, 2, 4, 5, 6, 7, 11, 12, 15, 17, 23, 24, 26,
        41, 50, 65, 66, 67, 77, 78, 80, 85, 86
    ])
    ind_messy = np.array([
        13, 20, 21, 28, 44, 49, 53, 63, 87, 91, 100, 101,
        102, 103, 105, 106, 107, 109, 110, 111, 112, 113,
        116, 120
    ])

    straight_folders = [cell_folders_all[i] for i in ind_straight]
    messy_folders    = [cell_folders_all[i] for i in ind_messy]
    messy_set = set(messy_folders)
    straight_set = set(straight_folders)

    def mode_of_cell(cell_name: str) -> str:
        if cell_name in messy_set:
            return "messy"
        if cell_name in straight_set:
            return "straight"
        return "other"

    # ------------ accumulators ------------
    # frames hist by k
    frames_k_overall  = np.zeros(max_protrusions_bin + 1, dtype=int)
    frames_k_messy    = np.zeros(max_protrusions_bin + 1, dtype=int)
    frames_k_straight = np.zeros(max_protrusions_bin + 1, dtype=int)

    # OT-rate values by k
    ot_by_k_overall  = [[] for _ in range(max_protrusions_bin + 1)]
    ot_by_k_messy    = [[] for _ in range(max_protrusions_bin + 1)]
    ot_by_k_straight = [[] for _ in range(max_protrusions_bin + 1)]

    def append_mode_list(mode: str, overall_list, messy_list, straight_list, k: int, value: float):
        kk = min(int(k), max_protrusions_bin)
        overall_list[kk].append(value)
        if mode == "messy":
            messy_list[kk].append(value)
        elif mode == "straight":
            straight_list[kk].append(value)

    # ------------ main loop over ALL cells once (как у тебя) ------------
    total_cells = len(cell_folders_all)
    for ii, cell_name in enumerate(cell_folders_all, start=1):
        mode = mode_of_cell(cell_name)
        print(f"\n=== {cell_name} ({ii}/{total_cells}) mode={mode} ===")

        cell_path = os.path.join(dataset_dir, cell_name)
        roi_dir   = os.path.join(roi_root_dir, cell_name, "roi_data")
        os.makedirs(roi_dir, exist_ok=True)
        pnt_path  = os.path.join(roi_dir, "peaks_troughs_list.npz")

        # generate if missing
        if not os.path.exists(pnt_path):
            try:
                idx_true = cell_idx_from_name(cell_name)
                cell, outlines, outlines_CMF = single_cell_tracking(dataset_dir, idx_true)
                peak_troughs_lineage(cell, outlines, outlines_CMF, roi_dir)
                print(f"[ok] generated pnt_list for {cell_name}")
            except Exception as e:
                print(f"[skip] generation failed for {cell_name}: {e}")
                continue

        # load pnt_list
        try:
            dat = np.load(pnt_path, allow_pickle=True)
            pnt_list = dat["pnt_list"] if "pnt_list" in dat else dat[list(dat.keys())[0]]
            pnt_list = np.asarray(pnt_list, dtype=float)
        except Exception as e:
            print(f"[skip] cannot load {pnt_path}: {e}")
            continue

        if pnt_list.size == 0:
            print(f"[skip] empty pnt_list for {cell_name}")
            continue

        # frames 0..max_frame (для гистограмм по кадрам, включая k=0)
        all_frame_ids = pnt_list[:, 1].astype(int)
        max_frame = int(all_frame_ids.max())
        frames_full = np.arange(0, max_frame + 1, dtype=int)

        # filter lineages by min_len
        lids, counts = np.unique(pnt_list[:, 7], return_counts=True)
        long_ids = lids[counts >= min_len]
        pnt_filt = pnt_list[np.isin(pnt_list[:, 7], long_ids)]

        # k_per_frame
        k_per_frame = np.zeros(max_frame + 1, dtype=int)
        if pnt_filt.size > 0:
            frame_ids   = pnt_filt[:, 1].astype(int)
            lineage_ids = pnt_filt[:, 7].astype(int)
            for fr in np.unique(frame_ids):
                if 0 <= fr <= max_frame:
                    k_per_frame[fr] = np.unique(lineage_ids[frame_ids == fr]).size

        # update frames histogram (include zeros)
        for fr in frames_full:
            k = int(k_per_frame[fr])
            kk = min(k, max_protrusions_bin)
            frames_k_overall[kk] += 1
            if mode == "messy":
                frames_k_messy[kk] += 1
            elif mode == "straight":
                frames_k_straight[kk] += 1

        # --------- OT accumulation by k (как speed/dtheta: просто добавляем значения) ---------
        cell_idx = cell_idx_from_name(cell_name)
        if cell_idx < 0 or cell_idx >= len(times):
            print(f"[warn] OT arrays do not contain cell_idx={cell_idx} for {cell_name}")
            continue

        T = len(times[cell_idx])
        if T < 2:
            print(f"[warn] too short OT times for {cell_name}")
            continue

        # синхронизация: OT можно только для fr=1..T-1, а k_per_frame определён до max_frame
        idx_max = min(T - 1, max_frame)

        for idx in range(1, idx_max + 1):
            fr = idx
            t_curr = get_times(cell_idx, idx)
            t_prev = get_times(cell_idx, idx - 1)
            dt = t_curr - t_prev
            if dt <= 0:
                continue

            rate = get_riemann_dist(cell_idx, idx) / dt

            k = int(k_per_frame[fr])
            append_mode_list(mode, ot_by_k_overall, ot_by_k_messy, ot_by_k_straight, k, float(rate))

    # ------------ mean±std arrays ------------
    ot_o_mean, ot_o_std, ot_o_n = mean_std_per_k(ot_by_k_overall)
    ot_m_mean, ot_m_std, ot_m_n = mean_std_per_k(ot_by_k_messy)
    ot_s_mean, ot_s_std, ot_s_n = mean_std_per_k(ot_by_k_straight)

    # ------------ output paths ------------
    stem, _ = os.path.splitext(output_png_path)
    out_overall_ot   = f"{stem}_overall_ot_meanstd_vs_k.png"
    out_overall_hist = f"{stem}_overall_frames_hist_k.png"
    out_ot_2modes    = f"{stem}_ot_meanstd_vs_k_2modes.png"

    ks = np.arange(max_protrusions_bin + 1)

    # =========================================================
    # (2) overall frames histogram (#frames vs k)  [СТОЛБИКИ]
    # =========================================================
    kmax_o = last_nonzero_idx(frames_k_overall)
    ks_hist = np.arange(0, kmax_o + 1)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(ks_hist, frames_k_overall[:kmax_o + 1])
    ax.set_xlabel("number of protrusions (k)")
    ax.set_ylabel("number of frames")
    ax.set_title(f"Overall: frames count vs k (min_len ≥ {min_len})")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_overall_hist, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # =========================================================
    # (1) overall OT mean±std vs k  [ТОЧКИ С ПОГРЕШНОСТЬЮ]
    # =========================================================
    valid = np.where(~np.isnan(ot_o_mean))[0]
    kmax = int(valid.max()) if valid.size > 0 else 0

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(ks[:kmax + 1], ot_o_mean[:kmax + 1], yerr=ot_o_std[:kmax + 1], fmt="o", capsize=4)
    ax.set_title(f"Overall: accumulated mean OT-rate vs k (min_len ≥ {min_len})")
    ax.set_xlabel("number of protrusions (k)")
    ax.set_ylabel("mean OT distance / Δt  ± std")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, max_protrusions_bin + 0.5)
    fig.tight_layout()
    fig.savefig(out_overall_ot, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # =========================================================
    # (4) 2-panel frames histogram by mode -> output_png_path  [СТОЛБИКИ]
    # =========================================================
    kmax_m = last_nonzero_idx(frames_k_messy)
    kmax_s = last_nonzero_idx(frames_k_straight)
    kmax = max(kmax_m, kmax_s)
    ks_hist = np.arange(0, kmax + 1)

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)

    ax0.bar(ks_hist, frames_k_messy[:kmax + 1])
    ax0.set_title(f"Messy: frames count vs k (min_len ≥ {min_len})")
    ax0.set_ylabel("# frames")
    ax0.grid(True, axis="y", alpha=0.3)

    ax1.bar(ks_hist, frames_k_straight[:kmax + 1])
    ax1.set_title(f"Straight: frames count vs k (min_len ≥ {min_len})")
    ax1.set_xlabel("number of protrusions (k)")
    ax1.set_ylabel("# frames")
    ax1.grid(True, axis="y", alpha=0.3)

    fig.savefig(output_png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # =========================================================
    # (3) 2-panel OT mean±std vs k by mode  [ТОЧКИ С ПОГРЕШНОСТЬЮ]
    # =========================================================
    valid_m = np.where(~np.isnan(ot_m_mean))[0]
    valid_s = np.where(~np.isnan(ot_s_mean))[0]
    kmax = int(max(valid_m.max() if valid_m.size else 0, valid_s.max() if valid_s.size else 0))

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)

    ax0.errorbar(ks[:kmax + 1], ot_m_mean[:kmax + 1], yerr=ot_m_std[:kmax + 1], fmt="o", capsize=4)
    ax0.set_title(f"Messy: accumulated mean OT-rate vs k (min_len ≥ {min_len})")
    ax0.set_ylabel("mean OT distance / Δt  ± std")
    ax0.grid(True, alpha=0.3)

    ax1.errorbar(ks[:kmax + 1], ot_s_mean[:kmax + 1], yerr=ot_s_std[:kmax + 1], fmt="o", capsize=4)
    ax1.set_title(f"Straight: accumulated mean OT-rate vs k (min_len ≥ {min_len})")
    ax1.set_xlabel("number of protrusions (k)")
    ax1.set_ylabel("mean OT distance / Δt  ± std")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.5, max_protrusions_bin + 0.5)

    fig.savefig(out_ot_2modes, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[OK] Saved overall OT mean±std vs k to {out_overall_ot}")
    print(f"[OK] Saved overall frames hist to {out_overall_hist}")
    print(f"[OK] Saved 2-mode OT mean±std vs k to {out_ot_2modes}")
    print(f"[OK] Saved 2-mode frames hist to {output_png_path}")

def generate_lineages_pdf_for_dataset_3criteria_onepage_plus_velocity_markers(
    dataset_dir: str,
    roi_root_dir: str,
    output_pdf_path: str,
    min_len: int = 10,
):
    """
    One page per cell:
      - 3 selected lineages (longest, best mean curvature, best mean height)
      - plus cell velocity direction (angle of centroid velocity)
    Each of the 4 overlays uses a different marker so overlaps are visible.

    Assumes pnt_list columns:
      col1 = frame_id
      col3 = theta (deg)
      col4 = height/dr
      col7 = lineage_id
      col8 = curvature
    """
    cell_folders = sorted(
        [f for f in os.listdir(dataset_dir) if f.startswith("cell_")],
        key=lambda x: int(x.split("_")[1])
    )

    os.makedirs(roi_root_dir, exist_ok=True)

    with PdfPages(output_pdf_path) as pdf:
        total_cells = len(cell_folders)
        print(f"[INFO] Found {total_cells} cells")

        for ci, cell_name in enumerate(cell_folders, start=1):
            print(f"\n=== {cell_name} ({ci}/{total_cells}) ===")

            cell_path = os.path.join(dataset_dir, cell_name)
            roi_dir = os.path.join(roi_root_dir, cell_name, "roi_data")
            os.makedirs(roi_dir, exist_ok=True)
            pnt_path = os.path.join(roi_dir, "peaks_troughs_list.npz")

            # --- generate if missing ---
            if not os.path.exists(pnt_path):
                try:
                    cell, outlines,outlines_CMF = single_cell_tracking(dataset_dir, ci - 1)
                    peak_troughs_lineage(cell, outlines, outlines_CMF,roi_dir)
                    print(f"[ok] generated pnt_list for {cell_name}")
                except Exception as e:
                    print(f"[skip] generation failed for {cell_name}: {e}")
                    continue

            # --- load ---
            try:
                dat = np.load(pnt_path, allow_pickle=True)
                pnt_list = dat["pnt_list"] if "pnt_list" in dat else dat[list(dat.keys())[0]]
                pnt_list = np.asarray(pnt_list, dtype=float)
            except Exception as e:
                print(f"[skip] cannot load {pnt_path}: {e}")
                continue

            if pnt_list.ndim != 2 or pnt_list.size == 0:
                print(f"[skip] empty/bad pnt_list for {cell_name}")
                continue

            if pnt_list.shape[1] < 9:
                print(f"[skip] pnt_list has {pnt_list.shape[1]} cols, expected ≥9 (needs curvature in col8) for {cell_name}")
                continue

            # --- filter by min_len ---
            lids_all = pnt_list[:, 7].astype(int)
            uniq, counts = np.unique(lids_all, return_counts=True)
            long_ids = uniq[(uniq != -1) & (counts >= min_len)]
            if long_ids.size == 0:
                print(f"[skip] no long lineages (≥{min_len}) for {cell_name}")
                continue

            pnt_long = pnt_list[np.isin(lids_all, long_ids)]

            # helper: pick lid by aggregate (max)
            def select_by_agg(pnt, agg_fn):
                lids = np.unique(pnt[:, 7].astype(int))
                lids = lids[lids != -1]
                best_lid, best_val = None, -np.inf
                for lid in lids:
                    arr = pnt[pnt[:, 7].astype(int) == lid]
                    val = agg_fn(arr)
                    if np.isnan(val):
                        continue
                    if val > best_val:
                        best_val = val
                        best_lid = int(lid)
                return best_lid, best_val

            # 1) longest by count
            lid_len, val_len = select_by_agg(pnt_long, lambda arr: float(arr.shape[0]))

            # 2) best mean curvature (col8)
            lid_curv, val_curv = select_by_agg(pnt_long, lambda arr: float(np.nanmean(arr[:, 8])))

            # 3) best mean height/dr (col4)
            lid_h, val_h = select_by_agg(pnt_long, lambda arr: float(np.nanmean(arr[:, 4])))

            selected = [
                ("Longest", lid_len, f"len={int(val_len)}"),
                ("Best mean curvature", lid_curv, f"mean_curv={np.abs(val_curv):.4g}"),
                ("Best mean height", lid_h, f"mean_h={val_h:.4g}"),
            ]

            # --- velocity direction line (centroid velocity angle) ---
            theta_vel = None
            frames_vel = None
            try:
                (
                    _all_outlines,
                    _,
                    _all_outlines_cMCF_topography,
                    _all_outlines_curvature,
                    _disk_coors,
                    centr,
                    fin_times,
                ) = conformal_representation(cell_path)

                centr = np.asarray(centr, dtype=float)
                if centr.ndim == 2 and centr.shape[0] >= 2:
                    v = np.diff(centr, axis=0)  # (T-1, 2)
                    theta_vel = (np.degrees(np.arctan2(v[:, 1], v[:, 0])) % 360.0)
                    frames_vel = np.arange(1, len(centr), dtype=float)  # velocity defined between frames
            except Exception as e:
                print(f"[warn] cannot compute centroid velocity angle for {cell_name}: {e}")

            # --- plotting: one page with 4 overlays, different markers ---
            marker_map = {
                "Longest": "o",               # circles
                "Best mean curvature": "x",   # crosses
                "Best mean height": "s",      # squares
            }
            color_map = {
                "Longest": "C0",
                "Best mean curvature": "C1",
                "Best mean height": "C2",
            }

            fig, ax = plt.subplots(figsize=(9, 3.8))
            any_plotted = False

            for label, lid, stat in selected:
                if lid is None:
                    continue

                pts = pnt_long[pnt_long[:, 7].astype(int) == int(lid)]
                if pts.size == 0:
                    continue

                m = marker_map[label]
                c = color_map[label]

                # Hollow markers for o/s; x doesn't have facecolor
                if m in ("o", "s"):
                    ax.scatter(
                        pts[:, 1], pts[:, 3],
                        s=30,
                        marker=m,
                        facecolors="none",
                        edgecolors=c,
                        linewidths=1.2,
                        label=f"{label}: lid={lid} ({stat})",
                        zorder=2,
                    )
                else:  # "x"
                    ax.scatter(
                        pts[:, 1], pts[:, 3],
                        s=30,
                        marker=m,
                        c=c,
                        linewidths=1.2,
                        label=f"{label}: lid={lid} ({stat})",
                        zorder=3,
                    )

                any_plotted = True

            # velocity overlay: black line + triangles
            if theta_vel is not None and frames_vel is not None and len(theta_vel) == len(frames_vel):
                theta_vel = gaussian_filter1d(theta_vel, 1, mode="nearest")
                ax.plot(
                    frames_vel, theta_vel,
                    color="k",
                    linewidth=1.2,
                    marker="^",
                    markersize=4,
                    label="Cell velocity direction",
                    zorder=4,
                )

            if not any_plotted:
                print(f"[skip] nothing to plot for {cell_name}")
                plt.close(fig)
                continue

            ax.set_xlabel("frame_id")
            ax.set_ylabel("θ (deg)")
            ax.set_ylim(0, 360)
            ax.set_yticks(np.arange(0, 361, 30))
            ax.set_title(f"{cell_name} | 3 selected lineages + velocity direction")
            ax.legend(loc="upper right", fontsize=8, frameon=True)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        print(f"\n[OK] Saved all lineage plots to {output_pdf_path}")




def generate_lineages_pdf_highlight_best_point_each_frame(
    dataset_dir: str,
    roi_root_dir: str,
    output_pdf_path: str,
    min_len: int = 10,
    prot_type_value: int = 1,   # 1 = protrusion/peak (если у тебя иначе — поменяй)
):
    """
    One page per cell:
      - plot ALL lineages (>= min_len) as background points
      - for EACH frame: highlight ONE protrusion point with
          (a) maximum height/dr   (col4)
          (b) maximum |curvature| (col8)
      - overlay centroid velocity direction angle

    pnt_list columns assumed:
      col1 frame_id
      col2 type
      col3 theta (deg)
      col4 height/dr
      col7 lineage_id
      col8 curvature
    """
    cell_folders = sorted(
        [f for f in os.listdir(dataset_dir) if f.startswith("cell_")],
        key=lambda x: int(x.split("_")[1])
    )

    os.makedirs(roi_root_dir, exist_ok=True)

    with PdfPages(output_pdf_path) as pdf:
        total_cells = len(cell_folders)
        print(f"[INFO] Found {total_cells} cells")

        for ci, cell_name in enumerate(cell_folders, start=1):
            print(f"\n=== {cell_name} ({ci}/{total_cells}) ===")

            cell_path = os.path.join(dataset_dir, cell_name)
            roi_dir = os.path.join(roi_root_dir, cell_name, "roi_data")
            os.makedirs(roi_dir, exist_ok=True)
            pnt_path = os.path.join(roi_dir, "peaks_troughs_list.npz")

            # --- generate if missing ---
            if not os.path.exists(pnt_path):
                try:
                    cell, outlines,outlines_CMF = single_cell_tracking(dataset_dir, ci - 1)
                    peak_troughs_lineage(cell, outlines, outlines_CMF,roi_dir)
                    print(f"[ok] generated pnt_list for {cell_name}")
                except Exception as e:
                    print(f"[skip] generation failed for {cell_name}: {e}")
                    continue

            # --- load ---
            try:
                dat = np.load(pnt_path, allow_pickle=True)
                pnt_list = dat["pnt_list"] if "pnt_list" in dat else dat[list(dat.keys())[0]]
                pnt_list = np.asarray(pnt_list, dtype=float)
            except Exception as e:
                print(f"[skip] cannot load {pnt_path}: {e}")
                continue

            if pnt_list.ndim != 2 or pnt_list.size == 0:
                print(f"[skip] empty/bad pnt_list for {cell_name}")
                continue

            if pnt_list.shape[1] < 9:
                print(f"[skip] pnt_list has {pnt_list.shape[1]} cols, expected ≥9 for {cell_name}")
                continue

            # --- keep only lineages with >= min_len ---
            lids_all = pnt_list[:, 7].astype(int)
            uniq, counts = np.unique(lids_all, return_counts=True)
            long_ids = uniq[(uniq != -1) & (counts >= min_len)]
            if long_ids.size == 0:
                print(f"[skip] no long lineages (≥{min_len}) for {cell_name}")
                continue

            pnt_long = pnt_list[np.isin(lids_all, long_ids)]

            # --- keep only protrusions for frame-wise selection (type == prot_type_value) ---
            is_prot = (pnt_long[:, 2].astype(int) == int(prot_type_value))
            prot = pnt_long[is_prot]
            if prot.size == 0:
                print(f"[skip] no protrusion points after filtering for {cell_name}")
                continue

            # --- per-frame best point by height and by |curvature| ---
            frames = np.unique(prot[:, 1].astype(int))

            best_by_height = []
            best_by_abs_curv = []

            for fr in frames:
                pts = prot[prot[:, 1].astype(int) == fr]
                if pts.size == 0:
                    continue

                # best by height/dr (col4)
                idx_h = int(np.nanargmax(pts[:, 4]))
                best_by_height.append(pts[idx_h])

                # best by |curvature| (col8)
                idx_c = int(np.nanargmax(np.abs(pts[:, 8])))
                best_by_abs_curv.append(pts[idx_c])

            best_by_height = np.asarray(best_by_height, dtype=float) if best_by_height else None
            best_by_abs_curv = np.asarray(best_by_abs_curv, dtype=float) if best_by_abs_curv else None

            # --- velocity direction (centroid velocity angle) ---
            theta_vel = None
            frames_vel = None
            try:
                (
                    _all_outlines,
                    _,
                    _all_outlines_cMCF_topography,
                    _all_outlines_curvature,
                    _disk_coors,
                    centr,
                    fin_times,
                ) = conformal_representation(cell_path)

                centr = np.asarray(centr, dtype=float)
                if centr.ndim == 2 and centr.shape[0] >= 2:
                    v = np.diff(centr, axis=0)
                    theta_vel = (np.degrees(np.arctan2(v[:, 1], v[:, 0])) % 360.0)
                    theta_vel = gaussian_filter1d(theta_vel, 1, mode="nearest")
                    frames_vel = np.arange(1, len(centr), dtype=float)
            except Exception as e:
                print(f"[warn] cannot compute centroid velocity angle for {cell_name}: {e}")

            # --- plot: one page ---
            fig, ax = plt.subplots(figsize=(9, 3.8))

            # background: all long lineages (all types)
            ax.scatter(
                pnt_long[:, 1], pnt_long[:, 3],
                s=8, marker=".",
                alpha=0.25,
                zorder=1,
                label=f"All lineages (≥{min_len})"
            )

            # highlight per-frame best height: circles outline
            if best_by_height is not None and best_by_height.size:
                ax.scatter(
                    best_by_height[:, 1], best_by_height[:, 3],
                    s=60, marker="o",
                    facecolors="none",
                    edgecolors="C2",
                    linewidths=1.6,
                    zorder=4,
                    label="Per-frame max height (protrusion)"
                )

            # highlight per-frame best |curv|: squares outline
            if best_by_abs_curv is not None and best_by_abs_curv.size:
                ax.scatter(
                    best_by_abs_curv[:, 1], best_by_abs_curv[:, 3],
                    s=70, marker="s",
                    facecolors="none",
                    edgecolors="C1",
                    linewidths=1.6,
                    zorder=5,
                    label="Per-frame max |curvature| (protrusion)"
                )

            # velocity overlay
            if theta_vel is not None and frames_vel is not None and len(theta_vel) == len(frames_vel):
                ax.plot(
                    frames_vel, theta_vel,
                    color="k",
                    linewidth=1.2,
                    marker="^",
                    markersize=4,
                    zorder=6,
                    label="Cell velocity direction"
                )

            ax.set_xlabel("frame_id")
            ax.set_ylabel("θ (deg)")
            ax.set_ylim(0, 360)
            ax.set_yticks(np.arange(0, 361, 30))
            ax.set_title(f"{cell_name} | per-frame highlighted protrusion points + velocity")
            ax.legend(loc="upper right", fontsize=8, frameon=True)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        print(f"\n[OK] Saved all lineage plots to {output_pdf_path}")

# generate_lineages_pdf_highlight_best_point_each_frame(
#      dataset_dir="/home/pavel/cell_morphology/filtered_data/cells_filtered",
#      roi_root_dir="/home/pavel/shape_analysis/roi_data_all",
#      output_pdf_path="/home/pavel/shape_analysis/lineages_dataset_curvature_filtered_best_curv_height_velocity.pdf",
#      min_len=10)


# # generate_lineages_count_vs_frame_pdf_for_dataset(
# #     dataset_dir="/home/pavel/cell_morphology/filtered_data/cells_filtered",
# #     roi_root_dir="/home/pavel/shape_analysis/roi_data_all",
# #     output_pdf_path="/home/pavel/shape_analysis/lineage_amount_per_time.pdf",
# #     min_len=10)


# accumulate_protrusion_statistics_2modes(
#     dataset_dir="/home/pavel/cell_morphology/filtered_data/cells_filtered",
#     roi_root_dir="/home/pavel/shape_analysis/roi_data_all",
#     output_png_path = "statistics_protrusion_frames_messy.png",
# )
# accumulate_protrusion_statistics_ot_2modes(
#     dataset_dir="/home/pavel/cell_morphology/filtered_data/cells_filtered",
#     roi_root_dir="/home/pavel/shape_analysis/roi_data_all",
#     output_png_path = "statistics_protrusion_frames_ot_messy.png",
# )



glob_filtered_folder = '/home/pavel/cell_morphology/filtered_data/cells_filtered/'

#peak_troughs_lineage(single_cell_tracking(glob_filtered_folder, 28)[0],single_cell_tracking(glob_filtered_folder, 28)[1], '/home/pavel/shape_analysis/roi_data')
cell, outlines,outlines_CMF = single_cell_tracking(glob_filtered_folder, 0)
peak_troughs_lineage(cell, outlines, outlines_CMF,'/home/pavel/shape_analysis/roi_data')
npz_path= "/home/pavel/shape_analysis/roi_data/peaks_troughs_list.npz"
# roi_dir = "/home/pavel/shape_analysis/roi_data"
# os.makedirs(roi_dir, exist_ok=True)
# cell_folders = sorted(
#     [d for d in os.listdir(glob_filtered_folder) if d.startswith("cell_")],
#     key=lambda x: int(x.split("_")[1])
# )

# for i, cell_name in enumerate(cell_folders):
#     print(f"\n=== {cell_name} (i={i})x ===")
#     cell, outlines, outlines_CMF = single_cell_tracking(glob_filtered_folder, i)

#     peak_troughs_lineage(cell, outlines, outlines_CMF, roi_dir)

#     npz_path = os.path.join(roi_dir, "peaks_troughs_list.npz")
#     out_pdf = f"{cell_name}_tracking.pdf"
#     polar_pdf_from_pntlist(cell, outlines, npz_path, out_pdf)

#     print(f"[OK] {cell_name} -> {out_pdf}")


#polar_pdf_from_cell(cell,"test1.pdf")

polar_pdf_from_pntlist(cell, outlines,npz_path, "cell_1_tracking.pdf")
#polar_pdf_from_lineages(npz_path, "test_2_ver_3.pdf")
#single_cell_tracking(glob_filtered_folder, 1)

#

#data = np.load('/home/pavel/shape_analysis/roi_data/peaks_troughs_ROI_list.npz', allow_pickle = True)
#print(data["arr_0"])
#print(data["arr_0"].shape)


#build_protrusion_counts_array("/home/pavel/shape_analysis/roi_data_all", n_cells=121)


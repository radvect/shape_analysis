import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def angle_between_vectors(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    cosine_angle = np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0)
    angle_rad = np.arccos(cosine_angle)
    return angle_rad


with open("W_all.pkl", "rb") as file:
    W = pickle.load(file)
with open("C_all.pkl", "rb") as file:
    C = pickle.load(file)
with open("T_all.pkl", "rb") as file:
    T = pickle.load(file)

phys_turn = []
phys_dist = []
shape_change = []
Disp_list = []
Wmax_list = []
dpratio_phys = []
dpratio_shape = []

for i in range(len(W)):
    sorted_Centroid = C[i]
    v = C[i]
    Wmat2 = W[i]
    sorted_Time = T[i]

    velocity_vectors = np.diff(sorted_Centroid, axis=0)
    angles = []
    for j in range(len(velocity_vectors) - 1):
        angle_rad = angle_between_vectors(velocity_vectors[j], velocity_vectors[j + 1])
        angles.append(np.degrees(angle_rad))

    sup_diag = np.diag(Wmat2, k=1)

    dv = []
    dist_inc = 0.0
    for k in range(1, v.shape[0]):
        inc = v[k, :] - v[k - 1, :]
        dist_inc += np.linalg.norm(inc, 2)
        dv.append(dist_inc)
    DCent = dv[-1]

    an = np.array(angles)
    sh = np.array(sup_diag)
    N = sorted_Time[-1] - sorted_Time[0]

    Disp_list.append(np.linalg.norm(sorted_Centroid[-1, :] - sorted_Centroid[0, :]) / N)
    Wmax_list.append(np.max(Wmat2) / N)

    Disp_phys = np.linalg.norm(sorted_Centroid[-1, :] - sorted_Centroid[0, :])
    Pathl_phys = DCent
    dpratio_phys.append(Disp_phys / Pathl_phys)

    Disp_shape = Wmat2[0, -1]
    Pathl_shape = np.sum(sh)
    dpratio_shape.append(Disp_shape / Pathl_shape)

    phys_turn.append(np.sum(an) / N)
    phys_dist.append(DCent / N)
    shape_change.append(np.sum(sh) / N)

p = np.array(phys_turn)
pd = np.array(phys_dist)
s = np.array(shape_change)
dis = np.array(Disp_list)
wmax = np.array(Wmax_list)
rphys = np.array(dpratio_phys)
rshape = np.array(dpratio_shape)


def ls_fit(x, y):
    """y = a x + b"""
    a, b = np.polyfit(x, y, 1)
    return a, b

def add_ls_line(x, y, ax, label="LS"):
    a, b = ls_fit(x, y)
    xs = np.linspace(np.min(x), np.max(x), 200)
    ax.plot(xs, a*xs + b, linewidth=2, label=f"{label}: y={a:.3g}x+{b:.3g}")
    return a, b

def add_point_labels(x, y, ax, fontsize=7, dx=0.0, dy=0.0):
    for idx, (xi, yi) in enumerate(zip(x, y)):
        ax.text(xi + dx, yi + dy, str(idx), fontsize=fontsize)

# ======= Цвета: выше/ниже МНК-линии на 1-м графике =======
a1, b1 = ls_fit(rphys, rshape)
residuals = rshape - (a1 * rphys + b1)

# выше линии -> green, ниже -> red
colors = np.where(residuals >= 0, "green", "red")

# =================== График 1: rphys vs rshape ===================
fig1, ax1 = plt.subplots()
ax1.scatter(rphys, rshape, c=colors)
ax1.set_xlabel("physical displacement / path", fontsize=20)
ax1.set_ylabel("shape displacement / path", fontsize=20)

add_ls_line(rphys, rshape, ax1, label="LS (plot1)")
add_point_labels(rphys, rshape, ax1, fontsize=7)

ax1.legend(fontsize=10)
plt.show()

# =================== График 2: pd vs s (те же цвета) ===================
fig2, ax2 = plt.subplots()
ax2.scatter(pd, s, c=colors)
ax2.set_xlabel("physical velocity", fontsize=20)
ax2.set_ylabel("shape velocity", fontsize=20)

add_ls_line(pd, s, ax2, label="LS (plot2)")
add_point_labels(pd, s, ax2, fontsize=7)

ax2.legend(fontsize=10)
plt.show()


# ====== given indices ======
ind_straight = np.array([ 0,  1,  2,  4,  5,  6,  7, 11, 12, 15, 17, 23, 24, 26,
                         41, 50, 65, 66, 67, 77, 78, 80, 85, 86], dtype=int)

ind_messy = np.array([ 13,  20,  21,  28,  44,  49,  53,  63,  87,  91, 100, 101,
                       102, 103, 105, 106, 107, 109, 110, 111, 112, 113, 116, 120], dtype=int)

# helper: label only selected points
def label_selected(ax, x, y, indices, fontsize=7):
    for idx in indices:
        ax.text(x[idx], y[idx], str(idx), fontsize=fontsize)

# =================== Plot 3: rphys vs rshape (only selected) ===================
fig3, ax3 = plt.subplots()

ax3.scatter(rphys[ind_straight], rshape[ind_straight], color="green", label="straight")
ax3.scatter(rphys[ind_messy],   rshape[ind_messy],   color="red",   label="messy")

ax3.set_xlabel("physical displacement / path", fontsize=20)
ax3.set_ylabel("shape displacement / path", fontsize=20)

label_selected(ax3, rphys, rshape, ind_straight, fontsize=7)
label_selected(ax3, rphys, rshape, ind_messy, fontsize=7)

ax3.legend(fontsize=12)
plt.show()

# =================== Plot 4: pd vs s (only selected) ===================
fig4, ax4 = plt.subplots()

ax4.scatter(pd[ind_straight], s[ind_straight], color="green", label="straight")
ax4.scatter(pd[ind_messy],    s[ind_messy],    color="red",   label="messy")

ax4.set_xlabel("physical velocity", fontsize=20)
ax4.set_ylabel("shape velocity", fontsize=20)

label_selected(ax4, pd, s, ind_straight, fontsize=7)
label_selected(ax4, pd, s, ind_messy, fontsize=7)

ax4.legend(fontsize=12)
plt.show()
import os
import numpy as np
import matplotlib.pyplot as plt


def plot_cell(n):
    
    cell_dir = 'cells'
    cell_path = os.path.join(cell_dir, os.listdir(cell_dir)[n-1])
    cell = os.listdir(cell_path)
    
    plt.figure()
    ax = plt.axes(projection='3d')
    tot_len = len(cell)
    colors = plt.cm.viridis(np.linspace(0,1,tot_len))
    for i,frame in enumerate(cell):
        time = np.load(os.path.join(cell_path, frame, 'time.npy'))
        outline = np.load(os.path.join(cell_path, frame, 'outline.npy'))
        centroid = np.load(os.path.join(cell_path, frame, 'centroid.npy'))
        print(outline, time)
        ax.plot3D(outline[:,0],outline[:,1], time*np.ones(len(outline[:,1])), c=colors[i] )
        # ax.scatter(centroid[0], centroid[1], time, c = 'k')
    plt.show()
        
    


if __name__ == "__main__":
    plot_cell(145)
    
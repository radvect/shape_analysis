import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

from mpl_toolkits.axes_grid1 import make_axes_locatable




Directory= "July6_plate1_xy02/" 





def motility_traj(direcs,plot=True,ret=True):
    dicname='Main_dictionnary.npz'

    listname='masks_list.npz'

    ROIdict='ROI_dict.npz'
    
    traj=[]
    areas = []
    for dir in direcs:
        path = os.path.join('results', dir)
        main_list=np.load(os.path.join(path, listname), allow_pickle=True)['arr_0']
        ROI_dict = np.load(os.path.join(path, ROIdict), allow_pickle=True)['arr_0'].item()
        main_dict=np.load(os.path.join(path, dicname), allow_pickle=True)['arr_0'].item()

        for ROI in ROI_dict:
            indices = ROI_dict[ROI]['Mask IDs']
            if len(indices)>=5 and ROI_dict[ROI]['Children']==[] and ROI_dict[ROI]['Parent']=='':
                position = np.array([list(main_dict[main_list[id][2]]['centroid'][main_list[id][3]-1]) for id in indices])
                time = np.array([main_dict[main_list[id][2]]['time'] for id in indices])
                area =  [int(main_dict[main_list[id][2]]['area'][main_list[id][3]-1]) for id in indices]
                var_area = np.array([area[i+1]/area[i] for i in range(len(area)-1)])
                area =  np.array(area)

                if 0.75<=np.min(var_area) and  np.max(var_area)<=1.33 and np.max(np.abs(position[:-1]-position[1:]))<100:
                    traj.append(np.column_stack((position-position[0,:],time))) 
                    areas.append(np.column_stack((area,time))) 

    if plot:
        plt.figure()
        plt.title('trajectories of the isolated cells')
        for elem in traj:
            plt.plot(elem[:,0],elem[:,1])
        plt.show()
        
        new_traj = np.concatenate(traj)
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        
        x = new_traj[:,0]
        y = new_traj[:,1]
        # the scatter plot:
        ax.scatter(x, y)
        ax.set_xlabel('distribution of the isolated cells')
        # Set aspect of the main axes.
        ax.set_aspect(1.)

        # create new axes on the right and on the top of the current axes
        divider = make_axes_locatable(ax)
        # below height and pad are in inches
        ax_histx = divider.append_axes("top", 1.2, pad=0.1, sharex=ax)
        ax_histy = divider.append_axes("right", 1.2, pad=0.1, sharey=ax)

        # make some labels invisible
        ax_histx.xaxis.set_tick_params(labelbottom=False)
        ax_histy.yaxis.set_tick_params(labelleft=False)

        # now determine nice limits by hand:
        # binwidth = 0.25
        # xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
        # lim = (int(xymax/binwidth) + 1)*binwidth

        # bins = np.arange(-lim, lim + binwidth, binwidth)
        ax_histx.hist(x, bins=80)
        ax_histy.hist(y, bins=80, orientation='horizontal')

        # the xaxis of ax_histx and yaxis of ax_histy are shared with ax,
        # thus there is no need to manually adjust the xlim and ylim of these
        # axis.

        

        plt.show()
    
    if ret:
        return traj, areas
    
    


def stats(direcs):

    distribution = []
    speed = []
    area_growth = []
    for dir in direcs:
        traj, area = motility_traj([dir],plot=False)
        distribution.append(np.concatenate(traj))
        
        for elem in traj:
            veloc = np.divide(np.linalg.norm(elem[1:,:-1]-elem[:-1,:-1] ,axis=1),elem[1:,2]-elem[:-1,2])
            speed.append(np.column_stack((veloc,elem[:-1,2])))
            
        for elem in area:
            veloc = np.divide(elem[1:,0]-elem[:-1,0],elem[1:,1]-elem[:-1,1])
            area_growth.append(np.column_stack((veloc,elem[:-1,1])))

    distribution = np.concatenate(distribution)
    speed = np.concatenate(speed)
    area_growth = np.concatenate(area_growth)
    
    MSDx = np.absolute(distribution[:,1:])
    MSDy = np.absolute(distribution[:,::2])
    MSD = np.column_stack((np.linalg.norm(distribution[:,:-1], axis = 1),distribution[:,-1]))


    plt.figure()
    res = np.array([[np.average(area_growth[area_growth[:,1]==i]),i] for i in np.unique(area_growth[:,1])])
    plt.scatter(res[:,1],res[:,0])
    plt.xlabel('time')
    plt.ylabel('mean area growth')
    A = np.vstack([res[:,1], np.ones(len(res[:,0]))]).T
    m, c = np.linalg.lstsq(A, res[:,0], rcond=None)[0]
    plt.plot(res[:,1], m*res[:,1] + c, 'r', label=f'Fitted line with slope {m:.2E}')
    

    
    
    plt.figure()
    res = np.array([[np.average(speed[speed[:,1]==i]**2),i] for i in np.unique(speed[:,1])])
    plt.plot(res[:,1],res[:,0], label = 'averaged squared instantaneous speed', color = 'green')
    alpha, beta, time = superdiff_parameters(res)
    plt.plot(time, beta*np.power(time, alpha), color = 'purple', linestyle= 'dotted', label =  r"squared speed fit $\beta t^{\alpha} $ with $(\alpha,\beta)$ = "+f" {alpha:.2E} and {beta:.2E}")
    
    
    res = np.array([[np.average(MSD[MSD[:,1]==i]**2),i] for i in np.unique(MSD[:,1])])
    plt.scatter(res[:,1],res[:,0])
    plt.xlabel('time')
    plt.ylabel('mean squared distance')
    alpha, beta, time = superdiff_parameters(res)
    plt.title('MSD of the cells with power approximation (L2 norm)')
    plt.plot(time, beta*np.power(time, alpha), color = 'r', label =  r"MSD fit $\beta t^{\alpha} $ with $(\alpha,\beta)$ = "+f" {alpha:.2E} and {beta:.2E}")
    
    plt.legend()

    
    plt.figure()
    plt.title('MSD of the cells with power approximation (L2 norm)')
    plt.xlabel('time, log scale')
    plt.ylabel('mean squared distance, log scale')
    plt.scatter(np.log(res[1:,1]),np.log(res[1:,0]))
    plt.plot(np.log(time[1:]), np.log(beta*np.power(time[1:], alpha)), color = 'r' )
    
    alpha, beta, time = superdiff_parameters_log(res)
    plt.figure()
    plt.scatter(res[:,1],res[:,0])
    plt.xlabel('time')
    plt.ylabel('mean squared distance')
    plt.title('MSD of the cells with power approximation (log-L2 norm)')
    plt.plot(time, beta*np.power(time, alpha), color = 'r', label =  r"$\beta t^{\alpha} $ with $(\alpha,\beta)$ = "+f" {alpha:.2E} and {beta:.2E}")
    
    plt.legend()

    
    plt.figure()
    plt.title('MSD of the cells with power approximation (log-L2 norm)')
    plt.xlabel('time, log scale')
    plt.ylabel('mean squared distance, log scale')
    plt.scatter(np.log(res[1:,1]),np.log(res[1:,0]))
    plt.plot(np.log(time), np.log(beta*np.power(time, alpha)) , color = 'r')
    
    plt.figure()
    plt.title('MSD along x axis')
    res = np.array([[np.average(MSDx[MSDx[:,1]==i]**2),i] for i in np.unique(MSDx[:,1])])
    plt.scatter(res[:,1],res[:,0])
    plt.xlabel('time')
    plt.ylabel('mean squared distance')
    
    
    plt.figure()
    plt.title('MSD along y axis')
    res = np.array([[np.average(MSDy[MSDy[:,1]==i]**2),i] for i in np.unique(MSDy[:,1])])
    plt.scatter(res[:,1],res[:,0])
    plt.xlabel('time')
    plt.ylabel('mean squared distance')

    
    plt.figure()
    plt.title(r'Time dependent correlation $Corr(X_t,Y_t)$')
    res = []
    for i in np.unique(distribution[:,2]):
        if i==0:
            res.append([0,0])
        else:
            distx = distribution[distribution[:,2]==i][:,0]
            disty = distribution[distribution[:,2]==i][:,1]
            res.append([np.average((distx-np.average(distx))*(disty-np.average(disty)))/(np.std(distx)*np.std(disty)),i])
                       
    res = np.array(res)
    plt.scatter(res[:,1],res[:,0])
    plt.xlabel('time')
    plt.ylabel('correlation')
    
    plt.figure()
    plt.title(r'Time dependent expected value and  standard deviations')
    res = []
    res2=[]
    for i in np.unique(distribution[:,2]):
        distx = distribution[distribution[:,2]==i][:,0]
        disty = distribution[distribution[:,2]==i][:,1]
        res.append([np.average(distx),np.average(disty),i])
        res2.append([np.std(distx),np.std(disty),i])
                       
    res = np.array(res)
    res2 = np.array(res2)
    plt.scatter(res[:,2],res[:,0], color = 'b', label= 'x expected value')
    plt.scatter(res[:,2],res[:,1], color = 'g', label= 'y expected value')
    plt.scatter(res2[:,2],res2[:,0], color = 'r', label= 'x standard deviation')
    plt.scatter(res2[:,2],res2[:,1], color = 'k', label= 'y standard deviation')
    plt.xlabel('time')
    plt.ylabel('expected value')
    plt.legend()
    # plt.show()    
    
        
def compute_power(alpha,data,time):
    beta = np.dot(data, np.power(time,alpha))/np.linalg.norm(np.power(time,alpha))**2
    return np.linalg.norm(data-beta*np.power(time,alpha))
    
    
def superdiff_parameters(MSD):
    data = MSD[:,0]
    time = MSD[:,1] 
    
    res = opt.minimize(compute_power, 1.1, args=(data,time))
    alpha = res.x[0]
    beta = np.dot(data, np.power(time,alpha))/np.linalg.norm(np.power(time,alpha))**2
    return alpha, beta, time

def superdiff_parameters_log(MSD):
    MSD = MSD[1:,:]
    data = np.log(MSD[:,0])
    avg_data = np.average (data)
    time = np.log(MSD[:,1])
    avg_time = np.average (time)
    
    alpha = np.dot(data-avg_data, time-avg_time)/np.linalg.norm(time-avg_time)**2
    beta = np.exp(avg_data-alpha*avg_time)
    
    return alpha, beta, MSD[:,1]
    
    
    
if __name__ == "__main__":
    dir_list=os.listdir('results')
    # dir_list=['July6_plate1_xy02',
    # 'July6_plate1_xy05',
    # 'July6_plate1_xy06',
    # 'July7_plate1_xy01',
    # 'July7_plate1_xy02']
    # motility_traj(dir_list)
    stats(dir_list)
    plt.show()
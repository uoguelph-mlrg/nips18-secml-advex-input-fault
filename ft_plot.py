import numpy as np
import matplotlib
matplotlib.rc('text', usetex=True)
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# indices for axis 1 in data
SNR = 0
IYT = 1
ACC = 2

w_p = 3.
w_s = 2.5
font_primary = 24
font_second = 20
font_legend = 16
SNR_XLIM = 120


def fault_tolerance_unique_obj_ADef(data_list, legend=False, save=False, labels=None, plot_name=None):

    XLIM = 3

    fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))

    pcolor_list = ['navy', 'maroon', 'forestgreen', 'black']
    if legend:
        assert len(data_list) == len(labels)

    for i, data in enumerate(data_list):
        ax1.plot(data[:, SNR], data[:, IYT], label=labels[i], c=pcolor_list[i], linewidth=w_p)

    ax1.set_xlim(0,XLIM)
    ax1.set_ylim(0., 3.3)
    ax1.set_ylabel(r'$I(T; Y)$', fontsize=font_primary, labelpad=8)#, color='navy')
    ax1.set_xlabel(r'Max. Norm', fontsize=font_primary)

    # Major ticks every 10, minor ticks every 5
    #major_ticks_x = np.arange(5, XLIM+1, 5)
    major_ticks_x = np.arange(1, XLIM+1, 1)
    minor_ticks_x = np.arange(0.5, XLIM+1, 0.5)
    ax1.set_xticks(major_ticks_x)
    ax1.set_xticks(minor_ticks_x, minor=True)
    ax1.set_yticks([0., 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    ax1.set_yticks([.25, .75, 1.25, 1.75, 2.25, 2.75, 3.0], minor=True)


    # And a corresponding grid
    ax1.grid(which='both')
    if legend:
        plt.legend(fontsize=font_legend, shadow='true', edgecolor='k')

    '''
    ax2 = ax1.twinx()
    #scolor_list = ['lightsteelblue', 'indianred', 'lightsalmon']
    for i, data in enumerate(data_list):
        ax2.plot(data[:, SNR], data[:, ACC] * 100, linewidth=w_s, c=pcolor_list[i], linestyle='--')
    ax2.set_ylabel(r'Accuracy (\%)', fontsize=font_primary, labelpad=10)
    ax2.tick_params('y', colors='k', direction='in', labelsize=font_second, length=0)
    ax2.set_ylim(0, 100)
    ax2.set_yticks([10, 30, 50, 70, 90])
    '''

    # Or if you want different settings for the grids:
    ax1.grid(which='minor', linestyle='-', linewidth=0.5, c='lightgray')
    ax1.grid(which='major', linestyle='-', c='darkgray')
    ax1.tick_params(which='major', direction='in', labelsize=font_second, length=8)
    ax1.tick_params(which='minor', direction='in', labelsize=font_second, length=5)
    ax1.set_facecolor('white')
    plt.tight_layout()
    plt.show()
    if save:
        fig.savefig(plot_name, bbox_inches='tight', format='eps')


def fault_tolerance_unique_obj(data_list, legend=False, axis_labels='both', save=False, min_snr=0, max_snr=SNR_XLIM, labels=None, modelname=None):
    """
    Figure 1. fault tolerance plot for unique attack objectives for NIPS SECML.

    @param data_list: takes up to four different data series
                     (if more are needed, add more colours to pcolor_list.
    @param legend: boolean, if legend should be used
    @param axis_labels: either 'both', 'mi', or 'acc''
    @save boolean: boolean, if plot should be saved to file as *.eps
    @label: list of series labels, to be used if save=True
    @modelname: descriptive name to be supplied if save=True
    """
    fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))

    pcolor_list = ['lightgray', 'navy', 'maroon', 'forestgreen']
    if legend:
        assert len(data_list) == len(labels)
    for i, data in enumerate(data_list):
        ax1.plot(data[:, SNR], data[:, IYT], label=labels[i], c=pcolor_list[i], linewidth=w_p)
    ax1.set_xlim(max_snr, min_snr + 1)
    ax1.set_xlabel(r'SNR', fontsize=font_primary)
    ax1.set_ylim(0., 3.3)
    if axis_labels == 'both' or axis_labels == 'mi':
        ax1.set_ylabel(r'$I(T; Y)$', fontsize=font_primary, labelpad=8)#, color='navy')
    else:
        ax1.yaxis.set_ticklabels([])
    # Major ticks every 10, minor ticks every 5
    major_ticks_x = np.arange(10, max_snr + 1, 10)[::-1]
    minor_ticks_x = np.arange(5, max_snr + 1, 5)[::-1]
    ax1.set_xticks(major_ticks_x)
    ax1.set_xticks(minor_ticks_x, minor=True)
    ax1.set_yticks([.25, .75, 1.25, 1.75, 2.25, 2.75, 3.0], minor=True)
    ax1.set_yticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

    # And a corresponding grid
    ax1.grid(which='both')
    if legend:
        plt.legend(fontsize=font_legend, edgecolor='k', loc='lower left') # borderaxespad=.75, )

    ax2 = ax1.twinx()
    #scolor_list = ['lightsteelblue', 'indianred', 'lightsalmon']
    for i, data in enumerate(data_list):
        ax2.plot(data[:, SNR], data[:, ACC] * 100, linewidth=w_s, c=pcolor_list[i], linestyle='--')
    ax2.tick_params('y', colors='k', direction='in', labelsize=font_second, length=0)
    ax2.set_ylim(0, 100)
    if axis_labels == 'both' or axis_labels == 'acc':
        ax2.set_ylabel(r'Accuracy (\%)', fontsize=font_primary, labelpad=10)
        ax2.set_yticks([10, 30, 50, 70, 90])
    else:
        ax2.set_yticks([])
    #ax2.axvline(x=5, ymin=0, ymax=1, c='pink') # rose SNR criterion

    # Or if you want different settings for the grids:
    ax1.grid(which='minor', linestyle='-', linewidth=0.5, c='lightgray')
    ax1.grid(which='major', linestyle='-', c='darkgray')
    ax1.tick_params(which='major', direction='in', labelsize=font_second, length=5)
    ax1.tick_params(which='minor', direction='in', labelsize=font_second, length=3)
    ax1.set_facecolor('white')
    plt.tight_layout()
    plt.show()
    if save:
        PLT_NAME = modelname + 'fault.eps'
        fig.savefig(PLT_NAME, bbox_inches='tight', pad_inches=0.1, format='eps')


def fault_tolerance_plot_rot_from_list_30(rot, data_list, legend=False, save=False, labels=None, modelname=None):
    """
    Figure 2. b) fault tolerance plot for rotation for NIPS SECML.

    @param rot: x-axis steps in radians
    @param data_list:
    @param legend: boolean, if legend should be used
    @save boolean: boolean, if plot should be saved to file as *.eps
    @label: list of series labels, to be used if save=True
    @modelname: descriptive name to be supplied if save=True
    """
    fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))

    pcolor_list = ['navy', 'maroon', 'forestgreen', 'lightgray']
    if legend:
        assert len(data_list) == len(labels)
    for i, data in enumerate(data_list):
        ax1.plot(rot * (180 / np.pi), data[:, IYT], label=labels[i],c=pcolor_list[i], linewidth=w_p)

    ax1.set_xlim(0, (np.pi / 6) * (180 / np.pi))
    ax1.set_ylim(0., 3.3)
    ax1.set_ylabel(r'$I(T; Y)$', fontsize=font_primary, labelpad=10)
    ax1.set_xlabel(r'Rotation (Deg.)', fontsize=font_primary)

    # Major ticks every 10, minor ticks every 5
    major_ticks_x = np.arange(0, 31, 10)[::-1]
    minor_ticks_x = np.arange(0, 31, 5)[::-1]
    ax1.set_xticks(major_ticks_x)
    ax1.set_xticks(minor_ticks_x, minor=True)
    ax1.set_yticks([0., 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    ax1.set_yticks([.25, .75, 1.25, 1.75, 2.25, 2.75, 3.0], minor=True)

    # And a corresponding grid
    ax1.grid(which='both')
    if legend:
        plt.legend(fontsize=font_legend, shadow='true', edgecolor='k')
        #plt.legend(fontsize=font_legend, shadow='true', edgecolor='k', bbox_to_anchor=(0.94, 0.55))

    ax2 = ax1.twinx()
    for i, data in enumerate(data_list):
        ax2.plot(rot * (180 / np.pi), data[:, ACC] * 100, linewidth=w_s, c=pcolor_list[i], linestyle='--')
    ax2.set_ylabel(r'Accuracy (\%)', fontsize=font_primary, labelpad=10)
    ax2.tick_params('y', colors='k', direction='in', labelsize=font_second, length=0)
    ax2.set_ylim(0, 100)
    ax2.set_yticks([10, 30, 50, 70, 90])

    # Or if you want different settings for the grids:
    ax1.grid(which='minor', linestyle='-', linewidth=0.5, c='lightgray')
    ax1.grid(which='major', linestyle='-', c='darkgray')
    ax1.tick_params(which='major', direction='in', labelsize=font_second, length=8)
    ax1.tick_params(which='minor', direction='in', labelsize=font_second, length=5)
    ax1.set_facecolor('white')
    plt.tight_layout()
    plt.show()
    if save:
        PLT_NAME = modelname + '_fault_rot30.eps'
        fig.savefig(PLT_NAME, bbox_inches='tight', pad_inches=0.1, format='eps')


def fault_tolerance_plot_rot_from_list(rot, data_list, legend=False, save=False, max_rot=180, labels=None, modelname=None):
    """
    @param rot: x-axis steps in radians
    @param data_list:
    @param legend: boolean, if legend should be used
    @save boolean: boolean, if plot should be saved to file as *.eps
    @label: list of series labels, to be used if save=True
    @modelname: descriptive name to be supplied if save=True
    """
    fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))

    pcolor_list = ['navy', 'maroon', 'forestgreen', 'lightgray']
    if legend:
        assert len(data_list) == len(labels)
    for i, data in enumerate(data_list):
        ax1.plot(rot * (180 / np.pi), data[:, IYT], label=labels[i],c=pcolor_list[i], linewidth=w_p)

    ax1.set_xlim(0, max_rot)
    ax1.set_ylim(0., 3.3)
    ax1.set_ylabel(r'$I(T; Y)$', fontsize=font_primary, labelpad=10)
    ax1.set_xlabel(r'Rotation (Degrees)', fontsize=font_primary)
    # Major ticks every 10, minor ticks every 5
    major_ticks_x = np.arange(0, max_rot + 1, 60)[::-1]
    minor_ticks_x = np.arange(0, max_rot + 1, 20)[::-1]
    ax1.set_xticks(major_ticks_x)
    ax1.set_xticks(minor_ticks_x, minor=True)
    ax1.set_yticks([0., 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    ax1.set_yticks([.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0], minor=True)

    # And a corresponding grid
    ax1.grid(which='both')
    if legend:
        plt.legend(fontsize=font_legend, shadow='true', edgecolor='k', bbox_to_anchor=(0.94, 0.55))
    ax2 = ax1.twinx()
    for i, data in enumerate(data_list):
        ax2.plot(rot * (180 / np.pi), data[:, ACC] * 100, linewidth=w_s, c=pcolor_list[i], linestyle='--')
    ax2.set_ylabel(r'Accuracy (\%)', fontsize=font_primary, labelpad=15)
    ax2.tick_params('y', colors='k', direction='in', labelsize=font_primary, length=0)
    ax2.set_ylim(0, 100)
    ax2.set_yticks([10, 30, 50, 70, 90])

    # Or if you want different settings for the grids:
    ax1.grid(which='minor', linestyle='-', linewidth=0.5, c='lightgray')
    ax1.grid(which='major', linestyle='-', c='darkgray')
    ax1.tick_params(which='major', direction='in', labelsize=font_primary, length=8)
    ax1.tick_params(which='minor', direction='in', labelsize=font_primary, length=5)
    ax1.set_facecolor('white')
    plt.show()
    if save:
        PLT_NAME = modelname + '_fault_rot360.eps'
        fig.savefig(PLT_NAME, bbox_inches='tight', pad_inches=0.1, format='eps')


def fault_tolerance_plot_from_list(data_list, legend=False, save=False, max_snr=SNR_XLIM, labels=None, modelname=None):
    """
    For reproducing Fig. 5 in "A Rate-Distortion Theory of Adversarial Examples"
    @param data_list:
    @param legend: boolean, if legend should be used
    @save boolean: boolean, if plot should be saved to file as *.eps
    @label: list of series labels, to be used if save=True
    @modelname: descriptive name to be supplied if save=True
    """
    fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))

    line_sty = ['-', '--', ':']
    if legend:
        assert len(data_list) == len(labels)
    for i, data in enumerate(data_list):
        ax1.plot(data[:, SNR], data[:, IYT], label=labels[i],
                 c='navy', linewidth=w_p, linestyle=line_sty[i])

    ax1.set_xlim(max_snr, 1)
    ax1.set_ylim(0.5, 3.0)
    ax1.set_ylabel(r'$I(T; Y)$', fontsize=font_primary, color='navy')
    ax1.set_xlabel(r'SNR', fontsize=font_primary)
    # Major ticks every 10, minor ticks every 5
    major_ticks_x = np.arange(10, max_snr + 1, 20)[::-1]
    minor_ticks_x = np.arange(5, max_snr + 1, 10)[::-1]
    ax1.set_xticks(major_ticks_x)
    ax1.set_xticks(minor_ticks_x, minor=True)
    ax1.set_yticks([1.0, 1.5, 2.0, 2.5, 3.0])
    ax1.set_yticks([.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0], minor=True)

    # And a corresponding grid
    ax1.grid(which='both')
    if legend:
        plt.legend(fontsize=font_legend, shadow='true', edgecolor='k')
    ax2 = ax1.twinx()
    for i, data in enumerate(data_list):
        ax2.plot(data[:, SNR], data[:, ACC] * 100, linewidth=w_s,
                 c='lightsteelblue', linestyle=line_sty[i])
    ax2.set_ylabel('Test Accuracy (\%)', color='lightsteelblue', fontsize=font_second, labelpad=8)
    ax2.tick_params('y', colors='k', direction='in', labelsize=font_second, length=5)
    ax2.set_ylim(0, 100)
    ax2.set_yticks([10, 30, 50, 70, 90])

    # Or if you want different settings for the grids:
    ax1.grid(which='minor', linestyle='-', linewidth=0.5, c='lightgray')
    ax1.grid(which='major', linestyle='-', c='darkgray')
    ax1.tick_params(which='major', direction='in', labelsize=font_primary, length=8)
    ax1.tick_params(which='minor', direction='in', labelsize=font_primary, length=5)
    ax1.set_facecolor('white')
    plt.show()
    if save:
        PLT_NAME = modelname + '_fault.eps'
        fig.savefig(PLT_NAME, bbox_inches='tight', pad_inches=0.1, format='eps')


def fault_tolerance_plot(data, legend=False, save=False, max_snr=SNR_XLIM, label=None, modelname=None):


    fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))

    ax1.plot(data[:, SNR], data[:, IYT], label=label, c='navy', linewidth=w_p)
    ax1.set_xlim(max_snr, 1)
    ax1.set_ylim(0., 3.33)

    ax1.set_ylabel(r'$I(T; Y)$', fontsize=font_primary, color='navy')
    ax1.set_xlabel(r'SNR', fontsize=font_primary)


    # Major ticks every 10, minor ticks every 5
    major_ticks_x = np.arange(10, max_snr + 1, 10)[::-1]
    minor_ticks_x = np.arange(5, max_snr  + 1, 5)[::-1]

    ax1.set_xticks(major_ticks_x)
    ax1.set_xticks(minor_ticks_x, minor=True)
    ax1.set_yticks([0., 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.33])
    ax1.set_yticks([.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.33], minor=True)

    # And a corresponding grid
    ax1.grid(which='both')

    #plt.legend(fontsize=font_legend, facecolor='antiquewhite', shadow='true', edgecolor='k')
    if legend:
        plt.legend(fontsize=font_legend, shadow='true', edgecolor='k')

    ax2 = ax1.twinx()

    ax2.plot(data[:, SNR], data[:, ACC] * 100, linewidth=w_s, c='lightsteelblue')
    #ax2.plot(ft_inf[:, SNR], ft_inf[:, ACC]*100, linewidth=w_s, c='lightsteelblue', linestyle='--')
    #ax2.plot(ft_noise[:, SNR], ft_noise[:, ACC]*100, linewidth=w_s, c='lightsteelblue', linestyle=':')

    ax2.set_ylabel('Test Accuracy (\%)', color='lightsteelblue', fontsize=font_second, labelpad=8)
    ax2.tick_params('y', colors='k', direction='in', labelsize=font_second, length=5)
    ax2.set_ylim(0, 100)
    ax2.set_yticks([10, 30, 50, 70, 90])

    # Or if you want different settings for the grids:
    ax1.grid(which='minor', linestyle='-', linewidth=0.5, c='lightgray')
    ax1.grid(which='major', linestyle='-', c='darkgray')
    ax1.tick_params(which='major', direction='in', labelsize=font_primary, length=8)
    ax1.tick_params(which='minor', direction='in', labelsize=font_primary, length=5)

    #ax1.set_facecolor('oldlace')
    ax1.set_facecolor('white')

    plt.show()

    if save:
        PLT_NAME = modelname + '_fault.eps'
        fig.savefig(PLT_NAME, bbox_inches='tight', pad_inches=0.1, format='eps')


def fault_tolerance_rotation_plot(rot, data, legend=False, save=False, label=None, modelname=None):

    fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))

    ax1.plot(rot * (180 / np.pi), data[:, IYT], label=label, c='navy', linewidth=w_p)

    #ax1.set_xlim(65, 1)
    ax1.set_xlim(0, np.pi/6 * (180 / np.pi))
    ax1.set_ylim(0.5, 3.0)

    ax1.set_ylabel(r'$I(T; Y)$', fontsize=font_primary, color='navy')
    ax1.set_xlabel(r'Rotation (Deg.)', fontsize=font_primary)

    # Major ticks every 10, minor ticks every 5
    major_ticks_x = np.arange(0, 31, 10)[::-1]
    minor_ticks_x = np.arange(0, 31, 5)[::-1]

    ax1.set_xticks(major_ticks_x)
    ax1.set_xticks(minor_ticks_x, minor=True)
    ax1.set_yticks([1.0, 1.5, 2.0, 2.5, 3.0])
    ax1.set_yticks([.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0], minor=True)

    # And a corresponding grid
    ax1.grid(which='both')

    #plt.legend(fontsize=font_legend, facecolor='antiquewhite', shadow='true', edgecolor='k')
    if legend:
        plt.legend(fontsize=font_legend, shadow='true', edgecolor='k')

    ax2 = ax1.twinx()

    ax2.plot(rot * (180 / np.pi), data[:, ACC] * 100, linewidth=w_s, c='lightsteelblue')

    ax2.set_ylabel('Test Accuracy (\%)', color='lightsteelblue', fontsize=font_second, labelpad=8)
    ax2.tick_params('y', colors='k', direction='in', labelsize=font_second, length=5)
    ax2.set_ylim(0, 100)
    ax2.set_yticks([10, 30, 50, 70, 90])

    # Or if you want different settings for the grids:
    ax1.grid(which='minor', linestyle='-', linewidth=0.5, c='lightgray')
    ax1.grid(which='major', linestyle='-', c='darkgray')
    ax1.tick_params(which='major', direction='in', labelsize=font_primary, length=8)
    ax1.tick_params(which='minor', direction='in', labelsize=font_primary, length=5)

    #ax1.set_facecolor('oldlace')
    ax1.set_facecolor('white')

    plt.show()

    if save:
        PLT_NAME = modelname + '_fault.eps'
        fig.savefig(PLT_NAME, bbox_inches='tight', pad_inches=0.1, format='eps')


def grid_visual(data):
    """
    This function displays a grid of images to show full misclassification
    :param data: grid data of the form;
        [nb_classes : nb_classes : img_rows : img_cols : nb_channels]
    :return: if necessary, the matplot figure to reuse
    """
    # build a rectangle in axes coords
    left, width = .25, .5
    bottom, height = .25, .5
    right = left + width
    top = bottom + height
    fontsize = 18
    import matplotlib.pyplot as plt

    # Ensure interactive mode is disabled and initialize our graph
    plt.ioff()
    figure = plt.figure(figsize=(8, 8))
    figure.canvas.set_window_title('Cleverhans: Grid Visualization')

    # Add the images to the plot
    num_cols = data.shape[0]
    num_rows = data.shape[1]
    #num_channels = data.shape[4]
    current_row = 0
    for y in range(num_rows):
        for x in range(num_cols):
            figure.add_subplot(num_rows, num_cols, (x + 1) + (y * num_cols))
            #plt.axis('off')
            ax = plt.gca()
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.tick_params(axis='both', length=0)
            plt.imshow(data[x, y, :, :], cmap='gray')
            if y == 0:
                ax.set_title('%d' % x, fontsize=fontsize)
            if x == 0:
                ax.set_ylabel('%d' % y, rotation='horizontal', va='center',
                              fontsize=fontsize, labelpad=10)
                            
    #plt.tight_layout()
    #figure.text(0.5, 1.01, r'\textbf{Target Class}', ha='center', fontsize=24)
    #figure.text(-0.02, 0.5*(bottom+top), r'\textbf{Source Class}', 
    #         ha='center', va='center', fontsize=24, rotation='vertical')
    # Draw the plot and return
    plt.show()
    return figure        

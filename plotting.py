from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


def surface_plot_function(function, pos_range=512, neg_range=-512):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Make data.
    x = np.arange(neg_range, pos_range, 1)
    y = np.arange(neg_range, pos_range, 1)
    X, Y = np.meshgrid(x, y)
    Z = function([X,Y])
    # Plot the surface.
    # surf = ax.plot_trisurf(x, y, Z, cmap="Accent",
    #                        linewidth=0.05)
    # ax = plt.axes(projection='3d')

    xLabel = ax.set_xlabel('\n x_1', linespacing=3.2)
    yLabel = ax.set_ylabel('\n x_2', linespacing=3.1)
    zLabel = ax.set_zlabel('\n f(x_1, x_2)', linespacing=3.4)
    surf = ax.plot_surface(X, Y, Z, cmap=plt.cm.jet, linewidth=0)
    # Customize the z axis.
    # plt.cm.jet
    # ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

def contour_plot_function(function, pos_range=512, neg_range=-512, route = None, title="", min_x=None, populations=None):
    fig = plt.figure()
    x = np.arange(neg_range, pos_range, 0.5)
    y = np.arange(neg_range, pos_range, 0.5)
    X, Y = np.meshgrid(x, y)
    Z = function([X,Y])
    breaks = np.linspace(-1100, 1100, 30)
    contour = plt.contour(X, Y, Z,
                     breaks,
                     cmap='jet',
                     )
    fig.colorbar(contour, ticks=breaks)

    plt.xlabel('x_1')
    plt.ylabel('x_2')
    plt.grid()
    if title == "":
        plt.title('Egg Holder 2D')
    else:
        plt.title(title)
    # Make an option to plot the optimisation route:
    # if route:
    for (x,y) in route:
        plt.plot(x,y, 'm1', markersize=12)

    for generation in populations:
        for (x,y) in generation:
            plt.plot(x,y, 'm4', markersize=12)
    try:
        plt.plot(min_x[0], min_x[1], 'co', markersize=20)
    except Exception as e:
        raise
    plt.show()


def plot_generations(function, pos_range=512, neg_range=-512, title="", population=None):
    fig = plt.figure()
    x = np.arange(neg_range, pos_range, 0.5)
    y = np.arange(neg_range, pos_range, 0.5)
    X, Y = np.meshgrid(x, y)
    Z = function([X,Y])
    breaks = np.linspace(-1100, 1100, 30)
    contour = plt.contour(X, Y, Z,
                     breaks,
                     cmap='jet',
                     )
    fig.colorbar(contour, ticks=breaks)
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    plt.grid()
    if title == "":
        plt.title('Egg Holder 2D')
    else:
        plt.title(title)
    for (x,y) in population:
        plt.plot(x,y, 'm4', markersize=12)
    plt.show()




if __name__ == '__main__':
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Make data.
    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X**2 + Y**2)
    Z = np.sin(R)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    # Customize the z axis.
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

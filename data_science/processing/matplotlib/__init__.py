import matplotlib
import matplotlib.pyplot as plt
import numpy as np

'''
learning material : https://www.runoob.com/matplotlib/matplotlib-tutorial.html
api : https://matplotlib.org/stable/api/index.html
'''

# matplotlib is composed of three modules : pyplot, pylab, oop
# where :
# - pyplot is for script configuration
# - pylab is for interactive configuration
# - oop is in object-oriented paradigm
# their roles are always the same : plotting !
# hence we're only going to study one of them
# since the other two are similar and less important

# don't forget plt.show() !!!

'plot'
xs = np.array([1, 8])
ys = np.array([3, 10])
plt.plot(xs, ys, 'r-')  # the same as Matlab
plt.plot(ys)  # default xs : 0, 1, 2, ...
zs = np.array([11, 23])
plt.plot(xs, ys, xs, zs)  # more figure

'marker'
# markersize，abbr. ms：define size of marker point.
# markerfacecolor，abbr. mfc：define inner color of marker point.
# markeredgecolor，abbr. mec：define edge color of marker point.
# to define style: ms, mfc, mec, marker, ls, color, linewidth, ...
plt.plot(ys, marker='o', ms=20, ls='-')

'label and title'
plt.title("beautiful chart")
plt.xlabel("xswl")
plt.ylabel("yyds")

'font'
# print(matplotlib.font_manager.fontManager.ttflist)
# font = matplotlib.font_manager.FontProperties(fname='path-to-.otf')
# plt.title('new font', fontproperties=font)
# plt.rcParams['font.family']=['STFangsong']
# font_dict = {'color': 'blue', 'size': 20}
# plt.title('new font with style', fontproperties=font, fontdict=font_dict)

'grid'
plt.grid(b=True, which='major', axis='both', linestyle='--')

'multi-plot'
plt.subplot(1, 2, 1)  # one row, two columns, the first as current
plt.subplot(1, 2, 2)
plt.suptitle('super title')

x = np.linspace(0, 2 * np.pi, 400)
y = np.sin(x ** 2)

fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_title('one plot')

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(x, y)
ax1.set_title('two plots')
ax2.scatter(x, y)

fig, axs = plt.subplots(2, 2, subplot_kw=dict(projection='polar'))
axs[0, 0].plot(x, y)
axs[1, 1].scatter(x, y)

plt.subplots(2, 2, sharex='col')
plt.subplots(2, 2, sharey='row')
plt.subplots(2, 2, sharey='all', sharex='all')
plt.subplots(2, 2, sharey=True, sharex=True)  # the same as former one
fig, ax = plt.subplots(num=10, clear=True)  # create ten plots, clear existing plots.

'scatter'
x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y = np.array([1, 4, 9, 16, 7, 11, 23, 18])
colors = np.array(["red", "green", "black", "orange", "purple", "beige", "cyan", "magenta"])
sizes = np.array([20, 50, 100, 200, 500, 1000, 60, 90])
plt.scatter(x, y, s=sizes, c=colors)
colors = np.array([0, 10, 20, 30, 40, 45, 50, 55, 60, 70, 80, 90, 100])
plt.scatter(x, y, c=colors, cmap='viridis')
# cmap='viridis': use viridis color bar: which represents each color by a number
plt.colorbar()  # show colorbar


'bar'
x = np.array(["Runoob-1", "Runoob-2", "Runoob-3", "C-RUNOOB"])
y = np.array([12, 22, 6, 18])
plt.bar(x,y)
plt.barh(x,y)

'pie'
y = np.array([35, 25, 25, 15])
plt.pie(y,
        labels=['A','B','C','D'], # set pie label
        colors=["#d5695d", "#5d8ca8", "#65a479", "#a564c9"], # set pie color
        explode=(0, 0.2, 0, 0), # section B extrudes, more distant as larger
        autopct='%.2f%%', # format output percentage
       )

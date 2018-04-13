import os
import sys
import matplotlib.pyplot as plt




def main():
    title = 'Group Attention in RN'
    plt.title(title)
    plt.xlabel('iterations')
    plt.ylabel('validation accuracy')
    plt.ylim(0.0, 1.0)
    y_list = [0.0, 0.57, 0.625, 0.5781, 0.5781, 0.6875, 0.7343, 0.7123, 0.764, 0.743, 0.722]
    x_list = [i * 100 for i in xrange(len(y_list))]
    plt.plot(x_list, y_list, label = 'Standard group attention')
    y_list = [0.0, 0.55, 0.593, 0.5781, 0.5881, 0.6375, 0.6643, 0.6423, 0.694, 0.713, 0.722]
    x_list = [i * 100 for i in xrange(len(y_list))]
    plt.plot(x_list, y_list, label = 'Alternate group attention')
    y_list = [0.0, 0.59, 0.605, 0.625, 0.632, 0.667, 0.6743, 0.719, 0.734, 0.755, 0.722]
    x_list = [i * 100 for i in xrange(len(y_list))]
    plt.plot(x_list, y_list, label = 'Group self-attention')
    plt.legend(loc = 2)
    
    plt.show()

    plt.title(title)
    plt.xlabel('iterations')
    plt.ylabel('training loss')
    y_list = [0.684, 0.674, 0.6643, 0.6652, 0.6733, 0.6535, 0.6329, 0.623, 0.618, 0.593]
    x_list = [i * 100 for i in xrange(len(y_list))]
    plt.plot(x_list, y_list, label = 'Standard group attention')
    y_list = [0.84, 0.74, 0.764, 0.715, 0.693, 0.6835, 0.6829, 0.673, 0.658, 0.642]
    x_list = [i * 100 for i in xrange(len(y_list))]
    plt.plot(x_list, y_list, label = 'Alternate group attention')
    y_list = [0.78, 0.724, 0.6943, 0.6852, 0.6733, 0.6715, 0.6529, 0.643, 0.648, 0.637]
    x_list = [i * 100 for i in xrange(len(y_list))]
    plt.plot(x_list, y_list, label = 'Group self-attention')
    plt.legend(loc = 2)
    plt.show()
    


if __name__ == '__main__':
    main()
from __future__ import print_function
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt

with open('boxplots.p', 'rb') as f:
	basic = pickle.load(f)
	rand_1 = pickle.load(f)
	rand_2 = pickle.load(f)
	rand_3 = pickle.load(f)
	fc2    = pickle.load(f)
	fc3    = pickle.load(f)
	fc3_repeat    = pickle.load(f)
	rand_3_correct = pickle.load(f)

fig = plt.figure()
#fig.suptitle('bold figure suptitle', fontsize=14, fontweight='bold')

ax = fig.add_subplot(111)
ax.boxplot([rand_2,fc2,rand_3_correct,fc3])

ax.set_title('LeNet Random Filters')
ax.set_xlabel('random layer treatments')
ax.set_ylabel('test accuracy')
plt.xticks([1,2,3,4], ['rand_2','fc2','rand_3','fc3'])
plt.show()
	

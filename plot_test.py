from __future__ import print_function
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt

with open('test.p', 'rb') as f:
	test_accuracy = pickle.load(f)

fig = plt.figure()
#fig.suptitle('bold figure suptitle', fontsize=14, fontweight='bold')

ax = fig.add_subplot(111)
ax.boxplot(test_accuracy)

ax.set_title('Test accuracy')
ax.set_xlabel('NN Treatments')
ax.set_ylabel('test_accuracy')
plt.xticks([1], ['basic'])
plt.show()
	

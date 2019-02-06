def a():
     f = open('./kss/transcript.txt', 'r')
     ff = open('./kss/metadata.txt', 'w')

     for i in f.readlines():
          a = i.split('|')
          b = a[0].split('/')[1].replace('.wav', '') + '|' + a[1] + '|' + a[2] + '\n'
          ff.write(b)

import numpy as np
print(np.log10(1))
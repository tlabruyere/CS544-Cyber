import os,glob,array
import numpy as np
import scipy.misc as misc

i = 0
os.chdir('train/')
for file in glob.glob('*.bytes'):
	f = open(file,'rb')
	ln = os.path.getsize(file)
	width = int(ln**0.5)
	rem = ln%width
	a = array.array("B")
	a.fromfile(f,ln-rem)
	f.close()
	g = np.reshape(a,(len(a)/width,width))
	g = np.uint8(g)
	os.chdir('..')
	os.chdir('trainimg/')
	misc.imsave(file + '.png', g)
	os.chdir('..')
	os.chdir('train/')
	i+=1
	print('train image: '+ str(i) + ' complete!')
	
i = 0
os.chdir('..')
os.chdir('test/')
for file in glob.glob('*.bytes'):
        f = open(file,'rb')
        ln = os.path.getsize(file)
        width = int(ln**0.5)
        rem = ln%width
        a = array.array("B")
        a.fromfile(f,ln-rem)
        f.close()
        g = np.reshape(a,(len(a)/width,width))
        g = np.uint8(g)
        os.chdir('..')
        os.chdir('testimg/')
        misc.imsave(file + '.png', g)   
        os.chdir('..')
        os.chdir('test/')   	
        i+=1
        print('test image: '+ str(i) + ' complete!')

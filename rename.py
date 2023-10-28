import os

path = 'valid'

j = 1
for i in os.listdir(path):
    if not os.path.exists(path+'/'+i):
        os.rename(path+'/'+i, path+'/'+str(j)+'.png')
    j+=1
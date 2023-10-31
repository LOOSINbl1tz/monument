import os
import shutil

input_dir = os.path.join('data_temp')
dest_dir = os.path.join('data')

for i in os.listdir(input_dir):
    folder_dir = os.path.join(input_dir,i)
    
    # for n,j in enumerate(os.listdir(folder_dir)):
    #     input_rename = os.path.join(folder_dir,j)
    #     output_name = os.path.join(folder_dir,str(n+1)+'.png')
    #     os.rename(input_rename,output_name)

    mul = {'test':1,'train':39,'val':10}

    for q in mul:
        out_folder_dir = os.path.join(dest_dir,q,i)
        if not os.path.exists(out_folder_dir):
            os.mkdir(out_folder_dir)

        for n,j in enumerate(os.listdir(folder_dir)):
            if n+1> mul[q]:
                break
            input_file = os.path.join(folder_dir,j)
            out_file = os.path.join(dest_dir,q,i,j)
            print(q,input_file,'   ',out_file)
            shutil.move(input_file,out_file)
    
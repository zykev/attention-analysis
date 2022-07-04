

import shutil, os

# whole_dir = 'D:/Project/child_eyetrace/Data/image_group1'
#
# for name in os.listdir(whole_dir):
#     old_dir = os.path.join(whole_dir, name)
#
#     new_dir = os.path.join('D:/Project/child_eyetrace/Data/img_group1', name.split('_')[0])
#
#
#     if not os.path.exists(new_dir):
#         os.makedirs(new_dir)
#     old_dir_list = list(os.listdir(old_dir))
#     old_dir_list.sort(key=lambda x: int(x.split('.')[0]))
#
#     current_length = len(os.listdir(new_dir))
#     for count, i in enumerate(old_dir_list):
#         old_dir_path = os.path.join(old_dir, i)
#         new_dir_path = os.path.join(new_dir, str(current_length + count + 1) + '.jpg')
#         shutil.copy(old_dir_path, new_dir_path)


# single dir
old_dir = 'E:/Project/child_eyetrace/Data/image3_group0/czh_3'

new_dir = 'E:/Project/child_eyetrace/Data/img3_group0/czh'


if not os.path.exists(new_dir):
    os.makedirs(new_dir)
old_dir_list = list(os.listdir(old_dir))
old_dir_list.sort(key=lambda x: int(x.split('.')[0]))

current_length = len(os.listdir(new_dir))
for count, i in enumerate(old_dir_list):
    old_dir_path = os.path.join(old_dir, i)
    new_dir_path = os.path.join(new_dir, str(current_length + count + 1) + '.jpg')
    shutil.copy(old_dir_path, new_dir_path)
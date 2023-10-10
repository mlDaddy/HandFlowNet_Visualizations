import numpy as np

def read_obj(file_path):
    out_list = []

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v'):
                # Split the line and extract float values
                values = line.strip().split(' ')
                out_list.append([float(val) for val in values[1:]])
    return out_list

def read_obj_folder(folder_path, num_frames):
    out_list = []
    for i in range(num_frames):
        filename = folder_path + '/' + f'{i:06.0f}_gt.obj'
        out_list.append(read_obj(filename))
    return np.array(out_list)

def read_npz_floder(folder_path, num_frames):
    joint3d_list = []
    seg_list = []
    for i in range(num_frames):
        filename = folder_path + '/' + f'labels_{i:06.0f}.npz'
        f = np.load(filename)
        joint3d_list.append(f['joint_3d.npy'][0])
        arr = f['seg']
        arr[arr<255] = 0
        seg_list.append(arr)
    return np.array(joint3d_list), np.array(seg_list)

def read_MV_npz_floder(folder_path, num_frames):
    out_list = []
    for i in range(num_frames):
        filename = folder_path + '/' + f'{i:02.0f}.npz'
        f = np.load(filename)
        out_list.append(f)
    return out_list

def read_SV_npz_floder(folder_path, num_frames):
    out_list = []
    for i in range(num_frames):
        filename = folder_path + '/' + f'pcd_{i:06.0f}.npz'
        f = np.load(filename)
        out_list.append(f)
    return out_list


# Example usage
if __name__ == "__main__":
    print(read_obj_folder('Annotations', 72).shape)
    print(read_npz_floder('joints', 72).shape)

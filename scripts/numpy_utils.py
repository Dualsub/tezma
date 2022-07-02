import numpy as np

def load_array(filename : str) -> np.ndarray:
    """
    It reads the first 12 bytes of the file to get the size, type code, and order of the array, then
    reads the next 4 * order bytes to get the shape of the array, then reads the rest of the file to get
    the data
    
    :param filename: The name of the file to load the array from
    :type filename: str
    :return: A numpy array
    """
    data : np.ndarray = None
    with open(filename, 'rb') as f:
        size = int.from_bytes(f.read(4), byteorder='little')
        type_code = int.from_bytes(f.read(4), byteorder='little')
        order = int.from_bytes(f.read(4), byteorder='little')

        print(f'Loading array of size {size} with type code {type_code} and order {order}')

        shape = []
        for i in range(order):
            shape.append(int.from_bytes(f.read(4), byteorder='little'))

        data = np.frombuffer(f.read(size * 4), dtype=np.float32)
        data = data.reshape(shape) 

    return data

def save_array(data : np.ndarray, filename : str) -> None:
    """
    It writes the size of the array, the type of the array, the number of dimensions, the size of each
    dimension, and then the data itself
    
    :param data: the numpy array to be saved
    :type data: np.ndarray
    :param filename: the name of the file to save the data to
    :type filename: str
    """

    with open(filename, 'wb') as f:
        f.write(int(data.size).to_bytes(4, byteorder='little'))
        f.write(int(9).to_bytes(4, byteorder='little'))
        f.write(int(len(data.shape)).to_bytes(4, byteorder='little'))
        for i in range(len(data.shape)):
            f.write(int(data.shape[i]).to_bytes(4, byteorder='little'))

        f.write(data.tobytes())

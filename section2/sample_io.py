import struct
import sys


with open(sys.argv[1], 'rb') as f:
    # ヘッダーの読み込み
    while True:
        line = f.readline()
        print(line)
        if b'end_header' in line:
            break
        if b'vertex ' in line:
            num_vertices = int(line.split(b' ')[-1])
        if b'face' in line:
            num_faces = int(line.split(b' ')[-1])

    print(num_vertices)
    print(num_faces)

    for _i in range(num_vertices):
        for _j in range(3):
            print(struct.unpack('f', f.read(4))[0], end=' ')
        print()

    for _i in range(num_faces):
        n = struct.unpack('B', f.read(1))[0]
        for _j in range(n):
            print(struct.unpack('i', f.read(4))[0], end=' ')
        print()

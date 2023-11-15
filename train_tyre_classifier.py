from src.data.io import DataIO


def main():
    dataset_path = r'C:\data\typro\archive\Digital images of defective and good condition tyres'
    data = DataIO.read_data_points(dataset_path)
    print('[INFO] found {0} data points to train on'.format(len(data)))


if __name__ == '__main__':
    main()

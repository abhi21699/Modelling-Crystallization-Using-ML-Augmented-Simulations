import matplotlib.pyplot as plt
cord=[[ -4.0915,  12.3158],
        [ -4.0744,  12.3401],
        [ -4.0599,  12.3253],
        [ -4.0641,  12.3177],
        [ -4.0892,  12.3113],
        [ -4.0272,  12.2955],
        [ -4.0133,  12.2759],
        [ -4.0214,  12.2932],
        [ -4.0471,  12.3504],
        [ -4.0670,  12.3833],
        [ -4.1061,  12.3592],
        [ -4.0294,  12.3530],
        [ -4.0339,  12.3039],
        [ -4.0697,  12.3143],
        [ -4.0472,  12.3105],
        [ -4.0992,  12.2569],
        [ -4.0840,  12.2785],
        [ -4.0224,  12.2999],
        [ -4.0643,  12.3213],
        [ -4.0907,  12.3057],
        [-13.0770,   9.0286],
        [-10.8025,  10.5703],
        [ -8.8682,  12.1980],
        [-12.4408,   9.6862],
        [-11.3956,  10.5880],
        [-12.4649,   9.2760],
        [ -9.7017,  11.6814],
        [-10.4370,  10.9249],
        [-12.3800,   9.5549],
        [-12.4949,   9.0145],
        [-12.3733,   9.6623],
        [-10.2493,  11.0179],
        [-11.5140,   9.9375],
        [-11.3224,  10.3544],
        [-10.3870,  11.1555],
        [-11.7645,   9.9209],
        [-12.7671,   9.3902],
        [-12.1264,   9.9069],
        [-11.7994,  10.0524],
        [-11.7946,  10.0701]]
def plot_coordinates(coordinates):
    half_len = len(coordinates) // 2

    x_values_1st_half = [coord[0] for coord in coordinates[:half_len]]
    y_values_1st_half = [coord[1] for coord in coordinates[:half_len]]

    x_values_2nd_half = [coord[0] for coord in coordinates[half_len:]]
    y_values_2nd_half = [coord[1] for coord in coordinates[half_len:]]

    plt.scatter(x_values_1st_half, y_values_1st_half, color='blue', label='First Half')
    plt.scatter(x_values_2nd_half, y_values_2nd_half, color='red', label='Second Half')

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Scatter Plot of Coordinates')
    plt.grid(True)
    plt.legend()
    plt.show()

# Example usage:
# coordinates_list = [[1, 2], [3, 5], [7, 8], [4, 3], [9, 6], [2, 9], [5, 1], [8, 4], [3, 7], [6, 2]]
plot_coordinates(cord)

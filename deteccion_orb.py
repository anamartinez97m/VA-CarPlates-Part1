import cv2
import math
import numpy as np
import os


def get_distance_between_points(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# First point is the KeyPoint and the second is the center
def get_angle_between_keypoint_and_center(x1, y1, x2, y2):
    cat_op = abs(y1 - y2)
    hipoth = get_distance_between_points(x1, y1, x2, y2)
    if hipoth != 0:
        return math.degrees(np.arcsin(cat_op/hipoth))
    return math.degrees(np.arcsin(cat_op/0.001))


def polar_coord_to_cart(r, ang):
    ang = math.radians(ang)
    return round(r * math.cos(ang)), round(r * math.sin(ang))


def add_vote_to_acc_matrix(x, y, size_d0, size_d1, mod, ang_mod, ang_d0, ang_d1, matrix_accumulation_evidence, height_img, width_img):
    mod_scaled = size_d1/size_d0*mod
    angles = (ang_mod + ang_d0 - ang_d1)
    cart_x, cart_y = polar_coord_to_cart(mod_scaled, angles)
    vote_x = int(x + cart_x)
    vote_y = int(y + cart_y)

    if 0 <= vote_x < width_img and 0 <= vote_y < height_img:
        matrix_accumulation_evidence[vote_x][vote_y] = matrix_accumulation_evidence[vote_x][vote_y] + 1


def train():
    (train_images_center_x, train_images_center_y) = (225, 110)

    # nfeatures es el numero de keypoints a detectar
    # nleves es el nivel de la piramide
    # scaleFactor es el factor de escala
    detector = cv2.ORB_create(nfeatures = 100, nlevels = 4, scaleFactor = 1.3)

    train_mods = []
    train_kp_angles = []
    train_descriptors = []
    train_sizes = []
    train_orientations = []

    directory = './train'
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            img_route = os.path.join(directory, filename)
            img = cv2.imread(img_route, 0)
            kp, des = detector.detectAndCompute(img, None)

            for row_number in range(len(kp)):
                (kp_x, kp_y) = kp[row_number].pt
                train_mods.append(get_distance_between_points(kp_x, kp_y, train_images_center_x, train_images_center_y))
                train_kp_angles.append(
                    get_angle_between_keypoint_and_center(kp_x, kp_y, train_images_center_x, train_images_center_y))
                train_descriptors.append(des[row_number])
                train_sizes.append(kp[row_number].size)
                train_orientations.append(kp[row_number].angle)

    train_mods = np.array(train_mods, dtype=np.uint8)
    train_kp_angles = np.array(train_kp_angles)
    train_descriptors = np.array(train_descriptors, dtype=np.uint8)
    train_sizes = np.array(train_sizes)
    train_orientations = np.array(train_orientations)

    return detector, train_mods, train_kp_angles, train_descriptors, train_sizes, train_orientations


def search_object(img_test_original, detector, train_mods, train_kp_angles, train_descriptors, train_sizes, train_orientations):
    # Búsqueda del objeto de test
    img_test = cv2.cvtColor(img_test_original, cv2.COLOR_BGR2GRAY)
    kp_test, des_test = detector.detectAndCompute(img_test, None)
    height_test, width_test = img_test.shape
    center_test_x, center_test_y = (width_test / 2, height_test / 2)

    test_coordinates = []
    test_mods = []
    test_kp_angles = []
    test_descriptors = []
    test_sizes = []
    test_orientations = []

    for row_n in range(len(kp_test)):
        (kp_x_test, kp_y_test) = kp_test[row_n].pt
        test_coordinates.append((kp_x_test, kp_y_test))
        test_mods.append(get_distance_between_points(kp_x_test, kp_y_test, center_test_x, center_test_y))
        test_kp_angles.append(get_angle_between_keypoint_and_center(kp_x_test, kp_y_test, center_test_x, center_test_y))
        test_descriptors.append(des_test[row_n])
        test_sizes.append(kp_test[row_n].size)
        test_orientations.append(kp_test[row_n].angle)

    test_descriptors = np.array(test_descriptors)
    test_sizes = np.array(test_sizes)
    test_orientations = np.array(test_orientations)

    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=3, multi_probe_lever=1)
    search_params = dict(checks=-1)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    for d in train_descriptors:
        flann.add([d])

    flann.train()

    matrix_accumulation_evidence = np.zeros((width_test, height_test), dtype=np.uint8)
    results = flann.knnMatch(test_descriptors, train_descriptors, k=5)
    for r in results:
        for m in r:
            x, y = test_coordinates[m.queryIdx]
            add_vote_to_acc_matrix(x, y, train_sizes[m.trainIdx], test_sizes[m.queryIdx], train_mods[m.trainIdx],
                                   train_kp_angles[m.trainIdx], train_orientations[m.trainIdx],
                                   test_orientations[m.queryIdx],
                                   matrix_accumulation_evidence, height_test, width_test)

    max_number = 0
    best_position = (0, 0)

    for row_number in range(len(matrix_accumulation_evidence)):
        for col_number in range(len(matrix_accumulation_evidence[row_number])):
            if matrix_accumulation_evidence[row_number][col_number] > max_number:
                max_number = matrix_accumulation_evidence[row_number][col_number]
                best_position = (row_number, col_number)

    cv2.circle(img_test_original, best_position, 20, (0, 255, 0), -1)
    cv2.imshow("Car", img_test_original)


def main():
    detector, train_mods, train_kp_angles, train_descriptors, train_sizes, train_orientations = train()
    directory = './test'
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            img_route = os.path.join(directory, filename)
            img_test = cv2.imread(img_route)
            search_object(img_test, detector, train_mods, train_kp_angles, train_descriptors, train_sizes, train_orientations)
            cv2.waitKey()


if __name__ == "__main__":
    main()

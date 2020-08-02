import cv2
from deteccion_haar import detect_and_display
from deteccion_orb import train, search_object


# Deteccion de video con el algoritmo ORB
def deteccion_algoritmo_orb(frame, detector, train_mods, train_kp_angles, train_descriptors, train_sizes, train_orientations):
    search_object(frame, detector, train_mods, train_kp_angles, train_descriptors, train_sizes, train_orientations)


def deteccion_algoritmo_haar(frame):
    detect_and_display(frame)


def main():
    detector, train_mods, train_kp_angles, train_descriptors, train_sizes, train_orientations = train()

    # Deteccion de video con el algoritmo cv2.CascadeClassifier
    video1 = cv2.VideoCapture("video1.wmv")
    video2 = cv2.VideoCapture("video2.wmv")

    if not video1.isOpened or not video2.isOpened:
        exit(0)

    while True:
        ret1, frame1 = video1.read()

        if frame1 is None:
            break

        deteccion_algoritmo_haar(frame1)

        if cv2.waitKey(10) == 27:
            break

    while True:
        ret2, frame2 = video2.read()

        if frame2 is None:
            break

        deteccion_algoritmo_haar(frame2)

        if cv2.waitKey(10) == 27:
            break

    while True:
        ret3, frame3 = video1.read()

        if frame3 is None:
            break

        deteccion_algoritmo_orb(frame3, detector, train_mods, train_kp_angles, train_descriptors, train_sizes,
                                train_orientations)

        if cv2.waitKey(10) == 27:
            break

    while True:
        ret4, frame4 = video2.read()

        if frame4 is None:
            break

        deteccion_algoritmo_orb(frame4, detector, train_mods, train_kp_angles, train_descriptors, train_sizes,
                                train_orientations)

        if cv2.waitKey(10) == 27:
            break


if __name__ == "__main__":
    main()

import numpy as np
import cv2

def get_top_k_objects_polygon(
    segment_img, origin_x, origin_y, im_width, im_height, IDX2CLASS, max_n_objects=1
):
    MY_LABELS = {'small_Vehicle': 'car', 'large_Vehicle': 'car', 'plane': 'plane', 'storage_tank': 'building'}
    vals = list(set(segment_img.flatten()))
    vals = [x for x in vals if x != 0]
    if len(vals) == 0:
        return [{
            "label": "",
            "shape_type": "",
            "points": [],
        }]

    # find largest object

    all_labels = []
    all_contours = []
    all_contours_areas = []
    for val in vals:
        mask = np.zeros_like(segment_img, dtype=np.uint8)
        # smooth image
        mask[segment_img == val] = 255
        # kernel = np.ones((7, 7), np.float32) / 49
        # mask = cv2.filter2D(mask, -1, kernel)
        # mask[mask >= 127] = 255
        # mask[mask < 127] = 0
        # mask[mask == 255] = 1
        # kernel = np.ones((3, 3), np.uint8)
        # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # end smooth image

        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # contours = [x for x in contours if cv2.contourArea(x) > 20]
        contours_areas = [cv2.contourArea(x) for x in contours]
        all_contours.extend(contours)
        all_contours_areas.extend(contours_areas)
        all_labels.extend([val] * len(contours))

    # sort objects by areas (number of pixels of each object)
    sorted_inds = np.argsort(all_contours_areas)[::-1][:max_n_objects]
    selected_contours = [all_contours[i] for i in sorted_inds]
    selected_labels = [all_labels[i] for i in sorted_inds]

    # Iterate over contours, smooth and draw polygons
    all_boxes = []
    for contour, label in zip(selected_contours, selected_labels):
        # smooth contour
        contour = contour.reshape(-1, 2)
        new_contour = [contour[0]]
        for i in range(1, len(contour)):
            prev_pt = new_contour[-1]
            dist = np.sqrt(np.sum((contour[i] - prev_pt) ** 2))
            if dist > 10:
                new_contour.append(contour[i])
        if len(new_contour) >= 3:
            # filter too small objects which is often due to noisy
            contour = new_contour

        box = {

            "label": MY_LABELS[IDX2CLASS[label]],
            "shape_type": "polygon",
            "points": [
                [
                    int(origin_x + int(x * im_width / mask.shape[1])),
                    int(origin_y + int(y * im_height / mask.shape[0])),
                ]
                for x, y in contour
            ],
        }
        all_boxes.append(box)

    return all_boxes

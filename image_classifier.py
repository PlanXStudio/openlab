import cv2
import numpy as np
import os, glob, sys


def clamp(value):
    if value < 0:
        return 0
    elif value > 300:
        return 300
    else:
        return value


def extract_timestamp(filename):
    # data_line = os.path.basename(filename).split("_")[2].split(" ")[0].replace("-", "")
    time_line = os.path.basename(filename).split(" ")[1].split(".")[0:2]
    time_line = "".join(time_line).replace("_", "")
    # timestamp = int(data_line + time_line)
    return time_line


def click_event(event, x, y, flags, params):
    global count, _count, masked_img
    global width, height
    global goal_x, goal_y

    if event == cv2.EVENT_LBUTTONDOWN:
        if 130 < y < 280:
            if count < 2:
                image_tmp = cv2.circle(masked_img, (x, 260), 5, (0, 255, 0), 2)
                cv2.imshow("frame", image_tmp)
                count += 1
                coord_x.append(x)

                if count == 2:
                    mean_x = int(sum(coord_x) / len(coord_x))
                    goal_x = int((2 * mean_x) - (width / 2))
                    goal_x = clamp(goal_x)

                    image_tmp = cv2.circle(masked_img, (goal_x, 205), 5, (255, 255, 0), 2)
                    cv2.imshow("frame", image_tmp)
    # elif event == cv2.EVENT_RBUTTONDOWN:
    #     if 130 < y < 280:
    #         if _count < 2:
    #             image_tmp = cv2.circle(masked_img, (x, y), 5, (125, 125, 125), 2)
    #             cv2.imshow("frame", image_tmp)
    #             _count += 1
    #             _coord_x.append(x)

    #             if _count == 2:
    #                 mean_x = int(sum(_coord_x) / len(_coord_x))
    #                 goal_y = int((2 * mean_x) - (width / 2))
    #                 goal_y = clamp(goal_y)

    #                 image_tmp = cv2.circle(masked_img, (goal_y, mean_y), 5, (0, 255, 0), 2)
    #                 cv2.imshow("frame", image_tmp)


if __name__ == "__main__":
    print("############Image Classifier Instrunction############")
    print("a: Auto save on/off \ts: Save \td: Delete \nz: Previous Frame \tx: Reset Frame \tc: Next Frame")
    print("#####################################################")
    file_list = sorted(glob.glob("track_dataset/*.jpg"), key=extract_timestamp)
    auto_save = False
    full_page = len(file_list)
    dash_length = 5
    prop = 0
    page = 0

    print("Current page is (%d / %d)" % (page + 1, full_page))

    while page < full_page:
        try:
            image = cv2.imread(file_list[page])
            coord_x = []
            fSave = False
            count = 0
            _count = 0

            height = image.shape[0]
            width = image.shape[1]

            mask = np.zeros(image.shape, dtype=np.uint8)
            mask[130:280, :] = mask[130:280, :] + image[130:280, :]
            masked_img = cv2.addWeighted(image, 0.3, mask, 1, 0)

            masked_img = cv2.line(masked_img, (0, 130), (300, 130), (0, 0, 255), 1)
            masked_img = cv2.line(masked_img, (0, 280), (300, 280), (0, 0, 255), 1)

            current_length = 0

            while current_length < 300:
                start = (current_length, 260)
                end = ((current_length + dash_length), 260)
                cv2.line(masked_img, start, end, (0, 255, 0), 1)
                current_length = current_length + (dash_length * 3)

            cv2.namedWindow("frame")
            cv2.imshow("frame", masked_img)
            cv2.setMouseCallback("frame", click_event)

            while True:
                prop = cv2.getWindowProperty("frame", cv2.WND_PROP_AUTOSIZE)
                input = cv2.waitKeyEx(10)

                if prop < 0:
                    break

                if input in [27, 97, 99, 100, 115, 120, 122, 2424832, 2555904]:
                    if input == 115 or input == 97:
                        if input == 97:
                            if auto_save:
                                auto_save = False
                            else:
                                auto_save = True
                        elif input == 115:
                            if count >= 2:
                                cv2.destroyAllWindows()
                                fSave = True
                                page = page + 1
                                if page >= full_page:
                                    print("That was the last picture. The process is ceased.")
                                else:
                                    print("Current page is (%d / %d)" % (page + 1, full_page))
                                break
                            else:
                                print("Set two points before you save")
                    else:
                        cv2.destroyAllWindows()

                        if input == 100:
                            os.remove(file_list[page])
                            print(file_list[page] + " is deleted")
                        elif input == 2424832 or input == 122:
                            if page > 0:
                                page = page - 1
                                print("Current page is (%d / %d)" % (page + 1, full_page))
                            else:
                                print("This is the first picture")
                                page = page
                        elif input == 2555904 or input == 99:
                            if page < (full_page - 1):
                                page = page + 1
                                print("Current page is (%d / %d)" % (page + 1, full_page))
                            else:
                                print("This is the last picture")
                                page = page
                        elif input == 27:
                            page = full_page + 1
                            sys.exit()
                        break

                if auto_save:
                    if count >= 2:
                        cv2.destroyAllWindows()
                        fSave = True
                        page = page + 1
                        if page >= full_page:
                            print("That was the last picture. The process is ceased.")
                        else:
                            print("Current page is (%d / %d)" % (page + 1, full_page))
                        break
            try:
                if fSave:
                    components = file_list[page].split("_")[3:6]
                    components = components[0] + "_" + components[1] + "_" + components[2]
                    new_directory = "new_dataset/"

                    if _count == 2:
                        new_filename = str(goal_x) + "_" + str(goal_y) + "_" + components
                    else:
                        new_filename = str(goal_x) + "_" + str(goal_x) + "_" + components

                    destination_file = os.path.join(new_directory, new_filename)

                    os.makedirs(os.path.dirname(destination_file), exist_ok=True)

                    cv2.imwrite(destination_file, image)
            except Exception as e:
                pass

            if prop < 0:
                break
        except:
            if len(glob.glob("track_dataset/*.jpg")) < full_page:
                file_list = glob.glob("track_dataset/*.jpg")
                full_page = len(file_list)

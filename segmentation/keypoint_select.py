import numpy as np
import cv2


class ObjectManualSegmentation:
    def __init__(self, n_cameras=3):
        self._paused = False
        self.num_cameras = n_cameras

    @staticmethod
    def draw_reticle(img, u, v, label_color):
        """
        Draws a reticle on the image at the given (u,v) position

        :param img:
        :type img:
        :param u:
        :type u:
        :param v:
        :type v:
        :param label_color:
        :type label_color:
        :return:
        :rtype:
        """
        # cast to int
        u = int(u)
        v = int(v)

        white = (255, 255, 255)
        cv2.circle(img, (u, v), 10, label_color, 1)
        cv2.circle(img, (u, v), 11, white, 1)
        cv2.circle(img, (u, v), 12, label_color, 1)
        cv2.line(img, (u, v + 1), (u, v + 3), white, 1)
        cv2.line(img, (u + 1, v), (u + 3, v), white, 1)
        cv2.line(img, (u, v - 1), (u, v - 3), white, 1)
        cv2.line(img, (u - 1, v), (u - 3, v), white, 1)

    @staticmethod
    def draw_reticles(img,
              u_vec,
              v_vec,
              label_color=None,
              label_color_list=None,
              ):
        # draws multiple reticles
        n = len(u_vec)
        for i in range(n):
            u = u_vec[i]
            v = v_vec[i]

            color = None
            if label_color is not None:
                color = label_color
            else:
                color = label_color_list[i]

            draw_reticle(img, u, v, color)

    def set_keypoints_mouse_callback(self, event, u, v, flags, param):
        if self._paused:
            return

        current_data = self._source_data[self.current_idx]

        selected_pt = None
        if self.drawn_src_image is None:
            self.drawn_src_image = cv2.cvtColor(np.copy(self._source_data[self.current_idx]['rgb']), cv2.COLOR_RGB2BGR)

        # # draw all the keypoints we have already drawn 
        for i, uv in enumerate(self._source_data[self.current_idx]['uv']):
            pu, pv = uv[0], uv[1]
            cv2.circle(self.drawn_src_image, (pu, pv), radius=3, color=(0, 0, 255), thickness=-1)

            # print keypoint name next to point 
            text = str(i)
            H, W = self.drawn_src_image.shape[:-1]
            org = (pu - int(0.05*W)), (pv - int(0.05*H))
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (0, 255, 0) # green
            thickness = 2 # pixels
            cv2.putText(self.drawn_src_image, text, org, font, fontScale, color, thickness)


        # check to see if the left mouse button was released
        if event == cv2.EVENT_LBUTTONDOWN:
            selected_pt = (u, v)
            selected_pt_world_3d = current_data['pcd_world_img'][v, u]
            self.keypoints_list[self.current_idx].append(selected_pt)
            self.points_3d_list[self.current_idx].append(selected_pt_world_3d)

            current_data['uv'].append(selected_pt)

            print('3D value at selected point: ', selected_pt_world_3d)

            cv2.circle(self.drawn_src_image, (u, v), radius=3, color=(0, 0, 255), thickness=-1)

        drawn_image = np.copy(self.drawn_src_image)
        self.draw_reticle(drawn_image, u, v, [0, 255, 0])
        cv2.imshow('source', drawn_image)

        self.drawn_src_image = None

    def set_source_data_keypoints(self, data, n_pts=1, *args, **kwargs):
        self._source_data = data  # list of dicts with keys: 'rgb', 'pcd_world_img'

        self.current_idx = 0

        self.drawn_src_image = None
        self.data_list = []
        self.keypoints_list = []
        self.points_3d_list = []

        self._source_data[self.current_idx]['uv'] = []
        self.keypoints_list.append([])
        self.points_3d_list.append([])

        self.set_keypoints_mouse_callback(None, 0, 0, None, None)

        cv2.setMouseCallback('source', self.set_keypoints_mouse_callback)

        n_cams = self.num_cameras
        print('Current source index: %d' % self.current_idx)
        self.uv_name = str(0)
        while True:
            k = cv2.waitKey(20) & 0xFF
            val = (k - 255)/1000.0
            if k == 27:
                break
            elif k == ord('n'):
                self.current_idx += 1
                if self.current_idx >= len(self._source_data):
                    print('Done!')
                    break
                self._source_data[self.current_idx]['uv'] = []
                self.keypoints_list.append([])
                self.points_3d_list.append([])
                self.drawn_src_image = None
                print('Current source index: %d' % self.current_idx)
            elif k == ord('b'):
                self.current_idx -= 1
                if self.current_idx < 0:
                    self.current_idx = 0
                self.drawn_src_image = None
                print('Current source index: %d' % self.current_idx)
            else:
                pass

        cv2.destroyAllWindows()

        return -1


def main(args):
    manual_seg = ObjectManualSegmentation()
    manual_seg.set_source_data_keypoints(source_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()
    main(args)

import cv2
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from line_profiler import profile

class PoseDetector:
    @profile
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=True)
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=2, circle_radius=2)
    
    @profile
    def detect_poses(self, person_img_path, output_path):
        # Load person image
        image = cv2.imread(person_img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process with MediaPipe
        results = self.pose.process(image_rgb)

        # Create transparent image
        transparent = Image.new("RGBA", (image.shape[1], image.shape[0]), (0, 0, 0, 0))

        if results.pose_landmarks:
            # Create blank canvas to draw skeleton
            canvas = image.copy()
            canvas[:] = (0, 0, 0)

            # Draw landmarks and connections
            self.mp_drawing.draw_landmarks(
                canvas,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.drawing_spec,
                connection_drawing_spec=self.drawing_spec
            )

            # Convert to RGBA (with transparency)
            canvas_rgba = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGBA)
            # Make black background transparent
            canvas_rgba[(canvas_rgba[:, :, :3] == [0, 0, 0]).all(axis=2)] = (0, 0, 0, 0)

            # Save result
            result_img = Image.fromarray(canvas_rgba)
            result_img.save(output_path)

        else:
            print("No pose detected!")        
        return None
    
    @profile
    def detect_pose_on_half(self, image_half):
        image_rgb = cv2.cvtColor(image_half, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        # Blank canvas for skeleton
        skeleton = np.zeros_like(image_half)

        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                skeleton,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.drawing_spec,
                connection_drawing_spec=self.drawing_spec
            )

        return skeleton
    
    @profile
    def process_dual_pose(self, input_path, output_path):
        # Load full image
        image = cv2.imread(input_path)
        h, w, _ = image.shape
        mid = w // 2

        left_half = image[:, :mid]
        right_half = image[:, mid:]

        # Detect poses on both halves
        left_skeleton = self.detect_pose_on_half(left_half)
        right_skeleton = self.detect_pose_on_half(right_half)

        # Combine back into one full-size image
        combined = np.zeros_like(image)
        combined[:, :mid] = left_skeleton
        combined[:, mid:] = right_skeleton

        # Convert to RGBA and make black transparent
        combined_rgba = cv2.cvtColor(combined, cv2.COLOR_BGR2RGBA)
        combined_rgba[(combined_rgba[:, :, :3] == [0, 0, 0]).all(axis=2)] = (0, 0, 0, 0)

        # Save final image
        result_img = Image.fromarray(combined_rgba)
        result_img.save(output_path)    
    
    @profile
    def process_dual_pose_overlay(self, input_path, output_skeleton_path, output_overlay_path):
        # Load full image
        image = cv2.imread(input_path)
        h, w, _ = image.shape
        mid = w // 2

        left_half = image[:, :mid]
        right_half = image[:, mid:]

        # Detect skeletons
        left_skeleton = self.detect_pose_on_half(left_half)
        right_skeleton = self.detect_pose_on_half(right_half)

        # Combine skeletons
        skeleton_full = np.zeros_like(image)
        skeleton_full[:, :mid] = left_skeleton
        skeleton_full[:, mid:] = right_skeleton

        # Transparent skeleton output
        skeleton_rgba = cv2.cvtColor(skeleton_full, cv2.COLOR_BGR2RGBA)
        skeleton_rgba[(skeleton_rgba[:, :, :3] == [0, 0, 0]).all(axis=2)] = (0, 0, 0, 0)
        Image.fromarray(skeleton_rgba).save(output_skeleton_path)

        # Overlay skeletons on original image
        overlay_bgr = cv2.addWeighted(image, 1.0, skeleton_full, 1.0, 0)
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
        Image.fromarray(overlay_rgb).save(output_overlay_path)        
        
    @profile
    def detect_pose_on_half_distance(self, image_half, x_offset=0):
        image_rgb = cv2.cvtColor(image_half, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        skeleton = np.zeros_like(image_half)
        x_coords = []

        if results.pose_landmarks:
            # Draw skeleton
            self.mp_drawing.draw_landmarks(
                skeleton,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.drawing_spec,
                connection_drawing_spec=self.drawing_spec
            )

            # Collect x coordinates (normalized)
            h, w, _ = image_half.shape
            for lm in results.pose_landmarks.landmark:
                if lm.visibility > 0.5:  # filter for visible landmarks
                    x_coords.append(int(lm.x * w) + x_offset)

        return skeleton, x_coords        
    
    @profile
    def process_dual_pose_distance(self, image_or_filename, output_skeleton_path, output_overlay_path):                
        # Load full image
        
        image = None
        
        if isinstance(image_or_filename, str):
            image = cv2.imread(image_or_filename)
        else:
            image = image_or_filename
            
        h, w, _ = image.shape
        mid = w // 2
        extended_mid = min(mid + 230, w)
        
        left_half = image[:, :mid]
        right_half = image[:, mid:]

        # Detect skeletons and landmark positions
        left_skeleton, left_xs = self.detect_pose_on_half_distance(left_half, x_offset=0)
        # Retry with extended left half if no landmarks found
                
        if len(left_xs) < 20:
            print("not left_xs")
            extended_left_half = image[:, :extended_mid]
            left_skeleton, left_xs = self.detect_pose_on_half_distance(extended_left_half, x_offset=0)

            # Resize or crop skeleton back to original left_half width
            left_skeleton = left_skeleton[:, :mid]  # ensure it fits the left half


        right_skeleton, right_xs = self.detect_pose_on_half_distance(right_half, x_offset=mid)
        if len(right_xs) == 0:
            print("not right_xs")
            
        # Combine skeletons
        skeleton_full = np.zeros_like(image)
        skeleton_full[:, :mid] = left_skeleton
        skeleton_full[:, mid:] = right_skeleton

        # Convert to RGBA with transparency
        skeleton_rgba = cv2.cvtColor(skeleton_full, cv2.COLOR_BGR2RGBA)
        skeleton_rgba[(skeleton_rgba[:, :, :3] == [0, 0, 0]).all(axis=2)] = (0, 0, 0, 0)

        text = None
        text_position = None
        # Draw pixel distance if both players detected
        if left_xs and right_xs:
            left_max_x = max(left_xs)
            right_min_x = min(right_xs)
            pixel_distance = right_min_x - left_max_x

            # Draw on transparent image using PIL
            img_pil = Image.fromarray(skeleton_rgba)
            draw = ImageDraw.Draw(img_pil)
            if (pixel_distance < 110):
                text = f"{pixel_distance}px short range"
            elif (425 <= pixel_distance <= 475):
                text = f"{pixel_distance}px round start range"                
            elif (pixel_distance > 500):
                text = f"{pixel_distance}px long range"
            else:
                text = f"{pixel_distance}px mid range"
                
            text_position = (w // 2 - 100, 200)

            # Optional: use truetype font if you want better styling
            try:
                font = ImageFont.truetype("arial.ttf", 68)
            except:
                font = ImageFont.load_default()

            draw.text(text_position, text, fill=(255, 255, 255, 255), font=font)
            skeleton_rgba = np.array(img_pil)

        # Save skeleton image with transparency and text
        Image.fromarray(skeleton_rgba).save(output_skeleton_path)

        #Also generate overlay image
        overlay_bgr = cv2.addWeighted(image, 1.0, skeleton_full, 1.0, 0)
        if left_xs and right_xs:
            color_outline = (0,0,0)
            thickness = 5
            scale = 4
            text_position = (w // 2 - 300, 200)
            
            for dx in [-5, 0, 5]:
                for dy in [-5, 0, 5]:
                    if dx != 0 or dy != 0:
                        pos = (text_position[0] + dx, text_position[1] + dy)
                        cv2.putText(overlay_bgr, text, pos, cv2.FONT_HERSHEY_PLAIN, scale, color_outline, thickness, cv2.LINE_AA)
                        
            cv2.putText(
                overlay_bgr,
                text,
                text_position,
                cv2.FONT_HERSHEY_PLAIN,
                scale,               # font scale
                (255, 255, 255),   # text color (white)
                thickness,                 # thickness
                cv2.LINE_AA
            )

        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
            
        Image.fromarray(overlay_rgb).save(output_overlay_path)        
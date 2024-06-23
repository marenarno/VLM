import cv2
import numpy as np

def display_text_window(frame, response, font_color=(0, 0, 0), background_color=(255, 255, 255)):
    # Scale factor for creating a smaller window
    scale_factor = 0.5
    high_res_factor = 2  # High-resolution scale factor for text quality

    # Create a high-resolution blank image for the text
    high_res_width = int(frame.shape[1] * scale_factor * high_res_factor)
    high_res_height = int(frame.shape[0] * scale_factor * high_res_factor)
    text_img = np.zeros((high_res_height, high_res_width, 3), np.uint8)
    text_img[:] = background_color

    # Set font parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0  # Font scale for high resolution
    line_type = 2

    # Add "Assistant: " to the response
    response = "Assistant: " + response

    # Split the response into lines that fit within the text box
    max_width = high_res_width - 40  # Adjust for high-resolution padding
    words = response.split()
    lines = []
    current_line = ""

    for word in words:
        # Check the width of the current line plus the new word
        if cv2.getTextSize(current_line + word, font, font_scale, line_type)[0][0] <= max_width:
            current_line += word + " "
        else:
            lines.append(current_line)
            current_line = word + " "

    # Add the last line
    lines.append(current_line)

    # Calculate the total height of the text
    line_height = cv2.getTextSize("Tg", font, font_scale, line_type)[0][1] + 10
    total_text_height = len(lines) * line_height

    # Calculate the starting y coordinate to center the text
    text_y = (high_res_height - total_text_height) // 2 + line_height

    # Draw the text on the high-resolution text image
    for line in lines:
        cv2.putText(text_img, line, (20, text_y), font, font_scale, font_color, line_type)
        text_y += line_height

    # Resize the text image back down to the desired smaller size
    text_img = cv2.resize(text_img, (int(frame.shape[1] * scale_factor), int(frame.shape[0] * scale_factor)), interpolation=cv2.INTER_AREA)

    return text_img

# # Testing
# frame_path = '../frames/frames_97s/frame_0007.jpg'
# response = "Please sit down on a chair with your back supported and your feet flat on the floor."

# # Read the frame
# frame = cv2.imread(frame_path)
# if frame is None:
#     print("Error: Could not load image.")
# else:
#     text_window = display_text_window(frame, response)

#     # Resize frame to match the new text window size while maintaining aspect ratio
#     new_frame_size = (text_window.shape[1], text_window.shape[0])
#     resized_frame = cv2.resize(frame, new_frame_size)

#     # Show the frame and the text window side by side
#     combined_img = np.hstack((resized_frame, text_window))
#     cv2.imshow('Frame and Text', combined_img)
#     cv2.waitKey(0)  # Wait for a key press to close the window
#     cv2.destroyAllWindows()


import cv2

# Convert image to grayscale
def convert_image_to_grayscale(input_img_path, output_img_path):
    # Read the input image
    img = cv2.imread(input_img_path)

    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Save the grayscale image
    cv2.imwrite(output_img_path, gray_img)

if __name__ == "__main__":
    input_img_path = r"images/sf2screen_fight.png"
    output_img_path = r"images/sf2screen_fight_gray.png"
    convert_image_to_grayscale(input_img_path, output_img_path)